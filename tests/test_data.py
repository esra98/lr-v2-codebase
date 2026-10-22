import os
import random

import pytest
import torch

from avsr.data.dataset import AVDataset, BucketBatchSampler, collate, split_batch
from avsr.data.manifest import (
    FilterConfig,
    Row,
    is_hallucinated_phrase,
    judge_label,
    kept_rows,
    read_clip_table,
    read_manifest,
    write_clip_table,
    write_manifest,
)
from avsr.data.recurrence import (
    BURST,
    LOOP,
    SENTENCE,
    RecurrenceConfig,
    find_recurrent,
    has_internal_loop,
    similar,
)
from avsr.data.transforms import (
    AdaptiveTimeMask,
    AddNoise,
    AudioTransform,
    CenterCrop,
    PartitionTimeMask,
    RandomCropFlip,
    VideoTransform,
    speed_perturb,
    to_grayscale,
)
from avsr.text.labels import Utterance, Word
from avsr.text.normalize import normalize_text
from avsr.text.tokenizer import SentencePieceTokenizer

# ---------------------------------------------------------------- transforms


def test_time_mask_keeps_shape_and_masks_something():
    random.seed(0)
    x = torch.ones(100, 1, 8, 8)
    out = AdaptiveTimeMask(max_len=10, stride=25)(x)
    assert out.shape == x.shape
    masked = (out.flatten(1).sum(1) == 0).sum().item()
    assert 0 < masked <= 40
    assert x.min() == 1  # input untouched


def test_time_mask_never_wipes_a_short_clip():
    for seed in range(50):
        random.seed(seed)
        out = AdaptiveTimeMask(max_len=10, stride=25)(torch.ones(12, 1, 4, 4))
        assert (out.flatten(1).sum(1) > 0).sum() >= 6


@pytest.mark.parametrize("length", [0, 1])
def test_time_mask_degenerate_lengths(length):
    x = torch.ones(length, 1, 4, 4)
    assert AdaptiveTimeMask(10, 25)(x).shape == x.shape


def test_ptm_masks_two_cells_in_a_fifth_of_the_frames():
    random.seed(0)
    x = torch.ones(50, 1, 88, 88)
    out = PartitionTimeMask(grid=4, alpha=0.2, num_partitions=2)(x)
    assert x.min() == 1
    touched = [t for t in range(50) if out[t].min() == 0]
    assert len(touched) == 10
    assert touched == list(range(touched[0], touched[0] + 10))  # one contiguous span
    for t in touched:
        assert (out[t] == 0).sum().item() == 2 * 22 * 22


def test_ptm_cells_change_between_frames():
    random.seed(1)
    out = PartitionTimeMask(grid=4, alpha=1.0, num_partitions=2)(torch.ones(30, 1, 88, 88))
    patterns = {tuple((out[t, 0] == 0).flatten().tolist()) for t in range(30)}
    assert len(patterns) > 5


def test_ptm_mean_fill_and_grid_that_does_not_divide_the_frame():
    random.seed(0)
    x = torch.rand(20, 1, 90, 90)
    out = PartitionTimeMask(grid=4, alpha=0.5, num_partitions=3, fill="mean")(x)
    assert out.shape == x.shape
    assert torch.isclose(out, x.mean()).sum() > 0


def test_ptm_probability_zero_is_identity():
    x = torch.rand(20, 1, 88, 88)
    assert torch.equal(PartitionTimeMask(p=0.0)(x), x)


def test_ptm_with_its_own_generator_is_reproducible():
    x = torch.rand(40, 3, 96, 96)
    mask = PartitionTimeMask()
    a = mask(x, random.Random(5))
    random.seed(123)  # the module-level generator must not matter
    b = mask(x, random.Random(5))
    assert torch.equal(a, b)
    assert not torch.equal(a, mask(x, random.Random(6)))


def test_ptm_single_frame_clip():
    out = PartitionTimeMask()(torch.ones(1, 1, 88, 88))
    assert out.shape == (1, 1, 88, 88)


@pytest.mark.parametrize("kwargs", [{"num_partitions": 0}, {"num_partitions": 17}, {"alpha": 0.0}, {"alpha": 1.5}])
def test_ptm_rejects_bad_parameters(kwargs):
    with pytest.raises(ValueError):
        PartitionTimeMask(**kwargs)


def test_crop_is_the_same_for_every_frame():
    random.seed(0)
    frame = torch.arange(96 * 96, dtype=torch.float32).reshape(1, 1, 96, 96)
    out = RandomCropFlip(88)(frame.repeat(10, 1, 1, 1))
    assert out.shape == (10, 1, 88, 88)
    assert all(torch.equal(out[0], out[t]) for t in range(10))


def test_center_crop_and_too_small_input():
    assert CenterCrop(88)(torch.zeros(3, 1, 96, 96)).shape == (3, 1, 88, 88)
    with pytest.raises(ValueError):
        CenterCrop(88)(torch.zeros(3, 1, 80, 80))


def test_crop_size_zero_keeps_the_stored_frames():
    random.seed(0)
    x = torch.rand(5, 1, 98, 98)
    assert CenterCrop(0)(x) is x
    out = RandomCropFlip(0, flip_p=1.0)(x)
    assert torch.equal(out, x.flip(-1))


def test_crop_larger_than_the_stored_frames_explains_the_fix():
    with pytest.raises(ValueError, match="set it to 0"):
        RandomCropFlip(100)(torch.zeros(3, 1, 96, 96))


def test_to_grayscale():
    video = torch.zeros(2, 4, 4, 3, dtype=torch.uint8)
    video[..., 1] = 255
    grey = to_grayscale(video)
    assert grey.shape == (2, 1, 4, 4)
    assert torch.allclose(grey, torch.full_like(grey, 0.587))


@pytest.mark.parametrize("factor", [0.9, 1.0, 1.1])
def test_speed_perturb_keeps_audio_and_video_in_sync(factor):
    video, audio = torch.rand(50, 1, 8, 8), torch.rand(50 * 640, 1)
    v, a = speed_perturb(video, audio, factor)
    assert v.size(0) == round(50 / factor)
    assert a.size(0) == v.size(0) * 640
    assert v.shape[1:] == video.shape[1:]


def test_speed_perturb_single_modality():
    v, a = speed_perturb(torch.rand(50, 1, 8, 8), None, 1.1)
    assert a is None and v.size(0) == 45
    v, a = speed_perturb(None, torch.rand(50 * 640, 1), 0.9)
    assert v is None and a.size(0) == 56 * 640


def test_add_noise_hits_the_requested_snr():
    random.seed(0)
    torch.manual_seed(0)
    speech = torch.randn(16000, 1)
    noisy = AddNoise(snr_levels=[10])(speech)
    noise = noisy - speech
    snr = 10 * torch.log10(speech.pow(2).mean() / noise.pow(2).mean())
    assert abs(snr.item() - 10) < 0.1


def test_add_noise_clean_level_and_silence():
    speech = torch.randn(1600, 1)
    assert torch.equal(AddNoise(snr_levels=[None])(speech), speech)
    silence = torch.zeros(1600, 1)
    assert torch.equal(AddNoise(snr_levels=[0])(silence), silence)


def test_video_transform_train_and_eval(toy_cfg):
    cfg = toy_cfg()
    x = torch.rand(40, 1, 96, 96)
    assert VideoTransform("train", cfg.augment.video)(x).shape == (40, 1, 88, 88)
    a = VideoTransform("val", cfg.augment.video)(x)
    b = VideoTransform("val", cfg.augment.video)(x)
    assert torch.equal(a, b)  # evaluation is deterministic


def test_audio_transform_normalises(toy_cfg):
    out = AudioTransform("val", toy_cfg().augment.audio)(torch.randn(6400, 1) * 0.01 + 3)
    assert abs(out.mean().item()) < 1e-4 and abs(out.std().item() - 1) < 1e-3
    silent = AudioTransform("val", toy_cfg().augment.audio)(torch.zeros(6400, 1))
    assert torch.isfinite(silent).all()


# ------------------------------------------------------------------- dataset


@pytest.fixture
def tokenizer(toy_data):
    return SentencePieceTokenizer(toy_data["spm"])


def _dataset(toy_data, tokenizer, modality, subset="train", **kwargs):
    return AVDataset(
        os.path.join(toy_data["manifests"], f"{subset}.tsv"),
        toy_data["processed"],
        modality,
        tokenizer,
        **kwargs,
    )


def test_grey_from_the_brightness_plane_matches_grey_from_rgb(toy_data):
    from avsr.preprocess import video_io

    path = os.path.join(toy_data["processed"], "s1", "00003.mp4")
    direct = torch.from_numpy(video_io.read_video_gray(path))
    via_rgb = to_grayscale(torch.from_numpy(video_io.read_video(path))).squeeze(1)
    assert direct.shape == via_rgb.shape and direct.dtype == torch.float32
    assert 0.0 <= direct.min() and direct.max() <= 1.0
    assert (direct - via_rgb).abs().mean() < 2 / 255
    assert direct.std() > 0.1  # a real picture, not a constant


@pytest.mark.parametrize("modality", ["video", "audio", "audiovisual"])
def test_dataset_items(toy_data, tokenizer, modality):
    ds = _dataset(toy_data, tokenizer, modality)
    item = ds[0]
    frames = ds.rows[0].num_frames
    assert ("video" in item) == (modality != "audio")
    assert ("audio" in item) == (modality != "video")
    if "video" in item:
        assert item["video"].shape == (frames, 1, 96, 96)
    if "audio" in item:
        assert item["audio"].shape == (frames * 640, 1)
    assert item["target"].dtype == torch.long


def test_dataset_rejects_unknown_modality(toy_data, tokenizer):
    with pytest.raises(ValueError):
        _dataset(toy_data, tokenizer, "lips")


def test_dataset_missing_file_returns_none(toy_data, tokenizer):
    ds = _dataset(toy_data, tokenizer, "video")
    ds.rows[0].clip_id = "does_not_exist"
    assert ds[0] is None
    batch = collate([ds[0], ds[1]])
    assert len(batch["utt_ids"]) == 1
    assert collate([None, None]) is None


def test_dataset_drops_utterances_with_more_tokens_than_frames(toy_data, tokenizer, tmp_path):
    rows = read_manifest(os.path.join(toy_data["manifests"], "train.tsv"))
    rows[0].num_frames = 2
    path = str(tmp_path / "train.tsv")
    write_manifest(path, rows)
    ds = AVDataset(path, toy_data["processed"], "video", tokenizer)
    assert len(ds) == len(rows) - 1


def test_dataset_limit_and_max_frames(toy_data, tokenizer):
    assert len(_dataset(toy_data, tokenizer, "video", limit=5)) == 5
    ds = _dataset(toy_data, tokenizer, "video", max_frames=45)
    assert max(ds.lengths) <= 45


def test_collate_pads_and_reports_lengths(toy_data, tokenizer):
    ds = _dataset(toy_data, tokenizer, "audiovisual")
    batch = collate([ds[i] for i in range(4)])
    lengths = batch["video_lengths"]
    assert batch["video"].shape[:2] == (4, int(lengths.max()))
    assert torch.equal(batch["audio_lengths"], lengths * 640)
    assert (batch["targets"] == -1).any() or len(set(batch["target_lengths"].tolist())) == 1
    shortest = int(lengths.argmin())
    assert batch["video"][shortest, int(lengths[shortest]) :].abs().sum() == 0


def test_split_batch_keeps_every_clip_and_trims_padding(toy_data, tokenizer):
    ds = _dataset(toy_data, tokenizer, "audiovisual")
    batch = collate([ds[i] for i in range(5)])
    first, second = split_batch(batch)
    assert first["utt_ids"] + second["utt_ids"] == batch["utt_ids"]
    for half, part in ((first, slice(0, 2)), (second, slice(2, 5))):
        lengths = batch["video_lengths"][part]
        assert torch.equal(half["video_lengths"], lengths)
        assert half["video"].shape[:2] == (len(lengths), int(lengths.max()))
        assert half["audio"].shape[1] == int(lengths.max()) * 640
        assert half["targets"].shape[1] == int(half["target_lengths"].max())
        for k, n in enumerate(lengths):
            assert torch.equal(half["video"][k, :n], batch["video"][part][k, :n])
    with pytest.raises(ValueError):
        split_batch(first if len(first["utt_ids"]) == 1 else collate([ds[0]]))


# ------------------------------------------------------------------- sampler


def test_sampler_covers_every_clip_once_and_respects_the_frame_budget():
    rng = random.Random(0)
    lengths = [rng.randint(12, 300) for _ in range(500)]
    sampler = BucketBatchSampler(lengths, max_frames=600, max_batch_size=16, num_buckets=10)
    batches = list(sampler)
    assert sorted(i for b in batches for i in b) == list(range(500))
    for batch in batches:
        assert max(lengths[i] for i in batch) * len(batch) <= 600
        assert len(batch) <= 16
    assert len(sampler) == len(batches)


def test_sampler_has_its_own_batches_for_every_epoch():
    lengths = list(range(20, 120))
    first = BucketBatchSampler(lengths, 600, num_buckets=5, seed=1, epoch=0)
    second = BucketBatchSampler(lengths, 600, num_buckets=5, seed=1, epoch=1)
    assert list(first) != list(second)
    assert list(first) == list(first)  # the batches of an epoch do not change
    assert list(BucketBatchSampler(lengths, 600, num_buckets=5, seed=1, epoch=0)) == list(first)
    assert list(BucketBatchSampler(lengths, 600, num_buckets=5, seed=2, epoch=0)) != list(first)
    first.set_epoch(1)
    assert list(first) == list(second)


def test_sampler_length_is_the_number_of_batches_of_that_epoch():
    """The count differs between epochs; a stale count makes a trainer skip validation."""
    rng = random.Random(3)
    lengths = [rng.randint(12, 250) for _ in range(3000)]
    counts = set()
    for epoch in range(12):
        sampler = BucketBatchSampler(lengths, max_frames=252, max_batch_size=16, num_buckets=50, seed=42, epoch=epoch)
        batches = list(sampler)
        assert len(sampler) == len(batches)
        assert sorted(i for b in batches for i in b) == list(range(3000))
        counts.add(len(batches))
    assert len(counts) > 1


def test_sampler_without_shuffle_is_stable():
    a = BucketBatchSampler(list(range(20, 120)), 600, shuffle=False, epoch=0)
    b = BucketBatchSampler(list(range(20, 120)), 600, shuffle=False, epoch=5)
    assert list(a) == list(b)


def test_sampler_rejects_clips_longer_than_the_budget():
    with pytest.raises(ValueError):
        BucketBatchSampler([100, 700], max_frames=600)
    with pytest.raises(ValueError):
        BucketBatchSampler([], max_frames=600)


# ------------------------------------------------------------------ manifest


def _utterance(text, conf=0.95, end=1.96, words=None):
    """An utterance with its words spread evenly over ``end`` seconds."""
    if words is None:
        parts = text.split()
        step = (end - 0.04) / len(parts)
        words = [(w, 0.04 + k * step, 0.04 + (k + 1) * step, conf) for k, w in enumerate(parts)]
    return Utterance(text, [Word(w, s, e, c) for w, s, e, c in words])


def _judge(session_texts, cfg=None):
    """Verdict of every utterance of one session, in order (None = kept)."""
    cfg = cfg or FilterConfig()
    ids = [f"{i:05d}" for i in range(len(session_texts))]
    texts = [normalize_text(u.text) for u in session_texts]
    recurrent = find_recurrent({"s1": list(zip(ids, texts))}, cfg.recurrence)
    return [judge_label(u, t, recurrent.get(("s1", i)), cfg) for u, t, i in zip(session_texts, texts, ids)]


def test_label_verdicts():
    squeezed = [("GENEL", 0.04, 1.9, 0.95), ("KURULU", 1.9, 1.92, 0.97), ("SELAMLIYORUM", 1.92, 1.96, 0.9)]
    utterances = [
        _utterance("SAYIN BAŞKAN TEŞEKKÜRLER"),
        _utterance("DEĞERLI ARKADAŞLAR", conf=0.4),
        _utterance("GENEL KURULU SELAMLIYORUM", words=squeezed),
        _utterance("...", words=[("...", 0.04, 1.9, 0.95)]),
        _utterance("ALTYAZI M.K."),
    ]
    assert _judge(utterances) == [None, "low_confidence", "squeezed_words", "empty_text", "hallucinated_phrase"]
    assert judge_label(None, "", None, FilterConfig()) == "bad_label_file"


def test_recurrent_clips_are_rejected_but_their_neighbours_are_kept():
    texts = ["BIR ŞEY SÖYLEYECEĞIM", "ABONE KALIN LÜTFEN", "ABONE KALIN LÜTFEN", "ABONE KALIN LÜTFEN", "BAŞKA BIR CÜMLE"]
    utterances = [_utterance(t) for t in texts]
    assert _judge(utterances) == [None, BURST, BURST, BURST, None]
    assert not any(_judge(utterances, FilterConfig(drop_recurrent=False)))


def _table_row(clip_id, verdict="kept", split="train", session="s1"):
    kept = verdict == "kept"
    return {
        "session": session,
        "clip_id": clip_id,
        "verdict": verdict,
        "split": split if kept else "",
        "masked": ("yes" if split == "train" else "no") if kept else "",
        "num_frames": 50 if kept else "",
        "mean_conf": "0.9100",
        "detection_rate": "1.000" if kept else "",
        "yaw": "0.1000" if kept else "",
        "mouth_motion": "0.01000" if kept else "",
        "text": "BIR ŞEY",
        "error": "",
    }


def test_clip_table_roundtrip_and_kept_rows(tmp_path):
    rows = [
        _table_row("00000"),
        _table_row("00001", verdict="recurrent_burst"),
        _table_row("00002", split="val"),
        _table_row("00003", split="test", session="s2"),
    ]
    path = str(tmp_path / "out" / "data.clips.tsv")
    write_clip_table(path, rows)
    table = read_clip_table(path)
    assert [r["verdict"] for r in table] == ["kept", "recurrent_burst", "kept", "kept"]
    assert [r.utt_id for r in kept_rows(table)] == ["s1/00000", "s1/00002", "s2/00003"]
    assert [r.utt_id for r in kept_rows(table, "val")] == ["s1/00002"]
    assert kept_rows(table, "train")[0].num_frames == 50


def test_read_clip_table_rejects_wrong_columns(tmp_path):
    path = tmp_path / "t.tsv"
    path.write_text("session\tclip_id\ns1\t00000\n", encoding="utf-8")
    with pytest.raises(ValueError):
        read_clip_table(str(path))


@pytest.mark.parametrize(
    "text, expected",
    [
        ("ABONE OL", True),
        ("ALTYAZI M K", True),
        ("LÜTFEN ABONE OL", True),
        ("IZLEDIĞINIZ IÇIN TEŞEKKÜR EDERIM", True),
        ("ELEKTRIĞE ABONE OLAN VATANDAŞLARIMIZ", False),  # OLAN is a different word
        ("VATANDAŞLAR DOĞAL GAZA ABONE OL DENILEREK YILLARCA BEKLETILDI", False),  # a small part of the sentence
        ("TEŞEKKÜR EDERIM", False),
        ("", False),
    ],
)
def test_hallucinated_phrase_must_dominate_the_transcript(text, expected):
    assert is_hallucinated_phrase(text) is expected


# ---------------------------------------------------------------- recurrence


def _session(texts):
    return [(f"{i:05d}", t) for i, t in enumerate(texts)]


def _distinct(n, tag=""):
    """Transcripts that share no word with each other."""
    return [f"{tag}A{i} {tag}B{i} {tag}C{i} {tag}D{i} {tag}E{i}" for i in range(n)]


def test_burst_of_identical_transcripts_is_flagged():
    texts = ["ILK CÜMLE BURADA"] + ["ABONE OL"] * 6 + ["SON CÜMLE BURADA"]
    flags = find_recurrent({"s": _session(texts)})
    assert {k[1] for k in flags} == {f"{i:05d}" for i in range(1, 7)}
    assert set(flags.values()) == {BURST}


def test_near_identical_transcripts_count_as_the_same():
    a = "BIR NISAN BIN DOKUZ YÜZ YIRMI BIRDE GIRESUNUZ"
    b = "BIR NISAN BIN DOKUZ YÜZ YIRMI BIRDE GIRESUNUMUZ"
    flags = find_recurrent({"s": _session(["BAŞKA BIR ŞEY", a, "ARADA FARKLI", b])})
    assert {k[1] for k in flags} == {"00001", "00003"}
    cfg = RecurrenceConfig()
    assert similar(a.split(), b.split(), cfg)
    # genuine neighbours that differ in one word are too short for fuzzy matching
    assert not similar("IKI BIN ON SEKIZ".split(), "IKI BIN ON DOKUZ".split(), cfg)
    assert not similar("IÇINDE INSAN YOK".split(), "IÇINDE TARIH YOK".split(), cfg)
    assert not similar("BU KANUN BIZIM IÇIN ÇOK ÖNEMLI".split(), "BU ÜLKE HEPIMIZ IÇIN ÇOK GÜZEL".split(), cfg)


def test_genuine_frequent_phrases_are_kept():
    """A formula that recurs far apart and in many sessions is real speech."""
    sessions = {}
    for s in range(5):
        texts = _distinct(60, tag=f"S{s}")
        for position in (3, 20, 41):
            texts[position] = "TEŞEKKÜR EDERIM SAYIN BAŞKAN"
        sessions[f"s{s}"] = _session(texts)
    assert find_recurrent(sessions) == {}


def test_sentence_that_mostly_occurs_in_bursts_is_rejected_everywhere():
    texts = _distinct(80)
    for position in (10, 11, 12, 13, 60):  # a burst, and one isolated occurrence
        texts[position] = "AKLINIZA GELEBILECEK HER TÜRLÜ"
    flags = find_recurrent({"s": _session(texts)})
    assert flags[("s", "00060")] == SENTENCE
    assert [flags[("s", f"{i:05d}")] for i in (10, 11, 12, 13)] == [BURST] * 4
    assert len(flags) == 5


def test_a_loop_in_one_session_does_not_condemn_the_sentence_elsewhere():
    looping = _distinct(40, tag="X")
    for position in range(5, 15):
        looping[position] = "TÜRKIYE BÜYÜK MILLET MECLISI"
    looping[30] = "TÜRKIYE BÜYÜK MILLET MECLISI"  # isolated, but in the looping session
    genuine = _distinct(40, tag="Y")
    genuine[7] = genuine[25] = "TÜRKIYE BÜYÜK MILLET MECLISI"
    flags = find_recurrent({"a": _session(looping), "b": _session(genuine)})
    assert {k for k in flags if k[0] == "b"} == set()
    assert flags[("a", "00030")] == SENTENCE
    assert len(flags) == 11


def test_bursts_do_not_cross_sessions():
    flags = find_recurrent({"a": _session(["X Y Z", "AYNI CÜMLE BURADA"]), "b": _session(["AYNI CÜMLE BURADA", "P Q R"])})
    assert flags == {}


def test_clips_outside_the_window_are_not_a_burst():
    texts = _distinct(20)
    texts[2] = texts[8] = "AYNI CÜMLE IKI KERE"
    assert find_recurrent({"s": _session(texts)}, RecurrenceConfig(window=5)) == {}
    assert len(find_recurrent({"s": _session(texts)}, RecurrenceConfig(window=6))) == 2


@pytest.mark.parametrize(
    "text, expected",
    [
        ("DA DA DA", True),
        ("BU DA DA DA DA OLDU", True),
        ("MECLIS MECLIS", False),
        ("ÇOK ÇOK TEŞEKKÜR EDERIM", False),
        ("VE BU VE BU VE BU", True),
        ("BIR IKI ÜÇ BIR IKI ÜÇ BIR IKI ÜÇ", True),
        ("BIR IKI ÜÇ DÖRT BEŞ BIR IKI ÜÇ DÖRT BEŞ", False),
        ("SAYIN BAŞKAN DEĞERLI MILLETVEKILLERI", False),
        ("", False),
    ],
)
def test_internal_loop(text, expected):
    assert has_internal_loop(text.split(), RecurrenceConfig()) is expected


def test_looping_transcript_is_flagged_even_when_alone():
    texts = ["NORMAL BIR CÜMLE", "EVVEL " * 40, "BAŞKA NORMAL CÜMLE"]
    flags = find_recurrent({"s": _session([t.strip() for t in texts])})
    assert flags == {("s", "00001"): LOOP}


def test_empty_transcripts_are_ignored():
    assert find_recurrent({"s": _session(["", "", "", "BIR CÜMLE"])}) == {}


def test_manifest_roundtrip(tmp_path):
    rows = [Row("s/00001", "s", "00001", 50, "BIR ŞEY", 0.91234, -0.25)]
    path = str(tmp_path / "m.tsv")
    write_manifest(path, rows)
    back = read_manifest(path)
    assert back[0].text == "BIR ŞEY" and back[0].num_frames == 50
    assert back[0].mean_conf == pytest.approx(0.9123)


def test_read_manifest_rejects_wrong_columns(tmp_path):
    path = tmp_path / "m.tsv"
    path.write_text("utt_id\ttext\nx\ty\n", encoding="utf-8")
    with pytest.raises(ValueError):
        read_manifest(str(path))

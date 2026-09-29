import io
import os
import sys
import tarfile

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "scripts"))

pytest.importorskip("cv2", reason="OpenCV is only installed where datasets are built")

import build_dataset  # noqa: E402
import build_manifest  # noqa: E402
import extract_raw  # noqa: E402
import verify_dataset  # noqa: E402


def _tarball(path, members, mode="w:gz"):
    with tarfile.open(path, mode) as tar:
        for name, content in members.items():
            info = tarfile.TarInfo(name)
            info.size = len(content)
            tar.addfile(info, io.BytesIO(content))


def test_extract_is_idempotent_and_leaves_the_source_alone(tmp_path, capsys):
    src, dest = tmp_path / "src", tmp_path / "dest"
    src.mkdir()
    tar_path = src / "video-1.tar.gz"
    _tarball(tar_path, {"s1/00000.txt": b"Text: A\n", "s1/00000.mp4": b"xx"})
    before = tar_path.read_bytes()

    extract_raw.extract(str(tar_path), str(dest))
    assert (dest / "s1" / "00000.txt").read_text() == "Text: A\n"
    (dest / "s1" / "00000.txt").write_text("edited")
    extract_raw.extract(str(tar_path), str(dest))  # second call must not overwrite
    assert (dest / "s1" / "00000.txt").read_text() == "edited"
    assert "already extracted" in capsys.readouterr().out
    assert tar_path.read_bytes() == before
    assert sorted(os.listdir(src)) == ["video-1.tar.gz"]


def test_extract_refuses_paths_outside_the_destination(tmp_path):
    src, dest = tmp_path / "src", tmp_path / "dest"
    src.mkdir()
    dest.mkdir()
    tar_path = src / "evil.tar.gz"
    with tarfile.open(tar_path, "w:gz") as tar:
        info = tarfile.TarInfo("../escaped.txt")
        info.size = 1
        tar.addfile(info, io.BytesIO(b"x"))
        link = tarfile.TarInfo("s1/link")
        link.type = tarfile.SYMTYPE
        link.linkname = "/etc/passwd"
        tar.addfile(link)
        ok = tarfile.TarInfo("s1/fine.txt")
        ok.size = 1
        tar.addfile(ok, io.BytesIO(b"y"))
    extract_raw.extract(str(tar_path), str(dest))
    assert not (tmp_path / "escaped.txt").exists()
    assert not (dest / "s1" / "link").exists()
    assert (dest / "s1" / "fine.txt").exists()


def test_extract_refuses_to_write_into_the_dataset(tmp_path, monkeypatch):
    src = tmp_path / "src"
    src.mkdir()
    _tarball(src / "video-1.tar.gz", {"s1/a.txt": b"x"})
    monkeypatch.setattr(sys, "argv", ["extract_raw.py", "--src", str(src), "--dest", str(src / "out")])
    with pytest.raises(SystemExit, match="source dataset"):
        extract_raw.main()
    assert sorted(os.listdir(src)) == ["video-1.tar.gz"]


def test_extract_refuses_two_tarballs_with_the_same_folder(tmp_path):
    src, dest = tmp_path / "src", tmp_path / "dest"
    src.mkdir()
    _tarball(src / "video-1.tar.gz", {"video-6/00000.txt": b"first"})
    _tarball(src / "video-2.tar.gz", {"video-6/00000.txt": b"second"})
    extract_raw.extract(str(src / "video-1.tar.gz"), str(dest))
    with pytest.raises(SystemExit, match="already extracted from video-1.tar.gz"):
        extract_raw.extract(str(src / "video-2.tar.gz"), str(dest))
    assert (dest / "video-6" / "00000.txt").read_bytes() == b"first"


def test_extract_resumes_after_an_interruption(tmp_path):
    src, dest = tmp_path / "src", tmp_path / "dest"
    src.mkdir()
    _tarball(src / "video-1.tar.gz", {"s1/00000.txt": b"a", "s1/00001.txt": b"b"})
    extract_raw.extract(str(src / "video-1.tar.gz"), str(dest))
    # as if the run had stopped before finishing: files present, no marker
    os.remove(dest / ".extracted" / "video-1.tar.gz.done")
    os.remove(dest / "s1" / "00001.txt")
    extract_raw.extract(str(src / "video-1.tar.gz"), str(dest))
    assert (dest / "s1" / "00001.txt").read_bytes() == b"b"


# ------------------------------------------------------------ build_dataset


# The pool needs module-level functions, so the fakes live here.
def _fake_work(job):
    if job[4] == "crash":
        os._exit(1)  # what a native crash inside the detector looks like
    if job[4] == "raise":
        raise RuntimeError("bad clip")
    return build_dataset.ClipResult(job[3], job[4], verdict="kept", num_frames=25)


def _no_init():
    pass


def test_run_jobs_survives_a_crashing_worker():
    ids = [f"{i:05d}" for i in range(40)]
    ids[17] = "crash"
    ids[23] = "raise"
    jobs = [("raw", "out", "cache", "s1", clip_id) for clip_id in ids]
    results = build_dataset.run_jobs(jobs, 3, work=_fake_work, initializer=_no_init, batch_size=8)
    by_id = {r.clip_id: r for r in results}
    assert len(results) == 40 and set(by_id) == set(ids)
    assert by_id["crash"].verdict == "error" and "crashed" in by_id["crash"].error
    assert by_id["raise"].verdict == "error" and "bad clip" in by_id["raise"].error
    assert sum(r.verdict == "kept" for r in results) == 38


def test_assign_splits():
    ids = [f"{i:05d}" for i in range(40)]
    splits = build_dataset.assign_splits("s1", ids, {"s9"}, 0.1)
    assert [splits[i] for i in ids] == ["train"] * 36 + ["val"] * 4
    assert set(build_dataset.assign_splits("s9", ids, {"s9"}, 0.1).values()) == {"test"}
    assert build_dataset.assign_splits("s1", [], {"s9"}, 0.1) == {}


LETTERS = "ABCÇDEFGĞHIJKLMNOÖPRSŞTUÜVYZ"


def _unique_sentence(i):
    """Four words that no other clip uses, so that nothing looks recurrent."""
    return " ".join("SÖZ" + LETTERS[i % 29] + LETTERS[(i // 29) % 29] + LETTERS[k] for k in range(4))


@pytest.fixture
def raw_corpus(tmp_path):
    """Two sessions in the layout of the source, with a cached landmark track per clip."""
    import numpy as np

    from avsr.preprocess import landmarks as lm
    from avsr.preprocess import video_io

    if not os.path.isfile(lm.DEFAULT_MODEL_PATH):
        pytest.skip("face landmark model not available")
    raw, cache = tmp_path / "raw", tmp_path / "cache"
    rng = np.random.default_rng(0)
    for session, count in (("s1", 24), ("s2", 6)):
        for i in range(count):
            clip_id, frames_n = f"{i:05d}", 50
            x = np.linspace(60, 200, 224, dtype=np.float32)
            frames = np.stack([np.tile(np.roll(x, 2 * t + i), (224, 1)) for t in range(frames_n)])
            frames = np.repeat(frames[..., None], 3, axis=-1).astype(np.uint8)
            prefix = raw / session / clip_id
            video_io.write_clip(str(prefix) + ".mp4", frames, rng.normal(0, 0.1, frames_n * 640))
            (raw / session / f"{clip_id}.aac").write_bytes(f"audio {session} {i}".encode())
            # clips 10 to 13 of the first session carry the same invented sentence
            if session == "s1" and 10 <= i <= 13:
                text = "AKLINIZA GELEBILECEK HER TÜRLÜ"
            else:
                text = _unique_sentence(i + (100 if session == "s2" else 0))
            words = text.split()
            step = 1.9 / len(words)
            lines = [f"Text:  {text}"] + [
                f"{w} {0.04 + k * step:.4f} {0.04 + (k + 1) * step:.4f} 0.9500" for k, w in enumerate(words)
            ]
            (raw / session / f"{clip_id}.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
            landmarks = np.zeros((frames_n, lm.NUM_LANDMARKS, 2), dtype=np.float32)
            landmarks[:, lm.MOUTH_OUTER] = (112, 160)
            landmarks[:, lm.FOREHEAD] = (112, 30)
            landmarks[:, lm.CHIN] = (112, 200)
            detected = np.ones(frames_n, dtype=bool)
            if session == "s1" and i == 5:
                detected[:30] = False  # a clip in which the face is mostly not found
            path = cache / "landmarks" / session / f"{clip_id}.npz"
            path.parent.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(path, landmarks=landmarks, detected=detected)
    return raw, cache, tmp_path / "dataset"


def _build(monkeypatch, raw_corpus, *extra):
    raw, cache, out = raw_corpus
    argv = ["build_dataset.py", "--raw", str(raw), "--cache", str(cache), "--out", str(out)]
    argv += ["--holdout-sessions", "s2", "--val-fraction", "0.1", "--workers", "2"]
    monkeypatch.setattr(sys, "argv", argv + list(extra))
    build_dataset.main()
    from avsr.data.manifest import read_clip_table

    return out, {(r["session"], r["clip_id"]): r for r in read_clip_table(str(out) + ".clips.tsv")}


def _has_mask(path):
    from avsr.preprocess import video_io

    video = video_io.read_video(str(path)).astype(int).sum(-1)
    cells = video.reshape(len(video), 4, 24, 4, 24).mean(axis=(2, 4))
    return sum((cells[t] < 30).sum() >= 2 for t in range(len(video)))


def test_build_dataset_end_to_end(monkeypatch, raw_corpus):
    out, table = _build(monkeypatch, raw_corpus)
    raw = raw_corpus[0]

    assert sorted(os.listdir(out)) == ["s1", "s2"]  # nothing but session folders
    left_out = {k for k, r in table.items() if r["verdict"] != "kept"}
    assert {k: table[k]["verdict"] for k in left_out} == {
        ("s1", "00005"): "no_face",
        ("s1", "00010"): "recurrent_burst",
        ("s1", "00011"): "recurrent_burst",
        ("s1", "00012"): "recurrent_burst",
        ("s1", "00013"): "recurrent_burst",
    }
    for session, clip_id in table:
        files = sorted(p.name for p in (out / session).glob(f"{clip_id}.*"))
        expected = [] if (session, clip_id) in left_out else [f"{clip_id}.aac", f"{clip_id}.mp4", f"{clip_id}.txt"]
        assert files == expected
        for ext in (".aac", ".txt"):
            if expected:
                assert (out / session / (clip_id + ext)).read_bytes() == (raw / session / (clip_id + ext)).read_bytes()

    splits = {k: r["split"] for k, r in table.items() if r["verdict"] == "kept"}
    assert {s for (session, _), s in splits.items() if session == "s2"} == {"test"}
    assert sorted(c for (session, c), s in splits.items() if s == "val") == ["00022", "00023"]
    for (session, clip_id), row in table.items():
        if row["verdict"] != "kept":
            continue
        masked_frames = _has_mask(out / session / f"{clip_id}.mp4")
        assert row["masked"] == ("yes" if row["split"] == "train" else "no")
        assert masked_frames == (10 if row["split"] == "train" else 0)


def test_build_dataset_can_be_rerun_and_guards_its_settings(monkeypatch, raw_corpus):
    out, first = _build(monkeypatch, raw_corpus)
    stamp = os.path.getmtime(out / "s1" / "00000.mp4")
    out, second = _build(monkeypatch, raw_corpus)
    assert os.path.getmtime(out / "s1" / "00000.mp4") == stamp  # finished clips are not rebuilt
    assert {k: r["verdict"] for k, r in first.items()} == {k: r["verdict"] for k, r in second.items()}
    with pytest.raises(SystemExit, match="different settings"):
        _build(monkeypatch, raw_corpus, "--roi-scale", "0.8")
    with pytest.raises(SystemExit, match="different settings"):
        _build(monkeypatch, raw_corpus, "--no-ptm")


def test_build_dataset_removes_clips_that_no_longer_qualify(monkeypatch, raw_corpus):
    out, _ = _build(monkeypatch, raw_corpus)
    raw = raw_corpus[0]
    assert (out / "s1" / "00002.mp4").exists()
    label = raw / "s1" / "00002.txt"
    label.write_text(label.read_text().replace("0.9500", "0.2000"))  # now a low-confidence label
    out, table = _build(monkeypatch, raw_corpus)
    assert table[("s1", "00002")]["verdict"] == "low_confidence"
    assert not list((out / "s1").glob("00002.*"))


def test_build_dataset_refuses_bad_arguments(monkeypatch, raw_corpus):
    raw, cache, out = raw_corpus
    base = ["build_dataset.py", "--raw", str(raw), "--cache", str(cache)]
    for argv, message in [
        (["--out", str(raw / "inside"), "--holdout-sessions", "s2"], "must not be inside"),
        (["--out", str(out), "--holdout-sessions", "nowhere"], "not found"),
        (["--out", str(out), "--holdout-sessions", "s1", "s2"], "nothing is left"),
        (["--out", str(out), "--holdout-sessions", "s2", "--out-size", "97"], "even"),
    ]:
        monkeypatch.setattr(sys, "argv", base + argv)
        with pytest.raises((SystemExit, ValueError), match=message):
            build_dataset.main()
    assert sorted(os.listdir(raw)) == ["s1", "s2"]


def test_manifests_and_verification(monkeypatch, raw_corpus, tmp_path, capsys):
    from avsr.data.manifest import read_manifest

    out, _ = _build(monkeypatch, raw_corpus)
    manifests = tmp_path / "manifests"
    monkeypatch.setattr(sys, "argv", ["build_manifest.py", "--dataset", str(out), "--out", str(manifests)])
    build_manifest.main()
    sizes = {s: len(read_manifest(str(manifests / f"{s}.tsv"))) for s in ("train", "val", "test")}
    assert sizes == {"train": 17, "val": 2, "test": 6}

    argv = ["verify_dataset.py", "--dataset", str(out), "--raw", str(raw_corpus[0]), "--cache", str(raw_corpus[1])]
    monkeypatch.setattr(sys, "argv", argv + ["--manifests", str(manifests)])
    monkeypatch.setattr(verify_dataset, "FAILED", [])
    with pytest.raises(SystemExit) as done:
        verify_dataset.main()
    assert done.value.code == 0, capsys.readouterr().out

    # damage must be noticed: one clip replaced by another, a stray file, an edited transcript
    capsys.readouterr()
    os.replace(out / "s1" / "00001.mp4", out / "s1" / "00000.mp4")
    (out / "s1" / "notes.bak").write_text("stray")
    with open(out / "s2" / "00000.txt", "a") as f:
        f.write("EK 1.0 1.1 0.9\n")
    monkeypatch.setattr(verify_dataset, "FAILED", [])
    monkeypatch.setattr(sys, "argv", argv + ["--manifests", str(manifests), "--sample", "100"])
    with pytest.raises(SystemExit) as done:
        verify_dataset.main()
    assert done.value.code == 1
    printed = capsys.readouterr().out
    for check in (
        "every clip has its three files",
        "dataset folder holds session folders",
        "transcripts are byte-identical",
        "every sampled clip can be decoded",
        # the clip that took the place of another carries that other clip's mask
        "masked cells are black and where the seed puts them",
    ):
        assert f"FAIL  {check}" in printed, check
    assert "PASS  aac files are byte-identical" in printed

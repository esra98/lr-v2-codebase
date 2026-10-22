import pytest
import torch

from avsr.text.labels import parse_label_file
from avsr.text.normalize import TURKISH_ALPHABET, normalize_text, number_to_words
from avsr.text.tokenizer import CharTokenizer, SentencePieceTokenizer, train_sentencepiece


@pytest.mark.parametrize(
    "n, expected",
    [
        (0, "SIFIR"),
        (1, "BİR"),
        (10, "ON"),
        (17, "ON YEDİ"),
        (100, "YÜZ"),
        (101, "YÜZ BİR"),
        (200, "İKİ YÜZ"),
        (1000, "BİN"),
        (1921, "BİN DOKUZ YÜZ YİRMİ BİR"),
        (2025, "İKİ BİN YİRMİ BEŞ"),
        (21000, "YİRMİ BİR BİN"),
        (1000000, "BİR MİLYON"),
        (1001000, "BİR MİLYON BİN"),
    ],
)
def test_number_to_words(n, expected):
    assert number_to_words(n) == expected


@pytest.mark.parametrize(
    "raw, expected",
    [
        ("MERSIN MILLETVEKILI", "MERSIN MILLETVEKILI"),
        ("BIR ŞEY.", "BIR ŞEY"),
        ("VE YAVUZ BIR ARALARINDAN,", "VE YAVUZ BIR ARALARINDAN"),
        ("17-25 ARALIĞI", "ON YEDİ YİRMİ BEŞ ARALIĞI"),
        ("1 NISAN 1921'DE GIRESUNUZ", "BİR NISAN BİN DOKUZ YÜZ YİRMİ BİRDE GIRESUNUZ"),
        ("%50", "YÜZDE ELLİ"),
        ("3,5 MILYON", "ÜÇ VİRGÜL BEŞ MILYON"),
        ("1.250.000 LIRA", "BİR MİLYON İKİ YÜZ ELLİ BİN LIRA"),
        ("TÜRKIYE'NIN", "TÜRKIYENIN"),
        ("İSTANBUL HÂLÂ", "İSTANBUL HALA"),
        ("istanbul ılık", "İSTANBUL ILIK"),
        ("NE?  NASIL!", "NE NASIL"),
        ("...", ""),
    ],
)
def test_normalize_text(raw, expected):
    assert normalize_text(raw) == expected


def test_normalize_can_reproduce_the_old_merged_alphabet():
    assert normalize_text("istanbul ılık", merge_dotted_i=True) == "ISTANBUL ILIK"
    assert "İ" not in normalize_text("1921 İZMİR", merge_dotted_i=True)


def test_alphabet_has_both_i_letters():
    assert "I" in TURKISH_ALPHABET and "İ" in TURKISH_ALPHABET and len(TURKISH_ALPHABET) == 29


def test_normalize_is_idempotent():
    once = normalize_text("1 NISAN 1921'DE, %5 ARTTI.")
    assert normalize_text(once) == once


def test_parse_label_file(tmp_path):
    p = tmp_path / "a.txt"
    p.write_text("Text:  17-25 ARALIĞI\n17-25 0.0400 0.5600 0.9750\nARALIĞI 0.5600 1.0400 0.9130\n", encoding="utf-8")
    utt = parse_label_file(p)
    assert utt.text == "17-25 ARALIĞI"
    assert [w.text for w in utt.words] == ["17-25", "ARALIĞI"]
    assert utt.speech_end == pytest.approx(1.04)
    assert utt.mean_conf == pytest.approx((0.975 + 0.913) / 2)
    assert utt.short_word_fraction() == 0.0


def test_parse_label_flags_squeezed_words(tmp_path):
    p = tmp_path / "a.txt"
    p.write_text(
        "Text:  MERSIN MILLETVEKILI BURHAN\n"
        "MERSIN 0.0400 0.2000 0.88\nMILLETVEKILI 0.2000 2.0200 0.99\nBURHAN 2.0200 2.0400 0.98\n",
        encoding="utf-8",
    )
    assert parse_label_file(p).short_word_fraction() == pytest.approx(1 / 3)


@pytest.mark.parametrize("content", ["", "MERSIN 0.1 0.2 0.9\n", "Text: A\nA 0.1 x 0.9\n", "Text: A\nA 0.1\n"])
def test_parse_label_rejects_malformed(tmp_path, content):
    p = tmp_path / "a.txt"
    p.write_text(content, encoding="utf-8")
    with pytest.raises(ValueError):
        parse_label_file(p)


def test_char_tokenizer_roundtrip():
    tok = CharTokenizer("ABCÇDEFGĞHIJKLMNOÖPRSŞTUÜVYZ")
    text = "BIR ŞEY ÇOK GÜZEL"
    ids = tok.encode(text)
    assert ids.min() > 0 and ids.max() < tok.eos_id
    assert tok.decode(ids) == text
    # padding, blank and eos never leak into decoded text
    noisy = torch.cat([ids, torch.tensor([tok.eos_id, 0, -1])])
    assert tok.decode(noisy) == text


def test_sentencepiece_tokenizer_roundtrip(tmp_path):
    corpus = ["BIR ŞEY ÇOK GÜZEL", "MERSIN MILLETVEKILI", "SAYIN BAŞKAN DEĞERLI MILLETVEKILLERI"] * 30
    model = train_sentencepiece(corpus, str(tmp_path / "spm"), vocab_size=40)
    tok = SentencePieceTokenizer(model)
    assert tok.token_list[0] == "<blank>" and tok.token_list[-1] == "<eos>"
    for text in corpus[:3]:
        ids = tok.encode(text)
        assert ids.min() > 0 and ids.max() < tok.eos_id
        assert tok.decode(ids) == text

"""Turkish transcript normalisation.

Transcripts are upper-cased with Turkish casing rules (``i`` becomes ``İ``,
``ı`` becomes ``I``), so the dotted and dotless I stay distinct and the
alphabet has the 29 Turkish letters plus Q, W, X for foreign words.

The first TR-LRS corpus (2026-09) had been upper-cased without those rules,
which merged both letters into ``I``. ``merge_dotted_i=True`` reproduces that
alphabet for work on the old clips; new corpora are cut from the Whisper
output in its original casing and do not need it.
"""

import re
import unicodedata

# Canonical alphabet after normalisation (upper-case, Turkish casing).
TURKISH_ALPHABET = "ABCÇDEFGĞHIİJKLMNOÖPRSŞTUÜVYZ"
FOREIGN_LETTERS = "QWX"

_CIRCUMFLEX = str.maketrans({"Â": "A", "Î": "I", "Û": "U", "â": "a", "î": "i", "û": "u"})

_ONES = ["", "BİR", "İKİ", "ÜÇ", "DÖRT", "BEŞ", "ALTI", "YEDİ", "SEKİZ", "DOKUZ"]
_TENS = ["", "ON", "YİRMİ", "OTUZ", "KIRK", "ELLİ", "ALTMIŞ", "YETMİŞ", "SEKSEN", "DOKSAN"]
_SCALES = [(10**12, "TRİLYON"), (10**9, "MİLYAR"), (10**6, "MİLYON"), (10**3, "BİN")]

_NUMBER_RE = re.compile(r"\d+(?:[.,]\d+)*")
_APOSTROPHES = "'’‘`´"


def _below_thousand(n):
    words = []
    hundreds, rest = divmod(n, 100)
    if hundreds:
        if hundreds > 1:
            words.append(_ONES[hundreds])
        words.append("YÜZ")
    tens, ones = divmod(rest, 10)
    if tens:
        words.append(_TENS[tens])
    if ones:
        words.append(_ONES[ones])
    return words


def number_to_words(n):
    """Spell a non-negative integer in Turkish (upper-case, Turkish casing)."""
    n = int(n)
    if n == 0:
        return "SIFIR"
    if n >= 10**15:
        # Too large to read as a quantity; spell digit by digit.
        return " ".join(number_to_words(int(d)) for d in str(n))
    words = []
    for value, name in _SCALES:
        count, n = divmod(n, value)
        if not count:
            continue
        # "bin" never takes a leading "bir" in Turkish (1000 = "bin").
        if not (value == 1000 and count == 1):
            words.extend(_below_thousand(count))
        words.append(name)
    words.extend(_below_thousand(n))
    return " ".join(words)


def _spell_number_token(token):
    """Spell a digit token that may contain thousands dots or a decimal comma."""
    token = token.strip(".,")
    if not token:
        return ""
    if "," in token:
        integer, _, fraction = token.partition(",")
        integer = integer.replace(".", "")
        fraction = fraction.replace(".", "").replace(",", "")
        if integer.isdigit() and fraction.isdigit():
            return f"{number_to_words(integer)} VİRGÜL {number_to_words(fraction)}"
        token = integer + fraction
    parts = token.split(".")
    # Dots followed by exactly three digits are thousands separators (1.250.000).
    if len(parts) > 1 and all(len(p) == 3 for p in parts[1:]):
        return number_to_words("".join(parts))
    return " ".join(number_to_words(p) for p in parts if p)


def spell_numbers(text):
    """Replace digit groups with Turkish words, keeping attached suffixes."""
    text = text.replace("%", " YÜZDE ")

    def repl(match):
        return f" {_spell_number_token(match.group(0))}\u0000"

    text = _NUMBER_RE.sub(repl, text)
    # A suffix written after an apostrophe attaches to the number word
    # ("1921'DE" -> "... YIRMI BIRDE"); anything else becomes a new word.
    text = re.sub("\u0000[" + _APOSTROPHES + "]", "", text)
    return text.replace("\u0000", " ")


def fold_dotted_i(text):
    """Map ``İ`` to ``I``, the alphabet of the first corpus (see module docstring)."""
    return text.replace("İ", "I")


def normalize_text(text, merge_dotted_i=False, spell_digits=True, keep_foreign=True):
    """Normalise a raw transcript to the canonical training alphabet.

    Returns an upper-case string containing only alphabet letters and single
    spaces. Returns an empty string when nothing usable remains.
    """
    text = unicodedata.normalize("NFC", text)
    text = text.translate(_CIRCUMFLEX)
    if spell_digits:
        text = spell_numbers(text)
    # Turkish-aware upper-casing, done before str.upper() can mangle it.
    text = text.replace("i", "İ").replace("ı", "I").upper()
    if merge_dotted_i:
        text = fold_dotted_i(text)
    for ch in _APOSTROPHES:
        text = text.replace(ch, "")
    allowed = set(TURKISH_ALPHABET)
    if keep_foreign:
        allowed.update(FOREIGN_LETTERS)
    text = "".join(ch if ch in allowed else " " for ch in text)
    return " ".join(text.split())

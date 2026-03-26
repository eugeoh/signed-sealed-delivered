"""Steganographic decoder — extract embedded hash bits from text and verify against expected hash."""

from __future__ import annotations

import re

from ssd.synonym_map import SYNONYM_PAIRS, PUNCTUATION_TOGGLES, build_synonym_lookup


def decode(text: str, expected_num_bits: int = 64) -> list[int]:
    """Extract embedded bits from *text*.

    Returns a list of 0/1 values recovered from synonym choices,
    punctuation toggles, and sentence spacing, in the same order
    they were encoded.
    """
    bits: list[int] = []
    bits.extend(_decode_synonyms(text, expected_num_bits))
    remaining = expected_num_bits - len(bits)
    if remaining > 0:
        bits.extend(_decode_punctuation(text, remaining))
    remaining = expected_num_bits - len(bits)
    if remaining > 0:
        bits.extend(_decode_sentence_spacing(text, remaining))
    return bits[:expected_num_bits]


# ---------------------------------------------------------------------------
# Synonym-based decoding
# ---------------------------------------------------------------------------

def _decode_synonyms(text: str, max_bits: int) -> list[int]:
    """Scan text for synonym-pair words and extract bits from each occurrence."""
    lookup = build_synonym_lookup()
    tokens = re.split(r"(\b)", text)
    bits: list[int] = []
    used_pairs: set[int] = set()

    for token in tokens:
        if len(bits) >= max_bits:
            break

        lower = token.lower()
        if lower in lookup:
            _partner, pair_idx, bit_value = lookup[lower]
            if pair_idx in used_pairs:
                continue
            used_pairs.add(pair_idx)
            bits.append(bit_value)

    return bits


# ---------------------------------------------------------------------------
# Punctuation-based decoding
# ---------------------------------------------------------------------------

def _decode_punctuation(text: str, max_bits: int) -> list[int]:
    """Extract bits from punctuation toggles."""
    bits: list[int] = []
    for toggle in PUNCTUATION_TOGGLES:
        if len(bits) >= max_bits:
            break

        has_zero = re.search(toggle["pattern_zero"], text)
        has_one = re.search(toggle["pattern_one"], text)

        if has_zero and not has_one:
            bits.append(0)
        elif has_one and not has_zero:
            bits.append(1)
        elif has_zero or has_one:
            bits.append(0)

    return bits


# ---------------------------------------------------------------------------
# Sentence-spacing decoding
# ---------------------------------------------------------------------------

_SENTENCE_BOUNDARY = re.compile(r"[.!?]([ ]+)")


def _decode_sentence_spacing(text: str, max_bits: int) -> list[int]:
    """Extract bits from inter-sentence spacing (1 space = 0, 2 spaces = 1)."""
    bits: list[int] = []
    for match in _SENTENCE_BOUNDARY.finditer(text):
        if len(bits) >= max_bits:
            break
        spaces = match.group(1)
        bits.append(0 if len(spaces) == 1 else 1)
    return bits


# ---------------------------------------------------------------------------
# Verification
# ---------------------------------------------------------------------------

def hamming_distance(a: list[int], b: list[int]) -> int:
    """Count the number of differing bits between two equal-length bit lists."""
    return sum(x != y for x, y in zip(a, b))


def verify(text: str, expected_bits: list[int], threshold: float = 0.25) -> tuple[bool, float]:
    """Verify that *text* contains the expected steganographic hash.

    Returns ``(match, confidence)`` where *confidence* is 1.0 minus the
    normalised Hamming distance (1.0 = perfect match, 0.0 = all bits differ).
    *match* is True when confidence >= (1 - threshold).
    """
    extracted = decode(text, len(expected_bits))

    # Pad extracted bits if fewer than expected
    while len(extracted) < len(expected_bits):
        extracted.append(0)

    dist = hamming_distance(extracted, expected_bits)
    normalised = dist / len(expected_bits) if expected_bits else 0.0
    confidence = 1.0 - normalised
    match = confidence >= (1.0 - threshold)
    return match, round(confidence, 4)

"""Steganographic encoder — embed a 64-bit style hash into text via synonym/punctuation choices."""

from __future__ import annotations

import re

from ssd.synonym_map import SYNONYM_PAIRS, PUNCTUATION_TOGGLES, build_synonym_lookup


def encode(text: str, hash_bits: list[int]) -> str:
    """Embed *hash_bits* (list of 0/1) into *text* using synonym swaps,
    punctuation toggles, and sentence-level micro-patterns.

    Returns the modified text with the hash steganographically embedded.
    """
    bits = list(hash_bits)  # copy so we don't mutate caller's list
    text = _encode_synonyms(text, bits)
    text = _encode_punctuation(text, bits)
    text = _encode_sentence_spacing(text, bits)
    return text


# ---------------------------------------------------------------------------
# Synonym-based encoding
# ---------------------------------------------------------------------------

def _encode_synonyms(text: str, bits: list[int]) -> str:
    """Walk through the text and replace synonym-pair words to encode bits.

    *bits* is mutated in-place — consumed bits are popped from the front.
    """
    lookup = build_synonym_lookup()

    # We process word-by-word, preserving non-word characters.
    tokens = re.split(r"(\b)", text)
    result: list[str] = []
    used_pairs: set[int] = set()  # only use each synonym pair once

    for token in tokens:
        if not bits:
            result.append(token)
            continue

        lower = token.lower()
        if lower in lookup:
            partner, pair_idx, current_bit = lookup[lower]
            if pair_idx in used_pairs:
                result.append(token)
                continue

            desired_bit = bits.pop(0)
            used_pairs.add(pair_idx)

            if desired_bit == current_bit:
                result.append(token)
            else:
                result.append(_match_case(token, partner))
        else:
            result.append(token)

    return "".join(result)


def _match_case(original: str, replacement: str) -> str:
    """Apply the casing pattern of *original* to *replacement*."""
    if original.isupper():
        return replacement.upper()
    if original[0].isupper():
        return replacement[0].upper() + replacement[1:]
    return replacement


# ---------------------------------------------------------------------------
# Punctuation-based encoding
# ---------------------------------------------------------------------------

def _encode_punctuation(text: str, bits: list[int]) -> str:
    """Use punctuation toggles to encode remaining bits."""
    for toggle in PUNCTUATION_TOGGLES:
        if not bits:
            break

        desired_bit = bits[0]

        has_zero = re.search(toggle["pattern_zero"], text)
        has_one = re.search(toggle["pattern_one"], text)

        if desired_bit == 0 and has_one and not has_zero:
            text = re.sub(
                toggle["pattern_one"], toggle["replace_zero"], text, count=1
            )
            bits.pop(0)
        elif desired_bit == 1 and has_zero and not has_one:
            text = re.sub(
                toggle["pattern_zero"], toggle["replace_one"], text, count=1
            )
            bits.pop(0)
        elif has_zero or has_one:
            if desired_bit == 0 and has_zero:
                bits.pop(0)
            elif desired_bit == 1 and has_one:
                bits.pop(0)

    return text


# ---------------------------------------------------------------------------
# Sentence-spacing encoding — encode bits into inter-sentence whitespace
# ---------------------------------------------------------------------------

# Each sentence boundary encodes 1 bit:
#   bit 0 → one space after period ("Foo. Bar")
#   bit 1 → two spaces after period ("Foo.  Bar")
# This is invisible in rendered HTML/markdown but preserved in plain text.

_SENTENCE_BOUNDARY = re.compile(r"([.!?])([ ]+)")


def _encode_sentence_spacing(text: str, bits: list[int]) -> str:
    """Encode remaining bits via single vs double space after sentence-ending punctuation."""
    if not bits:
        return text

    def _replacer(match: re.Match) -> str:
        if not bits:
            return match.group(0)
        bit = bits.pop(0)
        punct = match.group(1)
        return punct + (" " if bit == 0 else "  ")

    return _SENTENCE_BOUNDARY.sub(_replacer, text)

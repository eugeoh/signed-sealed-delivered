"""Deterministic style hashing for writing style profiles."""

from __future__ import annotations

import hashlib

import base58

from ssd.models import StyleProfile

# Min/max ranges for each of the 22 numeric features (in to_vector() order).
# Values outside these ranges are clamped to the boundary.
_FEATURE_RANGES: list[tuple[float, float]] = [
    (3.0, 40.0),    # mean_sentence_length_words
    (0.0, 20.0),    # std_sentence_length_words
    (15.0, 250.0),  # mean_sentence_length_chars
    (0.0, 120.0),   # std_sentence_length_chars
    (0.0, 1.0),     # question_ratio
    (0.0, 1.0),     # exclamation_ratio
    (0.0, 1.0),     # type_token_ratio
    (0.0, 1.0),     # hapax_legomena_ratio
    (1.0, 10.0),    # avg_word_length
    (1.0, 5.0),     # syllable_complexity
    (0.0, 0.5),     # comma_density
    (0.0, 0.1),     # semicolon_density
    (0.0, 0.2),     # dash_density
    (0.0, 0.1),     # ellipsis_frequency
    (0.0, 1.0),     # oxford_comma_score
    (0.0, 0.3),     # adjective_density
    (0.0, 0.2),     # adverb_density
    (0.0, 1.0),     # passive_voice_ratio
    (0.0, 0.5),     # subordinate_clause_frequency
    (0.0, 500.0),   # sentence_length_variance
    (0.0, 1.0),     # contraction_frequency
    (0.0, 1.0),     # formality_score
]


def quantize_profile(profile: StyleProfile, num_bins: int = 8) -> list[int]:
    """Quantize each numeric feature into discrete bins.

    Each feature value is mapped to an integer in [0, num_bins-1] based on
    predefined min/max ranges. Values outside the range are clamped.
    """
    vector = profile.to_vector()
    bins: list[int] = []
    for value, (lo, hi) in zip(vector, _FEATURE_RANGES):
        clamped = max(lo, min(hi, value))
        if hi == lo:
            bins.append(0)
        else:
            normalized = (clamped - lo) / (hi - lo)
            bin_index = int(normalized * num_bins)
            bin_index = min(bin_index, num_bins - 1)
            bins.append(bin_index)
    return bins


def compute_style_hash(
    profile: StyleProfile, hash_bits: int = 64, name: str | None = None
) -> bytes:
    """Produce a deterministic hash of the style profile and optional author name.

    Quantizes the profile, builds a canonical byte string from the bin
    indices (prefixed with the normalised name when provided), SHA-256
    hashes it, and returns the first ``hash_bits // 8`` bytes.
    """
    bins = quantize_profile(profile)
    canonical = bytes(bins)
    if name:
        canonical = name.strip().lower().encode("utf-8") + b":" + canonical
    digest = hashlib.sha256(canonical).digest()
    return digest[: hash_bits // 8]


def style_hash_to_bits(hash_bytes: bytes) -> list[int]:
    """Convert hash bytes to a list of individual bits (0 or 1)."""
    bits: list[int] = []
    for byte in hash_bytes:
        for i in range(7, -1, -1):
            bits.append((byte >> i) & 1)
    return bits


def style_signature(profile: StyleProfile, name: str | None = None) -> str:
    """Return a human-readable base58 style signature."""
    hash_bytes = compute_style_hash(profile, name=name)
    return base58.b58encode(hash_bytes).decode("ascii")

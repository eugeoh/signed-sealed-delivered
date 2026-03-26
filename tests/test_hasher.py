"""Tests for style_hasher module."""

from ssd.models import StyleProfile
from ssd.style_hasher import (
    compute_style_hash,
    quantize_profile,
    style_hash_to_bits,
    style_signature,
)


def _sample_profile() -> StyleProfile:
    return StyleProfile(
        mean_sentence_length_words=8.0,
        std_sentence_length_words=3.0,
        mean_sentence_length_chars=40.0,
        std_sentence_length_chars=15.0,
        question_ratio=0.1,
        exclamation_ratio=0.0,
        type_token_ratio=0.65,
        hapax_legomena_ratio=0.5,
        avg_word_length=4.2,
        syllable_complexity=1.4,
        comma_density=0.03,
        semicolon_density=0.0,
        dash_density=0.01,
        ellipsis_frequency=0.0,
        oxford_comma_score=0.5,
        adjective_density=0.08,
        adverb_density=0.04,
        passive_voice_ratio=0.1,
        subordinate_clause_frequency=0.15,
        sentence_length_variance=0.35,
        contraction_frequency=0.06,
        formality_score=0.45,
    )


def test_quantize_produces_correct_length():
    bins = quantize_profile(_sample_profile())
    assert len(bins) == 22


def test_quantize_values_in_range():
    bins = quantize_profile(_sample_profile(), num_bins=8)
    for b in bins:
        assert 0 <= b <= 7


def test_hash_deterministic():
    h1 = compute_style_hash(_sample_profile())
    h2 = compute_style_hash(_sample_profile())
    assert h1 == h2


def test_hash_length():
    h = compute_style_hash(_sample_profile(), hash_bits=64)
    assert len(h) == 8  # 64 bits = 8 bytes


def test_hash_to_bits():
    h = compute_style_hash(_sample_profile())
    bits = style_hash_to_bits(h)
    assert len(bits) == 64
    assert all(b in (0, 1) for b in bits)


def test_different_profiles_different_hashes():
    p1 = _sample_profile()
    p2 = _sample_profile()
    p2.mean_sentence_length_words = 25.0
    p2.formality_score = 0.9
    assert compute_style_hash(p1) != compute_style_hash(p2)


def test_style_signature_is_string():
    sig = style_signature(_sample_profile())
    assert isinstance(sig, str)
    assert len(sig) > 0

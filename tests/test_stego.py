"""Tests for stego_encoder and stego_decoder modules."""

from ssd.stego_encoder import encode
from ssd.stego_decoder import decode, verify, hamming_distance


SAMPLE_TEXT = (
    "However, the big changes that started recently are important. "
    "We also think this will help people quickly obtain the right answers. "
    "The fast response was enough to demonstrate the strong results. "
    "She began to show the entire team how to use the new approach. "
    "They frequently find that it is easy to get started."
)


def test_encode_decode_round_trip():
    """Encoded bits should be recoverable."""
    bits = [1, 0, 1, 1, 0, 0, 1, 0]
    encoded = encode(SAMPLE_TEXT, bits)
    extracted = decode(encoded, len(bits))
    # At least some bits should match
    matching = sum(a == b for a, b in zip(bits, extracted))
    assert matching >= len(bits) // 2  # at least half should survive


def test_encode_preserves_readability():
    """Encoded text should still be readable English."""
    bits = [0, 1, 0, 1, 0, 1, 0, 1]
    encoded = encode(SAMPLE_TEXT, bits)
    # Should still have roughly the same word count
    original_words = len(SAMPLE_TEXT.split())
    encoded_words = len(encoded.split())
    assert abs(original_words - encoded_words) <= 5


def test_hamming_distance():
    assert hamming_distance([0, 0, 1, 1], [0, 1, 1, 0]) == 2
    assert hamming_distance([1, 1, 1, 1], [1, 1, 1, 1]) == 0


def test_verify_matching_bits():
    bits = [1, 0, 1, 0, 1, 0, 1, 0]
    encoded = encode(SAMPLE_TEXT, bits)
    match, confidence = verify(encoded, bits)
    # With a generous threshold, should match
    assert confidence > 0.5


def test_verify_wrong_bits():
    """Completely wrong bits should have low confidence."""
    encode_bits = [1] * 16
    wrong_bits = [0] * 16
    encoded = encode(SAMPLE_TEXT, encode_bits)
    _match, confidence = verify(encoded, wrong_bits)
    # Confidence should be lower than a correct verification
    _, correct_conf = verify(encoded, encode_bits)
    assert correct_conf >= confidence


def test_encode_empty_bits():
    """Empty bit list should return text unchanged."""
    result = encode(SAMPLE_TEXT, [])
    assert result == SAMPLE_TEXT

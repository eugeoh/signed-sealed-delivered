"""Tests for style_extractor module."""

import pytest

from ssd.style_extractor import extract_style
from ssd.models import StyleProfile


HEMINGWAY_SAMPLE = (
    "He sat by the window. The rain fell. It was cold outside. "
    "He drank his coffee. It was black and strong. He didn't say anything. "
    "She looked at him. He looked away. The silence was heavy. "
    "He stood up and left. She stayed. The rain kept falling."
)

ACADEMIC_SAMPLE = (
    "The comprehensive analysis of multidimensional socioeconomic indicators "
    "reveals a statistically significant correlation between educational "
    "attainment and longitudinal health outcomes; furthermore, the methodology "
    "employed in this investigation — incorporating both quantitative surveys "
    "and qualitative ethnographic observations — demonstrates the necessity "
    "of interdisciplinary approaches when examining complex phenomena that "
    "transcend traditional disciplinary boundaries. Additionally, the "
    "implications of these findings extend beyond the immediate research "
    "context, suggesting that policymakers should consider the interconnected "
    "nature of social determinants when formulating evidence-based interventions."
)


def test_extract_returns_style_profile():
    profile = extract_style(HEMINGWAY_SAMPLE)
    assert isinstance(profile, StyleProfile)


def test_empty_text_returns_default():
    profile = extract_style("")
    assert profile.mean_sentence_length_words == 0.0


def test_hemingway_short_sentences():
    profile = extract_style(HEMINGWAY_SAMPLE)
    # Hemingway-style: expect short sentences
    assert profile.mean_sentence_length_words < 10


def test_academic_long_sentences():
    profile = extract_style(ACADEMIC_SAMPLE)
    # Academic: expect longer sentences
    assert profile.mean_sentence_length_words > 15


def test_deterministic():
    """Same input must produce the same profile."""
    p1 = extract_style(HEMINGWAY_SAMPLE)
    p2 = extract_style(HEMINGWAY_SAMPLE)
    assert p1.to_vector() == p2.to_vector()


def test_type_token_ratio_range():
    profile = extract_style(HEMINGWAY_SAMPLE)
    assert 0.0 <= profile.type_token_ratio <= 1.0


def test_contraction_detection():
    text = "I don't think it's right. We can't go. They won't stop."
    profile = extract_style(text)
    assert profile.contraction_frequency > 0


def test_to_vector_length():
    profile = extract_style(HEMINGWAY_SAMPLE)
    assert len(profile.to_vector()) == 22

"""Pydantic request/response schemas and core data models."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from pydantic import BaseModel, Field


# ---------------------------------------------------------------------------
# Core domain model
# ---------------------------------------------------------------------------

@dataclass
class StyleProfile:
    """Quantifiable stylometric feature vector extracted from a writing sample."""

    # Sentence structure
    mean_sentence_length_words: float = 0.0
    std_sentence_length_words: float = 0.0
    mean_sentence_length_chars: float = 0.0
    std_sentence_length_chars: float = 0.0
    question_ratio: float = 0.0
    exclamation_ratio: float = 0.0

    # Vocabulary
    type_token_ratio: float = 0.0
    hapax_legomena_ratio: float = 0.0
    avg_word_length: float = 0.0
    syllable_complexity: float = 0.0

    # Punctuation
    comma_density: float = 0.0
    semicolon_density: float = 0.0
    dash_density: float = 0.0
    ellipsis_frequency: float = 0.0
    oxford_comma_score: float = 0.0

    # Syntax / POS
    adjective_density: float = 0.0
    adverb_density: float = 0.0
    passive_voice_ratio: float = 0.0
    subordinate_clause_frequency: float = 0.0

    # Rhythm
    sentence_length_variance: float = 0.0

    # Lexical preference
    contraction_frequency: float = 0.0
    formality_score: float = 0.0

    # Distinctive words (not used in hashing, but useful for prompt composition)
    distinctive_words: list[str] = field(default_factory=list)

    def to_vector(self) -> list[float]:
        """Return the numeric features as an ordered list (excludes distinctive_words)."""
        return [
            self.mean_sentence_length_words,
            self.std_sentence_length_words,
            self.mean_sentence_length_chars,
            self.std_sentence_length_chars,
            self.question_ratio,
            self.exclamation_ratio,
            self.type_token_ratio,
            self.hapax_legomena_ratio,
            self.avg_word_length,
            self.syllable_complexity,
            self.comma_density,
            self.semicolon_density,
            self.dash_density,
            self.ellipsis_frequency,
            self.oxford_comma_score,
            self.adjective_density,
            self.adverb_density,
            self.passive_voice_ratio,
            self.subordinate_clause_frequency,
            self.sentence_length_variance,
            self.contraction_frequency,
            self.formality_score,
        ]

    def to_dict(self) -> dict[str, Any]:
        """Serialize to a plain dict."""
        return {
            "mean_sentence_length_words": self.mean_sentence_length_words,
            "std_sentence_length_words": self.std_sentence_length_words,
            "mean_sentence_length_chars": self.mean_sentence_length_chars,
            "std_sentence_length_chars": self.std_sentence_length_chars,
            "question_ratio": self.question_ratio,
            "exclamation_ratio": self.exclamation_ratio,
            "type_token_ratio": self.type_token_ratio,
            "hapax_legomena_ratio": self.hapax_legomena_ratio,
            "avg_word_length": self.avg_word_length,
            "syllable_complexity": self.syllable_complexity,
            "comma_density": self.comma_density,
            "semicolon_density": self.semicolon_density,
            "dash_density": self.dash_density,
            "ellipsis_frequency": self.ellipsis_frequency,
            "oxford_comma_score": self.oxford_comma_score,
            "adjective_density": self.adjective_density,
            "adverb_density": self.adverb_density,
            "passive_voice_ratio": self.passive_voice_ratio,
            "subordinate_clause_frequency": self.subordinate_clause_frequency,
            "sentence_length_variance": self.sentence_length_variance,
            "contraction_frequency": self.contraction_frequency,
            "formality_score": self.formality_score,
            "distinctive_words": self.distinctive_words,
        }


# ---------------------------------------------------------------------------
# API request / response schemas
# ---------------------------------------------------------------------------

class GenerateRequest(BaseModel):
    sample: str | None = Field(default=None, min_length=50, description="Writing sample from the author")
    style_hash: str | None = Field(default=None, description="Style hash from a previous /analyze call (used instead of sample)")
    prompt: str = Field(..., min_length=1, description="Topic / instruction for the generated text")
    author: str | None = Field(default=None, description="Author name to include in the signature block")
    model: str = Field(default="claude-sonnet-4-5-20250929", description="Anthropic model ID")
    api_key: str | None = Field(default=None, description="Anthropic API key (overrides ANTHROPIC_API_KEY env var)")


class GenerateResponse(BaseModel):
    text: str
    signature: str
    style_profile: dict[str, Any]


class VerifyRequest(BaseModel):
    text: str = Field(..., min_length=1, description="Text to verify")
    sample: str | None = Field(default=None, min_length=50, description="Writing sample of the claimed author")
    style_hash: str | None = Field(default=None, description="Style hash from a previous /analyze call (used instead of sample)")


class VerifyResponse(BaseModel):
    match: bool
    confidence: float
    extracted_hash: str
    expected_hash: str


class AnalyzeRequest(BaseModel):
    sample: str = Field(..., min_length=50, description="Writing sample to analyze")
    name: str | None = Field(default=None, description="Author name (included in hash computation)")


class AnalyzeResponse(BaseModel):
    style_profile: dict[str, Any]
    style_hash: str
    style_description: str

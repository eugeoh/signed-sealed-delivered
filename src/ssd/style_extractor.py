"""Extract stylometric features from text using spaCy."""

from __future__ import annotations

import re
from collections import Counter
from statistics import mean, stdev

import spacy
from spacy.language import Language

from ssd.models import StyleProfile

# ---------------------------------------------------------------------------
# Lazy spaCy model loader
# ---------------------------------------------------------------------------

_nlp: Language | None = None


def _get_nlp() -> Language:
    global _nlp
    if _nlp is None:
        _nlp = spacy.load("en_core_web_sm")
    return _nlp


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_VOWELS = set("aeiouy")


def _estimate_syllables(word: str) -> int:
    """Rough syllable count for an English word."""
    word = word.lower().strip()
    if len(word) <= 2:
        return 1
    # Remove trailing silent-e
    if word.endswith("e"):
        word = word[:-1]
    count = 0
    prev_vowel = False
    for ch in word:
        is_vowel = ch in _VOWELS
        if is_vowel and not prev_vowel:
            count += 1
        prev_vowel = is_vowel
    return max(count, 1)


def _safe_stdev(values: list[float]) -> float:
    if len(values) < 2:
        return 0.0
    return stdev(values)


_CONTRACTION_RE = re.compile(
    r"\b\w+['`\u2019](t|s|re|ve|ll|d|m)\b", re.IGNORECASE
)


# ---------------------------------------------------------------------------
# Main extraction
# ---------------------------------------------------------------------------


def extract_style(text: str) -> StyleProfile:
    """Analyse *text* and return a populated :class:`StyleProfile`."""
    if not text or not text.strip():
        return StyleProfile()

    nlp = _get_nlp()
    doc = nlp(text)

    # -- Tokenise into sentences and words --------------------------------
    sentences = list(doc.sents)
    num_sents = len(sentences)
    if num_sents == 0:
        return StyleProfile()

    # Words: alphabetic tokens only (for vocabulary metrics)
    words = [tok.text for tok in doc if tok.is_alpha]
    num_words = len(words)
    if num_words == 0:
        return StyleProfile()

    words_lower = [w.lower() for w in words]

    # -- Sentence structure ------------------------------------------------
    sent_lengths_words = [
        sum(1 for tok in sent if tok.is_alpha) for sent in sentences
    ]
    sent_lengths_chars = [len(sent.text.strip()) for sent in sentences]

    mean_sl_w = mean(sent_lengths_words) if sent_lengths_words else 0.0
    std_sl_w = _safe_stdev([float(v) for v in sent_lengths_words])
    mean_sl_c = mean(sent_lengths_chars) if sent_lengths_chars else 0.0
    std_sl_c = _safe_stdev([float(v) for v in sent_lengths_chars])

    question_count = sum(1 for s in sentences if s.text.strip().endswith("?"))
    exclamation_count = sum(
        1 for s in sentences if s.text.strip().endswith("!")
    )
    question_ratio = question_count / num_sents
    exclamation_ratio = exclamation_count / num_sents

    # -- Vocabulary --------------------------------------------------------
    word_freq = Counter(words_lower)
    num_types = len(word_freq)
    type_token_ratio = num_types / num_words
    hapax = sum(1 for w, c in word_freq.items() if c == 1)
    hapax_ratio = hapax / num_types if num_types else 0.0
    avg_word_len = mean([len(w) for w in words])
    syllable_counts = [_estimate_syllables(w) for w in words]
    syllable_complexity = mean(syllable_counts)

    # -- Punctuation -------------------------------------------------------
    comma_count = sum(1 for tok in doc if tok.text == ",")
    semicolon_count = sum(1 for tok in doc if tok.text == ";")
    dash_count = sum(
        1
        for tok in doc
        if tok.text in ("-", "\u2013", "\u2014", "--")
    )
    ellipsis_count = sum(
        1 for tok in doc if tok.text in ("\u2026",)
    ) + text.count("...")

    comma_density = comma_count / num_sents
    semicolon_density = semicolon_count / num_sents
    dash_density = dash_count / num_sents
    ellipsis_freq = ellipsis_count / num_sents

    # Oxford comma: look for ", X, and/or Y" patterns in the raw text
    oxford_hits = len(
        re.findall(r",\s+\w+,\s+(?:and|or)\s", text, re.IGNORECASE)
    )
    non_oxford_hits = len(
        re.findall(r",\s+\w+\s+(?:and|or)\s", text, re.IGNORECASE)
    )
    total_list_patterns = oxford_hits + non_oxford_hits
    oxford_score = (
        oxford_hits / total_list_patterns if total_list_patterns else 0.5
    )

    # -- Syntax / POS ------------------------------------------------------
    adj_count = sum(1 for tok in doc if tok.pos_ == "ADJ")
    adv_count = sum(1 for tok in doc if tok.pos_ == "ADV")
    adjective_density = adj_count / num_words
    adverb_density = adv_count / num_words

    # Passive voice: tokens with auxpass or nsubjpass dependency
    passive_sents = sum(
        1
        for sent in sentences
        if any(tok.dep_ in ("auxpass", "nsubjpass") for tok in sent)
    )
    passive_ratio = passive_sents / num_sents

    # Subordinate clauses: tokens with mark, advcl, acl dependency
    sub_clause_sents = sum(
        1
        for sent in sentences
        if any(tok.dep_ in ("mark", "advcl", "acl") for tok in sent)
    )
    sub_clause_freq = sub_clause_sents / num_sents

    # -- Rhythm ------------------------------------------------------------
    cv = (std_sl_w / mean_sl_w) if mean_sl_w > 0 else 0.0

    # -- Lexical preference ------------------------------------------------
    contraction_count = len(_CONTRACTION_RE.findall(text))
    contraction_freq = contraction_count / num_sents

    # Formality (Heylighen & Dewaele inspired)
    noun_count = sum(1 for tok in doc if tok.pos_ == "NOUN")
    prep_count = sum(1 for tok in doc if tok.pos_ == "ADP")
    pron_count = sum(1 for tok in doc if tok.pos_ == "PRON")
    verb_count = sum(1 for tok in doc if tok.pos_ == "VERB")
    formal_score_raw = (
        (noun_count + adj_count + prep_count)
        - (pron_count + adv_count + verb_count)
    )
    # Normalise into roughly [0, 1]
    formality_score = (formal_score_raw / num_words + 1) / 2
    formality_score = max(0.0, min(1.0, formality_score))

    # Distinctive words: top-15 most frequent content words (non-stop)
    content_freq = Counter(
        tok.lemma_.lower()
        for tok in doc
        if tok.is_alpha and not tok.is_stop and len(tok.text) > 2
    )
    distinctive = [w for w, _ in content_freq.most_common(15)]

    return StyleProfile(
        mean_sentence_length_words=round(mean_sl_w, 3),
        std_sentence_length_words=round(std_sl_w, 3),
        mean_sentence_length_chars=round(mean_sl_c, 3),
        std_sentence_length_chars=round(std_sl_c, 3),
        question_ratio=round(question_ratio, 4),
        exclamation_ratio=round(exclamation_ratio, 4),
        type_token_ratio=round(type_token_ratio, 4),
        hapax_legomena_ratio=round(hapax_ratio, 4),
        avg_word_length=round(avg_word_len, 3),
        syllable_complexity=round(syllable_complexity, 3),
        comma_density=round(comma_density, 4),
        semicolon_density=round(semicolon_density, 4),
        dash_density=round(dash_density, 4),
        ellipsis_frequency=round(ellipsis_freq, 4),
        oxford_comma_score=round(oxford_score, 4),
        adjective_density=round(adjective_density, 4),
        adverb_density=round(adverb_density, 4),
        passive_voice_ratio=round(passive_ratio, 4),
        subordinate_clause_frequency=round(sub_clause_freq, 4),
        sentence_length_variance=round(cv, 4),
        contraction_frequency=round(contraction_freq, 4),
        formality_score=round(formality_score, 4),
        distinctive_words=distinctive,
    )

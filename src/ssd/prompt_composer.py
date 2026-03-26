"""Translate a StyleProfile into natural-language writing instructions for an LLM."""

from __future__ import annotations

from ssd.models import StyleProfile


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _sentence_length_instruction(profile: StyleProfile) -> str:
    mean = profile.mean_sentence_length_words
    if mean < 10:
        desc = "short, punchy"
        range_hint = "averaging 5-10 words"
    elif mean < 18:
        desc = "moderate-length"
        range_hint = f"averaging around {int(mean)} words"
    else:
        desc = "long, flowing"
        range_hint = f"averaging {int(mean)}+ words"
    return f"Use {desc} sentences {range_hint}."


def _sentence_variation_instruction(profile: StyleProfile) -> str:
    std = profile.std_sentence_length_words
    variance = profile.sentence_length_variance
    if std < 4 and variance < 20:
        return "Keep sentence lengths fairly uniform."
    if std > 8 or variance > 60:
        return "Vary sentence length dramatically — mix very short and very long sentences."
    return "Vary sentence length moderately for natural rhythm."


def _question_exclamation_instruction(profile: StyleProfile) -> str:
    parts: list[str] = []
    if profile.question_ratio > 0.15:
        parts.append("Use rhetorical questions frequently.")
    elif profile.question_ratio > 0.05:
        parts.append("Include occasional questions.")
    else:
        parts.append("Rarely use questions.")

    if profile.exclamation_ratio > 0.10:
        parts.append("Use exclamation marks liberally for emphasis.")
    elif profile.exclamation_ratio > 0.03:
        parts.append("Use exclamation marks sparingly.")
    else:
        parts.append("Avoid exclamation marks.")
    return " ".join(parts)


def _vocabulary_instruction(profile: StyleProfile) -> str:
    avg_len = profile.avg_word_length
    ttr = profile.type_token_ratio
    syl = profile.syllable_complexity

    if avg_len < 4.5 and syl < 1.5:
        complexity = "simple, everyday"
    elif avg_len > 5.5 or syl > 2.0:
        complexity = "sophisticated, multisyllabic"
    else:
        complexity = "conversational but precise"

    if ttr > 0.7:
        variety = "Use a wide, varied vocabulary with few repeated words."
    elif ttr > 0.5:
        variety = "Use moderate vocabulary variety."
    else:
        variety = "Rely on a core set of familiar words; repetition is acceptable."

    return f"Vocabulary is {complexity}. {variety}"


def _punctuation_instruction(profile: StyleProfile) -> str:
    parts: list[str] = []

    if profile.comma_density > 0.06:
        parts.append("Use commas generously.")
    elif profile.comma_density < 0.02:
        parts.append("Use commas sparingly.")

    if profile.semicolon_density > 0.01:
        parts.append("Use semicolons where appropriate.")
    else:
        parts.append("Avoid semicolons.")

    if profile.dash_density > 0.02:
        parts.append("Use em-dashes liberally.")
    elif profile.dash_density > 0.005:
        parts.append("Use dashes occasionally.")
    else:
        parts.append("Avoid dashes.")

    if profile.ellipsis_frequency > 0.02:
        parts.append("Use ellipses for trailing thoughts.")
    elif profile.ellipsis_frequency > 0.005:
        parts.append("Use ellipses sparingly.")

    return " ".join(parts)


def _voice_instruction(profile: StyleProfile) -> str:
    ratio = profile.passive_voice_ratio
    if ratio > 0.3:
        return "Use passive voice frequently."
    if ratio > 0.15:
        return "Mix active and passive voice."
    return "Favor active voice."


def _syntax_instruction(profile: StyleProfile) -> str:
    freq = profile.subordinate_clause_frequency
    if freq > 0.3:
        return "Build complex sentences with subordinate clauses, relative clauses, and embedded phrases."
    if freq > 0.15:
        return "Use moderately complex sentence structures with occasional subordinate clauses."
    return "Keep sentence structures simple and direct."


def _adjective_adverb_instruction(profile: StyleProfile) -> str:
    adj = profile.adjective_density
    adv = profile.adverb_density
    combined = adj + adv
    if combined > 0.15:
        return "Use adjectives and adverbs richly to paint vivid descriptions."
    if combined > 0.08:
        return "Use adjectives and adverbs in moderation."
    return "Use adjectives and adverbs sparingly — prefer strong nouns and verbs."


def _contraction_instruction(profile: StyleProfile) -> str:
    freq = profile.contraction_frequency
    if freq > 0.05:
        return "Contractions are frequent."
    if freq > 0.02:
        return "Use contractions occasionally."
    return "Avoid contractions; use full forms."


def _formality_instruction(profile: StyleProfile) -> str:
    score = profile.formality_score
    if score > 0.65:
        return "Maintain a formal, polished tone."
    if score > 0.35:
        return "Use a neutral, balanced tone — neither overly formal nor casual."
    return "Write in a casual, approachable tone."


def _distinctive_words_instruction(profile: StyleProfile) -> str:
    words = profile.distinctive_words
    if not words:
        return ""
    sample = words[:10]
    quoted = ", ".join(f'"{w}"' for w in sample)
    return f"Where natural, incorporate characteristic words/phrases such as: {quoted}."


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def compose_style_prompt(profile: StyleProfile) -> str:
    """Convert a *StyleProfile* into natural-language writing instructions."""
    sections = [
        _sentence_length_instruction(profile),
        _sentence_variation_instruction(profile),
        _question_exclamation_instruction(profile),
        _vocabulary_instruction(profile),
        _punctuation_instruction(profile),
        _voice_instruction(profile),
        _syntax_instruction(profile),
        _adjective_adverb_instruction(profile),
        _contraction_instruction(profile),
        _formality_instruction(profile),
        _distinctive_words_instruction(profile),
    ]
    # Filter out empty strings and join
    return "\n".join(s for s in sections if s)


def _vocabulary_seed_instruction() -> str:
    """Encourage use of words from the synonym map to create stego choice points."""
    from ssd.synonym_map import SYNONYM_PAIRS

    # Pick a representative sample of common words from our pairs
    sample_words: list[str] = []
    for zero, one in SYNONYM_PAIRS[:40]:
        sample_words.extend([zero, one])
    quoted = ", ".join(sample_words[:30])
    return (
        f"Where natural, try to use common words like: {quoted}. "
        "These are just suggestions — only use them where they fit naturally."
    )


def compose_full_prompt(profile: StyleProfile, user_prompt: str) -> str:
    """Combine style instructions with the user's topic into a complete LLM prompt."""
    style_instructions = compose_style_prompt(profile)
    vocab_seed = _vocabulary_seed_instruction()
    return (
        "You are a ghostwriter. Your task is to write text that precisely "
        "mimics a specific author's style.\n\n"
        "## Style Instructions\n\n"
        f"{style_instructions}\n\n"
        "## Vocabulary Guidance\n\n"
        f"{vocab_seed}\n\n"
        "## Task\n\n"
        f"{user_prompt}\n\n"
        "Write the response now, following the style instructions above exactly. "
        "Do not mention or reference these instructions in your output."
    )

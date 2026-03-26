"""Synonym pairs and punctuation toggles for steganographic bit encoding.

Each synonym pair represents a 0-bit and 1-bit choice point. When encoding,
the encoder selects the appropriate variant based on the bit to embed.
When decoding, the decoder identifies which variant was used to recover the bit.
"""

from __future__ import annotations

# ---------------------------------------------------------------------------
# Synonym pairs: (zero_variant, one_variant)
# Both words in each pair should be interchangeable in most contexts.
# All entries are lowercase.
# ---------------------------------------------------------------------------

SYNONYM_PAIRS: list[tuple[str, str]] = [
    # --- adjectives ---
    ("big", "large"),
    ("small", "little"),
    ("fast", "quick"),
    ("happy", "glad"),
    ("sad", "unhappy"),
    ("important", "significant"),
    ("enough", "sufficient"),
    ("hard", "difficult"),
    ("easy", "simple"),
    ("smart", "intelligent"),
    ("brave", "courageous"),
    ("calm", "peaceful"),
    ("entire", "whole"),
    ("correct", "right"),
    ("strong", "powerful"),
    ("weak", "feeble"),
    ("rich", "wealthy"),
    ("poor", "impoverished"),
    ("strange", "odd"),
    ("beautiful", "gorgeous"),
    ("ugly", "hideous"),
    ("angry", "furious"),
    ("funny", "humorous"),
    ("shy", "timid"),
    ("rude", "impolite"),
    ("polite", "courteous"),
    ("genuine", "authentic"),
    ("fake", "counterfeit"),
    ("clear", "obvious"),
    ("vague", "ambiguous"),
    ("huge", "enormous"),
    ("tiny", "miniature"),
    ("ancient", "old"),
    ("modern", "contemporary"),
    ("essential", "crucial"),
    ("optional", "voluntary"),
    ("frequent", "common"),
    ("rare", "uncommon"),
    ("broad", "wide"),
    ("narrow", "slim"),
    # --- verbs ---
    ("begin", "start"),
    ("end", "finish"),
    ("show", "demonstrate"),
    ("use", "utilize"),
    ("help", "assist"),
    ("get", "obtain"),
    ("make", "create"),
    ("think", "believe"),
    ("give", "provide"),
    ("buy", "purchase"),
    ("pick", "choose"),
    ("keep", "retain"),
    ("fix", "repair"),
    ("try", "attempt"),
    ("ask", "inquire"),
    ("answer", "respond"),
    ("allow", "permit"),
    ("forbid", "prohibit"),
    ("find", "discover"),
    ("hide", "conceal"),
    ("gather", "collect"),
    ("remove", "eliminate"),
    ("raise", "increase"),
    ("lower", "decrease"),
    ("change", "modify"),
    ("need", "require"),
    ("want", "desire"),
    ("seem", "appear"),
    ("tell", "inform"),
    ("leave", "depart"),
    ("arrive", "reach"),
    ("build", "construct"),
    ("destroy", "demolish"),
    ("check", "verify"),
    ("teach", "educate"),
    ("learn", "study"),
    ("copy", "duplicate"),
    ("join", "connect"),
    ("split", "divide"),
    ("grab", "seize"),
    # --- adverbs ---
    ("quickly", "rapidly"),
    ("slowly", "gradually"),
    ("often", "frequently"),
    ("rarely", "seldom"),
    ("almost", "nearly"),
    ("totally", "completely"),
    ("usually", "typically"),
    ("mainly", "primarily"),
    ("really", "truly"),
    ("lately", "recently"),
    # --- conjunctions / transitions ---
    ("however", "nevertheless"),
    ("also", "additionally"),
    ("although", "though"),
    ("because", "since"),
    ("therefore", "thus"),
    ("moreover", "furthermore"),
    ("but", "yet"),
    ("so", "hence"),
    ("meanwhile", "simultaneously"),
    ("likewise", "similarly"),
]

# ---------------------------------------------------------------------------
# Punctuation toggles: each encodes one bit via a punctuation style choice.
# ---------------------------------------------------------------------------

PUNCTUATION_TOGGLES: list[dict] = [
    {
        "name": "oxford_comma",
        "pattern_zero": r",\s+and\b",
        "pattern_one": r"(?<!,)\s+and\b",
        "replace_zero": ", and",
        "replace_one": " and",
    },
    {
        "name": "em_dash_spacing",
        "pattern_zero": r"(?<! )\u2014(?! )",
        "pattern_one": r" \u2014 ",
        "replace_zero": "\u2014",
        "replace_one": " \u2014 ",
    },
    {
        "name": "semicolon_vs_period",
        "pattern_zero": r";\s+",
        "pattern_one": r"\.\s+",
        "replace_zero": "; ",
        "replace_one": ". ",
    },
    {
        "name": "double_quote_style",
        "pattern_zero": r'\u201c([^\u201d]*)\u201d',
        "pattern_one": r'"([^"]*)"',
        "replace_zero": "\u201c\\1\u201d",
        "replace_one": '"\\1"',
    },
    {
        "name": "ellipsis_style",
        "pattern_zero": r"\.\.\.",
        "pattern_one": r"\u2026",
        "replace_zero": "...",
        "replace_one": "\u2026",
    },
    {
        "name": "colon_spacing",
        "pattern_zero": r":\s{1}(?!\s)",
        "pattern_one": r":\s{2}",
        "replace_zero": ": ",
        "replace_one": ":  ",
    },
]


def build_synonym_lookup() -> dict[str, tuple[str, int, int]]:
    """Build a reverse lookup from any synonym word to its pair metadata.

    Returns a dict mapping each word to a tuple of:
        (partner_word, pair_index, bit_value)

    where bit_value is 0 for the first element and 1 for the second element
    of the pair at pair_index.
    """
    lookup: dict[str, tuple[str, int, int]] = {}
    for idx, (zero_word, one_word) in enumerate(SYNONYM_PAIRS):
        lookup[zero_word] = (one_word, idx, 0)
        lookup[one_word] = (zero_word, idx, 1)
    return lookup

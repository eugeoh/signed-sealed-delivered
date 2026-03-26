# Signed, Sealed, Delivered

Stylometric identity service — extract writing style, transfer it to new content, and embed verifiable signatures using steganography.

## What it does

1. **Analyze** a writing sample to extract a 22-feature style profile and a deterministic style hash
2. **Generate** new text in that style (via Claude) with the hash steganographically embedded
3. **Verify** that a piece of text was generated from a specific style profile by extracting the hidden signature

The style hash works like a public key for your writing voice. Publish it in your bio, then anyone can verify text you've produced.

## How it works

```
Writing Sample
  → Style Extractor (spaCy NLP → 22 features)
  → Style Hasher (quantize → SHA-256 → 64-bit hash)
  → Prompt Composer (features → natural-language writing instructions)
  → Claude API (generates styled text)
  → Stego Encoder (embeds hash via synonym/punctuation/spacing choices)
  → Signed text with verifiable authorship
```

**Steganographic encoding** uses three cascading layers:
- **Synonym substitution** — 100 word pairs (big↔large, begin↔start, etc.) where each choice encodes a bit
- **Punctuation toggles** — Oxford commas, em-dash spacing, quote styles, etc.
- **Sentence spacing** — single vs. double space after sentence-ending punctuation

All layers are invisible to readers. Verification uses Hamming distance with a configurable confidence threshold.

## Quickstart

```bash
# Clone and install
git clone https://github.com/eugeneheo/signed-sealed-delivered.git
cd signed-sealed-delivered
uv sync
python -m spacy download en_core_web_sm

# Set your Anthropic API key
echo 'ANTHROPIC_API_KEY="sk-ant-..."' > .env.local

# Start the server
uv run uvicorn ssd.api:app --reload

# Open http://localhost:8000 in your browser
```

## API

### `POST /analyze`

Extract style profile from a writing sample.

```json
{
  "sample": "Your writing sample here (min 50 chars)..."
}
```

Returns `style_profile` (22 features), `style_hash` (base58), and `style_description` (natural language).

### `POST /generate`

Generate styled text with an embedded signature.

```json
{
  "style_hash": "4j5kL9mNpQr2S3t",
  "prompt": "Write about the sea",
  "author": "Ernest Hemingway"
}
```

Or pass `sample` instead of `style_hash` to analyze on the fly. Returns `text` with embedded signature block, `signature`, and `style_profile`.

### `POST /verify`

Verify authorship of generated text.

```json
{
  "text": "The generated text to verify...",
  "style_hash": "4j5kL9mNpQr2S3t"
}
```

Returns `match` (bool), `confidence` (0–1), and `extracted_hash` vs `expected_hash`.

## Style features extracted

| Category | Features |
|---|---|
| Sentence structure | Mean/std sentence length (words & chars), question ratio, exclamation ratio |
| Vocabulary | Type-token ratio, hapax legomena ratio, avg word length, syllable complexity |
| Punctuation | Comma, semicolon, dash, ellipsis density; Oxford comma score |
| Syntax | Adjective/adverb density, passive voice ratio, subordinate clause frequency |
| Rhythm | Sentence length variance |
| Lexical preference | Contraction frequency, formality score, distinctive words |

## Project structure

```
src/ssd/
├── api.py              # FastAPI endpoints + static file serving
├── models.py           # StyleProfile dataclass + Pydantic schemas
├── style_extractor.py  # spaCy-based 22-feature extraction
├── style_hasher.py     # Quantization → SHA-256 → base58 signature
├── prompt_composer.py  # Style profile → LLM writing instructions
├── stego_encoder.py    # Synonym/punctuation/spacing bit embedding
├── stego_decoder.py    # Bit extraction + Hamming distance verification
├── synonym_map.py      # 100 synonym pairs + 6 punctuation toggles
└── static/
    └── index.html      # Single-page web UI
```

## Development

```bash
uv sync --extra dev
uv run pytest
```

## License

MIT

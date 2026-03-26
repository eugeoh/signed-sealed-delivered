"""FastAPI service wiring all stylometric modules together."""

from __future__ import annotations

from pathlib import Path

import anthropic
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import RedirectResponse
from fastapi.staticfiles import StaticFiles

# Load .env.local from project root (if present)
load_dotenv(Path(__file__).resolve().parents[2] / ".env.local")

from ssd.models import (
    AnalyzeRequest,
    AnalyzeResponse,
    GenerateRequest,
    GenerateResponse,
    VerifyRequest,
    VerifyResponse,
)
from ssd.prompt_composer import compose_full_prompt, compose_style_prompt
from ssd.stego_decoder import decode, verify
from ssd.stego_encoder import encode
from ssd.style_extractor import extract_style
from ssd.style_hasher import (
    compute_style_hash,
    style_hash_to_bits,
    style_signature,
)

app = FastAPI(
    title="Signed, Sealed, Delivered",
    description="Stylometric identity service — extract writing style, transfer it, and embed verifiable signatures.",
    version="0.1.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

_static_dir = Path(__file__).resolve().parent / "static"
app.mount("/static", StaticFiles(directory=_static_dir, html=True), name="static")


@app.get("/")
async def root():
    return RedirectResponse(url="/static/index.html")


_default_client: anthropic.Anthropic | None = None

# In-memory cache: style_hash (base58) -> (StyleProfile, style_description)
_profile_cache: dict[str, tuple[object, str]] = {}


def _resolve_profile(
    sample: str | None, style_hash: str | None
) -> tuple:
    """Return (profile, sig, description) from either a sample or a cached hash."""
    from ssd.models import StyleProfile

    if style_hash and style_hash in _profile_cache:
        profile, description = _profile_cache[style_hash]
        return profile, style_hash, description
    if sample:
        profile = extract_style(sample)
        sig = style_signature(profile)
        description = compose_style_prompt(profile)
        _profile_cache[sig] = (profile, description)
        return profile, sig, description
    if style_hash:
        raise HTTPException(
            status_code=404,
            detail=f"Style hash '{style_hash}' not found in cache. Analyze a sample first.",
        )
    raise HTTPException(
        status_code=422,
        detail="Provide either 'sample' or 'style_hash'.",
    )


def _get_client(api_key: str | None = None) -> anthropic.Anthropic:
    """Return an Anthropic client, using the provided key or falling back to env."""
    if api_key:
        return anthropic.Anthropic(api_key=api_key)
    global _default_client
    if _default_client is None:
        _default_client = anthropic.Anthropic()
    return _default_client


# ---------------------------------------------------------------------------
# POST /analyze
# ---------------------------------------------------------------------------

@app.post("/analyze", response_model=AnalyzeResponse)
async def analyze(req: AnalyzeRequest) -> AnalyzeResponse:
    """Extract style profile, hash, and description from a writing sample."""
    profile = extract_style(req.sample)
    sig = style_signature(profile)
    description = compose_style_prompt(profile)
    _profile_cache[sig] = (profile, description)
    return AnalyzeResponse(
        style_profile=profile.to_dict(),
        style_hash=sig,
        style_description=description,
    )


# ---------------------------------------------------------------------------
# POST /generate
# ---------------------------------------------------------------------------

@app.post("/generate", response_model=GenerateResponse)
async def generate(req: GenerateRequest) -> GenerateResponse:
    """Generate styled text with an embedded steganographic signature."""
    profile, sig, _desc = _resolve_profile(req.sample, req.style_hash)
    full_prompt = compose_full_prompt(profile, req.prompt)

    # Call the Anthropic API
    client = _get_client(req.api_key)
    try:
        message = client.messages.create(
            model=req.model,
            max_tokens=1024,
            messages=[{"role": "user", "content": full_prompt}],
        )
    except anthropic.APIError as exc:
        raise HTTPException(status_code=502, detail=f"LLM API error: {exc}") from exc

    raw_text = message.content[0].text

    # Embed the style hash steganographically
    hash_bytes = compute_style_hash(profile)
    hash_bits = style_hash_to_bits(hash_bytes)
    stego_text = encode(raw_text, hash_bits)

    # Append a visible signature block
    sig_lines = []
    if req.author:
        sig_lines.append(f"Author: {req.author}")
    sig_lines.append(f"Style-Signature: {sig}")
    output_text = f"{stego_text}\n\n---\n" + "\n".join(sig_lines)

    return GenerateResponse(
        text=output_text,
        signature=sig,
        style_profile=profile.to_dict(),
    )


# ---------------------------------------------------------------------------
# POST /verify
# ---------------------------------------------------------------------------

@app.post("/verify", response_model=VerifyResponse)
async def verify_endpoint(req: VerifyRequest) -> VerifyResponse:
    """Verify that text contains the steganographic signature of a claimed author."""
    profile, _sig, _desc = _resolve_profile(req.sample, req.style_hash)
    hash_bytes = compute_style_hash(profile)
    expected_bits = style_hash_to_bits(hash_bytes)

    # Strip the appended signature block before decoding
    text = req.text.split("\n---\nStyle-Signature:")[0].rstrip()

    match, confidence = verify(text, expected_bits)

    extracted_raw = decode(text, len(expected_bits))
    # Pad
    while len(extracted_raw) < len(expected_bits):
        extracted_raw.append(0)
    extracted_hex = _bits_to_hex(extracted_raw)
    expected_hex = _bits_to_hex(expected_bits)

    return VerifyResponse(
        match=match,
        confidence=confidence,
        extracted_hash=extracted_hex,
        expected_hash=expected_hex,
    )


def _bits_to_hex(bits: list[int]) -> str:
    """Convert a list of bits to a hex string."""
    value = 0
    for b in bits:
        value = (value << 1) | b
    return format(value, f"0{len(bits) // 4}x")

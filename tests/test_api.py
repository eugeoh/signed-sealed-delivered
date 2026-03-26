"""Tests for the FastAPI endpoints."""

import pytest
from fastapi.testclient import TestClient

from ssd.api import app

client = TestClient(app)

WRITING_SAMPLE = (
    "He sat by the window. The rain fell. It was cold outside. "
    "He drank his coffee. It was black and strong. He didn't say anything. "
    "She looked at him. He looked away. The silence was heavy. "
    "He stood up and left. She stayed. The rain kept falling. "
    "The dog barked once. Then it was quiet again."
)


def test_analyze_endpoint():
    resp = client.post("/analyze", json={"sample": WRITING_SAMPLE})
    assert resp.status_code == 200
    data = resp.json()
    assert "style_profile" in data
    assert "style_hash" in data
    assert "style_description" in data
    assert isinstance(data["style_hash"], str)


def test_analyze_deterministic():
    r1 = client.post("/analyze", json={"sample": WRITING_SAMPLE}).json()
    r2 = client.post("/analyze", json={"sample": WRITING_SAMPLE}).json()
    assert r1["style_hash"] == r2["style_hash"]


def test_analyze_rejects_short_sample():
    resp = client.post("/analyze", json={"sample": "Too short."})
    assert resp.status_code == 422


def test_verify_endpoint_structure():
    """Test that verify endpoint returns correct structure with crafted input."""
    # Use a text that contains synonym words to have bits to extract
    text = (
        "However, the big changes are important. We also think this will "
        "help people quickly obtain answers. The fast response was enough."
    )
    resp = client.post("/verify", json={"text": text, "sample": WRITING_SAMPLE})
    assert resp.status_code == 200
    data = resp.json()
    assert "match" in data
    assert "confidence" in data
    assert "extracted_hash" in data
    assert "expected_hash" in data
    assert isinstance(data["confidence"], float)

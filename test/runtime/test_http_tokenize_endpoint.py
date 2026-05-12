from __future__ import annotations

from types import SimpleNamespace

from fastapi.testclient import TestClient

from tokenspeed.runtime.entrypoints import http_server


class StubTokenizer:
    def encode(self, text):
        return [ord(char) for char in text]


def test_tokenize_endpoint_returns_prompt_token_ids(monkeypatch):
    monkeypatch.setattr(
        http_server,
        "_global_state",
        SimpleNamespace(tokenizer_manager=SimpleNamespace(tokenizer=StubTokenizer())),
    )

    response = TestClient(http_server.app).post("/tokenize", json={"prompt": "Az"})

    assert response.status_code == 200
    assert response.json() == {"tokens": [65, 122], "token_ids": [65, 122]}


def test_tokenize_endpoint_rejects_missing_tokenizer(monkeypatch):
    monkeypatch.setattr(
        http_server,
        "_global_state",
        SimpleNamespace(tokenizer_manager=SimpleNamespace(tokenizer=None)),
    )

    response = TestClient(http_server.app).post("/tokenize", json={"prompt": "Az"})

    assert response.status_code == 400
    assert response.json()["message"] == "tokenizer is not initialized"

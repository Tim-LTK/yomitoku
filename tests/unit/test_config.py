import pytest

from yomitoku_api.config import Settings


def test_allowed_origins_splits_csv(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("ALLOWED_ORIGINS", " https://a.test , https://b.test ")
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    cfg = Settings()
    assert cfg.allowed_origins == ["https://a.test", "https://b.test"]


def test_allowed_origins_empty_when_unset(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("ALLOWED_ORIGINS", raising=False)
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    cfg = Settings()
    assert cfg.allowed_origins == []

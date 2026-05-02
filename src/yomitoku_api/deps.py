"""Shared FastAPI dependencies."""

from functools import lru_cache

from yomitoku_api.config import Settings


@lru_cache
def get_settings_cached() -> Settings:
    """Process-wide settings — restart after changing `.env` or Railway variables."""

    return Settings()

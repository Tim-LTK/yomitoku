"""Cached Supabase REST client — required for SRS gap persistence."""

from __future__ import annotations

from supabase import Client, create_client

from yomitoku_api.deps import get_settings_cached

_client: Client | None = None


def get_supabase_client() -> Client:
    """Return a process-local `create_client` instance."""

    global _client
    if _client is not None:
        return _client

    settings = get_settings_cached()
    url_raw = settings.supabase_url
    key_raw = settings.supabase_anon_key
    url = (url_raw or "").strip()
    key = (key_raw or "").strip()
    if not url or not key:
        raise RuntimeError("SUPABASE_URL and SUPABASE_ANON_KEY must both be set.")

    _client = create_client(url, key)
    return _client

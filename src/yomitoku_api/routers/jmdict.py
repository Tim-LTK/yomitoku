"""
yomitoku_api/routers/jmdict.py — Phase 3.1
GET /jmdict/lookup?term={text} → JmdictLookupResult
"""

from __future__ import annotations

import logging
from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException
from supabase import Client

from yomitoku_api.config import Settings
from yomitoku_api.deps import get_settings_cached
from yomitoku_api.services.jmdict import JmdictLookupResult, lookup
from yomitoku_api.services.supabase_client import get_supabase_client

log = logging.getLogger(__name__)

router = APIRouter(prefix="/jmdict", tags=["jmdict"])

SettingsDep = Depends(get_settings_cached)


def require_supabase_client() -> Client:
    """503 when Supabase env vars absent."""
    try:
        return get_supabase_client()
    except RuntimeError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc


SupabaseDep = Annotated[Client, Depends(require_supabase_client)]


@router.get(
    "/lookup",
    response_model=JmdictLookupResult,
    summary="Look up a Japanese term in JMdict",
    description=(
        "Returns JLPT level, pitch accent, meanings, and parts of speech. "
        "source='jmdict' = from database. "
        "source='fallback_ai' = term not in DB, AI best-effort estimate."
    ),
)
def lookup_term(
    term: str,
    supabase: SupabaseDep,
    settings: Settings = SettingsDep,
) -> JmdictLookupResult:
    term = term.strip()
    if not term:
        raise HTTPException(status_code=422, detail="term must not be blank")

    log.info("jmdict lookup: %r", term)
    return lookup(term=term, supabase=supabase, settings=settings)

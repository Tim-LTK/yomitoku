"""HTTP router — Supabase-backed knowledge gaps plus Claude SRS spacing hints."""

from __future__ import annotations

import logging
from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException
from supabase import Client

from yomitoku_api.config import Settings
from yomitoku_api.deps import get_settings_cached
from yomitoku_api.schemas import (
    KnowledgeGap,
    KnowledgeGapPartial,
    PracticeResult,
    ProblemDetail,
    SrsComputeRequest,
    SrsComputeResponse,
)
from yomitoku_api.services import prompts as prompt_service
from yomitoku_api.services import srs_compute as srs_compute_gen
from yomitoku_api.services import srs_gaps as srs_gaps_service
from yomitoku_api.services import validate as validate_service
from yomitoku_api.services.supabase_client import get_supabase_client

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/srs", tags=["srs"])
SettingsDep = Depends(get_settings_cached)


def require_supabase_client() -> Client:
    """503 when Postgres bridge env vars absent — keeps cold starts deterministic."""

    try:
        return get_supabase_client()
    except RuntimeError as exc:
        raise HTTPException(
            status_code=503,
            detail=ProblemDetail(title="SRS storage unavailable", detail=str(exc)).model_dump(),
        ) from exc


SupabaseDep = Annotated[Client, Depends(require_supabase_client)]


@router.post("/gaps", response_model=KnowledgeGap, summary="Upsert one knowledge-gap row")
def upsert_gap_route(
    body: KnowledgeGap,
    client: SupabaseDep,
    _settings: Settings = SettingsDep,
) -> KnowledgeGap:
    srs_gaps_service.upsert_gap(client, body)
    return body


@router.get("/gaps", response_model=list[KnowledgeGap], summary="List stored knowledge gaps")
def list_gaps_route(client: SupabaseDep, _settings: Settings = SettingsDep) -> list[KnowledgeGap]:
    return srs_gaps_service.list_gaps(client)


@router.delete("/gaps/{gap_id}", status_code=204, summary="Delete gap by primary key")
def delete_gap_route(gap_id: str, client: SupabaseDep, _settings: Settings = SettingsDep) -> None:
    srs_gaps_service.delete_gap(client, gap_id)


@router.post(
    "/gaps/{gap_id}/results",
    response_model=KnowledgeGap,
    summary="Append one practice grading row tied to gap_id",
)
def append_result_route(
    gap_id: str,
    body: PracticeResult,
    client: SupabaseDep,
    _settings: Settings = SettingsDep,
) -> KnowledgeGap:
    gap = srs_gaps_service.append_practice_result(client, gap_id, body)
    if gap is None:
        raise HTTPException(
            status_code=404,
            detail=ProblemDetail(
                title="Gap not found",
                detail=f"No row exists with id={gap_id!r}.",
            ).model_dump(),
        )
    return gap


@router.patch(
    "/gaps/{gap_id}",
    response_model=KnowledgeGap,
    summary="Merge sparse JSON into an existing KnowledgeGap blob",
)
def patch_gap_route(
    gap_id: str,
    body: KnowledgeGapPartial,
    client: SupabaseDep,
    _settings: Settings = SettingsDep,
) -> KnowledgeGap:
    merged = srs_gaps_service.update_gap_partial(client, gap_id, body)
    if merged is None:
        raise HTTPException(
            status_code=404,
            detail=ProblemDetail(
                title="Gap not found",
                detail=f"No row exists with id={gap_id!r}.",
            ).model_dump(),
        )
    return merged


@router.post(
    "/compute",
    response_model=SrsComputeResponse,
    summary="Infer next review spacing from gap snapshot + chronological practice grades",
)
def compute_schedule_route(
    body: SrsComputeRequest,
    settings: Settings = SettingsDep,
) -> SrsComputeResponse:
    bundle = prompt_service.build_srs_compute_bundle(
        settings,
        gap=body.gap,
        results=body.results,
    )
    raw = srs_compute_gen.generate_srs_schedule(settings, bundle)
    validation = validate_service.validate_srs_compute(raw)
    if not validation.is_valid or validation.srs_compute is None:
        logger.info(
            "validation.failed.srs_compute",
            extra={"issue_count": len(validation.issues)},
        )
        raise HTTPException(
            status_code=422,
            detail=[issue.model_dump() for issue in validation.issues],
        )
    return validation.srs_compute

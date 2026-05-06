"""HTTP router for Phase 2 practice — session generate + batch submit."""

import logging

from fastapi import APIRouter, Depends, HTTPException

from yomitoku_api.config import Settings
from yomitoku_api.deps import get_settings_cached
from yomitoku_api.exceptions import PromptNotFoundError
from yomitoku_api.schemas import (
    PracticeGenerateRequest,
    PracticeItem,
    SessionResult,
    SessionSubmission,
)
from yomitoku_api.services import practice as practice_service
from yomitoku_api.services import prompts as prompt_service

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/practice", tags=["practice"])
SettingsDep = Depends(get_settings_cached)


@router.post(
    "/generate",
    response_model=list[PracticeItem],
    summary="Compose session drills from learner gaps",
)
def post_practice_generate(
    body: PracticeGenerateRequest,
    settings: Settings = SettingsDep,
) -> list[PracticeItem]:
    student_context = prompt_service.resolve_request_student_context(body.student_context)
    try:
        items = practice_service.compose_practice_session_items(
            settings,
            body.gaps,
            student_context=student_context,
        )
    except ValueError as exc:
        logger.info("practice.generate.validation", extra={"detail": str(exc)})
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except PromptNotFoundError:
        raise
    return items


@router.post(
    "/submit",
    response_model=SessionResult,
    summary="Batch-evaluate answers, tutor summary, SRS intervals",
)
def post_practice_submit(
    body: SessionSubmission,
    settings: Settings = SettingsDep,
) -> SessionResult:
    student_context = prompt_service.resolve_request_student_context(body.student_context)
    try:
        return practice_service.finalize_session_results(
            settings,
            body,
            student_context=student_context,
        )
    except ValueError as exc:
        logger.info("practice.submit.validation", extra={"detail": str(exc)})
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except PromptNotFoundError:
        raise

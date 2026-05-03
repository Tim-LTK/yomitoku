"""HTTP router for Phase 2 practice drills — discrete from `/analyse` and `/extract`."""

import logging

from fastapi import APIRouter, Depends, HTTPException

from yomitoku_api.config import Settings
from yomitoku_api.deps import get_settings_cached
from yomitoku_api.schemas import (
    PracticeEvaluateRequest,
    PracticeEvaluateResponse,
    PracticeGenerateRequest,
    PracticeGenerateResponse,
)
from yomitoku_api.services import practice as practice_gen
from yomitoku_api.services import prompts as prompt_service
from yomitoku_api.services import validate as validate_service

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/practice", tags=["practice"])
SettingsDep = Depends(get_settings_cached)


@router.post(
    "/generate",
    response_model=PracticeGenerateResponse,
    summary="SentenceBreakdown JSON → grounded practice items",
)
def post_practice_generate(
    body: PracticeGenerateRequest,
    settings: Settings = SettingsDep,
) -> PracticeGenerateResponse:
    student_context = prompt_service.resolve_request_student_context(body.student_context)
    bundle = prompt_service.build_practice_generate_bundle(
        settings,
        body.sentence_breakdown,
        student_context=student_context,
    )
    raw = practice_gen.generate_practice_items(settings, bundle)
    validation = validate_service.validate_practice_generation(raw)
    if not validation.is_valid or validation.practice_items is None:
        logger.info(
            "validation.failed.practice_generate",
            extra={"issue_count": len(validation.issues)},
        )
        raise HTTPException(
            status_code=422,
            detail=[issue.model_dump() for issue in validation.issues],
        )
    return PracticeGenerateResponse(items=validation.practice_items)


@router.post(
    "/evaluate",
    response_model=PracticeEvaluateResponse,
    summary="Evaluate learner answer vs practice item + breakdown snapshot",
)
def post_practice_evaluate(
    body: PracticeEvaluateRequest,
    settings: Settings = SettingsDep,
) -> PracticeEvaluateResponse:
    student_context = prompt_service.resolve_request_student_context(body.student_context)
    bundle = prompt_service.build_practice_evaluate_bundle(
        settings,
        breakdown=body.sentence_breakdown,
        practice_item=body.practice_item,
        user_answer=body.user_answer,
        student_context=student_context,
    )
    raw = practice_gen.evaluate_practice_attempt(settings, bundle)
    validation = validate_service.validate_practice_evaluation(raw)
    if not validation.is_valid or validation.practice_result is None:
        logger.info(
            "validation.failed.practice_evaluate",
            extra={"issue_count": len(validation.issues)},
        )
        raise HTTPException(
            status_code=422,
            detail=[issue.model_dump() for issue in validation.issues],
        )
    return PracticeEvaluateResponse(result=validation.practice_result)

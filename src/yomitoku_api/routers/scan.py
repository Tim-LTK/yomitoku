"""HTTP router — Phase 1.7 targeted scan + passage Q&A."""

import logging

from fastapi import APIRouter, Depends, HTTPException

from yomitoku_api.config import Settings
from yomitoku_api.deps import get_settings_cached
from yomitoku_api.schemas import AskRequest, AskResponse, ScanRequest, ScanResponse
from yomitoku_api.services import ask as ask_gen
from yomitoku_api.services import prompts as prompt_service
from yomitoku_api.services import scan as scan_gen
from yomitoku_api.services import validate as validate_service

logger = logging.getLogger(__name__)

router = APIRouter(tags=["scan"])
SettingsDep = Depends(get_settings_cached)


@router.post(
    "/scan",
    response_model=ScanResponse,
    summary="Targeted grammar / vocabulary / expression scan for a passage",
)
def post_scan(body: ScanRequest, settings: Settings = SettingsDep) -> ScanResponse:
    student_context = prompt_service.resolve_request_student_context(body.student_context)
    bundle = prompt_service.build_targeted_scan_bundle(
        settings,
        body.text.strip(),
        student_context=student_context,
    )
    raw = scan_gen.generate_targeted_scan(settings, bundle)
    validation = validate_service.validate_scan_generation(raw)
    if not validation.is_valid or validation.scan_result is None:
        logger.info(
            "validation.failed.targeted_scan",
            extra={"issue_count": len(validation.issues)},
        )
        raise HTTPException(
            status_code=422,
            detail=[issue.model_dump() for issue in validation.issues],
        )
    return ScanResponse.model_validate(validation.scan_result.model_dump())


@router.post(
    "/ask",
    response_model=AskResponse,
    summary="Answer a question grounded in a specific passage",
)
def post_ask(body: AskRequest, settings: Settings = SettingsDep) -> AskResponse:
    student_context = prompt_service.resolve_request_student_context(body.student_context)
    bundle = prompt_service.build_scan_ask_bundle(
        settings,
        passage=body.passage.strip(),
        question=body.question.strip(),
        student_context=student_context,
    )
    raw = ask_gen.generate_ask_answer(settings, bundle)
    validation = validate_service.validate_ask_generation(raw)
    if not validation.is_valid or validation.ask_response is None:
        logger.info(
            "validation.failed.scan_ask",
            extra={"issue_count": len(validation.issues)},
        )
        raise HTTPException(
            status_code=422,
            detail=[issue.model_dump() for issue in validation.issues],
        )
    return validation.ask_response

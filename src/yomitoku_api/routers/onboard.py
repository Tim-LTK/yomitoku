"""HTTP router — Phase 1.6 onboarding assessment."""

from __future__ import annotations

import json
import logging

from fastapi import APIRouter, Depends, HTTPException

from yomitoku_api.config import Settings
from yomitoku_api.deps import get_settings_cached
from yomitoku_api.schemas import OnboardingAssessRequest, StudentProfile
from yomitoku_api.services import onboard as onboard_gen
from yomitoku_api.services import prompts as prompt_service
from yomitoku_api.services import validate as validate_service

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/onboard", tags=["onboard"])
SettingsDep = Depends(get_settings_cached)


@router.post(
    "/assess",
    response_model=StudentProfile,
    summary="Placement answers → assessed learner profile",
)
def post_onboard_assess(
    body: OnboardingAssessRequest,
    settings: Settings = SettingsDep,
) -> StudentProfile:
    student_context = prompt_service.resolve_request_student_context(body.student_context)
    bundle = prompt_service.build_onboard_assess_bundle(
        settings,
        native_languages_json=json.dumps(body.native_languages, ensure_ascii=False),
        self_reported_level=body.self_reported_level,
        answers_json=json.dumps(body.answers.model_dump(), ensure_ascii=False),
        student_context=student_context,
    )
    raw = onboard_gen.generate_onboarding_assessment(settings, bundle)
    validation = validate_service.validate_onboarding_assessment(raw)
    if not validation.is_valid or validation.student_profile is None:
        logger.info(
            "validation.failed.onboard_assess",
            extra={"issue_count": len(validation.issues)},
        )
        raise HTTPException(
            status_code=422,
            detail=[issue.model_dump() for issue in validation.issues],
        )
    return validation.student_profile

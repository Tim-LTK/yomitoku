"""HTTP router for OCR / handwriting extraction."""

import logging

from fastapi import APIRouter, Depends, HTTPException

from yomitoku_api.config import Settings
from yomitoku_api.deps import get_settings_cached
from yomitoku_api.schemas import ExtractRequest, ExtractResponse
from yomitoku_api.services import extract as extract_gen
from yomitoku_api.services import prompts as prompt_service
from yomitoku_api.services import validate as validate_service

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/extract", tags=["extract"])
SettingsDep = Depends(get_settings_cached)


@router.post("", response_model=ExtractResponse, summary="Image → plaintext Japanese")
def post_extract(body: ExtractRequest, settings: Settings = SettingsDep) -> ExtractResponse:
    student_context = prompt_service.resolve_request_student_context(body.student_context)
    bundle = prompt_service.build_scan_extract_bundle(settings, student_context=student_context)
    b64_preview = body.image_base64[:100]
    logger.info(
        "extract.incoming_image",
        extra={"image_base64_start": b64_preview, "media_type": body.media_type},
    )
    raw = extract_gen.generate_text_from_image(
        settings,
        bundle,
        image_base64=body.image_base64,
        media_type=body.media_type,
    )
    validation = validate_service.validate_plain_extract_text(raw)
    if not validation.is_valid:
        raise HTTPException(
            status_code=422,
            detail=[issue.model_dump() for issue in validation.issues],
        )
    cleaned = validate_service.strip_code_fences(raw.raw_text).strip()
    return ExtractResponse(text=cleaned)

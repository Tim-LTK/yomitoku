"""Prompt layer — loads versioned `.txt` files and injects `STUDENT_CONTEXT`."""

from dataclasses import dataclass
from pathlib import Path

from yomitoku_api.config import Settings
from yomitoku_api.constants import STUDENT_CONTEXT
from yomitoku_api.exceptions import PromptNotFoundError


@dataclass(frozen=True)
class PromptBundle:
    """Anthropic system/user pair plus feature → prompt version metadata."""

    system: str
    user: str
    prompt_versions: dict[str, str]


def _read_utf8(settings: Settings, filename: str) -> str:
    path: Path = settings.prompts_dir / filename
    if not path.is_file():
        raise PromptNotFoundError(filename)
    return path.read_text(encoding="utf-8")


def _inject_student_context(fragment: str) -> str:
    """Replace template token `{STUDENT_CONTEXT}` wherever prompt authors placed it."""

    return fragment.replace("{STUDENT_CONTEXT}", STUDENT_CONTEXT)


def build_scan_extract_bundle(settings: Settings) -> PromptBundle:
    system = _inject_student_context(_read_utf8(settings, "scan_extract_v1_system.txt"))
    user = _inject_student_context(_read_utf8(settings, "scan_extract_v1_user.txt"))
    return PromptBundle(
        system=system,
        user=user,
        prompt_versions={"scan_extract": "v1"},
    )


def build_breakdown_analysis_bundle(settings: Settings, japanese_text: str) -> PromptBundle:
    system = _inject_student_context(_read_utf8(settings, "breakdown_analysis_v1_system.txt"))
    user_template = _read_utf8(settings, "breakdown_analysis_v1_user.txt")
    user = _inject_student_context(user_template).replace("{JAPANESE_TEXT}", japanese_text.strip())
    return PromptBundle(
        system=system,
        user=user,
        prompt_versions={"breakdown_analysis": "v1"},
    )

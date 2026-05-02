"""Application configuration from environment."""

from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

from yomitoku_api.constants import DEFAULT_ANTHROPIC_MODEL


def _src_root() -> Path:
    """`src/` — prompt files live in `src/prompts/` per project rules."""

    return Path(__file__).resolve().parents[1]


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    anthropic_api_key: str = Field(default="", validation_alias="ANTHROPIC_API_KEY")
    anthropic_model: str = Field(
        default=DEFAULT_ANTHROPIC_MODEL,
        validation_alias="YOMITOKU_ANTHROPIC_MODEL",
    )

    cors_allow_origins_csv: str = Field(
        default="",
        validation_alias="ALLOWED_ORIGINS",
    )

    @property
    def allowed_origins(self) -> list[str]:
        """Comma-separated browser origins for CORS; empty denies cross-origin callers."""

        text = self.cors_allow_origins_csv.strip()
        if not text:
            return []
        return [segment.strip() for segment in text.split(",") if segment.strip()]

    @property
    def prompts_dir(self) -> Path:
        return _src_root() / "prompts"

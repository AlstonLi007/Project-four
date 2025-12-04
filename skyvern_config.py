"""Configuration helpers for Skyvern browser automation.

These helpers keep Skyvern settings in one place so other modules can
initialize clients safely without scattering environment lookups.
"""
from __future__ import annotations

import os
from dataclasses import dataclass


SKYVERN_API_KEY_ENV = "SKYVERN_API_KEY"
SKYVERN_BASE_URL_ENV = "SKYVERN_BASE_URL"


@dataclass
class SkyvernSettings:
    """Runtime configuration for Skyvern API calls."""

    api_key: str
    base_url: str = "https://api.skyvern.com"


def get_skyvern_settings() -> SkyvernSettings:
    """Return Skyvern settings sourced from environment variables.

    Raises:
        RuntimeError: if the required API key is missing.
    """

    api_key = os.environ.get(SKYVERN_API_KEY_ENV)
    if not api_key:
        raise RuntimeError(
            f"Missing {SKYVERN_API_KEY_ENV} environment variable; set your Skyvern API key first."
        )

    base_url = os.environ.get(SKYVERN_BASE_URL_ENV, "https://api.skyvern.com")
    if not base_url.startswith(("http://", "https://")):
        raise RuntimeError(
            f"SKYVERN_BASE_URL must start with http(s); got {base_url!r}."
        )
    return SkyvernSettings(api_key=api_key, base_url=base_url)

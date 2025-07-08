"""Settings API endpoints for Langflow.

This module provides API endpoints for managing Langflow settings,
including NeMo Microservices configuration.
"""

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from langflow.services.deps import get_settings_service

router = APIRouter(prefix="/settings", tags=["Settings"])


class NeMoSettingsUpdate(BaseModel):
    """NeMo settings update model."""

    nemo_use_mock: bool | None = None
    nemo_api_key: str | None = None
    nemo_base_url: str | None = None


class NeMoSettingsResponse(BaseModel):
    """NeMo settings response model."""

    nemo_use_mock: bool
    nemo_base_url: str
    nemo_api_key: str | None = None


@router.get("/nemo", response_model=NeMoSettingsResponse)
async def get_nemo_settings():
    """Get current NeMo settings.

    Returns:
        Current NeMo configuration settings
    """
    try:
        settings_service = get_settings_service()
        return NeMoSettingsResponse(
            nemo_use_mock=settings_service.settings.nemo_use_mock,
            nemo_base_url=settings_service.settings.nemo_base_url,
            nemo_api_key=settings_service.settings.nemo_api_key,
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get NeMo settings: {e!s}") from e


@router.patch("/nemo", response_model=NeMoSettingsResponse)
async def update_nemo_settings(settings_update: NeMoSettingsUpdate):
    """Update NeMo settings.

    Args:
        settings_update: NeMo settings to update

    Returns:
        Updated NeMo configuration settings
    """
    try:
        settings_service = get_settings_service()

        # Update settings if provided
        if settings_update.nemo_use_mock is not None:
            settings_service.settings.nemo_use_mock = settings_update.nemo_use_mock

        if settings_update.nemo_base_url is not None:
            settings_service.settings.nemo_base_url = settings_update.nemo_base_url

        if settings_update.nemo_api_key is not None:
            settings_service.settings.nemo_api_key = settings_update.nemo_api_key

        # Return updated settings
        return NeMoSettingsResponse(
            nemo_use_mock=settings_service.settings.nemo_use_mock,
            nemo_base_url=settings_service.settings.nemo_base_url,
            nemo_api_key=settings_service.settings.nemo_api_key,
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to update NeMo settings: {e!s}") from e

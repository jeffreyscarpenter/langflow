"""Factory for NeMo Microservices service.

This module provides a factory that returns the NeMo Microservices
implementation based on configuration stored in global variables
with fallback to environment variables.
"""

import os
from contextlib import suppress
from uuid import UUID

from langflow.services.deps import get_settings_service, get_variable_service
from langflow.services.nemo_microservices_real import RealNeMoMicroservicesService


async def get_nemo_service(
    user_id: UUID | None = None, session=None, header_api_key: str | None = None, header_base_url: str | None = None
):
    """Get the NeMo service with configuration from headers, user's global variables, or environment variables.

    Args:
        user_id: The user ID to get NeMo configuration for (optional)
        session: Database session for accessing global variables (optional)
        header_api_key: API key from HTTP headers (optional)
        header_base_url: Base URL from HTTP headers (optional)

    Returns:
        The real NeMo service with configuration, or raises an error if configuration is missing
    """
    settings_service = get_settings_service()
    variable_service = get_variable_service()

    # Prioritize headers first, then global variables, then environment variables
    nemo_api_key = header_api_key
    nemo_base_url = header_base_url

    # Only try to get from global variables if user_id and session are provided and headers are not set
    if user_id and session:
        if nemo_api_key is None:
            with suppress(ValueError):
                # Try to get NEMO_API_KEY from global variables
                nemo_api_key = await variable_service.get_variable(user_id, "NEMO_API_KEY", "NEMO_API_KEY", session)

        if nemo_base_url is None:
            with suppress(ValueError):
                # Try to get NEMO_BASE_URL from global variables
                nemo_base_url = await variable_service.get_variable(user_id, "NEMO_BASE_URL", "NEMO_BASE_URL", session)

    # Fallback to environment variables if headers and global variables are not set
    # This follows the same pattern as other Langflow components
    if settings_service.settings.fallback_to_env_var:
        if nemo_api_key is None:
            nemo_api_key = os.getenv("NEMO_API_KEY")
            if nemo_api_key:
                from loguru import logger

                logger.info("Using environment variable NEMO_API_KEY for NeMo configuration")

        if nemo_base_url is None:
            nemo_base_url = os.getenv("NEMO_BASE_URL")
            if nemo_base_url:
                from loguru import logger

                logger.info("Using environment variable NEMO_BASE_URL for NeMo configuration")

    # If we have both API key and base URL, use the service
    if nemo_api_key and nemo_base_url:
        # Log the source of configuration for debugging
        if header_api_key and header_base_url:
            from loguru import logger

            logger.info("Using NeMo configuration from HTTP headers")
        return RealNeMoMicroservicesService(base_url=nemo_base_url, api_key=nemo_api_key)

    # If configuration is missing, raise an error
    missing_configs = []
    if not nemo_api_key:
        missing_configs.append("NEMO_API_KEY")
    if not nemo_base_url:
        missing_configs.append("NEMO_BASE_URL")

    error_msg = (
        f"NeMo Microservices configuration is incomplete. Missing: {', '.join(missing_configs)}. "
        "Please configure these as global variables or environment variables."
    )
    raise ValueError(error_msg)

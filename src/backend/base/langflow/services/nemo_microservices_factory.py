"""Factory for NeMo Microservices service selection.

This module provides a factory that returns either the mock or real
NeMo Microservices implementation based on configuration settings.
"""

from langflow.services.deps import get_settings_service
from langflow.services.nemo_microservices_mock import mock_nemo_service
from langflow.services.nemo_microservices_real import RealNeMoMicroservicesService


def get_nemo_service():
    """Get the appropriate NeMo service based on configuration.

    Returns:
        Either mock_nemo_service or RealNeMoMicroservicesService based on
        the nemo_use_mock setting.
    """
    settings_service = get_settings_service()

    if settings_service.settings.nemo_use_mock:
        return mock_nemo_service
    return RealNeMoMicroservicesService()


# Global instance for easy access
nemo_service = get_nemo_service()

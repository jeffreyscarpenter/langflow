"""Real NeMo Microservices client for production use.

This module provides a real implementation of NeMo Microservices APIs
that makes actual HTTP calls to NeMo services.

Includes real implementations for:
- NeMo Data Store (datasets and files)
- NeMo Entity Store (entities, projects, models)
- NeMo Customizer (job tracking and status)
- NeMo Evaluator (evaluation results)
"""

import logging
from typing import Any

import httpx
from fastapi import UploadFile
from httpx import codes

from langflow.services.deps import get_settings_service

logger = logging.getLogger(__name__)


class RealNeMoMicroservicesService:
    """Real implementation of NeMo Microservices APIs.

    Makes actual HTTP calls to NeMo services for production use.
    """

    def __init__(self):
        self.settings_service = get_settings_service()
        self.base_url = self.settings_service.settings.nemo_base_url
        self.api_key = self.settings_service.settings.nemo_api_key

        # Data Store URLs
        self.data_store_url = self.settings_service.settings.nemo_data_store_url
        self.entity_store_url = self.settings_service.settings.nemo_entity_store_url
        self.customizer_url = self.settings_service.settings.nemo_customizer_url
        self.evaluator_url = self.settings_service.settings.nemo_evaluator_url

    def _get_auth_headers(self) -> dict[str, str]:
        """Get headers with authentication token."""
        headers = {"accept": "application/json", "Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        return headers

    # =============================================================================
    # Dataset Management (Data Store)
    # =============================================================================

    async def list_datasets(self) -> list[dict[str, Any]]:
        """Get list of datasets from NeMo Data Store."""
        try:
            async with httpx.AsyncClient(timeout=30.0) as client:
                response = await client.get(f"{self.data_store_url}/v1/datasets", headers=self._get_auth_headers())
                response.raise_for_status()
                return response.json().get("data", [])
        except Exception:
            logger.exception("Failed to list datasets")
            raise

    async def create_dataset(
        self, name: str, description: str | None = None, dataset_type: str = "fileset"
    ) -> dict[str, Any]:
        """Create a new dataset in NeMo Data Store."""
        try:
            data = {
                "name": name,
                "description": description or "",
                "type": dataset_type,
            }

            async with httpx.AsyncClient(timeout=30.0) as client:
                response = await client.post(
                    f"{self.data_store_url}/v1/datasets", json=data, headers=self._get_auth_headers()
                )
                response.raise_for_status()
                return response.json()
        except Exception:
            logger.exception("Failed to create dataset")
            raise

    async def get_dataset(self, dataset_id: str) -> dict[str, Any] | None:
        """Get dataset details from NeMo Data Store."""
        try:
            async with httpx.AsyncClient(timeout=30.0) as client:
                response = await client.get(
                    f"{self.data_store_url}/v1/datasets/{dataset_id}", headers=self._get_auth_headers()
                )
                response.raise_for_status()
                return response.json()
        except httpx.HTTPStatusError as e:
            if e.response.status_code == codes.NOT_FOUND:
                return None
            raise
        except Exception:
            logger.exception("Failed to get dataset")
            raise

    async def delete_dataset(self, dataset_id: str) -> bool:
        """Delete a dataset from NeMo Data Store."""
        try:
            async with httpx.AsyncClient(timeout=30.0) as client:
                response = await client.delete(
                    f"{self.data_store_url}/v1/datasets/{dataset_id}", headers=self._get_auth_headers()
                )
                response.raise_for_status()
                return True
        except httpx.HTTPStatusError as e:
            if e.response.status_code == codes.NOT_FOUND:
                return False
            raise
        except Exception:
            logger.exception("Failed to delete dataset")
            raise

    async def upload_files(self, dataset_id: str, files: list[UploadFile]) -> dict[str, Any]:
        """Upload files to a NeMo dataset."""
        try:
            form_data = {}
            for _i, file in enumerate(files):
                form_data["files"] = (file.filename, file.file, file.content_type)

            async with httpx.AsyncClient(timeout=60.0) as client:
                response = await client.post(
                    f"{self.data_store_url}/v1/datasets/{dataset_id}/files",
                    files=form_data,
                    headers={"Authorization": f"Bearer {self.api_key}"} if self.api_key else {},
                )
                response.raise_for_status()
                return response.json()
        except Exception:
            logger.exception("Failed to upload files")
            raise

    async def get_dataset_files(self, dataset_id: str) -> list[dict[str, Any]]:
        """Get list of files in a NeMo dataset."""
        try:
            async with httpx.AsyncClient(timeout=30.0) as client:
                response = await client.get(
                    f"{self.data_store_url}/v1/datasets/{dataset_id}/files", headers=self._get_auth_headers()
                )
                response.raise_for_status()
                return response.json().get("data", [])
        except Exception:
            logger.exception("Failed to get dataset files")
            raise

    # =============================================================================
    # Job Management (Customizer)
    # =============================================================================

    async def get_customization_configs(self) -> dict[str, Any]:
        """Get available model configurations for customization."""
        try:
            async with httpx.AsyncClient(timeout=30.0) as client:
                response = await client.get(
                    f"{self.base_url}/v1/customization/configs", headers=self._get_auth_headers()
                )
                response.raise_for_status()
                return response.json()
        except Exception:
            logger.exception("Failed to get customization configs")
            raise

    async def create_customization_job(self, job_data: dict) -> dict[str, Any]:
        """Create a new customization job."""
        try:
            async with httpx.AsyncClient(timeout=30.0) as client:
                response = await client.post(
                    f"{self.base_url}/v1/customization/jobs", json=job_data, headers=self._get_auth_headers()
                )
                response.raise_for_status()
                return response.json()
        except Exception:
            logger.exception("Failed to create customization job")
            raise

    async def get_customizer_job_status(self, job_id: str) -> dict[str, Any] | None:
        """Get customization job status."""
        try:
            async with httpx.AsyncClient(timeout=30.0) as client:
                response = await client.get(
                    f"{self.base_url}/v1/customization/jobs/{job_id}/status", headers=self._get_auth_headers()
                )
                response.raise_for_status()
                return response.json()
        except httpx.HTTPStatusError as e:
            if e.response.status_code == codes.NOT_FOUND:
                return None
            raise
        except Exception:
            logger.exception("Failed to get job status")
            raise

    async def get_customizer_job_details(self, job_id: str) -> dict[str, Any] | None:
        """Get customization job details."""
        try:
            async with httpx.AsyncClient(timeout=30.0) as client:
                response = await client.get(
                    f"{self.base_url}/v1/customization/jobs/{job_id}", headers=self._get_auth_headers()
                )
                response.raise_for_status()
                return response.json()
        except httpx.HTTPStatusError as e:
            if e.response.status_code == codes.NOT_FOUND:
                return None
            raise
        except Exception:
            logger.exception("Failed to get job details")
            raise

    async def list_customizer_jobs(self) -> list[dict[str, Any]]:
        """List all customization jobs."""
        try:
            async with httpx.AsyncClient(timeout=30.0) as client:
                response = await client.get(f"{self.base_url}/v1/customization/jobs", headers=self._get_auth_headers())
                response.raise_for_status()
                return response.json().get("data", [])
        except Exception:
            logger.exception("Failed to list customization jobs")
            raise

    # =============================================================================
    # Job Tracking for Langflow Dashboard
    # =============================================================================

    async def track_customizer_job(self, job_id: str, metadata: dict | None = None) -> dict[str, Any]:
        """Track a customization job for dashboard monitoring."""
        # For real API, we just return success since tracking is handled by the real service
        return {
            "job_id": job_id,
            "tracked_at": "2024-01-01T00:00:00Z",
            "metadata": metadata or {},
            "message": f"Now tracking job {job_id}",
        }

    async def get_tracked_jobs(self) -> list[dict[str, Any]]:
        """Get tracked jobs for dashboard monitoring."""
        # For real API, return actual jobs from the service
        try:
            jobs = await self.list_customizer_jobs()
        except Exception:
            logger.exception("Failed to get tracked jobs")
            raise
        else:
            return [
                {
                    "job_id": job["id"],
                    "status": job["status"],
                    "created_at": job["created_at"],
                    "updated_at": job["updated_at"],
                    "config": job.get("config", {}).get("name", "Unknown Model"),
                    "dataset": job.get("dataset", "Unknown Dataset"),
                    "progress": job.get("status_details", {}).get("percentage_done", 0),
                    "output_model": job.get("output_model"),
                    "hyperparameters": job.get("hyperparameters"),
                    "custom_fields": job.get("custom_fields"),
                }
                for job in jobs
            ]

    async def stop_tracking_job(self, job_id: str) -> dict[str, Any]:
        """Stop tracking a job."""
        return {"message": f"Stopped tracking job {job_id}"}

    # =============================================================================
    # Evaluation Management (Evaluator)
    # =============================================================================

    async def create_evaluation_job(self, job_data: dict) -> dict[str, Any]:
        """Create a new evaluation job."""
        try:
            async with httpx.AsyncClient(timeout=30.0) as client:
                response = await client.post(
                    f"{self.evaluator_url}/v1/evaluation/jobs", json=job_data, headers=self._get_auth_headers()
                )
                response.raise_for_status()
                return response.json()
        except Exception:
            logger.exception("Failed to create evaluation job")
            raise

    async def get_evaluation_job(self, job_id: str) -> dict[str, Any] | None:
        """Get evaluation job details."""
        try:
            async with httpx.AsyncClient(timeout=30.0) as client:
                response = await client.get(
                    f"{self.evaluator_url}/v1/evaluation/jobs/{job_id}", headers=self._get_auth_headers()
                )
                response.raise_for_status()
                return response.json()
        except httpx.HTTPStatusError as e:
            if e.response.status_code == codes.NOT_FOUND:
                return None
            raise
        except Exception:
            logger.exception("Failed to get evaluation job")
            raise

    async def list_evaluation_jobs(self) -> list[dict[str, Any]]:
        """List all evaluation jobs."""
        try:
            async with httpx.AsyncClient(timeout=30.0) as client:
                response = await client.get(
                    f"{self.evaluator_url}/v1/evaluation/jobs", headers=self._get_auth_headers()
                )
                response.raise_for_status()
                return response.json().get("data", [])
        except Exception:
            logger.exception("Failed to list evaluation jobs")
            raise

    def cleanup(self):
        """Cleanup resources."""

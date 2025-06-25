"""Mock NeMo Microservices for testing and development.

This module provides a mock implementation of the NeMo Microservices APIs
to allow development and testing without requiring the actual NeMo services.

Includes mock implementations for:
- NeMo Data Store (datasets and files)
- NeMo Customizer (job tracking and status) - matches real API structure
- NeMo Evaluator (evaluation results)
"""

import asyncio
import logging
import shutil
import tempfile
import uuid
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

from fastapi import UploadFile


class MockNeMoMicroservicesService:
    """Mock implementation of NeMo Microservices APIs for testing.

    Simulates the behavior of:
    - NeMo Data Store API endpoints
    - NeMo Customizer API endpoints (matching real API structure)
    - NeMo Evaluator results (future)
    """

    def __init__(self):
        # In-memory storage for mock data
        self._datasets: dict[str, dict] = {}
        self._files: dict[str, list[dict]] = {}
        self._customizer_jobs: dict[str, dict] = {}  # Jobs from NeMo Customizer API format
        self._tracked_jobs: list[str] = []  # Job IDs we're tracking for monitoring
        self._temp_dir = Path(tempfile.mkdtemp(prefix="nemo_mock_"))

        # Initialize with some sample data
        self._initialize_sample_data()

    def _initialize_sample_data(self):
        """Initialize with sample datasets and jobs for testing."""
        now = datetime.now(timezone.utc)

        # Sample datasets
        sample_datasets = [
            {
                "id": "dataset-001",
                "name": "Sample Training Data",
                "description": "A sample dataset for model training",
                "type": "fileset",
                "created_at": (now - timedelta(days=7)).isoformat(),
                "updated_at": (now - timedelta(days=1)).isoformat(),
                "metadata": {
                    "file_count": 150,
                    "total_size": "2.3GB",
                    "format": "jsonl",
                    "tags": ["training", "sample"],
                },
            },
            {
                "id": "dataset-002",
                "name": "Evaluation Dataset",
                "description": "Dataset for model evaluation and testing",
                "type": "fileset",
                "created_at": (now - timedelta(days=5)).isoformat(),
                "updated_at": (now - timedelta(hours=6)).isoformat(),
                "metadata": {
                    "file_count": 75,
                    "total_size": "1.1GB",
                    "format": "jsonl",
                    "tags": ["evaluation", "test"],
                },
            },
            {
                "id": "dataset-003",
                "name": "Fine-tuning Data",
                "description": "Specialized dataset for fine-tuning models",
                "type": "fileset",
                "created_at": (now - timedelta(days=2)).isoformat(),
                "updated_at": now.isoformat(),
                "metadata": {
                    "file_count": 200,
                    "total_size": "3.7GB",
                    "format": "jsonl",
                    "tags": ["fine-tuning", "specialized"],
                },
            },
        ]

        for dataset in sample_datasets:
            self._datasets[dataset["id"]] = dataset
            self._files[dataset["id"]] = []

        # Sample NeMo Customizer jobs (matching real API structure)
        sample_jobs = [
            {
                "id": "cust-Pi95UoDbNcqwgkruAB8LY6",
                "created_at": (now - timedelta(hours=2)).isoformat(),
                "updated_at": (now - timedelta(minutes=15)).isoformat(),
                "namespace": "default",
                "config": {
                    "schema_version": "1.0",
                    "id": "58bee815-0473-45d7-a5e6-fc088f6142eb",
                    "namespace": "default",
                    "created_at": (now - timedelta(hours=2)).isoformat(),
                    "updated_at": (now - timedelta(hours=2)).isoformat(),
                    "custom_fields": {},
                    "name": "meta/llama-3.1-8b-instruct",
                    "base_model": "meta/llama-3.1-8b-instruct",
                    "model_path": "llama-3_1-8b-instruct",
                    "training_types": ["sft"],
                    "finetuning_types": ["lora"],
                    "precision": "bf16",
                    "num_gpus": 4,
                    "num_nodes": 1,
                    "micro_batch_size": 1,
                    "tensor_parallel_size": 1,
                    "max_seq_length": 4096,
                },
                "dataset": "default/test-dataset",
                "hyperparameters": {
                    "finetuning_type": "lora",
                    "training_type": "sft",
                    "batch_size": 8,
                    "epochs": 50,
                    "learning_rate": 0.0001,
                    "lora": {"adapter_dim": 8, "adapter_dropout": 0.1},
                },
                "output_model": "default/meta-llama-3.1-8b-instruct-test-dataset-lora@cust-Pi95UoDbNcqwgkruAB8LY6",
                "status": "running",
                "status_details": {
                    "created_at": (now - timedelta(hours=2)).isoformat(),
                    "updated_at": (now - timedelta(minutes=15)).isoformat(),
                    "steps_completed": 1250,
                    "epochs_completed": 25,
                    "percentage_done": 50,
                    "status_logs": [
                        {
                            "updated_at": (now - timedelta(hours=2)).isoformat(),
                            "message": "created",
                        },
                        {
                            "updated_at": (now - timedelta(hours=2, minutes=-2)).isoformat(),
                            "message": "PVCCreated",
                        },
                        {
                            "updated_at": (now - timedelta(hours=2, minutes=-2)).isoformat(),
                            "message": "EntityHandler_0_Created",
                        },
                        {
                            "updated_at": (now - timedelta(hours=2, minutes=-1)).isoformat(),
                            "message": "EntityHandler_0_Pending",
                        },
                        {
                            "updated_at": (now - timedelta(hours=2, minutes=-1)).isoformat(),
                            "message": "EntityHandler_0_Completed",
                        },
                        {
                            "updated_at": (now - timedelta(hours=2, minutes=-1)).isoformat(),
                            "message": "TrainingJobCreated",
                        },
                        {
                            "updated_at": (now - timedelta(hours=1, minutes=58)).isoformat(),
                            "message": "TrainingJobRunning",
                        },
                    ],
                    "training_loss": [
                        {"step": 10, "value": 2.45, "timestamp": (now - timedelta(hours=1, minutes=50)).isoformat()},
                        {"step": 20, "value": 2.32, "timestamp": (now - timedelta(hours=1, minutes=45)).isoformat()},
                        {"step": 30, "value": 2.18, "timestamp": (now - timedelta(hours=1, minutes=40)).isoformat()},
                        {"step": 40, "value": 2.05, "timestamp": (now - timedelta(hours=1, minutes=35)).isoformat()},
                        {"step": 50, "value": 1.95, "timestamp": (now - timedelta(hours=1, minutes=30)).isoformat()},
                    ],
                    "validation_loss": [
                        {"epoch": 1, "value": 2.12, "timestamp": (now - timedelta(hours=1, minutes=45)).isoformat()},
                        {"epoch": 2, "value": 1.98, "timestamp": (now - timedelta(hours=1, minutes=30)).isoformat()},
                        {"epoch": 3, "value": 1.85, "timestamp": (now - timedelta(hours=1, minutes=15)).isoformat()},
                    ],
                },
                "custom_fields": {},
            },
            {
                "id": "cust-YbmGLDpnZUPMGjqKZ2MaUy",
                "created_at": (now - timedelta(days=1)).isoformat(),
                "updated_at": (now - timedelta(hours=8)).isoformat(),
                "namespace": "default",
                "config": {
                    "schema_version": "1.0",
                    "id": "99bee815-0473-45d7-a5e6-fc088f6142eb",
                    "namespace": "default",
                    "created_at": (now - timedelta(days=1)).isoformat(),
                    "updated_at": (now - timedelta(days=1)).isoformat(),
                    "custom_fields": {},
                    "name": "meta/llama-3.2-1b-instruct",
                    "base_model": "meta/llama-3.2-1b-instruct",
                    "model_path": "llama-3_2-1b-instruct",
                    "training_types": ["sft"],
                    "finetuning_types": ["full"],
                    "precision": "bf16",
                    "num_gpus": 8,
                    "num_nodes": 1,
                    "micro_batch_size": 1,
                    "tensor_parallel_size": 1,
                    "max_seq_length": 4096,
                },
                "dataset": "default/fine-tuning-dataset",
                "hyperparameters": {
                    "finetuning_type": "full",
                    "training_type": "sft",
                    "batch_size": 16,
                    "epochs": 10,
                    "learning_rate": 0.00005,
                },
                "output_model": "default/meta-llama-3.2-1b-instruct-fine-tuning-dataset-full@cust-YbmGLDpnZUPMGjqKZ2MaUy",
                "status": "completed",
                "status_details": {
                    "created_at": (now - timedelta(days=1)).isoformat(),
                    "updated_at": (now - timedelta(hours=8)).isoformat(),
                    "steps_completed": 1000,
                    "epochs_completed": 10,
                    "percentage_done": 100,
                    "status_logs": [
                        {
                            "updated_at": (now - timedelta(days=1)).isoformat(),
                            "message": "created",
                        },
                        {
                            "updated_at": (now - timedelta(days=1, minutes=-2)).isoformat(),
                            "message": "PVCCreated",
                        },
                        {
                            "updated_at": (now - timedelta(days=1, minutes=-2)).isoformat(),
                            "message": "EntityHandler_0_Created",
                        },
                        {
                            "updated_at": (now - timedelta(days=1, minutes=-1)).isoformat(),
                            "message": "EntityHandler_0_Pending",
                        },
                        {
                            "updated_at": (now - timedelta(days=1, minutes=-1)).isoformat(),
                            "message": "EntityHandler_0_Completed",
                        },
                        {
                            "updated_at": (now - timedelta(days=1, minutes=-1)).isoformat(),
                            "message": "TrainingJobCreated",
                        },
                        {
                            "updated_at": (now - timedelta(hours=23, minutes=58)).isoformat(),
                            "message": "TrainingJobRunning",
                        },
                        {
                            "updated_at": (now - timedelta(hours=8)).isoformat(),
                            "message": "completed",
                        },
                    ],
                    "training_loss": [
                        {"step": 100, "value": 1.85, "timestamp": (now - timedelta(hours=12)).isoformat()},
                        {"step": 200, "value": 1.62, "timestamp": (now - timedelta(hours=10)).isoformat()},
                        {"step": 300, "value": 1.45, "timestamp": (now - timedelta(hours=9)).isoformat()},
                        {"step": 400, "value": 1.32, "timestamp": (now - timedelta(hours=8, minutes=30)).isoformat()},
                        {"step": 500, "value": 1.28, "timestamp": (now - timedelta(hours=8)).isoformat()},
                    ],
                    "validation_loss": [
                        {"epoch": 2, "value": 1.68, "timestamp": (now - timedelta(hours=11)).isoformat()},
                        {"epoch": 4, "value": 1.52, "timestamp": (now - timedelta(hours=9, minutes=30)).isoformat()},
                        {"epoch": 6, "value": 1.41, "timestamp": (now - timedelta(hours=8, minutes=45)).isoformat()},
                        {"epoch": 8, "value": 1.35, "timestamp": (now - timedelta(hours=8, minutes=15)).isoformat()},
                        {"epoch": 10, "value": 1.31, "timestamp": (now - timedelta(hours=8)).isoformat()},
                    ],
                },
                "custom_fields": {},
            },
            {
                "id": "cust-FailedJobExample123",
                "created_at": (now - timedelta(hours=4)).isoformat(),
                "updated_at": (now - timedelta(hours=3)).isoformat(),
                "namespace": "default",
                "config": {
                    "schema_version": "1.0",
                    "id": "77bee815-0473-45d7-a5e6-fc088f6142eb",
                    "namespace": "default",
                    "created_at": (now - timedelta(hours=4)).isoformat(),
                    "updated_at": (now - timedelta(hours=4)).isoformat(),
                    "custom_fields": {},
                    "name": "meta/llama-3.1-8b-instruct",
                    "base_model": "meta/llama-3.1-8b-instruct",
                    "model_path": "llama-3_1-8b-instruct",
                    "training_types": ["sft"],
                    "finetuning_types": ["lora"],
                    "precision": "bf16",
                    "num_gpus": 4,
                    "num_nodes": 1,
                    "micro_batch_size": 1,
                    "tensor_parallel_size": 1,
                    "max_seq_length": 4096,
                },
                "dataset": "default/problematic-dataset",
                "hyperparameters": {
                    "finetuning_type": "lora",
                    "training_type": "sft",
                    "batch_size": 8,
                    "epochs": 20,
                    "learning_rate": 0.0001,
                    "lora": {"adapter_dim": 8, "adapter_dropout": 0.1},
                },
                "output_model": "default/meta-llama-3.1-8b-instruct-problematic-dataset-lora@cust-FailedJobExample123",
                "status": "failed",
                "status_details": {
                    "created_at": (now - timedelta(hours=4)).isoformat(),
                    "updated_at": (now - timedelta(hours=3)).isoformat(),
                    "steps_completed": 242,
                    "epochs_completed": 1,
                    "percentage_done": 25,
                    "status_logs": [
                        {
                            "updated_at": (now - timedelta(hours=4)).isoformat(),
                            "message": "created",
                        },
                        {
                            "updated_at": (now - timedelta(hours=4, minutes=-2)).isoformat(),
                            "message": "PVCCreated",
                        },
                        {
                            "updated_at": (now - timedelta(hours=4, minutes=-2)).isoformat(),
                            "message": "EntityHandler_0_Created",
                        },
                        {
                            "updated_at": (now - timedelta(hours=4, minutes=-1)).isoformat(),
                            "message": "EntityHandler_0_Pending",
                        },
                        {
                            "updated_at": (now - timedelta(hours=4, minutes=-1)).isoformat(),
                            "message": "EntityHandler_0_Completed",
                        },
                        {
                            "updated_at": (now - timedelta(hours=4, minutes=-1)).isoformat(),
                            "message": "TrainingJobCreated",
                        },
                        {
                            "updated_at": (now - timedelta(hours=3, minutes=58)).isoformat(),
                            "message": "TrainingJobRunning",
                        },
                        {
                            "updated_at": (now - timedelta(hours=3, minutes=30)).isoformat(),
                            "message": "DataLoader worker (pid 2266) is killed by signal: Terminated.",
                            "detail": 'Traceback (most recent call last):\n  File "/usr/local/lib/python3.12/dist-packages/lightning/pytorch/trainer/call.py", line 46, in _call_and_handle_interrupt',
                        },
                        {
                            "updated_at": (now - timedelta(hours=3)).isoformat(),
                            "message": "failed",
                        },
                    ],
                    "training_loss": [
                        {"step": 10, "value": 2.85, "timestamp": (now - timedelta(hours=3, minutes=45)).isoformat()},
                        {"step": 20, "value": 2.92, "timestamp": (now - timedelta(hours=3, minutes=40)).isoformat()},
                        {"step": 30, "value": 2.88, "timestamp": (now - timedelta(hours=3, minutes=35)).isoformat()},
                    ],
                    "validation_loss": [
                        {"epoch": 1, "value": 2.89, "timestamp": (now - timedelta(hours=3, minutes=40)).isoformat()},
                    ],
                },
                "custom_fields": {},
            },
        ]

        for job in sample_jobs:
            self._customizer_jobs[job["id"]] = job
            self._tracked_jobs.append(job["id"])

    # =============================================================================
    # Dataset Management (Data Store)
    # =============================================================================

    async def list_datasets(self) -> list[dict[str, Any]]:
        """Mock implementation of GET /api/v1/datasets.

        Returns:
            List of dataset objects.
        """
        # Simulate network delay
        await asyncio.sleep(0.1)

        return list(self._datasets.values())

    async def create_dataset(
        self, name: str, description: str | None = None, dataset_type: str = "fileset"
    ) -> dict[str, Any]:
        """Mock implementation of POST /api/v1/datasets.

        Args:
            name: Dataset name.
            description: Optional description.
            dataset_type: Type of dataset (default: fileset).

        Returns:
            Created dataset object.
        """
        # Simulate network delay
        await asyncio.sleep(0.2)

        dataset_id = f"dataset-{str(uuid.uuid4())[:8]}"
        now = datetime.now(timezone.utc).isoformat()

        dataset = {
            "id": dataset_id,
            "name": name,
            "description": description,
            "type": dataset_type,
            "created_at": now,
            "updated_at": now,
            "metadata": {"file_count": 0, "total_size": "0B", "format": "jsonl", "tags": []},
        }

        self._datasets[dataset_id] = dataset
        self._files[dataset_id] = []

        return dataset

    async def get_dataset(self, dataset_id: str) -> dict[str, Any] | None:
        """Mock implementation of GET /api/v1/datasets/{dataset_id}.

        Args:
            dataset_id: ID of the dataset to retrieve.

        Returns:
            Dataset object or None if not found.
        """
        # Simulate network delay
        await asyncio.sleep(0.1)

        dataset = self._datasets.get(dataset_id)
        if dataset:
            # Include file information
            files = self._files.get(dataset_id, [])
            dataset["files"] = files
            dataset["file_count"] = len(files)

        return dataset

    async def delete_dataset(self, dataset_id: str) -> bool:
        """Mock implementation of DELETE /api/v1/datasets/{dataset_id}.

        Args:
            dataset_id: ID of the dataset to delete.

        Returns:
            True if deleted, False if not found.
        """
        # Simulate network delay
        await asyncio.sleep(0.3)

        if dataset_id in self._datasets:
            del self._datasets[dataset_id]
            if dataset_id in self._files:
                del self._files[dataset_id]
            return True
        return False

    async def upload_files(self, dataset_id: str, files: list[UploadFile]) -> dict[str, Any]:
        """Mock implementation of POST /api/v1/datasets/{dataset_id}/files.

        Args:
            dataset_id: ID of the dataset to upload files to.
            files: List of files to upload.

        Returns:
            Upload result with file information.
        """
        logger = logging.getLogger("nemo_microservices_mock.upload_files")
        try:
            logger.info("Received upload for dataset_id=%s, files=%s", dataset_id, [f.filename for f in files])
            # Simulate network delay
            await asyncio.sleep(0.5)

            if dataset_id not in self._datasets:
                error_msg = f"Dataset {dataset_id} not found"
                logger.error(error_msg)
                raise ValueError(error_msg)

            uploaded_files = []
            for file in files:
                logger.info("Processing file: %s, content_type=%s", file.filename, file.content_type)
                content = await file.read()
                file_info = {
                    "filename": file.filename,
                    "size": len(content),
                    "content_type": file.content_type or "application/octet-stream",
                    "uploaded_at": datetime.now(timezone.utc).isoformat(),
                }
                self._files[dataset_id].append(file_info)
                uploaded_files.append(file_info)

            # Update dataset metadata
            self._datasets[dataset_id]["metadata"]["file_count"] = len(self._files[dataset_id])
            self._datasets[dataset_id]["metadata"]["total_size"] = f"{sum(f['size'] for f in self._files[dataset_id])}B"
            self._datasets[dataset_id]["updated_at"] = datetime.now(timezone.utc).isoformat()

            logger.info("Successfully uploaded %d files to dataset %s", len(uploaded_files), dataset_id)
            return {"message": f"Successfully uploaded {len(uploaded_files)} files", "files": uploaded_files}
        except Exception as e:
            logger.exception("Error in upload_files")
            return {"message": f"Error: {e!s}", "files": []}

    async def get_dataset_files(self, dataset_id: str) -> list[dict[str, Any]]:
        """Mock implementation of GET /api/v1/datasets/{dataset_id}/files.

        Args:
            dataset_id: ID of the dataset.

        Returns:
            List of files in the dataset.
        """
        # Simulate network delay
        await asyncio.sleep(0.1)

        return self._files.get(dataset_id, [])

    # =============================================================================
    # NeMo Customizer API (Real API Structure)
    # =============================================================================

    async def get_customizer_job_status(self, job_id: str) -> dict[str, Any] | None:
        """Mock implementation of GET /v1/customization/jobs/{id}/status.

        This matches the real NeMo Customizer API endpoint that returns
        timestamped training and validation loss values.

        Args:
            job_id: NeMo Customizer job ID

        Returns:
            Job status with timestamped loss values or None if not found
        """
        # Simulate network delay
        await asyncio.sleep(0.1)

        job = self._customizer_jobs.get(job_id)
        if not job:
            return None

        # Return status-focused response with timestamped loss values
        return {
            "id": job["id"],
            "status": job["status"],
            "status_details": job["status_details"],
            "created_at": job["created_at"],
            "updated_at": job["updated_at"],
        }

    async def get_customizer_job_details(self, job_id: str) -> dict[str, Any] | None:
        """Mock implementation of GET /v1/customization/jobs/{id}.

        This matches the real NeMo Customizer API endpoint that returns
        comprehensive job information including configuration and status logs.

        Args:
            job_id: NeMo Customizer job ID

        Returns:
            Complete job details or None if not found
        """
        # Simulate network delay
        await asyncio.sleep(0.1)

        return self._customizer_jobs.get(job_id)

    async def list_customizer_jobs(self) -> list[dict[str, Any]]:
        """Mock implementation for listing all NeMo Customizer jobs.

        Returns:
            List of all customizer jobs
        """
        # Simulate network delay
        await asyncio.sleep(0.1)

        return list(self._customizer_jobs.values())

    # =============================================================================
    # Job Tracking for Langflow Dashboard
    # =============================================================================

    async def track_customizer_job(self, job_id: str, metadata: dict | None = None) -> dict[str, Any]:
        """Start tracking a NeMo Customizer job for dashboard monitoring.

        Args:
            job_id: NeMo Customizer job ID to track
            metadata: Optional metadata for tracking (e.g., user-friendly name)

        Returns:
            Tracking confirmation
        """
        # Simulate network delay
        await asyncio.sleep(0.1)

        if job_id not in self._tracked_jobs:
            self._tracked_jobs.append(job_id)

        return {
            "job_id": job_id,
            "tracked_at": datetime.now(timezone.utc).isoformat(),
            "metadata": metadata or {},
            "message": f"Now tracking job {job_id}",
        }

    async def get_tracked_jobs(self) -> list[dict[str, Any]]:
        """Get all jobs being tracked for dashboard monitoring.

        Returns:
            List of tracked job IDs with their current status
        """
        # Simulate network delay
        await asyncio.sleep(0.1)

        tracked_jobs = []
        for job_id in self._tracked_jobs:
            job = self._customizer_jobs.get(job_id)
            if job:
                tracked_jobs.append(
                    {
                        "job_id": job_id,
                        "status": job["status"],
                        "created_at": job["created_at"],
                        "updated_at": job["updated_at"],
                        "config": job.get("config", {}).get("name", "Unknown Model"),
                        "dataset": job.get("dataset", "Unknown Dataset"),
                        "progress": job.get("status_details", {}).get("percentage_done", 0),
                    }
                )

        return tracked_jobs

    async def stop_tracking_job(self, job_id: str) -> dict[str, Any]:
        """Stop tracking a job for dashboard monitoring.

        Args:
            job_id: Job ID to stop tracking

        Returns:
            Confirmation message
        """
        # Simulate network delay
        await asyncio.sleep(0.1)

        if job_id in self._tracked_jobs:
            self._tracked_jobs.remove(job_id)
            return {"message": f"Stopped tracking job {job_id}"}
        return {"message": f"Job {job_id} was not being tracked"}

    def cleanup(self):
        """Clean up temporary files and directories."""
        if self._temp_dir.exists():
            shutil.rmtree(self._temp_dir)


# Global instance for use in API
mock_nemo_service = MockNeMoMicroservicesService()

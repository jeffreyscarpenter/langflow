"""Mock NeMo Microservices for testing and development.

This module provides a mock implementation of the NeMo Microservices APIs
to allow development and testing without requiring the actual NeMo services.

Includes mock implementations for:
- NeMo Data Store (datasets and files)
- NeMo Customizer (job tracking and status)
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
    - NeMo Customizer job tracking
    - NeMo Evaluator results (future)
    """

    def __init__(self):
        # In-memory storage for mock data
        self._datasets: dict[str, dict] = {}
        self._files: dict[str, list[dict]] = {}
        self._jobs: dict[str, dict] = {}  # NEW: Job tracking
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

        # Sample customizer jobs
        sample_jobs = [
            {
                "id": "cust-ABC123XYZ789",
                "created_at": (now - timedelta(hours=2)).isoformat(),
                "updated_at": (now - timedelta(minutes=30)).isoformat(),
                "namespace": "default",
                "config": {
                    "name": "meta/llama-3.1-8b-instruct",
                    "base_model": "meta/llama-3.1-8b-instruct",
                    "training_types": ["sft"],
                    "finetuning_types": ["lora"],
                    "precision": "bf16",
                    "num_gpus": 4,
                },
                "dataset": "default/Sample Training Data",
                "hyperparameters": {
                    "training_type": "sft",
                    "finetuning_type": "lora",
                    "epochs": 10,
                    "batch_size": 16,
                    "learning_rate": 0.0001,
                    "lora": {"adapter_dim": 8, "adapter_dropout": 0.1},
                },
                "output_model": "default/my-fine-tuned-model@v1",
                "status": "running",
                "progress": {
                    "current_epoch": 6,
                    "total_epochs": 10,
                    "percentage": 60,
                    "training_loss": [
                        {"step": 10, "loss": 2.45, "timestamp": (now - timedelta(hours=1, minutes=45)).isoformat()},
                        {"step": 20, "loss": 2.32, "timestamp": (now - timedelta(hours=1, minutes=30)).isoformat()},
                        {"step": 30, "loss": 2.18, "timestamp": (now - timedelta(hours=1, minutes=15)).isoformat()},
                    ],
                    "validation_loss": [
                        {"epoch": 1, "loss": 2.18, "timestamp": (now - timedelta(hours=1, minutes=40)).isoformat()},
                        {"epoch": 2, "loss": 2.05, "timestamp": (now - timedelta(hours=1, minutes=20)).isoformat()},
                    ],
                },
                "component_id": "component-123",
                "created_by_component": "NVIDIANeMoCustomizer",
            },
            {
                "id": "cust-DEF456UVW012",
                "created_at": (now - timedelta(days=1)).isoformat(),
                "updated_at": (now - timedelta(hours=8)).isoformat(),
                "namespace": "default",
                "config": {
                    "name": "meta/llama-3.2-1b-instruct",
                    "base_model": "meta/llama-3.2-1b-instruct",
                    "training_types": ["sft"],
                    "finetuning_types": ["full"],
                    "precision": "bf16",
                    "num_gpus": 8,
                },
                "dataset": "default/Fine-tuning Data",
                "hyperparameters": {
                    "training_type": "sft",
                    "finetuning_type": "full",
                    "epochs": 5,
                    "batch_size": 32,
                    "learning_rate": 0.00005,
                },
                "output_model": "default/completed-model@v2",
                "status": "completed",
                "progress": {
                    "current_epoch": 5,
                    "total_epochs": 5,
                    "percentage": 100,
                    "training_loss": [
                        {"step": 50, "loss": 1.89, "timestamp": (now - timedelta(hours=10)).isoformat()},
                        {"step": 100, "loss": 1.65, "timestamp": (now - timedelta(hours=9)).isoformat()},
                    ],
                    "validation_loss": [
                        {"epoch": 5, "loss": 1.42, "timestamp": (now - timedelta(hours=8)).isoformat()},
                    ],
                },
                "component_id": "component-456",
                "created_by_component": "NVIDIANeMoCustomizer",
            },
        ]

        for job in sample_jobs:
            self._jobs[job["id"]] = job

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
    # Job Management (Customizer)
    # =============================================================================

    async def store_customizer_job(self, job_data: dict) -> dict[str, Any]:
        """Store job info from NeMo component for tracking.

        Args:
            job_data: Job information from NeMo Customizer component

        Returns:
            Stored job data
        """
        # Simulate network delay
        await asyncio.sleep(0.1)

        job_id = job_data["job_info"]["id"]
        stored_job = {"stored_at": datetime.now(timezone.utc).isoformat(), **job_data}

        self._jobs[job_id] = stored_job
        return stored_job

    async def get_customizer_jobs(self) -> list[dict[str, Any]]:
        """Get all tracked customizer jobs.

        Returns:
            List of job objects with status and metadata
        """
        # Simulate network delay
        await asyncio.sleep(0.1)

        return list(self._jobs.values())

    async def get_customizer_job(self, job_id: str) -> dict[str, Any] | None:
        """Get detailed information about a specific customizer job.

        Args:
            job_id: NeMo Customizer job ID

        Returns:
            Job details or None if not found
        """
        # Simulate network delay
        await asyncio.sleep(0.1)

        return self._jobs.get(job_id)

    def cleanup(self):
        """Clean up temporary files and directories."""
        if self._temp_dir.exists():
            shutil.rmtree(self._temp_dir)


# Global instance for use in API
mock_nemo_service = MockNeMoMicroservicesService()

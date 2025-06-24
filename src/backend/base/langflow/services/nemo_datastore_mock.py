"""Mock NeMo Data Store Service for testing and development.

This module provides a mock implementation of the NeMo Data Store API
to allow development and testing without requiring the actual NeMo services.
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


class MockNeMoDataStoreService:
    """Mock implementation of NeMo Data Store API for testing.

    Simulates the behavior of the actual NeMo Data Store API endpoints:
    - GET /api/v1/datasets
    - POST /api/v1/datasets
    - GET /api/v1/datasets/{dataset_id}
    - DELETE /api/v1/datasets/{dataset_id}
    - POST /api/v1/datasets/{dataset_id}/files
    """

    def __init__(self):
        # In-memory storage for mock data
        self._datasets: dict[str, dict] = {}
        self._files: dict[str, list[dict]] = {}
        self._temp_dir = Path(tempfile.mkdtemp(prefix="nemo_mock_"))

        # Initialize with some sample data
        self._initialize_sample_data()

    def _initialize_sample_data(self):
        """Initialize with sample datasets for testing."""
        now = datetime.now(timezone.utc)
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
        logger = logging.getLogger("nemo_datastore_mock.upload_files")
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

    def cleanup(self):
        """Clean up temporary files and directories."""
        if self._temp_dir.exists():
            shutil.rmtree(self._temp_dir)


# Global instance for use in API
mock_nemo_service = MockNeMoDataStoreService()

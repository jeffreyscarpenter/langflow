"""Mock NeMo Microservices for testing and development.

This module provides a mock implementation of the NeMo Microservices APIs
to allow development and testing without requiring the actual NeMo services.

Includes mock implementations for:
- NeMo Data Store (datasets and files)
- NeMo Entity Store (entities, projects, models)
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
    - NeMo Entity Store API endpoints
    - NeMo Customizer API endpoints (matching real API structure)
    - NeMo Evaluator results (future)
    """

    def __init__(self):
        # In-memory storage for mock data
        self._datasets: dict[str, dict] = {}
        self._files: dict[str, list[dict]] = {}
        self._customizer_jobs: dict[str, dict] = {}  # Jobs from NeMo Customizer API format
        self._tracked_jobs: list[str] = []  # Job IDs we're tracking for monitoring
        self._evaluation_jobs: dict[str, dict] = {}  # Jobs from NeMo Evaluator API format
        self._evaluation_configs: dict[str, dict] = {}  # Evaluation configurations
        self._evaluation_targets: dict[str, dict] = {}  # Evaluation targets

        # Entity Store mock data
        self._namespaces: dict[str, dict] = {}
        self._projects: dict[str, dict] = {}
        self._models: dict[str, dict] = {}
        self._entities: dict[str, dict] = {}  # Generic entity storage

        self._temp_dir = Path(tempfile.mkdtemp(prefix="nemo_mock_"))

        # Initialize with some sample data
        self._initialize_sample_data()

    def _initialize_sample_data(self):
        """Initialize with sample datasets, jobs, and entities for testing."""
        now = datetime.now(timezone.utc)

        # Sample namespaces
        sample_namespaces = [
            {
                "id": "default",
                "name": "default",
                "description": "Default namespace for development",
                "created_at": (now - timedelta(days=30)).isoformat(),
                "updated_at": (now - timedelta(days=1)).isoformat(),
            },
            {
                "id": "my-company",
                "name": "my-company",
                "description": "Company namespace for production",
                "created_at": (now - timedelta(days=15)).isoformat(),
                "updated_at": (now - timedelta(hours=6)).isoformat(),
            },
        ]

        for namespace in sample_namespaces:
            self._namespaces[namespace["id"]] = namespace

        # Sample projects
        sample_projects = [
            {
                "id": "project-001",
                "name": "Customer Support AI",
                "description": "AI models for customer support automation",
                "namespace": "my-company",
                "created_at": (now - timedelta(days=10)).isoformat(),
                "updated_at": (now - timedelta(days=2)).isoformat(),
            },
            {
                "id": "project-002",
                "name": "Content Generation",
                "description": "Models for automated content generation",
                "namespace": "my-company",
                "created_at": (now - timedelta(days=8)).isoformat(),
                "updated_at": (now - timedelta(hours=12)).isoformat(),
            },
        ]

        for project in sample_projects:
            self._projects[project["id"]] = project

        # Sample models
        sample_models = [
            {
                "id": "model-001",
                "name": "llama-3.1-8b-instruct",
                "description": "Base Llama 3.1 8B instruction-tuned model",
                "namespace": "default",
                "type": "base_model",
                "format": "nemo",
                "files_url": "hf://models/default/llama-3.1-8b-instruct",
                "created_at": (now - timedelta(days=20)).isoformat(),
                "updated_at": (now - timedelta(days=5)).isoformat(),
            },
            {
                "id": "model-002",
                "name": "customer-support-finetuned",
                "description": "Fine-tuned model for customer support",
                "namespace": "my-company",
                "type": "fine_tuned_model",
                "format": "nemo",
                "base_model": "default/llama-3.1-8b-instruct",
                "training_dataset": "my-company/customer-support-dataset",
                "files_url": "hf://models/my-company/customer-support-finetuned",
                "created_at": (now - timedelta(days=3)).isoformat(),
                "updated_at": (now - timedelta(hours=2)).isoformat(),
            },
        ]

        for model in sample_models:
            self._models[model["id"]] = model

        # Sample datasets
        sample_datasets = [
            {
                "id": "dataset-001",
                "name": "Sample Training Data",
                "description": "A sample dataset for model training",
                "type": "fileset",
                "namespace": "default",
                "files_url": "hf://datasets/default/sample-training-data",
                "format": "jsonl",
                "project": "default",
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
                "namespace": "default",
                "files_url": "hf://datasets/default/evaluation-dataset",
                "format": "jsonl",
                "project": "default",
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
                "namespace": "default",
                "files_url": "hf://datasets/default/fine-tuning-data",
                "format": "jsonl",
                "project": "default",
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
                "output_model": (
                    "default/meta-llama-3.2-1b-instruct-fine-tuning-dataset-full@cust-YbmGLDpnZUPMGjqKZ2MaUy"
                ),
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
                            "updated_at": (now - timedelta(days=1, minutes=-1)).isoformat(),
                            "message": "TrainingJobRunning",
                        },
                        {
                            "updated_at": (now - timedelta(hours=8)).isoformat(),
                            "message": "TrainingJobCompleted",
                        },
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
                            "updated_at": (now - timedelta(hours=3)).isoformat(),
                            "message": "TrainingJobFailed",
                        },
                    ],
                },
                "custom_fields": {},
            },
        ]

        for job in sample_jobs:
            self._customizer_jobs[job["id"]] = job

        # Sample NeMo Evaluator jobs (matching real API structure)
        sample_evaluator_jobs = [
            {
                "id": "eval-123456789abc",
                "created_at": (now - timedelta(hours=1)).isoformat(),
                "updated_at": (now - timedelta(minutes=30)).isoformat(),
                "namespace": "default",
                "target": "default/target-123",
                "config": "default/config-456",
                "tags": ["gsm8k", "math-evaluation"],
                "status": "running",
                "status_details": {
                    "created_at": (now - timedelta(hours=1)).isoformat(),
                    "updated_at": (now - timedelta(minutes=30)).isoformat(),
                    "message": "Evaluation job is running on GSM8K dataset",
                    "percentage_done": 60,
                },
            },
            {
                "id": "eval-987654321def",
                "created_at": (now - timedelta(days=1)).isoformat(),
                "updated_at": (now - timedelta(hours=6)).isoformat(),
                "namespace": "default",
                "target": "default/target-789",
                "config": "default/config-012",
                "tags": ["similarity-metrics", "accuracy", "bleu"],
                "status": "completed",
                "status_details": {
                    "created_at": (now - timedelta(days=1)).isoformat(),
                    "updated_at": (now - timedelta(hours=6)).isoformat(),
                    "message": "Similarity metrics evaluation completed successfully",
                    "percentage_done": 100,
                },
            },
            {
                "id": "eval-555666777ghi",
                "created_at": (now - timedelta(hours=3)).isoformat(),
                "updated_at": (now - timedelta(hours=2, minutes=45)).isoformat(),
                "namespace": "default",
                "target": "default/target-345",
                "config": "default/config-678",
                "tags": ["lm-eval-harness", "mmlu"],
                "status": "failed",
                "status_details": {
                    "created_at": (now - timedelta(hours=3)).isoformat(),
                    "updated_at": (now - timedelta(hours=2, minutes=45)).isoformat(),
                    "message": "Evaluation failed due to model endpoint timeout",
                    "percentage_done": 20,
                },
            },
        ]

        for job in sample_evaluator_jobs:
            self._evaluation_jobs[job["id"]] = job

    # =============================================================================
    # Entity Store Management
    # =============================================================================

    async def create_namespace(self, namespace_data: dict) -> dict[str, Any]:
        """Mock implementation of POST /v1/namespaces.

        Args:
            namespace_data: Namespace creation data

        Returns:
            Created namespace information
        """
        # Simulate network delay
        await asyncio.sleep(0.1)

        now = datetime.now(timezone.utc)
        namespace_id = namespace_data.get("namespace", str(uuid.uuid4()))

        namespace = {
            "id": namespace_id,
            "name": namespace_data.get("namespace", namespace_id),
            "description": namespace_data.get("description", ""),
            "created_at": now.isoformat(),
            "updated_at": now.isoformat(),
        }

        self._namespaces[namespace_id] = namespace
        return namespace

    async def get_namespace(self, namespace_id: str) -> dict[str, Any] | None:
        """Mock implementation of GET /v1/namespaces/{namespace_id}.

        Args:
            namespace_id: Namespace ID

        Returns:
            Namespace information or None if not found
        """
        # Simulate network delay
        await asyncio.sleep(0.1)

        return self._namespaces.get(namespace_id)

    async def list_namespaces(self) -> list[dict[str, Any]]:
        """Mock implementation of GET /v1/namespaces.

        Returns:
            List of all namespaces
        """
        # Simulate network delay
        await asyncio.sleep(0.1)

        return list(self._namespaces.values())

    async def create_project(self, project_data: dict) -> dict[str, Any]:
        """Mock implementation of POST /v1/projects.

        Args:
            project_data: Project creation data

        Returns:
            Created project information
        """
        # Simulate network delay
        await asyncio.sleep(0.1)

        now = datetime.now(timezone.utc)
        project_id = f"project-{str(uuid.uuid4())[:8]}"

        project = {
            "id": project_id,
            "name": project_data.get("name", ""),
            "description": project_data.get("description", ""),
            "namespace": project_data.get("namespace", "default"),
            "created_at": now.isoformat(),
            "updated_at": now.isoformat(),
        }

        self._projects[project_id] = project
        return project

    async def get_project(self, project_id: str) -> dict[str, Any] | None:
        """Mock implementation of GET /v1/projects/{project_id}.

        Args:
            project_id: Project ID

        Returns:
            Project information or None if not found
        """
        # Simulate network delay
        await asyncio.sleep(0.1)

        return self._projects.get(project_id)

    async def list_projects(self) -> list[dict[str, Any]]:
        """Mock implementation of GET /v1/projects.

        Returns:
            List of all projects
        """
        # Simulate network delay
        await asyncio.sleep(0.1)

        return list(self._projects.values())

    async def create_entity(self, entity_type: str, entity_data: dict) -> dict[str, Any]:
        """Mock implementation of POST /v1/{entity_type}.

        Generic entity creation for datasets, models, etc.

        Args:
            entity_type: Type of entity (datasets, models, etc.)
            entity_data: Entity creation data

        Returns:
            Created entity information
        """
        # Simulate network delay
        await asyncio.sleep(0.1)

        now = datetime.now(timezone.utc)
        entity_id = f"{entity_type}-{str(uuid.uuid4())[:8]}"

        # Base entity structure
        entity = {
            "id": entity_id,
            "name": entity_data.get("name", ""),
            "description": entity_data.get("description", ""),
            "namespace": entity_data.get("namespace", "default"),
            "created_at": now.isoformat(),
            "updated_at": now.isoformat(),
        }

        # Add entity-specific fields
        if entity_type == "datasets":
            entity.update(
                {
                    "type": "fileset",
                    "files_url": entity_data.get("files_url", ""),
                    "format": entity_data.get("format", "jsonl"),
                    "project": entity_data.get("project", ""),
                }
            )
        elif entity_type == "models":
            # Generate files_url if not provided
            files_url = entity_data.get("files_url", "")
            if not files_url:
                files_url = f"hf://models/{entity_data.get('namespace', 'default')}/{entity_data.get('name', '')}"

            entity.update(
                {
                    "type": entity_data.get("type", "base_model"),
                    "format": entity_data.get("format", "nemo"),
                    "files_url": files_url,
                    "base_model": entity_data.get("base_model", ""),
                    "training_dataset": entity_data.get("training_dataset", ""),
                }
            )

        # Add any additional fields from entity_data
        additional_fields = {
            key: value
            for key, value in entity_data.items()
            if key not in ["name", "description", "namespace"]  # Already handled above
        }
        entity.update(additional_fields)

        # Store in appropriate collection
        if entity_type == "datasets":
            self._datasets[entity_id] = entity
            self._files[entity_id] = []  # Initialize files list for datasets
        elif entity_type == "models":
            self._models[entity_id] = entity
        else:
            self._entities[entity_id] = entity

        return entity

    async def get_entity(self, entity_type: str, entity_id: str) -> dict[str, Any] | None:
        """Mock implementation of GET /v1/{entity_type}/{entity_id}.

        Generic entity retrieval.

        Args:
            entity_type: Type of entity (datasets, models, etc.)
            entity_id: Entity ID

        Returns:
            Entity information or None if not found
        """
        # Simulate network delay
        await asyncio.sleep(0.1)

        if entity_type == "datasets":
            return self._datasets.get(entity_id)
        if entity_type == "models":
            return self._models.get(entity_id)
        return self._entities.get(entity_id)

    async def list_entities(self, entity_type: str) -> list[dict[str, Any]]:
        """Mock implementation of GET /v1/{entity_type}.

        Generic entity listing.

        Args:
            entity_type: Type of entity (datasets, models, etc.)

        Returns:
            List of entities of the specified type
        """
        # Simulate network delay
        await asyncio.sleep(0.1)

        if entity_type == "datasets":
            return list(self._datasets.values())
        if entity_type == "models":
            return list(self._models.values())
        return list(self._entities.values())

    async def update_entity(self, entity_type: str, entity_id: str, updates: dict) -> dict[str, Any] | None:
        """Mock implementation of PUT /v1/{entity_type}/{entity_id}.

        Generic entity update.

        Args:
            entity_type: Type of entity (datasets, models, etc.)
            entity_id: Entity ID
            updates: Update data

        Returns:
            Updated entity information or None if not found
        """
        # Simulate network delay
        await asyncio.sleep(0.1)

        # Add a small delay to ensure timestamp difference
        await asyncio.sleep(0.01)
        now = datetime.now(timezone.utc)

        if entity_type == "datasets":
            entity = self._datasets.get(entity_id)
            if entity:
                entity.update(updates)
                entity["updated_at"] = now.isoformat()
                return entity
        elif entity_type == "models":
            entity = self._models.get(entity_id)
            if entity:
                entity.update(updates)
                entity["updated_at"] = now.isoformat()
                return entity
        else:
            entity = self._entities.get(entity_id)
            if entity:
                entity.update(updates)
                entity["updated_at"] = now.isoformat()
                return entity

        return None

    async def delete_entity(self, entity_type: str, entity_id: str) -> bool:
        """Mock implementation of DELETE /v1/{entity_type}/{entity_id}.

        Generic entity deletion.

        Args:
            entity_type: Type of entity (datasets, models, etc.)
            entity_id: Entity ID

        Returns:
            True if deleted, False if not found
        """
        # Simulate network delay
        await asyncio.sleep(0.1)

        if entity_type == "datasets":
            if entity_id in self._datasets:
                del self._datasets[entity_id]
                return True
        elif entity_type == "models":
            if entity_id in self._models:
                del self._models[entity_id]
                return True
        elif entity_id in self._entities:
            del self._entities[entity_id]
            return True

        return False

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
        self, name: str, description: str | None = None, namespace: str = "default", project: str | None = None
    ) -> dict[str, Any]:
        """Mock implementation of POST /api/v1/datasets with Entity Store integration.

        Args:
            name: Dataset name.
            description: Optional description.
            namespace: Namespace for the dataset.
            project: Project ID (optional).

        Returns:
            Created dataset object.
        """
        # Simulate network delay
        await asyncio.sleep(0.2)

        dataset_id = f"dataset-{str(uuid.uuid4())[:8]}"
        now = datetime.now(timezone.utc).isoformat()

        # Create files_url for Data Store reference
        files_url = f"hf://datasets/{namespace}/{name}"

        # Create the entity with the same ID we'll use for the dataset
        entity = {
            "id": dataset_id,
            "name": name,
            "description": description or "",
            "namespace": namespace,
            "created_at": now,
            "updated_at": now,
            "type": "fileset",
            "files_url": files_url,
            "format": "jsonl",
            "project": project or "",
        }

        # Store in both Entity Store and Data Store
        self._datasets[dataset_id] = entity
        self._files[dataset_id] = []

        # Add additional metadata for Data Store compatibility
        entity["metadata"] = {"file_count": 0, "total_size": "0B", "format": "jsonl", "tags": []}

        return entity

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

            # Update entity in Entity Store
            if dataset_id in self._datasets:
                dataset = self._datasets[dataset_id]
                await self.update_entity(
                    "datasets", dataset_id, {"updated_at": dataset["updated_at"], "metadata": dataset["metadata"]}
                )

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

    async def get_customization_configs(self) -> dict[str, Any]:
        """Mock implementation of GET /v1/customization/configs.

        This endpoint provides available model configurations for customization.
        Used by the NeMo Customizer component to populate dropdown options.

        Returns:
            Available model configurations with training and fine-tuning types
        """
        # No delay for maximum responsiveness
        # await asyncio.sleep(0.01)

        now = datetime.now(timezone.utc)

        # Sample model configurations that can be customized
        configs = [
            {
                "schema_version": "1.0",
                "id": "58bee815-0473-45d7-a5e6-fc088f6142eb",
                "namespace": "default",
                "created_at": (now - timedelta(days=30)).isoformat(),
                "updated_at": (now - timedelta(days=5)).isoformat(),
                "custom_fields": {},
                "name": "meta/llama-3.1-8b-instruct",
                "base_model": "meta/llama-3.1-8b-instruct",
                "model_path": "llama-3_1-8b-instruct",
                "training_types": ["sft", "dpo", "rm"],
                "finetuning_types": ["lora", "qlora", "full"],
                "precision": "bf16",
                "num_gpus": 4,
                "num_nodes": 1,
                "micro_batch_size": 1,
                "tensor_parallel_size": 1,
                "max_seq_length": 4096,
            },
            {
                "schema_version": "1.0",
                "id": "99bee815-0473-45d7-a5e6-fc088f6142eb",
                "namespace": "default",
                "created_at": (now - timedelta(days=25)).isoformat(),
                "updated_at": (now - timedelta(days=3)).isoformat(),
                "custom_fields": {},
                "name": "meta/llama-3.2-1b-instruct",
                "base_model": "meta/llama-3.2-1b-instruct",
                "model_path": "llama-3_2-1b-instruct",
                "training_types": ["sft", "dpo"],
                "finetuning_types": ["lora", "qlora", "full"],
                "precision": "bf16",
                "num_gpus": 2,
                "num_nodes": 1,
                "micro_batch_size": 1,
                "tensor_parallel_size": 1,
                "max_seq_length": 4096,
            },
            {
                "schema_version": "1.0",
                "id": "77bee815-0473-45d7-a5e6-fc088f6142eb",
                "namespace": "default",
                "created_at": (now - timedelta(days=20)).isoformat(),
                "updated_at": (now - timedelta(days=1)).isoformat(),
                "custom_fields": {},
                "name": "meta/llama-3.1-70b-instruct",
                "base_model": "meta/llama-3.1-70b-instruct",
                "model_path": "llama-3_1-70b-instruct",
                "training_types": ["sft", "dpo", "rm"],
                "finetuning_types": ["lora", "qlora"],
                "precision": "bf16",
                "num_gpus": 8,
                "num_nodes": 1,
                "micro_batch_size": 1,
                "tensor_parallel_size": 2,
                "max_seq_length": 4096,
            },
            {
                "schema_version": "1.0",
                "id": "44bee815-0473-45d7-a5e6-fc088f6142eb",
                "namespace": "default",
                "created_at": (now - timedelta(days=15)).isoformat(),
                "updated_at": (now - timedelta(hours=12)).isoformat(),
                "custom_fields": {},
                "name": "microsoft/DialoGPT-medium",
                "base_model": "microsoft/DialoGPT-medium",
                "model_path": "microsoft_DialoGPT-medium",
                "training_types": ["sft"],
                "finetuning_types": ["lora", "qlora", "full"],
                "precision": "bf16",
                "num_gpus": 2,
                "num_nodes": 1,
                "micro_batch_size": 1,
                "tensor_parallel_size": 1,
                "max_seq_length": 1024,
            },
            {
                "schema_version": "1.0",
                "id": "55bee815-0473-45d7-a5e6-fc088f6142eb",
                "namespace": "default",
                "created_at": (now - timedelta(days=10)).isoformat(),
                "updated_at": (now - timedelta(hours=6)).isoformat(),
                "custom_fields": {},
                "name": "gpt2",
                "base_model": "gpt2",
                "model_path": "gpt2",
                "training_types": ["sft"],
                "finetuning_types": ["lora", "qlora", "full"],
                "precision": "bf16",
                "num_gpus": 1,
                "num_nodes": 1,
                "micro_batch_size": 1,
                "tensor_parallel_size": 1,
                "max_seq_length": 512,
            },
        ]

        return {
            "data": configs,
            "total": len(configs),
            "page": 1,
            "size": len(configs),
        }

    async def create_customization_job(self, job_data: dict) -> dict[str, Any]:
        """Mock implementation of POST /v1/customization/jobs.

        This matches the real NeMo Customizer API endpoint that creates
        a new customization job.

        Args:
            job_data: Job configuration including model, dataset, hyperparameters, etc.

        Returns:
            Created job object with job ID and initial status
        """
        # Simulate network delay
        await asyncio.sleep(0.2)

        # Generate a unique job ID
        job_id = f"cust-{uuid.uuid4().hex[:24]}"
        now = datetime.now(timezone.utc)

        # Extract data from the request
        config_name = job_data.get("config", "Unknown Model")
        dataset_info = job_data.get("dataset", {})
        dataset_name = dataset_info.get("name", "Unknown Dataset")
        namespace = dataset_info.get("namespace", "default")
        hyperparameters = job_data.get("hyperparameters", {})
        output_model = job_data.get("output_model", f"{namespace}/{job_id}")

        # Create the job entry
        job_entry = {
            "id": job_id,
            "created_at": now.isoformat(),
            "updated_at": now.isoformat(),
            "namespace": namespace,
            "config": {
                "schema_version": "1.0",
                "id": str(uuid.uuid4()),
                "namespace": namespace,
                "created_at": now.isoformat(),
                "updated_at": now.isoformat(),
                "custom_fields": {},
                "name": config_name,
                "base_model": config_name,
                "model_path": config_name.replace("/", "_"),
                "training_types": [hyperparameters.get("training_type", "sft")],
                "finetuning_types": [hyperparameters.get("finetuning_type", "lora")],
                "precision": "bf16",
                "num_gpus": 4,
                "num_nodes": 1,
                "micro_batch_size": 1,
                "tensor_parallel_size": 1,
                "max_seq_length": 4096,
            },
            "dataset": f"{namespace}/{dataset_name}",
            "hyperparameters": hyperparameters,
            "output_model": output_model,
            "status": "created",
            "status_details": {
                "created_at": now.isoformat(),
                "updated_at": now.isoformat(),
                "steps_completed": 0,
                "epochs_completed": 0,
                "percentage_done": 0,
                "status_logs": [
                    {
                        "updated_at": now.isoformat(),
                        "message": "created",
                    },
                ],
                "training_loss": [],
                "validation_loss": [],
            },
            "custom_fields": {},
        }

        # Store the job
        self._customizer_jobs[job_id] = job_entry

        # Return the created job
        return job_entry

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

        # Add to tracked jobs list
        if job_id not in self._tracked_jobs:
            self._tracked_jobs.append(job_id)

        # Create a job entry if it doesn't exist
        if job_id not in self._customizer_jobs:
            now = datetime.now(timezone.utc)
            job_entry = {
                "id": job_id,
                "created_at": now.isoformat(),
                "updated_at": now.isoformat(),
                "namespace": metadata.get("namespace", "default") if metadata else "default",
                "config": {
                    "schema_version": "1.0",
                    "id": str(uuid.uuid4()),
                    "namespace": metadata.get("namespace", "default") if metadata else "default",
                    "created_at": now.isoformat(),
                    "updated_at": now.isoformat(),
                    "custom_fields": {},
                    "name": metadata.get("config", "Unknown Model") if metadata else "Unknown Model",
                    "base_model": metadata.get("config", "Unknown Model") if metadata else "Unknown Model",
                    "model_path": (
                        metadata.get("config", "unknown-model").replace("/", "_") if metadata else "unknown-model"
                    ),
                    "training_types": [metadata.get("training_type", "sft")] if metadata else ["sft"],
                    "finetuning_types": [metadata.get("fine_tuning_type", "lora")] if metadata else ["lora"],
                    "precision": "bf16",
                    "num_gpus": 4,
                    "num_nodes": 1,
                    "micro_batch_size": 1,
                    "tensor_parallel_size": 1,
                    "max_seq_length": 4096,
                },
                "dataset": metadata.get("dataset", "Unknown Dataset") if metadata else "Unknown Dataset",
                "hyperparameters": {
                    "finetuning_type": metadata.get("fine_tuning_type", "lora") if metadata else "lora",
                    "training_type": metadata.get("training_type", "sft") if metadata else "sft",
                    "batch_size": metadata.get("batch_size", 16) if metadata else 16,
                    "epochs": metadata.get("epochs", 5) if metadata else 5,
                    "learning_rate": metadata.get("learning_rate", 0.0001) if metadata else 0.0001,
                },
                "output_model": metadata.get("output_model", f"default/{job_id}") if metadata else f"default/{job_id}",
                "status": "created",
                "status_details": {
                    "created_at": now.isoformat(),
                    "updated_at": now.isoformat(),
                    "steps_completed": 0,
                    "epochs_completed": 0,
                    "percentage_done": 0,
                    "status_logs": [
                        {
                            "updated_at": now.isoformat(),
                            "message": "created",
                        },
                    ],
                    "training_loss": [],
                    "validation_loss": [],
                },
                "custom_fields": {},
            }

            # Add LoRA config if applicable
            if metadata and metadata.get("fine_tuning_type") == "lora":
                job_entry["hyperparameters"]["lora"] = {"adapter_dim": 16}

            self._customizer_jobs[job_id] = job_entry

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

    # =============================================================================
    # Evaluation Management (Evaluator) - Real NeMo API Structure
    # =============================================================================

    async def create_evaluation_job(self, job_data: dict) -> dict[str, Any]:
        """Mock implementation of POST /v1/evaluation/jobs.

        This matches the real NeMo Evaluator API endpoint for creating evaluation jobs.

        Args:
            job_data: Evaluation job configuration data

        Returns:
            Created evaluation job information
        """
        # Simulate network delay
        await asyncio.sleep(0.1)

        now = datetime.now(timezone.utc)
        job_id = f"eval-{uuid.uuid4().hex[:12]}"

        # Create evaluation job entry
        evaluation_job = {
            "id": job_id,
            "created_at": now.isoformat(),
            "updated_at": now.isoformat(),
            "namespace": job_data.get("namespace", "default"),
            "target": job_data.get("target", ""),
            "config": job_data.get("config", ""),
            "tags": job_data.get("tags", []),
            "status": "created",
            "status_details": {
                "created_at": now.isoformat(),
                "updated_at": now.isoformat(),
                "message": "Evaluation job created successfully",
            },
        }

        self._evaluation_jobs[job_id] = evaluation_job

        return evaluation_job

    async def create_evaluation_config(self, config_data: dict) -> dict[str, Any]:
        """Mock implementation of POST /v1/evaluation/configs.

        This matches the real NeMo Evaluator API endpoint for creating evaluation configurations.

        Args:
            config_data: Evaluation configuration data

        Returns:
            Created evaluation configuration information
        """
        # Simulate network delay
        await asyncio.sleep(0.1)

        now = datetime.now(timezone.utc)
        config_id = str(uuid.uuid4())

        # Create evaluation config entry
        evaluation_config = {
            "id": config_id,
            "created_at": now.isoformat(),
            "updated_at": now.isoformat(),
            "namespace": config_data.get("namespace", "default"),
            "type": config_data.get("type", "lm_eval_harness"),
            "tasks": config_data.get("tasks", []),
            "params": config_data.get("params", {}),
        }

        self._evaluation_configs[config_id] = evaluation_config

        return evaluation_config

    async def create_evaluation_target(self, target_data: dict) -> dict[str, Any]:
        """Mock implementation of POST /v1/evaluation/targets.

        This matches the real NeMo Evaluator API endpoint for creating evaluation targets.

        Args:
            target_data: Evaluation target configuration data

        Returns:
            Created evaluation target information
        """
        # Simulate network delay
        await asyncio.sleep(0.1)

        now = datetime.now(timezone.utc)
        target_id = str(uuid.uuid4())

        # Create evaluation target entry
        evaluation_target = {
            "id": target_id,
            "created_at": now.isoformat(),
            "updated_at": now.isoformat(),
            "namespace": target_data.get("namespace", "default"),
            "type": target_data.get("type", "model"),
            "model": target_data.get("model", {}),
        }

        self._evaluation_targets[target_id] = evaluation_target

        return evaluation_target

    async def get_evaluation_job(self, job_id: str) -> dict[str, Any] | None:
        """Mock implementation of GET /v1/evaluation/jobs/{job_id}.

        This matches the real NeMo Evaluator API endpoint for getting evaluation job details.

        Args:
            job_id: NeMo Evaluator job ID

        Returns:
            Evaluation job details or None if not found
        """
        # Simulate network delay
        await asyncio.sleep(0.1)

        return self._evaluation_jobs.get(job_id)

    async def list_evaluation_jobs(self) -> list[dict[str, Any]]:
        """Mock implementation of GET /v1/evaluation/jobs.

        This matches the real NeMo Evaluator API endpoint for listing evaluation jobs.

        Returns:
            List of all evaluation jobs
        """
        # Simulate network delay
        await asyncio.sleep(0.1)

        return list(self._evaluation_jobs.values())

    def cleanup(self):
        """Clean up temporary files and directories."""
        if self._temp_dir.exists():
            shutil.rmtree(self._temp_dir)


# Global instance for use in API
mock_nemo_service = MockNeMoMicroservicesService()

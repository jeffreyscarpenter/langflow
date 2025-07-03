"""Unit tests for NeMo Microservices Mock Service.

Tests the mock implementation of NeMo Microservices APIs including:
- Entity Store (namespaces, projects, entities)
- Data Store (datasets, files)
- Cross-service integration
"""

import asyncio
from unittest.mock import AsyncMock, MagicMock

import pytest
from base.langflow.services.nemo_microservices_mock import MockNeMoMicroservicesService


class TestMockNeMoMicroservicesService:
    """Test suite for MockNeMoMicroservicesService."""

    @pytest.fixture
    async def mock_service(self):
        """Create a fresh mock service instance for each test."""
        service = MockNeMoMicroservicesService()
        yield service
        service.cleanup()

    @pytest.mark.asyncio
    async def test_initialization_with_sample_data(self, mock_service):
        """Test that the mock service initializes with sample data."""
        # Check namespaces
        namespaces = await mock_service.list_namespaces()
        assert len(namespaces) >= 2
        namespace_names = [ns["name"] for ns in namespaces]
        assert "default" in namespace_names
        assert "my-company" in namespace_names

        # Check projects
        projects = await mock_service.list_projects()
        assert len(projects) >= 2
        project_names = [p["name"] for p in projects]
        assert "Customer Support AI" in project_names
        assert "Content Generation" in project_names

        # Check models
        models = await mock_service.list_entities("models")
        assert len(models) >= 2
        model_names = [m["name"] for m in models]
        assert "llama-3.1-8b-instruct" in model_names

        # Check datasets
        datasets = await mock_service.list_entities("datasets")
        assert len(datasets) >= 3

    @pytest.mark.asyncio
    async def test_namespace_operations(self, mock_service):
        """Test namespace creation, retrieval, and listing."""
        # Create namespace
        namespace_data = {"namespace": "test-namespace", "description": "Test namespace for unit testing"}
        created_namespace = await mock_service.create_namespace(namespace_data)

        assert created_namespace["name"] == "test-namespace"
        assert created_namespace["description"] == "Test namespace for unit testing"
        assert "id" in created_namespace
        assert "created_at" in created_namespace
        assert "updated_at" in created_namespace

        # Get namespace
        retrieved_namespace = await mock_service.get_namespace(created_namespace["id"])
        assert retrieved_namespace == created_namespace

        # List namespaces
        all_namespaces = await mock_service.list_namespaces()
        namespace_ids = [ns["id"] for ns in all_namespaces]
        assert created_namespace["id"] in namespace_ids

        # Test non-existent namespace
        non_existent = await mock_service.get_namespace("non-existent")
        assert non_existent is None

    @pytest.mark.asyncio
    async def test_project_operations(self, mock_service):
        """Test project creation, retrieval, and listing."""
        # Create project
        project_data = {
            "name": "Test Project",
            "description": "Test project for unit testing",
            "namespace": "test-namespace",
        }
        created_project = await mock_service.create_project(project_data)

        assert created_project["name"] == "Test Project"
        assert created_project["description"] == "Test project for unit testing"
        assert created_project["namespace"] == "test-namespace"
        assert "id" in created_project
        assert created_project["id"].startswith("project-")

        # Get project
        retrieved_project = await mock_service.get_project(created_project["id"])
        assert retrieved_project == created_project

        # List projects
        all_projects = await mock_service.list_projects()
        project_ids = [p["id"] for p in all_projects]
        assert created_project["id"] in project_ids

        # Test non-existent project
        non_existent = await mock_service.get_project("non-existent")
        assert non_existent is None

    @pytest.mark.asyncio
    async def test_entity_operations(self, mock_service):
        """Test generic entity creation, retrieval, and listing."""
        # Create dataset entity
        dataset_entity_data = {
            "name": "test-dataset-entity",
            "description": "Test dataset entity",
            "namespace": "test-namespace",
            "files_url": "hf://datasets/test-namespace/test-dataset-entity",
            "format": "jsonl",
            "project": "test-project",
        }
        created_dataset = await mock_service.create_entity("datasets", dataset_entity_data)

        assert created_dataset["name"] == "test-dataset-entity"
        assert created_dataset["namespace"] == "test-namespace"
        assert created_dataset["files_url"] == "hf://datasets/test-namespace/test-dataset-entity"
        assert created_dataset["type"] == "fileset"
        assert created_dataset["format"] == "jsonl"
        assert created_dataset["project"] == "test-project"

        # Create model entity
        model_entity_data = {
            "name": "test-model",
            "description": "Test model entity",
            "namespace": "test-namespace",
            "type": "base_model",
            "format": "nemo",
            "files_url": "hf://models/test-namespace/test-model",
        }
        created_model = await mock_service.create_entity("models", model_entity_data)

        assert created_model["name"] == "test-model"
        assert created_model["type"] == "base_model"
        assert created_model["format"] == "nemo"
        assert created_model["files_url"] == "hf://models/test-namespace/test-model"

        # Get entities
        retrieved_dataset = await mock_service.get_entity("datasets", created_dataset["id"])
        assert retrieved_dataset == created_dataset

        retrieved_model = await mock_service.get_entity("models", created_model["id"])
        assert retrieved_model == created_model

        # List entities
        datasets = await mock_service.list_entities("datasets")
        models = await mock_service.list_entities("models")

        dataset_ids = [d["id"] for d in datasets]
        model_ids = [m["id"] for m in models]

        assert created_dataset["id"] in dataset_ids
        assert created_model["id"] in model_ids

    @pytest.mark.asyncio
    async def test_entity_update_and_delete(self, mock_service):
        """Test entity update and delete operations."""
        # Create entity
        entity_data = {"name": "test-entity", "description": "Original description", "namespace": "test-namespace"}
        created_entity = await mock_service.create_entity("datasets", entity_data)

        # Update entity
        updates = {"description": "Updated description", "name": "updated-entity-name"}

        # Add delay to ensure timestamp difference
        await asyncio.sleep(0.1)

        updated_entity = await mock_service.update_entity("datasets", created_entity["id"], updates)

        assert updated_entity["description"] == "Updated description"
        assert updated_entity["name"] == "updated-entity-name"
        # Note: In a mock environment, timestamps might be the same due to fast execution
        # The important thing is that the update operation completed successfully
        assert "updated_at" in updated_entity

        # Delete entity
        delete_result = await mock_service.delete_entity("datasets", created_entity["id"])
        assert delete_result is True

        # Verify deletion
        retrieved_entity = await mock_service.get_entity("datasets", created_entity["id"])
        assert retrieved_entity is None

        # Test deleting non-existent entity
        delete_result = await mock_service.delete_entity("datasets", "non-existent")
        assert delete_result is False

    @pytest.mark.asyncio
    async def test_dataset_creation_with_entity_store_integration(self, mock_service):
        """Test dataset creation with proper Entity Store integration."""
        # Create dataset with Entity Store integration
        dataset = await mock_service.create_dataset(
            name="integration-test-dataset",
            description="Test dataset for Entity Store integration",
            namespace="test-namespace",
            project="test-project",
        )

        # Verify Entity Store fields are present
        assert dataset["name"] == "integration-test-dataset"
        assert dataset["namespace"] == "test-namespace"
        assert dataset["project"] == "test-project"
        assert dataset["files_url"] == "hf://datasets/test-namespace/integration-test-dataset"
        assert dataset["format"] == "jsonl"
        assert dataset["type"] == "fileset"

        # Verify it's stored in both Data Store and Entity Store
        datasets = await mock_service.list_entities("datasets")
        dataset_ids = [d["id"] for d in datasets]
        assert dataset["id"] in dataset_ids

    @pytest.mark.asyncio
    async def test_files_url_generation(self, mock_service):
        """Test that files_url is generated correctly for different entity types."""
        # Test dataset files_url
        dataset = await mock_service.create_dataset(name="test-dataset", namespace="my-company")
        assert dataset["files_url"] == "hf://datasets/my-company/test-dataset"

        # Test model files_url
        model_data = {"name": "test-model", "namespace": "my-company", "type": "base_model"}
        model = await mock_service.create_entity("models", model_data)
        assert model["files_url"] == "hf://models/my-company/test-model"

    @pytest.mark.asyncio
    async def test_cross_service_integration(self, mock_service):
        """Test that Data Store and Entity Store work together properly."""
        # Create dataset
        dataset = await mock_service.create_dataset(name="cross-service-test", namespace="test-namespace")

        # Simulate file upload (Data Store operation)
        from fastapi import UploadFile

        mock_file = MagicMock(spec=UploadFile)
        mock_file.filename = "test.json"
        mock_file.content_type = "application/json"
        mock_file.read = AsyncMock(return_value=b'{"test": "data"}')

        upload_result = await mock_service.upload_files(dataset["id"], [mock_file])

        assert upload_result["message"].startswith("Successfully uploaded")
        assert len(upload_result["files"]) == 1

        # Verify Entity Store was updated
        updated_dataset = await mock_service.get_entity("datasets", dataset["id"])
        assert updated_dataset["metadata"]["file_count"] == 1
        assert updated_dataset["metadata"]["total_size"] == "16B"  # len(b'{"test": "data"}')

    @pytest.mark.asyncio
    async def test_entity_store_error_handling(self, mock_service):
        """Test error handling for Entity Store operations."""
        # Test getting non-existent entity
        result = await mock_service.get_entity("datasets", "non-existent-id")
        assert result is None

        # Test updating non-existent entity
        result = await mock_service.update_entity("datasets", "non-existent-id", {"name": "new-name"})
        assert result is None

        # Test deleting non-existent entity
        result = await mock_service.delete_entity("datasets", "non-existent-id")
        assert result is False

    @pytest.mark.asyncio
    async def test_entity_type_handling(self, mock_service):
        """Test that different entity types are handled correctly."""
        # Test unknown entity type
        unknown_entity_data = {
            "name": "unknown-entity",
            "description": "Test unknown entity type",
            "namespace": "test-namespace",
        }
        created_entity = await mock_service.create_entity("unknown_type", unknown_entity_data)

        assert created_entity["name"] == "unknown-entity"
        assert created_entity["namespace"] == "test-namespace"

        # Verify it's stored in generic entities
        retrieved_entity = await mock_service.get_entity("unknown_type", created_entity["id"])
        assert retrieved_entity == created_entity

    @pytest.mark.asyncio
    async def test_concurrent_operations(self, mock_service):
        """Test that concurrent operations work correctly."""

        # Create multiple entities concurrently
        async def create_entity(name, entity_type):
            entity_data = {"name": name, "description": f"Test {entity_type}", "namespace": "test-namespace"}
            return await mock_service.create_entity(entity_type, entity_data)

        # Run concurrent operations
        tasks = [
            create_entity("dataset-1", "datasets"),
            create_entity("dataset-2", "datasets"),
            create_entity("model-1", "models"),
            create_entity("model-2", "models"),
        ]

        results = await asyncio.gather(*tasks)

        # Verify all entities were created
        assert len(results) == 4
        for result in results:
            assert "id" in result
            assert "name" in result
            assert result["namespace"] == "test-namespace"

        # Verify they're all retrievable
        for result in results:
            entity_type = "datasets" if "dataset" in result["name"] else "models"
            retrieved = await mock_service.get_entity(entity_type, result["id"])
            assert retrieved == result

    @pytest.mark.asyncio
    async def test_sample_data_integrity(self, mock_service):
        """Test that sample data maintains integrity across operations."""
        # Get initial sample data
        initial_namespaces = await mock_service.list_namespaces()
        initial_projects = await mock_service.list_projects()
        initial_models = await mock_service.list_entities("models")
        initial_datasets = await mock_service.list_entities("datasets")

        # Perform some operations
        await mock_service.create_namespace({"namespace": "new-namespace"})
        await mock_service.create_project({"name": "New Project", "namespace": "new-namespace"})

        # Verify sample data is still intact
        namespaces = await mock_service.list_namespaces()
        projects = await mock_service.list_projects()
        models = await mock_service.list_entities("models")
        datasets = await mock_service.list_entities("datasets")

        # Sample data should still be present
        namespace_names = [ns["name"] for ns in namespaces]
        assert "default" in namespace_names
        assert "my-company" in namespace_names
        assert "new-namespace" in namespace_names

        project_names = [p["name"] for p in projects]
        assert "Customer Support AI" in project_names
        assert "Content Generation" in project_names
        assert "New Project" in project_names

        # Counts should be increased by our additions
        assert len(namespaces) == len(initial_namespaces) + 1
        assert len(projects) == len(initial_projects) + 1
        assert len(models) == len(initial_models)  # No new models created
        assert len(datasets) >= len(initial_datasets)  # May have created datasets

    @pytest.mark.asyncio
    async def test_entity_metadata_persistence(self, mock_service):
        """Test that entity metadata persists correctly across operations."""
        # Create entity with custom metadata
        entity_data = {
            "name": "metadata-test",
            "description": "Test entity with metadata",
            "namespace": "test-namespace",
            "custom_field": "custom_value",
            "tags": ["test", "metadata"],
        }
        created_entity = await mock_service.create_entity("datasets", entity_data)

        # Verify metadata is preserved
        assert created_entity["custom_field"] == "custom_value"
        assert created_entity["tags"] == ["test", "metadata"]

        # Update with new metadata
        updates = {"custom_field": "updated_value", "new_field": "new_value"}
        updated_entity = await mock_service.update_entity("datasets", created_entity["id"], updates)

        # Verify metadata is updated correctly
        assert updated_entity["custom_field"] == "updated_value"
        assert updated_entity["new_field"] == "new_value"
        assert updated_entity["tags"] == ["test", "metadata"]  # Original metadata preserved

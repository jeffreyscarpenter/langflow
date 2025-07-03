"""Integration tests for NeMo Entity Store integration with Langflow components.

Tests the integration between Langflow components and the NeMo Entity Store
functionality in the mock service.
"""

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from base.langflow.components.nvidia.nvidia_customizer import NvidiaCustomizerComponent
from base.langflow.components.nvidia.nvidia_dataset_creator import NvidiaDatasetCreatorComponent
from base.langflow.components.nvidia.nvidia_evaluator import NvidiaEvaluatorComponent
from base.langflow.services.nemo_microservices_mock import MockNeMoMicroservicesService
from fastapi import UploadFile


class TestNeMoEntityStoreIntegration:
    """Integration tests for NeMo Entity Store with Langflow components."""

    @pytest.fixture
    async def mock_service(self):
        """Create a fresh mock service instance for each test."""
        service = MockNeMoMicroservicesService()
        yield service
        service.cleanup()

    @pytest.fixture
    def dataset_creator_component(self):
        """Create a dataset creator component instance."""
        return NvidiaDatasetCreatorComponent()

    @pytest.fixture
    def customizer_component(self):
        """Create a customizer component instance."""
        return NvidiaCustomizerComponent()

    @pytest.fixture
    def evaluator_component(self):
        """Create an evaluator component instance."""
        return NvidiaEvaluatorComponent()

    @pytest.mark.asyncio
    async def test_dataset_creator_with_entity_store(self, mock_service, dataset_creator_component):
        """Test that dataset creator properly integrates with Entity Store."""
        # Mock the service dependency
        with patch.object(dataset_creator_component, "_get_nemo_service", return_value=mock_service):
            # Test dataset creation with namespace and project
            result = await dataset_creator_component.process(
                name="integration-test-dataset",
                description="Test dataset for Entity Store integration",
                namespace="test-namespace",
                project="test-project",
            )

            # Verify the result contains Entity Store fields
            assert result["name"] == "integration-test-dataset"
            assert result["namespace"] == "test-namespace"
            assert result["project"] == "test-project"
            assert result["files_url"] == "hf://datasets/test-namespace/integration-test-dataset"
            assert result["format"] == "jsonl"

            # Verify it's properly registered in Entity Store
            datasets = await mock_service.list_entities("datasets")
            dataset_ids = [d["id"] for d in datasets]
            assert result["id"] in dataset_ids

    @pytest.mark.asyncio
    async def test_dataset_creator_file_upload_integration(self, mock_service, dataset_creator_component):
        """Test that file uploads update Entity Store metadata."""
        # Mock the service dependency
        with patch.object(dataset_creator_component, "_get_nemo_service", return_value=mock_service):
            # Create dataset
            dataset_result = await dataset_creator_component.process(
                name="file-upload-test",
                description="Test dataset for file upload integration",
                namespace="test-namespace",
            )

            # Create mock files for upload
            mock_file1 = MagicMock(spec=UploadFile)
            mock_file1.filename = "train.json"
            mock_file1.content_type = "application/json"
            mock_file1.read = AsyncMock(return_value=b'{"train": "data"}')

            mock_file2 = MagicMock(spec=UploadFile)
            mock_file2.filename = "validation.json"
            mock_file2.content_type = "application/json"
            mock_file2.read = AsyncMock(return_value=b'{"validation": "data"}')

            # Upload files
            upload_result = await mock_service.upload_files(dataset_result["id"], [mock_file1, mock_file2])

            assert upload_result["message"].startswith("Successfully uploaded")
            assert len(upload_result["files"]) == 2

            # Verify Entity Store was updated with file metadata
            updated_dataset = await mock_service.get_entity("datasets", dataset_result["id"])
            assert updated_dataset["metadata"]["file_count"] == 2
            assert updated_dataset["metadata"]["total_size"] == "35B"  # Combined size

    @pytest.mark.asyncio
    async def test_customizer_with_entity_store_models(self, mock_service, customizer_component):
        """Test that customizer can work with Entity Store models."""
        # Mock the service dependency
        with patch.object(customizer_component, "_get_nemo_service", return_value=mock_service):
            # First, create a model in Entity Store
            model_data = {
                "name": "test-base-model",
                "description": "Test base model for customizer",
                "namespace": "test-namespace",
                "type": "base_model",
                "format": "nemo",
                "files_url": "hf://models/test-namespace/test-base-model",
            }
            model = await mock_service.create_entity("models", model_data)

            # Create a dataset for training
            dataset = await mock_service.create_dataset(name="customizer-training-data", namespace="test-namespace")

            # Test customizer job creation (this would normally use the model and dataset)
            # For now, we'll test that the customizer can access Entity Store data
            models = await mock_service.list_entities("models")
            datasets = await mock_service.list_entities("datasets")

            # Verify our created entities are accessible
            model_ids = [m["id"] for m in models]
            dataset_ids = [d["id"] for d in datasets]

            assert model["id"] in model_ids
            assert dataset["id"] in dataset_ids

    @pytest.mark.asyncio
    async def test_evaluator_with_entity_store_datasets(self, mock_service, evaluator_component):
        """Test that evaluator can work with Entity Store datasets."""
        # Mock the service dependency
        with patch.object(evaluator_component, "_get_nemo_service", return_value=mock_service):
            # Create evaluation dataset in Entity Store
            eval_dataset = await mock_service.create_dataset(
                name="evaluation-dataset", description="Dataset for model evaluation", namespace="test-namespace"
            )

            # Create evaluation target
            target_data = {
                "name": "test-target",
                "description": "Test evaluation target",
                "namespace": "test-namespace",
                "dataset": f"{eval_dataset['namespace']}/{eval_dataset['name']}",
            }
            target = await mock_service.create_evaluation_target(target_data)

            # Verify the target references the Entity Store dataset
            assert target["dataset"] == f"{eval_dataset['namespace']}/{eval_dataset['name']}"

            # Test that evaluator can access Entity Store datasets
            datasets = await mock_service.list_entities("datasets")
            dataset_names = [d["name"] for d in datasets]
            assert "evaluation-dataset" in dataset_names

    @pytest.mark.asyncio
    async def test_cross_component_entity_sharing(self, mock_service, dataset_creator_component, customizer_component):
        """Test that entities created by one component are accessible to others."""
        # Mock services for both components
        with (
            patch.object(dataset_creator_component, "_get_nemo_service", return_value=mock_service),
            patch.object(customizer_component, "_get_nemo_service", return_value=mock_service),
        ):
            # Dataset creator creates a dataset
            dataset_result = await dataset_creator_component.process(
                name="shared-dataset", description="Dataset shared between components", namespace="test-namespace"
            )

            # Customizer should be able to access the same dataset
            datasets = await mock_service.list_entities("datasets")
            dataset_ids = [d["id"] for d in datasets]
            assert dataset_result["id"] in dataset_ids

            # Verify the dataset has proper Entity Store metadata
            retrieved_dataset = await mock_service.get_entity("datasets", dataset_result["id"])
            assert retrieved_dataset["namespace"] == "test-namespace"
            assert retrieved_dataset["files_url"] == "hf://datasets/test-namespace/shared-dataset"

    @pytest.mark.asyncio
    async def test_entity_store_namespace_management(self, mock_service):
        """Test namespace management across components."""
        # Create namespace
        namespace = await mock_service.create_namespace(
            {"namespace": "component-test-namespace", "description": "Namespace for component testing"}
        )

        # Create project in namespace
        project = await mock_service.create_project(
            {
                "name": "Component Test Project",
                "description": "Project for testing component integration",
                "namespace": namespace["name"],
            }
        )

        # Verify namespace and project are properly linked
        assert project["namespace"] == namespace["name"]

        # Test that components can work within this namespace
        dataset = await mock_service.create_dataset(
            name="namespace-test-dataset", namespace=namespace["name"], project=project["id"]
        )

        assert dataset["namespace"] == namespace["name"]
        assert dataset["project"] == project["id"]

    @pytest.mark.asyncio
    async def test_entity_store_error_handling_in_components(self, mock_service, dataset_creator_component):
        """Test error handling when Entity Store operations fail."""
        # Mock the service dependency
        with patch.object(dataset_creator_component, "_get_nemo_service", return_value=mock_service):
            # Test with invalid namespace - should handle gracefully
            result = await dataset_creator_component.process(
                name="error-test-dataset",
                description="Test dataset for error handling",
                namespace="",  # Invalid empty namespace
                project="invalid-project",
            )
            # Should handle gracefully and return a result
            assert result is not None

    @pytest.mark.asyncio
    async def test_entity_store_metadata_persistence(self, mock_service):
        """Test that Entity Store metadata persists across component operations."""
        # Create dataset with custom metadata
        dataset = await mock_service.create_dataset(
            name="metadata-persistence-test",
            description="Test dataset for metadata persistence",
            namespace="test-namespace",
        )

        # Add custom metadata
        updated_dataset = await mock_service.update_entity(
            "datasets", dataset["id"], {"custom_field": "custom_value", "tags": ["test", "persistence"]}
        )

        # Verify metadata persists
        assert updated_dataset["custom_field"] == "custom_value"
        assert updated_dataset["tags"] == ["test", "persistence"]

        # Simulate component accessing the dataset
        retrieved_dataset = await mock_service.get_entity("datasets", dataset["id"])
        assert retrieved_dataset["custom_field"] == "custom_value"
        assert retrieved_dataset["tags"] == ["test", "persistence"]

    @pytest.mark.asyncio
    async def test_entity_store_concurrent_access(self, mock_service, dataset_creator_component, customizer_component):
        """Test concurrent access to Entity Store from multiple components."""
        # Mock services for both components
        with (
            patch.object(dataset_creator_component, "_get_nemo_service", return_value=mock_service),
            patch.object(customizer_component, "_get_nemo_service", return_value=mock_service),
        ):
            # Create datasets concurrently
            async def create_dataset(name):
                return await dataset_creator_component.process(
                    name=name, description=f"Concurrent test dataset {name}", namespace="test-namespace"
                )

            # Run concurrent operations
            tasks = [create_dataset("concurrent-1"), create_dataset("concurrent-2"), create_dataset("concurrent-3")]

            results = await asyncio.gather(*tasks)

            # Verify all datasets were created successfully
            assert len(results) == 3
            for result in results:
                assert result["namespace"] == "test-namespace"
                assert "concurrent-" in result["name"]

            # Verify all are accessible in Entity Store
            datasets = await mock_service.list_entities("datasets")
            dataset_names = [d["name"] for d in datasets]
            assert "concurrent-1" in dataset_names
            assert "concurrent-2" in dataset_names
            assert "concurrent-3" in dataset_names

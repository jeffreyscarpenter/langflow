"""Unit tests for the NeMo Evaluator component."""

from unittest.mock import Mock, patch

import pytest
from langflow.components.nvidia.nvidia_evaluator import NvidiaEvaluatorComponent
from langflow.schema import Data


class TestNvidiaEvaluatorErrorHandling:
    """Test error handling scenarios for the NeMo Evaluator component."""

    @pytest.fixture
    def evaluator(self):
        """Create a test evaluator instance."""
        evaluator = NvidiaEvaluatorComponent()
        evaluator.auth_token = "token"  # noqa: S105
        evaluator.base_url = "https://test.api.com/nvidia/nemo"
        evaluator.namespace = "test_namespace"
        evaluator.inference_model_url = "https://test.nim.com"
        evaluator.log = Mock()  # Mock logging
        return evaluator

    @pytest.mark.asyncio
    async def test_missing_dataset_error(self, evaluator):
        """Test that appropriate error is raised when no dataset is provided."""
        # Set up evaluator with no dataset connection and no existing dataset
        evaluator.dataset = None
        evaluator.existing_dataset = None
        evaluator.evaluation_type = "Similarity Metrics"
        evaluator.llm_name = "test_model"

        # Test the _prepare_custom_evaluation_data method
        with patch.object(evaluator, "_create_evaluation_target"):
            with pytest.raises(ValueError, match="dataset connection|existing dataset") as exc_info:
                await evaluator._prepare_custom_evaluation_data("https://test.api.com")

            error_msg = str(exc_info.value)
            assert "dataset connection" in error_msg.lower() or "existing dataset" in error_msg.lower()

    @pytest.mark.asyncio
    async def test_dataset_connection_namespace_handling(self, evaluator):
        """Test that dataset connection namespace is used when available."""
        # Create a mock dataset with different namespace
        dataset_data = {"name": "test_dataset", "namespace": "different_namespace"}
        dataset_input = Data(data=dataset_data)

        evaluator.dataset = dataset_input
        evaluator.existing_dataset = None
        evaluator.evaluation_type = "Similarity Metrics"
        evaluator.llm_name = "test_model"

        # Mock the _create_evaluation_target method to capture the namespace
        captured_namespace = None

        async def mock_create_target(_, __, namespace):
            nonlocal captured_namespace
            captured_namespace = namespace
            return {"type": "model", "namespace": namespace, "model": {}}

        with patch.object(evaluator, "_create_evaluation_target", side_effect=mock_create_target):
            await evaluator._prepare_custom_evaluation_data("https://test.api.com")
            assert captured_namespace == "different_namespace"

    @pytest.mark.asyncio
    async def test_existing_dataset_namespace_handling(self, evaluator):
        """Test that component namespace is used for existing datasets."""
        evaluator.dataset = None
        evaluator.existing_dataset = "test_existing_dataset"
        evaluator.evaluation_type = "Similarity Metrics"
        evaluator.llm_name = "test_model"

        # Mock the _create_evaluation_target method to capture the namespace
        captured_namespace = None

        async def mock_create_target(_, __, namespace):
            nonlocal captured_namespace
            captured_namespace = namespace
            return {"type": "model", "namespace": namespace, "model": {}}

        with patch.object(evaluator, "_create_evaluation_target", side_effect=mock_create_target):
            await evaluator._prepare_custom_evaluation_data("https://test.api.com")
            assert captured_namespace == "test_namespace"

    @pytest.mark.asyncio
    async def test_invalid_dataset_data_error(self, evaluator):
        """Test error handling for invalid dataset Data objects."""
        # Test with non-Data object
        evaluator.dataset = "invalid_data"
        evaluator.existing_dataset = None
        evaluator.evaluation_type = "Similarity Metrics"
        evaluator.llm_name = "test_model"

        with patch.object(evaluator, "_create_evaluation_target"):
            with pytest.raises(ValueError, match="Data object") as exc_info:
                await evaluator._prepare_custom_evaluation_data("https://test.api.com")

            error_msg = str(exc_info.value)
            assert "Data object" in error_msg

    @pytest.mark.asyncio
    async def test_missing_dataset_fields_error(self, evaluator):
        """Test error handling for dataset Data objects missing required fields."""
        # Test with dataset missing required fields
        dataset_data = {"name": "test_dataset"}  # Missing namespace
        dataset_input = Data(data=dataset_data)

        evaluator.dataset = dataset_input
        evaluator.existing_dataset = None
        evaluator.evaluation_type = "Similarity Metrics"
        evaluator.llm_name = "test_model"

        with patch.object(evaluator, "_create_evaluation_target"):
            with pytest.raises(ValueError, match="namespace") as exc_info:
                await evaluator._prepare_custom_evaluation_data("https://test.api.com")

            error_msg = str(exc_info.value)
            assert "namespace" in error_msg.lower()

    @pytest.mark.asyncio
    async def test_dataset_connection_priority_over_existing_dataset(self, evaluator):
        """Test that dataset connection takes priority over existing dataset."""
        # Create a mock dataset
        dataset_data = {"name": "connected_dataset", "namespace": "connected_namespace"}
        dataset_input = Data(data=dataset_data)

        evaluator.dataset = dataset_input
        evaluator.existing_dataset = "existing_dataset"  # This should be ignored
        evaluator.evaluation_type = "Similarity Metrics"
        evaluator.llm_name = "test_model"

        # Mock the _create_evaluation_target method to capture the namespace
        captured_namespace = None

        async def mock_create_target(_, __, namespace):
            nonlocal captured_namespace
            captured_namespace = namespace
            return {"type": "model", "namespace": namespace, "model": {}}

        with patch.object(evaluator, "_create_evaluation_target", side_effect=mock_create_target):
            await evaluator._prepare_custom_evaluation_data("https://test.api.com")
            # Should use the connected dataset's namespace, not the component namespace
            assert captured_namespace == "connected_namespace"

    @pytest.mark.asyncio
    async def test_lm_evaluation_missing_hf_token(self, evaluator):
        """Test that LM evaluation raises ValueError for missing HuggingFace token."""
        evaluator.evaluation_type = "LM Evaluation Harness"
        evaluator.llm_name = "test_model"

        with patch.object(evaluator, "_create_evaluation_target"):
            with pytest.raises(ValueError, match="Missing hf token") as exc_info:
                await evaluator._prepare_lm_evaluation_data("https://test.api.com")

            error_msg = str(exc_info.value)
            assert "Missing hf token" in error_msg

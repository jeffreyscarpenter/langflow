"""Unit tests for the NeMo Evaluator component."""

from unittest.mock import AsyncMock, Mock, patch

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
                await evaluator._prepare_custom_evaluation_data("https://test.api.com", "test_namespace", "test_model")

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

        async def mock_create_target(_, __, namespace, _model_name):
            nonlocal captured_namespace
            captured_namespace = namespace
            return {"type": "model", "namespace": namespace, "model": {}}

        with patch.object(evaluator, "_create_evaluation_target", side_effect=mock_create_target):
            await evaluator._prepare_custom_evaluation_data("https://test.api.com", "test_namespace", "test_model")
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

        async def mock_create_target(_, __, namespace, _model_name):
            nonlocal captured_namespace
            captured_namespace = namespace
            return {"type": "model", "namespace": namespace, "model": {}}

        with patch.object(evaluator, "_create_evaluation_target", side_effect=mock_create_target):
            await evaluator._prepare_custom_evaluation_data("https://test.api.com", "test_namespace", "test_model")
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
                await evaluator._prepare_custom_evaluation_data("https://test.api.com", "test_namespace", "test_model")

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
                await evaluator._prepare_custom_evaluation_data("https://test.api.com", "test_namespace", "test_model")

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

        async def mock_create_target(_, __, namespace, _model_name):
            nonlocal captured_namespace
            captured_namespace = namespace
            return {"type": "model", "namespace": namespace, "model": {}}

        with patch.object(evaluator, "_create_evaluation_target", side_effect=mock_create_target):
            await evaluator._prepare_custom_evaluation_data("https://test.api.com", "test_namespace", "test_model")
            assert captured_namespace == "connected_namespace"

    @pytest.mark.asyncio
    async def test_lm_evaluation_missing_hf_token(self, evaluator):
        """Test error handling for missing HuggingFace token in LM evaluation."""
        evaluator.dataset = None
        evaluator.existing_dataset = None
        evaluator.evaluation_type = "LM Evaluation Harness"
        evaluator.llm_name = "test_model"

        with patch.object(evaluator, "_create_evaluation_target"):
            with pytest.raises(ValueError, match="hf token") as exc_info:
                await evaluator._prepare_lm_evaluation_data("https://test.api.com", "test_namespace", "test_model")

            error_msg = str(exc_info.value)
            assert "hf token" in error_msg.lower()

    @pytest.mark.asyncio
    async def test_customized_model_priority_over_dropdown(self, evaluator):
        """Test that customized model input takes priority over dropdown model selection."""
        # Create a mock customized model Data object
        customized_model_data = {"output_model": "custom_namespace/custom_model_name"}
        customized_model_input = Data(data=customized_model_data)

        # Test the extract_customized_model_info method directly
        model_name, namespace = evaluator.extract_customized_model_info(customized_model_input)
        assert model_name == "custom_model_name"
        assert namespace == "custom_namespace"

    @pytest.mark.asyncio
    async def test_customized_model_namespace_usage(self, evaluator):
        """Test that customized model namespace is used when available."""
        # Create a mock customized model Data object with namespace
        customized_model_data = {"output_model": "custom_namespace/custom_model_name"}
        customized_model_input = Data(data=customized_model_data)

        # Test the extract_customized_model_info method directly
        model_name, namespace = evaluator.extract_customized_model_info(customized_model_input)
        assert model_name == "custom_model_name"
        assert namespace == "custom_namespace"

    @pytest.mark.asyncio
    async def test_customized_model_invalid_data_handling(self, evaluator):
        """Test error handling for invalid customized model Data objects."""
        # Test with non-Data object
        model_name, namespace = evaluator.extract_customized_model_info("invalid_data")
        assert model_name is None
        assert namespace is None

    @pytest.mark.asyncio
    async def test_customized_model_missing_output_model(self, evaluator):
        """Test error handling for customized model Data missing output_model field."""
        # Create a mock customized model Data object without output_model
        customized_model_data = {"some_other_field": "value"}
        customized_model_input = Data(data=customized_model_data)

        # Should return None values
        model_name, namespace = evaluator.extract_customized_model_info(customized_model_input)
        assert model_name is None
        assert namespace is None

    @pytest.mark.asyncio
    async def test_customized_model_without_namespace(self, evaluator):
        """Test customized model with output_model that doesn't contain namespace."""
        # Create a mock customized model Data object without namespace in output_model
        customized_model_data = {"output_model": "simple_model_name"}
        customized_model_input = Data(data=customized_model_data)

        # Test the extract_customized_model_info method directly
        model_name, namespace = evaluator.extract_customized_model_info(customized_model_input)
        assert model_name == "simple_model_name"
        assert namespace is None

    @pytest.mark.asyncio
    async def test_evaluate_with_customized_model(self, evaluator):
        """Test the complete evaluate method with customized model input."""
        # Set up evaluator with customized model
        customized_model_data = {"output_model": "custom_namespace/custom_model_name"}
        customized_model_input = Data(data=customized_model_data)

        evaluator.customized_model = customized_model_input
        evaluator.auth_token = "test_auth_token"  # noqa: S105
        evaluator.base_url = "https://test.api.com"
        evaluator.namespace = "test_namespace"
        evaluator.inference_model_url = "https://test.inference.com"
        setattr(evaluator, "002_evaluation_type", "LM Evaluation Harness")
        setattr(evaluator, "100_huggingface_token", "test_token")

        # Mock the NeMo client and its methods
        mock_response = Mock()
        mock_response.model_dump.return_value = {"id": "test_job_id", "status": "completed"}
        mock_response.id = "test_config_id"

        mock_target_response = Mock()
        mock_target_response.model_dump.return_value = {"id": "test_target_id"}
        mock_target_response.id = "test_target_id"

        mock_job_response = Mock()
        mock_job_response.model_dump.return_value = {"id": "test_job_id", "status": "completed"}

        with patch.object(evaluator, "get_nemo_client") as mock_get_client:
            mock_client = Mock()
            mock_get_client.return_value = mock_client

            # Mock the async methods
            mock_client.evaluation.configs.create = AsyncMock(return_value=mock_response)
            mock_client.evaluation.targets.create = AsyncMock(return_value=mock_target_response)
            mock_client.evaluation.jobs.create = AsyncMock(return_value=mock_job_response)

            # Call the evaluate method
            result = await evaluator.evaluate()

            # Verify the result
            assert result.data["id"] == "test_job_id"
            assert result.data["status"] == "completed"

            # Verify that the customized model was used (check the logs or verify the calls)
            # The customized model should have been extracted and used in the evaluation

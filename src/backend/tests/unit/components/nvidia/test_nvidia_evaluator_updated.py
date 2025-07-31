"""Test script for updated NeMo Evaluator component."""

from unittest.mock import AsyncMock, Mock, patch

import pytest
from langflow.components.nvidia.nvidia_evaluator import NvidiaEvaluatorComponent


@pytest.mark.asyncio
async def test_evaluator_component_initialization():
    """Test that the evaluator component initializes correctly."""
    component = NvidiaEvaluatorComponent()

    # Check that required attributes are initialized
    assert hasattr(component, "auth_token")
    assert hasattr(component, "base_url")
    assert hasattr(component, "namespace")

    # Check that inputs are properly defined
    input_names = [input_field.name for input_field in component.inputs]
    assert "target" in input_names
    assert "config" in input_names
    assert "auth_token" in input_names
    assert "base_url" in input_names


@pytest.mark.asyncio
async def test_fetch_available_evaluation_targets():
    """Test fetching evaluation targets."""
    component = NvidiaEvaluatorComponent()
    component.auth_token = "test-token"  # noqa: S105
    component.base_url = "https://test.api.com/nvidia/nemo"

    # Mock the NeMo client response
    mock_target = Mock()
    mock_target.name = "test-target"
    mock_target.id = "target-123"
    mock_target.type = "model"
    mock_target.created = "2024-01-01T00:00:00Z"
    mock_target.updated = "2024-01-02T00:00:00Z"

    mock_response = Mock()
    mock_response.data = [mock_target]

    with patch.object(component, "get_nemo_client") as mock_client:
        mock_nemo_client = AsyncMock()
        mock_nemo_client.evaluation.targets.list.return_value = mock_response
        mock_client.return_value = mock_nemo_client

        targets, metadata = await component.fetch_available_evaluation_targets()

        assert targets == ["test-target"]
        assert len(metadata) == 1
        assert metadata[0]["id"] == "123"
        assert metadata[0]["type"] == "MODEL"
        assert metadata[0]["icon"] == "NVIDIA"


@pytest.mark.asyncio
async def test_fetch_available_evaluation_configs():
    """Test fetching evaluation configurations."""
    component = NvidiaEvaluatorComponent()
    component.auth_token = "test-token"  # noqa: S105
    component.base_url = "https://test.api.com/nvidia/nemo"

    # Mock the NeMo client response
    mock_config = Mock()
    mock_config.name = "test-config"
    mock_config.type = "lm_eval_harness"

    # Create a mock Pydantic-like object for params
    mock_params = Mock()
    mock_params.model_dump.return_value = {"tasks": {"gsm8k": {"params": {"num_fewshot": 5}}}, "hf_token": "test-token"}
    mock_config.params = mock_params
    mock_config.created = "2024-01-01T00:00:00Z"
    mock_config.updated = "2024-01-02T00:00:00Z"

    mock_response = Mock()
    mock_response.data = [mock_config]

    with patch.object(component, "get_nemo_client") as mock_client:
        mock_nemo_client = AsyncMock()
        mock_nemo_client.evaluation.configs.list.return_value = mock_response
        mock_client.return_value = mock_nemo_client

        configs, metadata = await component.fetch_available_evaluation_configs()

        assert configs == ["test-config"]
        assert len(metadata) == 1
        assert metadata[0]["type"] == "Lm Eval Harness"
        assert metadata[0]["task"] == "gsm8k"
        assert metadata[0]["icon"] == "Settings"


@pytest.mark.asyncio
async def test_create_new_evaluation_config():
    """Test creating a new evaluation configuration."""
    component = NvidiaEvaluatorComponent()
    component.auth_token = "test-token"  # noqa: S105
    component.namespace = "default"

    # Mock config data
    config_data = {
        "02_config_name": "test-config",
        "01_evaluation_type": "LM Evaluation Harness",
        "03_task_name": "gsm8k",
        "04_hf_token": "test-hf-token",
        "05_few_shot_examples": 5,
        "06_batch_size": 16,
        "07_bootstrap_iterations": 100000,
        "08_limit": -1,
        "09_top_p": 0.0,
        "10_top_k": 1,
        "11_temperature": 0.1,
        "12_tokens_to_generate": 1024,
    }

    # Mock the NeMo client response
    mock_response = Mock()
    mock_response.id = "config-123"

    with patch.object(component, "get_nemo_client") as mock_client:
        mock_nemo_client = AsyncMock()
        mock_nemo_client.evaluation.configs.create.return_value = mock_response
        mock_client.return_value = mock_nemo_client

        config_id = await component._create_new_evaluation_config(config_data)

        assert config_id == "config-123"
        mock_nemo_client.evaluation.configs.create.assert_called_once()


@pytest.mark.asyncio
async def test_get_target_id():
    """Test getting target ID from target name."""
    component = NvidiaEvaluatorComponent()
    component.auth_token = "test-token"  # noqa: S105

    # Mock the NeMo client response
    mock_target = Mock()
    mock_target.name = "test-target"
    mock_target.id = "target-123"

    mock_response = Mock()
    mock_response.data = [mock_target]

    with patch.object(component, "get_nemo_client") as mock_client:
        mock_nemo_client = AsyncMock()
        mock_nemo_client.evaluation.targets.list.return_value = mock_response
        mock_client.return_value = mock_nemo_client

        target_id = await component._get_target_id("test-target")

        assert target_id == "target-123"


@pytest.mark.asyncio
async def test_get_config_id():
    """Test getting config ID from config name."""
    component = NvidiaEvaluatorComponent()
    component.auth_token = "test-token"  # noqa: S105

    # Mock the NeMo client response
    mock_config = Mock()
    mock_config.name = "test-config"
    mock_config.id = "config-123"

    mock_response = Mock()
    mock_response.data = [mock_config]

    with patch.object(component, "get_nemo_client") as mock_client:
        mock_nemo_client = AsyncMock()
        mock_nemo_client.evaluation.configs.list.return_value = mock_response
        mock_client.return_value = mock_nemo_client

        config_id = await component._get_config_id("test-config")

        assert config_id == "config-123"


@pytest.mark.asyncio
async def test_validate_target_config_compatibility():
    """Test target-config compatibility validation."""
    component = NvidiaEvaluatorComponent()
    component.auth_token = "test-token"  # noqa: S105

    # Mock the NeMo client responses
    mock_target_response = Mock()
    mock_target_response.type = "model"

    mock_config_response = Mock()
    mock_config_response.type = "lm_eval_harness"

    with patch.object(component, "get_nemo_client") as mock_client:
        mock_nemo_client = AsyncMock()
        mock_nemo_client.evaluation.targets.retrieve.return_value = mock_target_response
        mock_nemo_client.evaluation.configs.retrieve.return_value = mock_config_response
        mock_client.return_value = mock_nemo_client

        # Should not raise an exception for compatible types
        await component._validate_target_config_compatibility("target-123", "config-123")


def test_build_target_metadata():
    """Test building target metadata."""
    component = NvidiaEvaluatorComponent()

    metadata = component._build_target_metadata(
        target_id="target-123", target_type="model", created="2024-01-01T00:00:00Z", updated="2024-01-02T00:00:00Z"
    )

    assert metadata["id"] == "123"
    assert metadata["type"] == "MODEL"
    assert metadata["created"] == "2024-01-01T00:00:00Z"
    assert metadata["updated"] == "2024-01-02T00:00:00Z"
    assert metadata["icon"] == "NVIDIA"


def test_build_config_metadata():
    """Test building config metadata."""
    component = NvidiaEvaluatorComponent()

    # Create a mock Pydantic-like object for params
    mock_params = Mock()
    mock_params.model_dump.return_value = {"tasks": {"gsm8k": {"params": {"num_fewshot": 5}}}, "hf_token": "test-token"}

    metadata = component._build_config_metadata(
        config_type="lm_eval_harness",
        config_params=mock_params,
        created="2024-01-01T00:00:00Z",
        updated="2024-01-02T00:00:00Z",
    )

    assert metadata["type"] == "Lm Eval Harness"
    assert metadata["task"] == "gsm8k"
    assert metadata["created"] == "2024-01-01T00:00:00Z"
    assert metadata["updated"] == "2024-01-02T00:00:00Z"
    assert metadata["icon"] == "Settings"


if __name__ == "__main__":
    pytest.main([__file__])

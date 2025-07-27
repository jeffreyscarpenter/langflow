"""Test the new NeMo Customizer component with target/config pattern."""

from unittest.mock import AsyncMock, Mock, patch

import pytest
from langflow.components.nvidia.nvidia_customizer import NvidiaCustomizerComponent


@pytest.fixture
def customizer_component():
    """Create a test instance of the customizer component."""
    return NvidiaCustomizerComponent(
        auth_token="test-token",
        base_url="https://test.api.com/nvidia/nemo",
        namespace="test-namespace",
        target="test-model@v1",
        config="test-config@v1",
        fine_tuned_model_name="test-output@v1",
        existing_dataset="test-dataset",
        epochs=3,
        batch_size=8,
        learning_rate=0.0001,
    )


@pytest.mark.asyncio
async def test_component_initialization(customizer_component):
    """Test that the component initializes correctly."""
    assert customizer_component.auth_token == "test-token"
    assert customizer_component.base_url == "https://test.api.com/nvidia/nemo"
    assert customizer_component.namespace == "test-namespace"
    assert customizer_component.target == "test-model@v1"
    assert customizer_component.config == "test-config@v1"
    assert customizer_component.fine_tuned_model_name == "test-output@v1"


@pytest.mark.asyncio
async def test_fetch_available_targets(customizer_component):
    """Test fetching available targets."""
    mock_response = Mock()
    mock_target1 = Mock()
    mock_target1.name = "model1@v1"
    mock_target1.id = "target-1"
    mock_target2 = Mock()
    mock_target2.name = "model2@v2"
    mock_target2.id = "target-2"
    mock_response.data = [mock_target1, mock_target2]

    with patch.object(customizer_component, "get_nemo_client") as mock_client:
        mock_client.return_value.customization.targets.list = AsyncMock(return_value=mock_response)

        targets = await customizer_component.fetch_available_targets()

        assert targets == ["model1@v1", "model2@v2"]
        mock_client.return_value.customization.targets.list.assert_called_once()


@pytest.mark.asyncio
async def test_fetch_available_configs(customizer_component):
    """Test fetching available configs for a target."""
    mock_response = Mock()
    mock_config1 = Mock()
    mock_config1.name = "test-model@v1-config1"
    mock_config1.id = "config-1"
    mock_target1 = Mock()
    mock_target1.id = "target-123"
    mock_config1.target = mock_target1

    mock_config2 = Mock()
    mock_config2.name = "test-model@v1-config2"
    mock_config2.id = "config-2"
    mock_target2 = Mock()
    mock_target2.id = "target-123"
    mock_config2.target = mock_target2

    mock_response.data = [mock_config1, mock_config2]

    with (
        patch.object(customizer_component, "get_nemo_client") as mock_client,
        patch.object(customizer_component, "_get_target_id") as mock_get_target_id,
    ):
        mock_client.return_value.customization.configs.list = AsyncMock(return_value=mock_response)
        mock_get_target_id.return_value = "target-123"

        configs = await customizer_component.fetch_available_configs("test-model@v1")

        # Should filter configs that match the target
        assert len(configs) == 2
        assert "test-model@v1-config1" in configs
        assert "test-model@v1-config2" in configs
        mock_client.return_value.customization.configs.list.assert_called_once()


@pytest.mark.asyncio
async def test_create_new_config(customizer_component):
    """Test creating a new configuration."""
    config_data = {
        "01_config_name": "test-config@v1",
        "02_training_type": "sft",
        "03_finetuning_type": "lora",
        "04_max_seq_length": 4096,
        "05_prompt_template": "{prompt} {completion}",
        "06_training_precision": "bf16-mixed",
        "07_lora_adapter_dim": 32,
        "08_lora_alpha": 16,
        "09_lora_target_modules": "q_proj,v_proj",
    }

    mock_response = Mock()
    mock_response.id = "new-config-id"

    with patch.object(customizer_component, "get_nemo_client") as mock_client:
        mock_client.return_value.customization.configs.create = AsyncMock(return_value=mock_response)

        config_id = await customizer_component._create_new_config(config_data)

        assert config_id == "new-config-id"
        mock_client.return_value.customization.configs.create.assert_called_once()


@pytest.mark.asyncio
async def test_get_target_id(customizer_component):
    """Test getting target ID from target name."""
    mock_response = Mock()
    mock_target1 = Mock()
    mock_target1.name = "test-model@v1"
    mock_target1.id = "target-123"
    mock_target2 = Mock()
    mock_target2.name = "other-model@v1"
    mock_target2.id = "target-456"
    mock_response.data = [mock_target1, mock_target2]

    with patch.object(customizer_component, "get_nemo_client") as mock_client:
        mock_client.return_value.customization.targets.list = AsyncMock(return_value=mock_response)

        target_id = await customizer_component._get_target_id("test-model@v1")

        assert target_id == "target-123"


@pytest.mark.asyncio
async def test_extract_dataset_info(customizer_component):
    """Test extracting dataset information from Data object."""
    from langflow.schema import Data

    dataset_data = {
        "dataset_name": "test-dataset",
        "namespace": "test-namespace",
    }
    dataset_input = Data(data=dataset_data)

    dataset_name, dataset_namespace = customizer_component.extract_dataset_info(dataset_input)

    assert dataset_name == "test-dataset"
    assert dataset_namespace == "test-namespace"


@pytest.mark.asyncio
async def test_customize_with_existing_config(customizer_component):
    """Test creating a customization job with existing config."""
    # Set dataset to None so it uses existing_dataset
    customizer_component.dataset = None

    mock_job_response = Mock()
    mock_job_response.model_dump.return_value = {
        "id": "job-123",
        "status": "pending",
        "name": "customization-test-output@v1",
    }

    with patch.object(customizer_component, "get_nemo_client") as mock_client:
        mock_client.return_value.customization.jobs.create = AsyncMock(return_value=mock_job_response)

        # Mock target ID lookup
        with patch.object(customizer_component, "_get_target_id", return_value="target-123"):
            result = await customizer_component.customize()

            assert result.data["id"] == "job-123"
            mock_client.return_value.customization.jobs.create.assert_called_once()


@pytest.mark.asyncio
async def test_customize_with_new_config(customizer_component):
    """Test creating a customization job with new config from dialog."""
    # Set config to a dict (simulating dialog data)
    customizer_component.config = {
        "01_config_name": "new-config@v1",
        "02_training_type": "sft",
        "03_finetuning_type": "lora",
        "04_max_seq_length": 4096,
        "05_prompt_template": "{prompt} {completion}",
        "06_training_precision": "bf16-mixed",
        "07_lora_adapter_dim": 32,
        "08_lora_alpha": 16,
        "09_lora_target_modules": "q_proj,v_proj",
    }

    # Set dataset to None so it uses existing_dataset
    customizer_component.dataset = None

    mock_config_response = Mock()
    mock_config_response.id = "new-config-id"

    mock_job_response = Mock()
    mock_job_response.model_dump.return_value = {
        "id": "job-123",
        "status": "pending",
        "name": "customization-test-output@v1",
    }

    with patch.object(customizer_component, "get_nemo_client") as mock_client:
        mock_client.return_value.customization.configs.create = AsyncMock(return_value=mock_config_response)
        mock_client.return_value.customization.jobs.create = AsyncMock(return_value=mock_job_response)

        # Mock target ID lookup
        with patch.object(customizer_component, "_get_target_id", return_value="target-123"):
            result = await customizer_component.customize()

            assert result.data["id"] == "job-123"
            # Should create config first, then job
            mock_client.return_value.customization.configs.create.assert_called_once()
            mock_client.return_value.customization.jobs.create.assert_called_once()


@pytest.mark.asyncio
async def test_customize_missing_required_fields():
    """Test that customize raises error for missing required fields."""
    component = NvidiaCustomizerComponent()

    with pytest.raises(ValueError, match="Missing authentication token"):
        await component.customize()


@pytest.mark.asyncio
async def test_customize_missing_dataset():
    """Test that customize raises error when no dataset is provided."""
    component = NvidiaCustomizerComponent(
        auth_token="test-token",
        base_url="https://test.api.com/nvidia/nemo",
        namespace="test-namespace",
        target="test-model@v1",
        config="test-config@v1",
        fine_tuned_model_name="test-output@v1",
    )

    with pytest.raises(ValueError, match="Either dataset connection or existing dataset selection must be provided"):
        await component.customize()


@pytest.mark.asyncio
async def test_wait_for_job_completion(customizer_component):
    """Test waiting for job completion."""
    mock_response = Mock()
    mock_response.model_dump.return_value = {
        "id": "job-123",
        "status": "completed",
    }

    with patch.object(customizer_component, "get_nemo_client") as mock_client:
        mock_client.return_value.customization.jobs.get = AsyncMock(return_value=mock_response)

        result = await customizer_component.wait_for_job_completion("job-123", max_wait_time_minutes=1)

        assert result["status"] == "completed"
        mock_client.return_value.customization.jobs.get.assert_called_once()


@pytest.mark.asyncio
async def test_wait_for_job_completion_timeout(customizer_component):
    """Test job completion timeout."""
    mock_response = Mock()
    mock_response.model_dump.return_value = {
        "id": "job-123",
        "status": "running",
    }

    with patch.object(customizer_component, "get_nemo_client") as mock_client:
        mock_client.return_value.customization.jobs.get = AsyncMock(return_value=mock_response)

        with pytest.raises(TimeoutError):
            await customizer_component.wait_for_job_completion("job-123", max_wait_time_minutes=0)


@pytest.mark.asyncio
async def test_wait_for_job_completion_failed(customizer_component):
    """Test job completion with failed status."""
    mock_response = Mock()
    mock_response.model_dump.return_value = {
        "id": "job-123",
        "status": "failed",
    }

    with patch.object(customizer_component, "get_nemo_client") as mock_client:
        mock_client.return_value.customization.jobs.get = AsyncMock(return_value=mock_response)

        with pytest.raises(ValueError, match="Job job-123 failed"):
            await customizer_component.wait_for_job_completion("job-123", max_wait_time_minutes=1)


@pytest.mark.asyncio
async def test_fetch_existing_datasets(customizer_component):
    """Test fetching existing datasets."""
    mock_response = Mock()
    mock_dataset1 = Mock()
    mock_dataset1.name = "dataset1"
    mock_dataset2 = Mock()
    mock_dataset2.name = "dataset2"
    mock_response.data = [mock_dataset1, mock_dataset2]

    with patch.object(customizer_component, "get_nemo_client") as mock_client:
        mock_client.return_value.datasets.list = AsyncMock(return_value=mock_response)

        datasets = await customizer_component.fetch_existing_datasets()

        assert len(datasets) == 2
        assert "dataset1" in datasets
        assert "dataset2" in datasets
        mock_client.return_value.datasets.list.assert_called_once()

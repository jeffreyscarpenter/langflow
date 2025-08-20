from unittest.mock import AsyncMock, Mock, patch

import pytest
from langflow.base.nvidia.nemo_guardrails_base import (
    GuardrailsConfigInput,
)
from langflow.components.nvidia.nemo_guardrails_validator import (
    NVIDIANeMoGuardrailsValidator,
)


@pytest.fixture
def validator():
    """Create a test instance of the NeMo Guardrails Validator component."""
    return NVIDIANeMoGuardrailsValidator(
        base_url="https://test.api.nvidia.com/nemo",
        auth_token="test_token",  # noqa: S106
        namespace="test_namespace",
        guardrails_config="test_config",
        validation_mode="input",
    )


class TestGuardrailsConfigInput:
    """Test the GuardrailsConfigInput dataclass."""

    def test_guardrails_config_input_structure(self):
        """Test that GuardrailsConfigInput has the correct structure."""
        config_input = GuardrailsConfigInput()

        assert config_input.functionality == "create"
        assert "data" in config_input.fields
        assert "node" in config_input.fields["data"]
        assert config_input.fields["data"]["node"]["name"] == "create_guardrails_config"
        assert "template" in config_input.fields["data"]["node"]


class TestNVIDIANeMoGuardrailsValidator:
    """Test the NVIDIANeMoGuardrailsValidator component."""

    def test_init(self, validator):
        """Test component initialization."""
        assert validator.base_url == "https://test.api.nvidia.com/nemo"
        assert validator.auth_token == "test_token"  # noqa: S105
        assert validator.namespace == "test_namespace"
        assert validator.guardrails_config == "test_config"
        assert validator.validation_mode == "input"

    def test_reset_dialog_state(self, validator):
        """Test dialog state reset."""
        validator._dialog_state = "config_creation"
        validator._reset_dialog_state()
        assert validator._dialog_state == "config_selection"

    def test_get_auth_headers_with_token(self, validator):
        """Test authentication headers generation with token."""
        headers = validator.get_auth_headers()

        assert headers["accept"] == "application/json"
        assert headers["Content-Type"] == "application/json"
        assert headers["Authorization"] == "Bearer test_token"

    def test_get_auth_headers_without_token(self, validator):
        """Test authentication headers generation without token."""
        validator.auth_token = ""
        headers = validator.get_auth_headers()

        assert headers["accept"] == "application/json"
        assert headers["Content-Type"] == "application/json"
        assert "Authorization" not in headers

    def test_get_nemo_client(self, validator):
        """Test NeMo client creation."""
        client = validator.get_nemo_client()
        assert client is not None

    @pytest.mark.asyncio
    async def test_fetch_guardrails_configs_success(self, validator):
        """Test successful config fetching."""
        # Create mock configs
        mock_config1 = Mock()
        mock_config1.name = "config1"
        mock_config1.description = "First guardrails configuration"
        mock_config1.created = "2024-01-01T00:00:00Z"
        mock_config1.updated = "2024-01-02T00:00:00Z"

        mock_config2 = Mock()
        mock_config2.name = "config2"
        mock_config2.description = "Second guardrails configuration"
        mock_config2.created = "2024-01-03T00:00:00Z"
        mock_config2.updated = None

        # Create mock response
        mock_response = Mock()
        mock_response.data = [mock_config1, mock_config2]

        with patch.object(validator, "get_nemo_client") as mock_get_client:
            mock_client = AsyncMock()
            mock_client.guardrail.configs.list.return_value = mock_response
            mock_get_client.return_value = mock_client

            configs, metadata = await validator.fetch_guardrails_configs()

            assert configs == ["config1", "config2"]
            assert len(metadata) == 2
            assert metadata[0]["icon"] == "Settings"
            assert metadata[0]["description"] == "First guardrails configuration"
            assert metadata[0]["created"] == "2024-01-01T00:00:00Z"
            assert metadata[0]["updated"] == "2024-01-02T00:00:00Z"
            assert metadata[1]["description"] == "Second guardrails configuration"
            assert metadata[1]["created"] == "2024-01-03T00:00:00Z"
            assert "updated" not in metadata[1]

    @pytest.mark.asyncio
    async def test_fetch_guardrails_configs_empty(self, validator):
        """Test config fetching with empty response."""
        mock_response = Mock()
        mock_response.data = []

        with patch.object(validator, "get_nemo_client") as mock_get_client:
            mock_client = AsyncMock()
            mock_client.guardrail.configs.list.return_value = mock_response
            mock_get_client.return_value = mock_client

            configs, metadata = await validator.fetch_guardrails_configs()

            assert configs == []
            assert metadata == []

    @pytest.mark.asyncio
    async def test_fetch_guardrails_configs_error(self, validator):
        """Test config fetching with error."""
        with patch.object(validator, "get_nemo_client") as mock_get_client:
            mock_client = AsyncMock()
            mock_client.guardrail.configs.list.side_effect = Exception("API Error")
            mock_get_client.return_value = mock_client

            configs, metadata = await validator.fetch_guardrails_configs()

            assert configs == []
            assert metadata == []

    @pytest.mark.asyncio
    async def test_create_guardrails_config_success(self, validator):
        """Test successful config creation."""
        config_data = {
            "01_config_name": "test_config",
            "02_config_description": "Test description",
            "03_rail_types": ["content_safety_input"],
            "04_content_safety_prompt": "Test prompt",
        }

        # Create mock response
        mock_response = Mock()
        mock_response.name = "config_123"

        with patch.object(validator, "get_nemo_client") as mock_get_client:
            mock_client = AsyncMock()
            mock_client.guardrail.configs.create.return_value = mock_response
            mock_get_client.return_value = mock_client

            config_id = await validator.create_guardrails_config(config_data)

            assert config_id == "config_123"

    @pytest.mark.asyncio
    async def test_create_guardrails_config_missing_name(self, validator):
        """Test config creation with missing name."""
        config_data = {"03_rail_types": ["content_safety_input"]}

        with pytest.raises(ValueError, match="Config name is required"):
            await validator.create_guardrails_config(config_data)

    def test_build_guardrails_params_content_safety(self, validator):
        """Test parameter building for content safety rails."""
        config_data = {"04_content_safety_prompt": "Custom safety prompt"}
        rail_types = ["content_safety_input", "content_safety_output"]

        params = validator._build_guardrails_params(config_data, rail_types)

        assert "content safety check input" in params["rails"]["input"]["flows"]
        assert "content safety check output" in params["rails"]["output"]["flows"]
        assert len(params["prompts"]) == 2
        assert params["prompts"][0]["content"] == "Custom safety prompt"

    def test_build_guardrails_params_topic_control(self, validator):
        """Test parameter building for topic control rails."""
        config_data = {"05_topic_control_prompt": "Custom topic prompt"}
        rail_types = ["topic_control"]

        params = validator._build_guardrails_params(config_data, rail_types)

        assert "topic safety check input" in params["rails"]["input"]["flows"]
        assert len(params["prompts"]) == 1
        assert params["prompts"][0]["content"] == "Custom topic prompt"

    def test_build_guardrails_params_self_check(self, validator):
        """Test parameter building for self check rails."""
        config_data = {"06_self_check_prompt": "Custom self check prompt"}
        rail_types = ["self_check_input", "self_check_output"]

        params = validator._build_guardrails_params(config_data, rail_types)

        assert "self check input" in params["rails"]["input"]["flows"]
        assert "self check output" in params["rails"]["output"]["flows"]
        assert len(params["prompts"]) == 2
        assert params["prompts"][0]["content"] == "Custom self check prompt"

    def test_build_guardrails_params_jailbreak_detection(self, validator):
        """Test parameter building for jailbreak detection rails."""
        config_data = {}
        rail_types = ["jailbreak_detection"]

        params = validator._build_guardrails_params(config_data, rail_types)

        assert "jailbreak detection" in params["rails"]["input"]["flows"]
        assert len(params["prompts"]) == 0

    @pytest.mark.asyncio
    async def test_update_build_config_guardrails_config(self, validator):
        """Test build config update for guardrails config."""
        build_config = {"guardrails_config": {"options": [], "options_metadata": []}}

        with patch.object(validator, "fetch_guardrails_configs", new_callable=AsyncMock) as mock_fetch:
            mock_fetch.return_value = (["config1", "config2"], [{"icon": "Settings"}, {"icon": "Settings"}])

            await validator.update_build_config(build_config, None, "guardrails_config")

            assert build_config["guardrails_config"]["options"] == ["config1", "config2"]
            assert len(build_config["guardrails_config"]["options_metadata"]) == 2

    @pytest.mark.asyncio
    async def test_update_build_config_config_creation(self, validator):
        """Test build config update for config creation."""
        config_data = {
            "01_config_name": "new_config",
            "02_config_description": "Test description",
            "03_rail_types": ["content_safety_input"],
            "04_content_safety_prompt": "Test prompt",
        }

        build_config = {"guardrails_config": {"options": [], "options_metadata": []}}

        with patch.object(validator, "create_guardrails_config", new_callable=AsyncMock) as mock_create:
            mock_create.return_value = "new_config"

            with patch.object(validator, "fetch_guardrails_configs", new_callable=AsyncMock) as mock_fetch:
                mock_fetch.return_value = (["new_config"], [{"icon": "Settings"}])

                result = await validator.update_build_config(build_config, config_data, "guardrails_config")

                assert result == "new_config"
                assert build_config["guardrails_config"]["value"] == "new_config"

    @pytest.mark.asyncio
    async def test_process_input_validation_success(self, validator):
        """Test successful input validation."""
        validator.input_value = "Test input"
        validator.validation_mode = "input"

        # Mock validation check response
        mock_check_response = Mock()
        mock_check_response.status = "allowed"

        with patch.object(validator, "get_nemo_client") as mock_get_client:
            mock_client = AsyncMock()
            mock_client.guardrail.checks.create.return_value = mock_check_response
            mock_get_client.return_value = mock_client

            result = await validator.process()

            assert "validated_output" in result
            assert result["validated_output"].text == "Test input"
            assert validator.status == "Input validated successfully"

    @pytest.mark.asyncio
    async def test_process_output_validation_success(self, validator):
        """Test successful output validation."""
        validator.input_value = "Test output"
        validator.validation_mode = "output"

        # Mock validation check response
        mock_check_response = Mock()
        mock_check_response.status = "allowed"

        with patch.object(validator, "get_nemo_client") as mock_get_client:
            mock_client = AsyncMock()
            mock_client.guardrail.checks.create.return_value = mock_check_response
            mock_get_client.return_value = mock_client

            result = await validator.process()

            assert "validated_output" in result
            assert result["validated_output"].text == "Test output"
            assert validator.status == "Output validated successfully"

    @pytest.mark.asyncio
    async def test_process_validation_blocked(self, validator):
        """Test validation when input is blocked."""
        validator.input_value = "Blocked input"
        validator.validation_mode = "input"

        # Mock validation check response
        mock_check_response = Mock()
        mock_check_response.status = "blocked"

        with patch.object(validator, "get_nemo_client") as mock_get_client:
            mock_client = AsyncMock()
            mock_client.guardrail.checks.create.return_value = mock_check_response
            mock_get_client.return_value = mock_client

            result = await validator.process()

            assert "validation_error" in result
            assert result["validation_error"].text == "I cannot process that input."
            assert validator.status == "Input blocked by guardrails"

    @pytest.mark.asyncio
    async def test_process_empty_input(self, validator):
        """Test processing with empty input."""
        validator.input_value = ""
        validator.validation_mode = "input"

        with pytest.raises(ValueError, match="The message you want to validate is empty"):
            await validator.process()

    @pytest.mark.asyncio
    async def test_process_with_system_message(self, validator):
        """Test processing with system message."""
        validator.input_value = "Test input"
        validator.system_message = "System: Be helpful"
        validator.validation_mode = "input"

        # Mock validation check response
        mock_check_response = Mock()
        mock_check_response.status = "allowed"

        with patch.object(validator, "get_nemo_client") as mock_get_client:
            mock_client = AsyncMock()
            mock_client.guardrail.checks.create.return_value = mock_check_response
            mock_get_client.return_value = mock_client

            result = await validator.process()

            assert "validated_output" in result
            assert result["validated_output"].text == "System: Be helpful\n\nTest input"

    def test_get_exception_message(self, validator):
        """Test exception message extraction."""
        # Test with exception that has body attribute
        mock_exception = Mock()
        mock_exception.body = {"message": "Test error message"}

        message = validator._get_nemo_exception_message(mock_exception)
        assert message == "Test error message"

        # Test with exception without body
        mock_exception2 = Mock()
        mock_exception2.body = None

        message = validator._get_nemo_exception_message(mock_exception2)
        assert message is None

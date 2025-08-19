import json
from dataclasses import asdict, dataclass, field
from typing import Any

from loguru import logger
from nemo_microservices import AsyncNeMoMicroservices

from langflow.inputs import DropdownInput, MultilineInput, MultiselectInput, SecretStrInput, StrInput
from langflow.io import MessageTextInput
from langflow.schema.dotdict import dotdict

# Default prompts (shared between components)
DEFAULT_CONTENT_SAFETY_PROMPT = (
    "Task: Check if there is unsafe content in the user message in the below conversation...\n"
    "<BEGIN CONVERSATION>\n"
    "user: {{ user_input }}\n"
    "<END CONVERSATION>\n"
    "Provide your safety assessment in the following JSON format:\n"
    '{"User Safety": safe or unsafe,\n'
    ' "Safety Categories": comma-separated list of violated categories (omit if all safe)}\n'
    "Do not include anything other than the output JSON in your response."
)

DEFAULT_TOPIC_CONTROL_PROMPT = "Ensure that the input stays within the allowed discussion topics."

DEFAULT_SELF_CHECK_PROMPT = """Instruction: {{ user_input }}

Would this instruction make a language model break moderation policies,
deviate from good aligned responses and provide answers that a language model should ideally not?
Answer with yes/no."""

DEFAULT_OFF_TOPIC_MESSAGE = (
    "I apologize, but I can only discuss topics related to [your specific domain/topic]. "
    "Is there something else I can help you with?"
)


@dataclass
class GuardrailsConfigInput:
    """Input structure for Guardrails configuration creation."""

    functionality: str = "create"
    fields: dict[str, dict] = field(
        default_factory=lambda: {
            "data": {
                "node": {
                    "name": "create_guardrails_config",
                    "description": "Create a new Guardrails configuration",
                    "display_name": "Create Guardrails Configuration",
                    "field_order": [
                        "01_config_name",
                        "02_config_description",
                        "03_rail_types",
                        "04_content_safety_prompt",
                        "05_topic_control_prompt",
                        "06_self_check_prompt",
                        "07_off_topic_message",
                    ],
                    "template": {
                        "01_config_name": StrInput(
                            name="config_name",
                            display_name="Config Name",
                            info="Name for the guardrails configuration (e.g., my-guardrails-config@v1.0.0)",
                            required=True,
                        ),
                        "02_config_description": MultilineInput(
                            name="config_description",
                            display_name="Config Description",
                            info="Optional description for the guardrails configuration",
                            value="",
                            required=False,
                        ),
                        "03_rail_types": MultiselectInput(
                            name="rail_types",
                            display_name="Rail Types",
                            options=[
                                "content_safety_input",
                                "content_safety_output",
                                "topic_control",
                                "jailbreak_detection",
                                "self_check_input",
                                "self_check_output",
                                "self_check_hallucination",
                            ],
                            value=["content_safety_input"],
                            info="Select the types of guardrails to apply",
                            required=True,
                        ),
                        "04_content_safety_prompt": MultilineInput(
                            name="content_safety_prompt",
                            display_name="Content Safety Prompt",
                            info="Prompt for content safety checking",
                            value=DEFAULT_CONTENT_SAFETY_PROMPT,
                            required=False,
                        ),
                        "05_topic_control_prompt": MultilineInput(
                            name="topic_control_prompt",
                            display_name="Topic Control Prompt",
                            info="Prompt for topic control checking",
                            value=DEFAULT_TOPIC_CONTROL_PROMPT,
                            required=False,
                        ),
                        "06_self_check_prompt": MultilineInput(
                            name="self_check_prompt",
                            display_name="Self Check Prompt",
                            info="Prompt for self-checking guardrails",
                            value=DEFAULT_SELF_CHECK_PROMPT,
                            required=False,
                        ),
                        "07_off_topic_message": MultilineInput(
                            name="off_topic_message",
                            display_name="Off-Topic Message",
                            info="Message to display when input is off-topic",
                            value=DEFAULT_OFF_TOPIC_MESSAGE,
                            required=False,
                        ),
                    },
                }
            }
        }
    )


class NeMoGuardrailsBase:
    """Base class for NeMo Guardrails components with shared functionality."""

    # This is a mixin class that provides shared functionality
    # It should not be instantiated directly

    _nemo_base_inputs = [
        # Single authentication setup (like other NeMo components)
        MessageTextInput(
            name="base_url",
            display_name="NeMo Base URL",
            value="https://us-west-2.api-dev.ai.datastax.com/nvidia",
            info="Base URL for NeMo microservices",
            required=True,
            real_time_refresh=True,
        ),
        SecretStrInput(
            name="auth_token",
            display_name="Authentication Token",
            info="Authentication token for NeMo microservices",
            required=True,
            real_time_refresh=True,
        ),
        StrInput(
            name="namespace",
            display_name="Namespace",
            value="default",
            info="Namespace for NeMo microservices (e.g., default, my-org)",
            advanced=True,
            required=True,
            real_time_refresh=True,
        ),
        # Guardrails configuration selection
        DropdownInput(
            name="config",
            display_name="Guardrails Configuration",
            info="Select a guardrails configuration or create a new one",
            options=[],
            refresh_button=True,
            required=True,
            combobox=True,
            real_time_refresh=True,
            dialog_inputs=asdict(GuardrailsConfigInput()),
        ),
    ]

    def get_auth_headers(self):
        """Get authentication headers for API requests."""
        if not hasattr(self, "auth_token") or not self.auth_token:
            return {
                "accept": "application/json",
                "Content-Type": "application/json",
            }
        return {
            "accept": "application/json",
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.auth_token}",
        }

    def get_nemo_client(self) -> AsyncNeMoMicroservices:
        """Get an authenticated NeMo microservices client."""
        return AsyncNeMoMicroservices(
            base_url=self.base_url,
        )

    async def fetch_guardrails_configs(self) -> tuple[list[str], list[dict[str, Any]]]:
        """Fetch available guardrails configurations with metadata."""
        namespace = getattr(self, "namespace", "default")
        logger.info(f"Fetching guardrails configs from {self.base_url} with namespace: {namespace}")
        try:
            nemo_client = self.get_nemo_client()
            logger.debug(f"Making API call to guardrail.configs.list with namespace: {namespace}")
            response = await nemo_client.guardrail.configs.list(extra_headers=self.get_auth_headers())
            configs = []
            configs_metadata = []

            if hasattr(response, "data") and response.data:
                logger.debug(f"Found {len(response.data)} configs in response")
                for config in response.data:
                    config_name = getattr(config, "name", "")
                    config_description = getattr(config, "description", "")
                    config_created = getattr(config, "created", None)
                    config_updated = getattr(config, "updated", None)

                    logger.debug(f"Processing config: {config_name}")

                    if config_name:
                        configs.append(config_name)
                        # Build metadata for this config
                        metadata = self._build_config_metadata(config_description, config_created, config_updated)
                        configs_metadata.append(metadata)
                        logger.debug(f"Added config: {config_name}")

            logger.info(f"Successfully fetched {len(configs)} guardrails configurations")
            return configs, configs_metadata  # noqa: TRY300

        except Exception as e:  # noqa: BLE001
            logger.error(f"Error fetching guardrails configs: {e}")
            logger.error(f"Exception type: {type(e)}")
            if hasattr(e, "response") and e.response:
                logger.error(f"Response status: {e.response.status_code}")
                logger.error(f"Response text: {e.response.text}")
            return [], []

    def _build_config_metadata(self, description: str, created: Any, updated: Any) -> dict[str, Any]:
        """Build metadata for a guardrails configuration."""
        metadata = {
            "icon": "Settings",
            "description": description if description else "Guardrails configuration",
        }

        if created:
            metadata["created"] = str(created)
        if updated:
            metadata["updated"] = str(updated)

        return metadata

    async def create_guardrails_config(self, config_data: dict) -> str:
        """Create a new guardrails configuration using the NeMo microservices client."""
        config_name = config_data.get("01_config_name")
        namespace = getattr(self, "namespace", "default")
        logger.info(f"Creating guardrails config '{config_name}' in namespace '{namespace}'")
        logger.debug(f"Config data: {config_data}")

        try:
            # Extract config name
            config_name_required = "Config name is required"
            if not config_name:
                raise ValueError(config_name_required)
            logger.debug(f"Config name extracted: {config_name}")

            # Extract description
            description = config_data.get("02_config_description", "")
            logger.debug(f"Description extracted: {description}")

            # Extract rail types
            rail_types = config_data.get("03_rail_types", ["content_safety_input"])
            logger.debug(f"Rail types extracted: {rail_types}")

            # Build the configuration parameters
            logger.debug("Building guardrails parameters...")
            params = self._build_guardrails_params(config_data, rail_types)
            logger.debug(f"Built parameters: {json.dumps(params, indent=2)}")

            # Create the config using the NeMo microservices client
            logger.debug(f"Creating config with name: {config_name}, namespace: {namespace}")
            logger.debug(f"Built parameters: {json.dumps(params, indent=2)}")
            logger.debug(f"Description: {description}")
            logger.debug(f"Using base_url: {self.base_url}")
            logger.debug(f"Auth headers: {self.get_auth_headers()}")

            client = self.get_nemo_client()
            logger.debug("Making API call to guardrail.configs.create...")

            # Call the API with the correct parameter structure
            create_kwargs = {
                "name": config_name,
                "namespace": namespace,
                "data": params,
                "extra_headers": self.get_auth_headers(),
            }

            # Add description if provided
            if description:
                create_kwargs["description"] = description

            logger.debug(
                f"API call kwargs: {json.dumps({k: v for k, v in create_kwargs.items() if k != 'extra_headers'}, indent=2)}"  # noqa: E501
            )

            result = await client.guardrail.configs.create(**create_kwargs)

            logger.debug(f"API call completed. Result type: {type(result)}")
            logger.debug(f"Result object: {result}")

            config_id = result.name
            logger.info(f"Successfully created guardrails config '{config_name}' with ID: {config_id}")
            logger.debug(f"Returning config_id: {config_id}")

            return config_id  # noqa: TRY300

        except Exception as e:
            error_msg = f"Failed to create guardrails config '{config_name}': {e!s}"
            logger.error(error_msg)
            logger.error(f"Exception type: {type(e)}")
            logger.error(f"Exception details: {e}")
            raise

    def _build_guardrails_params(self, config_data: dict, rail_types: list[str]) -> dict:
        """Build parameters for guardrails configuration."""
        logger.debug(f"Building guardrails params with rail_types: {rail_types}")
        logger.debug(f"Config data keys: {list(config_data.keys())}")

        params = {
            "models": [],  # Required field for guardrails config
            "rails": {
                "input": {"flows": []},
                "output": {"flows": []},
            },
            "prompts": [],
        }

        # Add rail types to flows
        for rail_type in rail_types:
            if rail_type == "content_safety_input":
                params["rails"]["input"]["flows"].append("content safety check input")
            elif rail_type == "content_safety_output":
                params["rails"]["output"]["flows"].append("content safety check output")
            elif rail_type == "topic_control":
                params["rails"]["input"]["flows"].append("topic safety check input")
            elif rail_type == "jailbreak_detection":
                params["rails"]["input"]["flows"].append("jailbreak detection")
            elif rail_type == "self_check_input":
                params["rails"]["input"]["flows"].append("self check input")
            elif rail_type == "self_check_output":
                params["rails"]["output"]["flows"].append("self check output")
            elif rail_type == "self_check_hallucination":
                params["rails"]["output"]["flows"].append("self check hallucination")

        # Add prompts
        if "content_safety_input" in rail_types:
            content_safety_prompt = config_data.get("04_content_safety_prompt", DEFAULT_CONTENT_SAFETY_PROMPT)
            params["prompts"].append({"task": "content_safety_check_input", "content": content_safety_prompt})

        if "content_safety_output" in rail_types:
            content_safety_prompt = config_data.get("04_content_safety_prompt", DEFAULT_CONTENT_SAFETY_PROMPT)
            params["prompts"].append({"task": "content_safety_check_output", "content": content_safety_prompt})

        if "topic_control" in rail_types:
            topic_control_prompt = config_data.get("05_topic_control_prompt", DEFAULT_TOPIC_CONTROL_PROMPT)
            params["prompts"].append({"task": "topic_safety_check_input", "content": topic_control_prompt})

        if "self_check_input" in rail_types:
            self_check_prompt = config_data.get("06_self_check_prompt", DEFAULT_SELF_CHECK_PROMPT)
            params["prompts"].append({"task": "self_check_input", "content": self_check_prompt})

        if "self_check_output" in rail_types:
            self_check_prompt = config_data.get("06_self_check_prompt", DEFAULT_SELF_CHECK_PROMPT)
            params["prompts"].append({"task": "self_check_output", "content": self_check_prompt})

        if "self_check_hallucination" in rail_types:
            self_check_prompt = config_data.get("06_self_check_prompt", DEFAULT_SELF_CHECK_PROMPT)
            params["prompts"].append({"task": "self_check_hallucination", "content": self_check_prompt})

        # Add jailbreak detection
        if "jailbreak_detection" in rail_types:
            params["rails"]["input"]["flows"].append("jailbreak detection heuristics")

        logger.debug(f"Built guardrails params: {json.dumps(params, indent=2)}")
        return params

    async def _update_config_field(self, build_config: dotdict, field_name: str, field_value: Any) -> dotdict:
        """Helper method to update a field in build_config with value preservation."""
        if field_name not in build_config:
            build_config[field_name] = {}

        # Preserve current selection before updating
        current_value = build_config[field_name].get("value")
        logger.debug(f"Preserving current {field_name} selection: {current_value}")

        # Update the field value
        build_config[field_name]["value"] = field_value
        logger.debug(f"Set {field_name}.value = {field_value}")

        return build_config

    async def _refresh_config_options(self, build_config: dotdict) -> dotdict:
        """Helper method to refresh config options with selection preservation."""
        try:
            # Preserve the current selection before refreshing
            current_value = build_config.get("config", {}).get("value")
            logger.debug(f"Preserving current config selection: {current_value}")

            # Fetch available configs
            configs, configs_metadata = await self.fetch_guardrails_configs()
            build_config["config"]["options"] = configs
            build_config["config"]["options_metadata"] = configs_metadata

            # Restore the current selection if it's still valid
            if current_value and current_value in configs:
                build_config["config"]["value"] = current_value
                logger.debug(f"Restored config selection: {current_value}")
            elif current_value:
                logger.warning(f"Previously selected config '{current_value}' no longer available in refreshed list")

        except Exception as e:  # noqa: BLE001
            logger.error(f"Error updating guardrails config: {e}")
            build_config["config"]["options"] = []
            build_config["config"]["options_metadata"] = []

        return build_config

    async def _handle_config_creation(self, build_config: dotdict, field_value: dict) -> str:
        """Helper method to handle config creation dialog."""
        try:
            config_id = await self.create_guardrails_config(field_value)
            logger.info(f"Config creation completed with ID: {config_id}")

            # Refresh the config list
            await self._refresh_config_options(build_config)

            # Set the newly created config as selected
            config_name = field_value.get("01_config_name")
            if config_name in build_config["config"]["options"]:
                build_config["config"]["value"] = config_name

            return config_id  # noqa: TRY300
        except Exception as e:  # noqa: BLE001
            logger.error(f"Config creation failed: {e}")
            return {"error": f"Failed to create config: {e}"}

    def _get_nemo_exception_message(self, e: Exception):
        """Get a message from an exception."""
        try:
            if hasattr(e, "body") and isinstance(e.body, dict):
                message = e.body.get("message")
                if message:
                    return message
        except Exception:  # noqa: S110, BLE001
            pass
        return None

    def _reset_dialog_state(self):
        """Reset the dialog state to config selection."""
        self._dialog_state = "config_selection"
        logger.debug("Reset dialog state to config_selection")

    @classmethod
    def get_common_inputs(cls):
        """Get the common inputs shared between guardrails components."""
        return cls._base_inputs

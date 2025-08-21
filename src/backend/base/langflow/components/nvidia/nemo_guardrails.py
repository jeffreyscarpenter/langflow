import json
from dataclasses import asdict, dataclass, field
from typing import Any

import httpx
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from loguru import logger
from nemo_microservices import AsyncNeMoMicroservices
from pydantic import Field

from langflow.base.models.model import LCModelComponent
from langflow.field_typing import LanguageModel
from langflow.inputs import (
    DropdownInput,
    FloatInput,
    IntInput,
    MultilineInput,
    MultiselectInput,
    SecretStrInput,
    StrInput,
    TabInput,
)
from langflow.io import MessageTextInput, Output
from langflow.schema.dotdict import dotdict
from langflow.schema.message import Message

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


class GuardrailsMicroserviceModel(BaseChatModel):
    """Language model implementation that uses the guardrails microservice."""

    base_url: str = Field(description="Base URL for NeMo microservices")
    auth_token: str = Field(description="Authentication token for NeMo microservices")
    config_id: str = Field(description="Guardrails configuration ID")
    model_name: str = Field(description="Model name to use")
    stream: bool = Field(default=False, description="Whether to stream responses")
    max_tokens: int = Field(default=1024, description="Maximum tokens to generate")
    temperature: float = Field(default=0.7, description="Temperature for generation")
    top_p: float = Field(default=0.9, description="Top-p for generation")
    client: Any = Field(default=None, description="NeMo microservices client")

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.client = AsyncNeMoMicroservices(base_url=self.base_url)
        logger.info(
            f"Initialized GuardrailsMicroserviceModel with config_id: {self.config_id}, "
            f"model: {self.model_name}, stream: {self.stream}"
        )
        logger.debug(
            f"LLM parameters: max_tokens={self.max_tokens}, temperature={self.temperature}, top_p={self.top_p}"
        )

    def get_auth_headers(self):
        """Get authentication headers for API requests."""
        return {
            "accept": "application/json",
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.auth_token}",
        }

    def invoke(self, inputs, **kwargs):
        """Sync invoke method - delegates to _generate."""
        # Convert inputs to messages format for _generate
        if isinstance(inputs, list):
            messages = inputs
        elif isinstance(inputs, dict) and "messages" in inputs:
            # Convert dict messages to LangChain message objects
            messages = []
            for msg in inputs["messages"]:
                if isinstance(msg, dict):
                    if msg.get("role") == "user":
                        messages.append(HumanMessage(content=msg["content"]))
                    elif msg.get("role") == "system":
                        messages.append(SystemMessage(content=msg["content"]))
                    elif msg.get("role") == "assistant":
                        messages.append(AIMessage(content=msg["content"]))
                    else:
                        messages.append(HumanMessage(content=str(msg.get("content", ""))))
                else:
                    messages.append(msg)
        else:
            messages = [HumanMessage(content=str(inputs))]

        # Use _generate method which handles sync execution
        result = self._generate(messages, **kwargs)
        return result.generations[0].message

    async def ainvoke(self, inputs, **kwargs):
        """Async invoke method."""
        return await self._ainvoke_impl(inputs, **kwargs)

    async def _ainvoke_impl(self, inputs, **kwargs):
        """Async invoke implementation."""
        # Convert LangChain messages to the format expected by NeMo API
        if isinstance(inputs, list):
            # Convert LangChain messages to dict format
            messages = []
            for msg in inputs:
                if isinstance(msg, HumanMessage):
                    messages.append({"role": "user", "content": msg.content})
                elif isinstance(msg, SystemMessage):
                    messages.append({"role": "system", "content": msg.content})
                elif isinstance(msg, AIMessage):
                    messages.append({"role": "assistant", "content": msg.content})
                else:
                    messages.append({"role": "user", "content": str(msg.content)})
        else:
            # Handle dict format (backward compatibility)
            messages = inputs.get("messages", [])

        logger.info(
            f"Invoking guardrails microservice with config_id: {self.config_id}, "
            f"model: {self.model_name}, stream: {self.stream}"
        )
        logger.debug(f"Input messages: {messages}")

        try:
            # Prepare the request payload with only supported parameters
            payload = {
                "model": self.model_name,
                "messages": messages,
                "guardrails": {"config_id": self.config_id},
                "max_tokens": self.max_tokens,
                "temperature": self.temperature,
                "top_p": self.top_p,
                "stream": self.stream,
                **kwargs,
            }

            if self.stream:
                # For streaming, we'll use the chat completions endpoint
                chat_url = f"{self.base_url}/v1/guardrail/chat/completions"
                logger.debug(f"Making streaming request to: {chat_url}")
                logger.debug(f"Request payload: {json.dumps(payload, indent=2)}")

                async with httpx.AsyncClient() as client:
                    response = await client.post(chat_url, json=payload, headers=self.get_auth_headers(), timeout=30.0)
                    logger.debug(f"Streaming response status: {response.status_code}")
                    response.raise_for_status()
                    result = response.json()
                    logger.debug(f"Streaming response: {json.dumps(result, indent=2)}")
                    return result
            else:
                # For non-streaming, use the microservice client
                logger.debug(f"Making non-streaming request with payload: {json.dumps(payload, indent=2)}")

                # Prepare parameters for the client call
                client_params = {
                    "model": self.model_name,
                    "messages": messages,
                    "guardrails": {"config_id": self.config_id},
                    "extra_headers": self.get_auth_headers(),
                    "max_tokens": self.max_tokens,
                    "temperature": self.temperature,
                    "top_p": self.top_p,
                    "stream": self.stream,
                    **kwargs,
                }

                response = await self.client.guardrail.chat.completions.create(**client_params)
                logger.debug(f"Non-streaming response received: {type(response)}")

                # Extract content from response
                if hasattr(response, "choices") and response.choices:
                    content = response.choices[0].message.content
                elif hasattr(response, "content"):
                    content = response.content
                elif isinstance(response, dict) and "choices" in response:
                    content = response["choices"][0]["message"]["content"]
                else:
                    content = str(response)

                # Create AIMessage with metadata
                metadata = {}
                if hasattr(response, "usage"):
                    metadata["usage"] = response.usage
                if hasattr(response, "model"):
                    metadata["model"] = response.model

                return AIMessage(content=content, response_metadata=metadata)
        except httpx.HTTPStatusError as exc:
            logger.error(f"HTTP error {exc.response.status_code} during guardrails inference")
            logger.error(f"Response text: {exc.response.text}")
            raise
        except httpx.RequestError as exc:
            logger.error(f"Request error during guardrails inference: {exc}")
            raise
        except Exception as e:
            logger.error(f"Unexpected error during guardrails inference: {e}")
            raise

    def _generate(self, messages, stop=None, run_manager=None, **kwargs):  # noqa: ARG002
        """Required abstract method for BaseChatModel."""
        import asyncio

        from langchain_core.outputs import ChatGeneration, ChatResult

        # Convert LangChain messages to our format and run async invoke
        try:
            # Check if we're already in an async context
            try:
                asyncio.get_running_loop()
                # We're in an async context, use ThreadPoolExecutor
                import concurrent.futures

                with concurrent.futures.ThreadPoolExecutor() as executor:
                    future = executor.submit(lambda: asyncio.run(self._ainvoke_impl(messages, **kwargs)))
                    result = future.result()
            except RuntimeError:
                # No event loop running, we can create one
                result = asyncio.run(self._ainvoke_impl(messages, **kwargs))

            # Convert AIMessage result to ChatResult format
            generation = ChatGeneration(message=result)
            return ChatResult(generations=[generation])

        except Exception as e:
            logger.error(f"Error in _generate: {e}")
            raise

    async def _agenerate(self, messages, stop=None, run_manager=None, **kwargs):  # noqa: ARG002
        """Async version of _generate method."""
        from langchain_core.outputs import ChatGeneration, ChatResult

        try:
            # Use our async invoke method directly
            result = await self._ainvoke_impl(messages, **kwargs)

            # Convert AIMessage result to ChatResult format
            generation = ChatGeneration(message=result)
            return ChatResult(generations=[generation])

        except Exception as e:
            logger.error(f"Error in _agenerate: {e}")
            raise

    @property
    def _llm_type(self) -> str:
        """Required property for BaseChatModel."""
        return "nemo_guardrails"

    def with_config(self, config, **kwargs):  # noqa: ARG002
        """Support for LangChain configuration."""
        # Create a new instance with updated config
        return GuardrailsMicroserviceModel(
            base_url=self.base_url,
            auth_token=self.auth_token,
            config_id=self.config_id,
            model_name=self.model_name,
            stream=self.stream,
            max_tokens=self.max_tokens,
            temperature=self.temperature,
            top_p=self.top_p,
        )

    def bind_tools(self, tools, **kwargs):  # noqa: ARG002
        """Support for tool binding (if needed)."""
        # For now, return self as tools may not be supported in guardrails
        return self


class NVIDIANeMoGuardrailsComponent(LCModelComponent):
    display_name = "NeMo Guardrails"
    description = (
        "Apply guardrails to LLM interactions using the NeMo Guardrails microservice. "
        "Select a guardrails configuration and model to apply safety checks."
    )
    icon = "NVIDIA"
    name = "NVIDIANemoGuardrails"
    beta = True

    inputs = [
        *LCModelComponent._base_inputs,
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
        # Mode selection
        TabInput(
            name="mode",
            display_name="Mode",
            options=["chat", "check"],
            value="chat",
            info="Chat mode: Generate responses with guardrails. Check mode: Validate input/output only.",
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
        # LLM parameters (only shown in chat mode)
        IntInput(
            name="max_tokens",
            display_name="Max Tokens",
            info="Maximum number of tokens to generate.",
            advanced=True,
            value=1024,
        ),
        FloatInput(
            name="temperature",
            display_name="Temperature",
            info="Controls randomness in the response. Lower values are more deterministic.",
            advanced=True,
            value=0.7,
        ),
        FloatInput(
            name="top_p",
            display_name="Top P",
            info="Controls diversity via nucleus sampling. Lower values are more focused.",
            advanced=True,
            value=0.9,
        ),
        DropdownInput(
            name="model",
            display_name="Model",
            info="Select a model to use with the guardrails configuration",
            options=[],
            refresh_button=True,
            required=True,
            combobox=True,
            real_time_refresh=True,
        ),
        # Validation mode (only shown in check mode)
        DropdownInput(
            name="validation_mode",
            display_name="Validation Mode",
            options=["input", "output"],
            value="input",
            info="Validate input (before LLM) or output (after LLM) - only used in check mode",
            required=False,
            show=False,  # Initially hidden, shown in check mode
        ),
    ]

    # Default outputs (will be updated dynamically based on mode)
    outputs = [
        Output(display_name="Model Response", name="text_output", method="text_response", dynamic=True),
        Output(display_name="Language Model", name="model_output", method="build_model", dynamic=True),
    ]

    def update_outputs(self, frontend_node: dict, field_name: str, field_value: Any) -> dict:
        """Dynamically show only the relevant outputs based on the selected mode."""
        logger.info(f"update_outputs called with field_name: {field_name}, field_value: {field_value}")

        # Handle initial state - if no mode is set, default to chat mode
        if field_name == "mode" or (field_name is None and "outputs" not in frontend_node):
            # Get the current mode value, default to "chat"
            current_mode = field_value if field_name == "mode" else "chat"
            logger.info(f"Setting outputs for mode: {current_mode}")

            # Start with empty outputs
            frontend_node["outputs"] = []

            if current_mode == "chat":
                # In chat mode: show LLM outputs
                frontend_node["outputs"] = [
                    Output(
                        display_name="Model Response",
                        name="text_output",
                        method="text_response",
                        dynamic=True,
                    ),
                    Output(
                        display_name="Language Model",
                        name="model_output",
                        method="build_model",
                        dynamic=True,
                    ),
                ]
            elif current_mode == "check":
                # In check mode: show single validation output
                frontend_node["outputs"] = [
                    Output(
                        display_name="Validated Output",
                        name="validated_output",
                        method="validated_output",
                        dynamic=True,
                    ),
                ]

            logger.info(f"Updated frontend_node outputs for {current_mode} mode: {frontend_node['outputs']}")
        return frontend_node

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

    async def update_build_config(
        self, build_config: dotdict, field_value: Any, field_name: str | None = None
    ) -> dotdict | str:
        """Update build configuration for the guardrails component."""
        logger.info(f"Updating build config for field: {field_name}, value: {field_value}")

        # Handle mode changes - update visibility of inputs and outputs
        if field_name == "mode":
            mode = field_value
            logger.info(f"Mode changed to: {mode}")

            # Update visibility of LLM-specific inputs
            if mode == "chat":
                build_config["max_tokens"]["show"] = True
                build_config["temperature"]["show"] = True
                build_config["top_p"]["show"] = True
                build_config["model"]["show"] = True
                build_config["validation_mode"]["show"] = False
            else:  # check mode
                build_config["max_tokens"]["show"] = False
                build_config["temperature"]["show"] = False
                build_config["top_p"]["show"] = False
                build_config["model"]["show"] = False
                build_config["validation_mode"]["show"] = True

            # Update outputs based on mode
            if "outputs" not in build_config:
                build_config["outputs"] = []

            if mode == "chat":
                # In chat mode: show LLM outputs
                build_config["outputs"] = [
                    Output(
                        display_name="Model Response",
                        name="text_output",
                        method="text_response",
                        dynamic=True,
                    ),
                    Output(
                        display_name="Language Model",
                        name="model_output",
                        method="build_model",
                        dynamic=True,
                    ),
                ]
            elif mode == "check":
                # In check mode: show single validation output
                build_config["outputs"] = [
                    Output(
                        display_name="Validated Output",
                        name="validated_output",
                        method="validated_output",
                        dynamic=True,
                    ),
                ]

            logger.info(f"Updated outputs for {mode} mode: {build_config['outputs']}")

        # Handle config creation dialog
        if field_name == "config" and isinstance(field_value, dict) and "01_config_name" in field_value:
            try:
                config_id = await self.create_guardrails_config(field_value)
                logger.info(f"Config creation completed with ID: {config_id}")

                # Refresh the config list
                configs, configs_metadata = await self.fetch_guardrails_configs()
                build_config["config"]["options"] = configs
                build_config["config"]["options_metadata"] = configs_metadata

                # Set the newly created config as selected
                config_name = field_value.get("01_config_name")
                if config_name in configs:
                    build_config["config"]["value"] = config_name
                else:
                    pass
                return config_id  # noqa: TRY300
            except Exception as e:  # noqa: BLE001
                logger.error(f"Config creation failed: {e}")
                return {"error": f"Failed to create config: {e}"}

        # Handle config refresh
        if field_name == "config" and (field_value is None or field_value == ""):
            try:
                # Preserve current selection
                current_value = build_config.get("config", {}).get("value")
                logger.debug(f"Config refresh - preserving current value: {current_value}")

                # Fetch available configs
                configs, configs_metadata = await self.fetch_guardrails_configs()
                build_config["config"]["options"] = configs
                build_config["config"]["options_metadata"] = configs_metadata

                # Restore selection if still valid
                if current_value and current_value in configs:
                    build_config["config"]["value"] = current_value
                    logger.debug(f"Config refresh - restored selection: {current_value}")
                else:
                    logger.debug(
                        f"Config refresh - no valid selection to restore. "
                        f"current_value: {current_value}, available: {configs}"
                    )
            except Exception as e:  # noqa: BLE001
                logger.error(f"Error refreshing configs: {e}")
                build_config["config"]["options"] = []
                build_config["config"]["options_metadata"] = []

        # Handle existing config selection
        if field_name == "config" and isinstance(field_value, str):
            logger.debug(f"Config selection: {field_value}")
            build_config["config"]["value"] = field_value
            return build_config

        # Handle model refresh
        if field_name == "model" and (field_value is None or field_value == ""):
            logger.debug("Model refresh requested")
            return await self._handle_model_refresh(build_config)

        return build_config

    async def process(self) -> dict[str, Message]:
        """Process the input through guardrails validation (for check mode)."""
        logger.info("Starting guardrails validation process")

        # Prepare input
        input_text = ""
        if hasattr(self, "system_message") and self.system_message:
            input_text += f"{self.system_message}\n\n"
        if hasattr(self, "input_value") and self.input_value:
            if isinstance(self.input_value, Message):
                input_text += self.input_value.text
            else:
                input_text += str(self.input_value)

        logger.debug(f"Prepared input text: {input_text[:200]}...")  # Log first 200 chars

        empty_message_error = "The message you want to validate is empty."
        if not input_text.strip():
            logger.error("Empty input text provided")
            raise ValueError(empty_message_error)

        validation_mode = getattr(self, "validation_mode", "input")
        logger.info(f"Processing validation in {validation_mode} mode")

        try:
            # Validate using guardrail.chat.completions with guardrails
            client = self.get_nemo_client()

            logger.debug("Making API call to guardrail.chat.completions for validation")

            # Determine message role based on validation mode
            role = "user" if validation_mode == "input" else "assistant"

            # Use a minimal completion request to test validation
            validation_response = await client.guardrail.chat.completions.create(
                messages=[{"role": role, "content": input_text}],
                model="gpt-4o",  # Use a default model for validation
                guardrails={"config_id": self.config},
                max_tokens=1,  # Minimal tokens to just test validation
                extra_headers=self.get_auth_headers(),
            )

            logger.debug(f"Validation response: {validation_response}")

            # Check if the response indicates blocking
            # The response should contain information about whether the input was blocked
            if hasattr(validation_response, "choices") and validation_response.choices:
                choice = validation_response.choices[0]
                if hasattr(choice, "finish_reason") and choice.finish_reason == "guardrail_blocked":
                    logger.info(f"{validation_mode.capitalize()} blocked by guardrails")
                    self.status = f"{validation_mode.capitalize()} blocked by guardrails"
                    # Return error message with error=True and category="error"
                    return {
                        "validated_output": Message(
                            text=f"I cannot process that {validation_mode}.", error=True, category="error"
                        )
                    }

            # If validation passes, return the original input with error=False and category="message"
            logger.info(f"{validation_mode.capitalize()} passed guardrails validation")
            self.status = f"{validation_mode.capitalize()} validated successfully"
            return {"validated_output": Message(text=input_text, error=False, category="message")}

        except Exception as e:
            logger.error(f"Error in validation: {e}")
            logger.error(f"Exception type: {type(e)}")
            if hasattr(e, "response") and e.response:
                logger.error(f"Response status: {e.response.status_code}")
                logger.error(f"Response text: {e.response.text}")
            if message := self._get_nemo_exception_message(e):
                logger.error(f"Exception message: {message}")
                raise ValueError(message) from e
            raise

    async def validated_output(self) -> Message:
        """Return the validated output as a Message (for check mode).

        Returns a single Message that contains either:
        - The validated input (error=False, category="message") when validation passes
        - An error message (error=True, category="error") when validation fails

        Downstream components can check the Message's error field to determine validation status.
        """
        result = await self.process()
        return result.get("validated_output", Message(text="Validation completed", error=False, category="message"))

    async def fetch_guardrails_models(self) -> list[str]:
        """Fetch available models for guardrails using the general models endpoint."""
        logger.info("Fetching available models using general models endpoint")
        try:
            client = self.get_nemo_client()
            logger.debug("Using NeMo microservices client to fetch models")

            # Use the client's models resource
            models_response = await client.models.list(extra_headers=self.get_auth_headers())
            logger.debug(f"Models response: {models_response}")

            # Import ChatNVIDIA's model filtering logic
            try:
                from langchain_nvidia_ai_endpoints._statics import determine_model
            except ImportError:
                logger.warning("langchain-nvidia-ai-endpoints not available, falling back to basic filtering")
                return self._fallback_model_filtering(models_response)

            model_names = []
            total_models = 0
            chat_models = 0

            if hasattr(models_response, "data") and models_response.data:
                for model in models_response.data:
                    total_models += 1
                    model_id = None

                    # Extract model ID
                    if hasattr(model, "id") and model.id:
                        model_id = model.id
                    elif hasattr(model, "name") and model.name:
                        model_id = model.name

                    if model_id:
                        # Use ChatNVIDIA's lookup table to determine model type
                        known_model = determine_model(model_id)
                        if known_model and known_model.model_type == "chat":
                            model_names.append(model_id)
                            chat_models += 1
                            logger.debug(f"Added chat model: {model_id}")
                        elif known_model:
                            logger.debug(f"Skipped {known_model.model_type} model: {model_id}")
                        # Unknown model - use name-based filtering as fallback
                        elif self._is_likely_chat_model(model_id):
                            model_names.append(model_id)
                            chat_models += 1
                            logger.debug(f"Added likely chat model (fallback): {model_id}")
                        else:
                            logger.debug(f"Skipped unknown model: {model_id}")

            logger.info(f"Found {chat_models} chat models out of {total_models} total models")
            return model_names  # noqa: TRY300

        except Exception as exc:  # noqa: BLE001
            logger.error(f"Error fetching models using NeMo microservices client: {exc}")
            return []

    def _is_likely_chat_model(self, model_id: str) -> bool:
        """Fallback method to determine if a model is likely a chat model based on name."""
        model_id_lower = model_id.lower()

        # Exclude known non-chat models
        if any(keyword in model_id_lower for keyword in ["embed", "embedqa", "rerank", "rerankqa"]):
            return False

        # Include likely chat models
        chat_indicators = ["instruct", "chat", "completion", "nemotron", "llama-3", "gpt", "claude"]
        return any(indicator in model_id_lower for indicator in chat_indicators)

    def _fallback_model_filtering(self, models_response) -> list[str]:
        """Fallback method when ChatNVIDIA's lookup table is not available."""
        logger.info("Using fallback model filtering")
        model_names = []

        if hasattr(models_response, "data") and models_response.data:
            for model in models_response.data:
                model_id = None
                if hasattr(model, "id") and model.id:
                    model_id = model.id
                elif hasattr(model, "name") and model.name:
                    model_id = model.name

                if model_id and self._is_likely_chat_model(model_id):
                    model_names.append(model_id)

        return model_names

    async def _handle_model_refresh(self, build_config: dotdict) -> dotdict:
        """Handle model refresh with selection preservation."""
        logger.info("Handling model refresh request")

        try:
            # Preserve the current selection before refreshing
            current_value = build_config.get("model", {}).get("value")
            logger.debug(f"Preserving current model selection: {current_value}")

            # Fetch all available models for guardrails
            logger.debug("Refreshing available models for guardrails")
            models = await self.fetch_guardrails_models()
            build_config["model"]["options"] = models

            # Restore the current selection if it's still valid
            if current_value and current_value in models:
                build_config["model"]["value"] = current_value
                logger.debug(f"Restored model selection: {current_value}")
            elif models and not current_value:
                # Only set default if no current selection
                build_config["model"]["value"] = models[0]
                logger.debug(f"Set default model selection: {models[0]}")
            elif current_value:
                logger.warning(f"Previously selected model '{current_value}' no longer available in refreshed list")
                # Clear the value when the selected model is no longer available
                build_config["model"]["value"] = ""
            else:
                # No models available, clear the value
                build_config["model"]["value"] = ""

            logger.info(f"Refreshed {len(models)} available models for guardrails")

        except Exception as e:  # noqa: BLE001
            logger.error(f"Error refreshing models: {e}")
            build_config["model"]["options"] = []
            build_config["model"]["value"] = ""

        return build_config

    def build_model(self) -> LanguageModel:
        """Build a language model that uses the guardrails microservice."""
        mode = getattr(self, "mode", "chat")

        if mode == "check":
            error_msg = "Check mode does not provide a language model. Use the validation outputs instead."
            raise NotImplementedError(error_msg)

        logger.info(
            f"Building guardrails model with config: {getattr(self, 'config', 'None')}, "
            f"model: {getattr(self, 'model', 'None')}"
        )

        # Validate configuration
        base_url_required = "Base URL is required"
        auth_token_required = "Authentication token is required"  # noqa: S105
        config_required = "Guardrails configuration is required"
        namespace_required = "Namespace is required"
        model_required = "Model selection is required"

        if not hasattr(self, "model") or not self.model:
            logger.error("Model selection is required but not set")
            raise ValueError(model_required)

        # temp fix for config
        config = self.config or "self-check"

        if not hasattr(self, "config") or not self.config:
            logger.error("Guardrails configuration is required but not set")
            raise ValueError(config_required)

        if not hasattr(self, "base_url") or not self.base_url:
            logger.error("Base URL is required but not set")
            raise ValueError(base_url_required)

        if not hasattr(self, "auth_token") or not self.auth_token:
            logger.error("Authentication token is required but not set")
            raise ValueError(auth_token_required)

        if not hasattr(self, "namespace") or not self.namespace:
            logger.error("Namespace is required but not set")
            raise ValueError(namespace_required)

        logger.info(
            f"Creating GuardrailsMicroserviceModel with base_url: {self.base_url}, "
            f"config_id: {self.config}, model: {self.model}"
        )

        return GuardrailsMicroserviceModel(
            base_url=self.base_url,
            auth_token=self.auth_token,
            config_id=config,
            model_name=self.model,
            stream=self.stream,
            max_tokens=getattr(self, "max_tokens", 1024),
            temperature=getattr(self, "temperature", 0.7),
            top_p=getattr(self, "top_p", 0.9),
        )

    async def text_response(self) -> Message:
        """Handle text response based on mode."""
        mode = getattr(self, "mode", "chat")

        if mode == "check":
            # In check mode, perform validation and return the validated output
            result = await self.process()
            return result.get("validated_output", Message(text="Validation completed", error=False, category="message"))
        # In chat mode, use the normal LLM response
        return await super().text_response()

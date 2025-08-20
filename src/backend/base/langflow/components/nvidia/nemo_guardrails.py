import json
from typing import Any

import httpx
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from loguru import logger
from nemo_microservices import AsyncNeMoMicroservices
from pydantic import Field

from langflow.base.models.model import LCModelComponent
from langflow.base.nvidia.nemo_guardrails_base import NeMoGuardrailsBase
from langflow.field_typing import LanguageModel
from langflow.inputs import DropdownInput, FloatInput, IntInput
from langflow.schema.dotdict import dotdict
from langflow.schema.message import Message


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


class NVIDIANeMoGuardrailsComponent(LCModelComponent, NeMoGuardrailsBase):
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
        *NeMoGuardrailsBase._nemo_base_inputs,
        # LLM parameters
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
    ]

    async def fetch_guardrails_models(self) -> list[str]:
        """Fetch available models for guardrails using the NeMo microservices client."""
        logger.info("Fetching available models for guardrails using NeMo microservices client")
        try:
            client = self.get_nemo_client()
            logger.debug("Using NeMo microservices client to fetch models")

            # Use the client's models resource
            models_response = await client.guardrail.models.list(extra_headers=self.get_auth_headers())
            logger.debug(f"Models response: {models_response}")

            model_names = []
            if hasattr(models_response, "data") and models_response.data:
                for model in models_response.data:
                    logger.debug(f"Processing model: {model}")
                    # Check for both 'name' and 'id' attributes (some APIs use different field names)
                    if hasattr(model, "name") and model.name:
                        model_names.append(model.name)
                        logger.debug(f"Added model by name: {model.name}")
                    elif hasattr(model, "id") and model.id:
                        model_names.append(model.id)
                        logger.debug(f"Added model by id: {model.id}")

            logger.debug(f"Found {len(model_names)} available models for guardrails")
            return model_names  # noqa: TRY300

        except Exception as exc:  # noqa: BLE001
            logger.error(f"Error fetching models using NeMo microservices client: {exc}")
            return []

    async def update_build_config(
        self, build_config: dotdict, field_value: Any, field_name: str | None = None
    ) -> dotdict | str:
        """Update build configuration for the guardrails component."""
        logger.info(f"Updating build config for field: {field_name}, value: {field_value}")

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
        logger.info(
            f"Building guardrails model with config: {getattr(self, 'config', 'None')}, "
            f"model: {getattr(self, 'model', 'None')}"
        )

        # Validate configuration
        base_url_required = "Base URL is required"
        auth_token_required = "Authentication token is required"  # noqa: S105
        namespace_required = "Namespace is required"
        model_required = "Model selection is required"

        if not hasattr(self, "model") or not self.model:
            logger.error("Model selection is required but not set")
            raise ValueError(model_required)

        # temp fix for config
        config = self.config or "self-check"

        # if not hasattr(self, "config") or not self.config:
        #    logger.error("Guardrails configuration is required but not set")
        #    raise ValueError(config_required)

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
        """Use base class text_response method since our model now returns AIMessage objects."""
        return await super().text_response()

import json
from typing import Any

import httpx
from loguru import logger
from nemo_microservices import AsyncNeMoMicroservices

from langflow.base.models.model import LCModelComponent
from langflow.components.nvidia.nemo_guardrails_base import NeMoGuardrailsBase
from langflow.field_typing import LanguageModel
from langflow.inputs import BoolInput, DropdownInput, FloatInput, IntInput
from langflow.schema.dotdict import dotdict
from langflow.schema.message import MESSAGE_SENDER_AI, Message


class GuardrailsMicroserviceModel:
    """Language model implementation that uses the guardrails microservice."""

    def __init__(
        self,
        base_url: str,
        auth_token: str,
        config_id: str,
        model_name: str,
        *,
        stream: bool = False,
        max_tokens: int = 1024,
        temperature: float = 0.7,
        top_p: float = 0.9,
        top_k: int = 40,
        seed: int | None = None,
        frequency_penalty: float = 0.0,
        presence_penalty: float = 0.0,
        json_mode: bool = False,
    ):
        self.base_url = base_url
        self.auth_token = auth_token
        self.config_id = config_id
        self.model_name = model_name
        self.stream = stream
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.top_p = top_p
        self.top_k = top_k
        self.seed = seed
        self.frequency_penalty = frequency_penalty
        self.presence_penalty = presence_penalty
        self.json_mode = json_mode
        self.client = AsyncNeMoMicroservices(base_url=base_url)
        logger.info(
            f"Initialized GuardrailsMicroserviceModel with config_id: {config_id}, "
            f"model: {model_name}, stream: {stream}"
        )
        logger.debug(
            f"LLM parameters: max_tokens={max_tokens}, temperature={temperature}, "
            f"top_p={top_p}, top_k={top_k}, seed={seed}"
        )

    def get_auth_headers(self):
        """Get authentication headers for API requests."""
        return {
            "accept": "application/json",
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.auth_token}",
        }

    async def invoke(self, inputs: dict, **kwargs):
        """Invoke the guardrails microservice for inference."""
        messages = inputs.get("messages", [])
        logger.info(
            f"Invoking guardrails microservice with config_id: {self.config_id}, "
            f"model: {self.model_name}, stream: {self.stream}"
        )
        logger.debug(f"Input messages: {messages}")

        try:
            # Prepare the request payload with all LLM parameters
            payload = {
                "model": self.model_name,
                "messages": messages,
                "guardrails": {"config_id": self.config_id},
                "max_tokens": self.max_tokens,
                "temperature": self.temperature,
                "top_p": self.top_p,
                "top_k": self.top_k,
                "frequency_penalty": self.frequency_penalty,
                "presence_penalty": self.presence_penalty,
                **kwargs,
            }

            # Add optional parameters only if they have values
            if self.seed is not None:
                payload["seed"] = self.seed
            if self.json_mode:
                payload["response_format"] = {"type": "json_object"}

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
                    "top_k": self.top_k,
                    "frequency_penalty": self.frequency_penalty,
                    "presence_penalty": self.presence_penalty,
                    **kwargs,
                }

                # Add optional parameters
                if self.seed is not None:
                    client_params["seed"] = self.seed
                if self.json_mode:
                    client_params["response_format"] = {"type": "json_object"}

                response = await self.client.guardrail.chat.completions.create(**client_params)
                logger.debug(f"Non-streaming response received: {type(response)}")
                return response
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

    def with_config(self, config, **kwargs):  # noqa: ARG002
        """Support for LangChain configuration."""
        # Create a new instance with updated config
        return GuardrailsMicroserviceModel(
            self.base_url, self.auth_token, self.config_id, self.model_name, stream=self.stream
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
        IntInput(
            name="top_k",
            display_name="Top K",
            info="Limits the number of tokens considered for each step.",
            advanced=True,
            value=40,
        ),
        IntInput(
            name="seed",
            display_name="Seed",
            info="Random seed for reproducible results.",
            advanced=True,
            value=None,
        ),
        FloatInput(
            name="frequency_penalty",
            display_name="Frequency Penalty",
            info="Reduces repetition of common tokens.",
            advanced=True,
            value=0.0,
        ),
        FloatInput(
            name="presence_penalty",
            display_name="Presence Penalty",
            info="Reduces repetition of any token.",
            advanced=True,
            value=0.0,
        ),
        BoolInput(
            name="json_mode",
            display_name="JSON Mode",
            info="Force the model to respond with valid JSON.",
            advanced=True,
            value=False,
        ),
        # Model selection (from available models in the config)
        DropdownInput(
            name="model",
            display_name="Model",
            info="Select a model to use with the guardrails configuration",
            options=[],
            refresh_button=True,
            required=True,
            combobox=True,
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

                # Fetch available configs
                configs, configs_metadata = await self.fetch_guardrails_configs()
                build_config["config"]["options"] = configs
                build_config["config"]["options_metadata"] = configs_metadata

                # Restore selection if still valid
                if current_value and current_value in configs:
                    build_config["config"]["value"] = current_value
            except Exception as e:  # noqa: BLE001
                logger.error(f"Error refreshing configs: {e}")
                build_config["config"]["options"] = []
                build_config["config"]["options_metadata"] = []

        # Handle model refresh
        if field_name == "model" and (field_value is None or field_value == ""):
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
        config_required = "Guardrails configuration is required"
        base_url_required = "Base URL is required"
        auth_token_required = "Authentication token is required"  # noqa: S105
        namespace_required = "Namespace is required"
        model_required = "Model selection is required"

        if not hasattr(self, "model") or not self.model:
            logger.error("Model selection is required but not set")
            raise ValueError(model_required)

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
            config_id=self.config,
            model_name=self.model,
            stream=self.stream,
            max_tokens=getattr(self, "max_tokens", 1024),
            temperature=getattr(self, "temperature", 0.7),
            top_p=getattr(self, "top_p", 0.9),
            top_k=getattr(self, "top_k", 40),
            seed=getattr(self, "seed", None),
            frequency_penalty=getattr(self, "frequency_penalty", 0.0),
            presence_penalty=getattr(self, "presence_penalty", 0.0),
            json_mode=getattr(self, "json_mode", False),
        )

    async def text_response(self) -> Message:
        """Custom text_response method with proper LangChain tracing."""
        logger.info("Starting text_response method")

        # Prepare input
        input_text = ""
        if self.system_message:
            input_text += f"System: {self.system_message}\n\n"
        if self.input_value:
            if isinstance(self.input_value, Message):
                input_text += self.input_value.text
            else:
                input_text += str(self.input_value)

        logger.debug(f"Prepared input text: {input_text[:200]}...")  # Log first 200 chars

        empty_message_error = "The message you want to send to the model is empty."
        if not input_text.strip():
            logger.error("Empty input text provided")
            raise ValueError(empty_message_error)

        logger.info("Processing with guardrails-wrapped LLM")
        return await self._wrapped_mode_response(input_text)

    async def _wrapped_mode_response(self, input_text: str) -> Message:
        """Handle wrapped mode: LLM + guardrails together."""
        logger.info("Processing wrapped mode response")
        output = self.build_model()

        # Create messages format for the microservice
        messages = [{"role": "user", "content": input_text}]
        logger.debug(f"Created messages: {messages}")

        try:
            # Configure with LangChain callbacks
            output = output.with_config(
                {
                    "run_name": self.display_name,
                    "project_name": self.get_project_name(),
                    "callbacks": self.get_langchain_callbacks(),
                }
            )

            lf_message = None
            result = None

            if self.stream:
                # Handle streaming
                if self.is_connected_to_chat_output():
                    # Add a Message for streaming
                    if hasattr(self, "graph"):
                        session_id = self.graph.session_id
                    elif hasattr(self, "_session_id"):
                        session_id = self._session_id
                    else:
                        session_id = None

                    # For streaming, we need to handle the response format differently
                    result = await output.invoke({"messages": messages})

                    # Extract text from streaming response
                    if isinstance(result, dict) and "choices" in result:
                        stream_text = result["choices"][0]["message"]["content"]
                    else:
                        stream_text = str(result)

                    model_message = Message(
                        text=stream_text,
                        sender=MESSAGE_SENDER_AI,
                        sender_name="AI",
                        properties={"icon": self.icon, "state": "partial"},
                        session_id=session_id,
                    )
                    model_message.properties.source = self._build_source(self._id, self.display_name, self)
                    lf_message = await self.send_message(model_message)
                    result = lf_message.text
                else:
                    # Non-chat streaming
                    result = await output.invoke({"messages": messages})
                    if isinstance(result, dict) and "choices" in result:
                        result = result["choices"][0]["message"]["content"]
            else:
                # Non-streaming
                result = await output.invoke({"messages": messages})
                if isinstance(result, dict) and "choices" in result:
                    result = result["choices"][0]["message"]["content"]
                elif hasattr(result, "content"):
                    result = result.content
                # elif isinstance(result, str):
                #     result = result  # This is redundant

            # Set status
            if isinstance(result, dict):
                self.status = json.dumps(result, indent=4)
            else:
                self.status = result

        except Exception as e:
            logger.error(f"Error in text_response: {e}")
            logger.error(f"Exception type: {type(e)}")
            if hasattr(e, "response") and e.response:
                logger.error(f"Response status: {e.response.status_code}")
                logger.error(f"Response text: {e.response.text}")
            if message := self._get_nemo_exception_message(e):
                logger.error(f"Exception message: {message}")
                raise ValueError(message) from e
            raise

        logger.info("text_response completed successfully")
        return lf_message or Message(text=result)

from typing import Any

from loguru import logger

from langflow.base.models.model import Component
from langflow.components.nvidia.nemo_guardrails_base import NeMoGuardrailsBase
from langflow.inputs import DropdownInput, MessageInput, MultilineInput
from langflow.io import Output
from langflow.schema.dotdict import dotdict
from langflow.schema.message import Message


class NVIDIANeMoGuardrailsValidator(NeMoGuardrailsBase, Component):
    display_name = "NeMo Guardrails Validator"
    description = (
        "Validate input/output using NVIDIA NeMo Guardrails microservice. "
        "This component performs validation only - it does not generate responses. "
        "Use this to validate input before sending to an external LLM, or validate LLM responses."
    )
    documentation: str = "https://docs.nvidia.com/nemo/microservices/latest/guardrails/index.html"
    icon = "Shield"
    name = "NVIDIANeMoGuardrailsValidator"

    inputs = [
        MessageInput(name="input_value", display_name="Input"),
        MultilineInput(
            name="system_message",
            display_name="System Message",
            info="System message to include in validation context.",
            advanced=True,
        ),
        *NeMoGuardrailsBase._nemo_base_inputs,
        # Validation mode
        DropdownInput(
            name="validation_mode",
            display_name="Validation Mode",
            options=["input", "output"],
            value="input",
            info="Validate input (before LLM) or output (after LLM)",
            required=True,
        ),
    ]

    outputs = [
        Output(display_name="Validated Output", name="validated_output", method="process"),
        Output(display_name="Validation Error", name="validation_error", method="process"),
    ]

    async def update_build_config(
        self, build_config: dotdict, field_value: Any, field_name: str | None = None
    ) -> dotdict | str:
        """Update build configuration for the guardrails validator component."""
        logger.info(f"Updating build config for field: {field_name}, value: {field_value}")

        # Handle config creation dialog
        if field_name == "config" and isinstance(field_value, dict) and "01_config_name" in field_value:
            return await self._handle_config_creation(build_config, field_value)

        # Handle config refresh
        if field_name == "config":
            await self._refresh_config_options(build_config)
            return build_config

        # Handle basic field value setting
        if field_name and field_value is not None:
            await self._update_config_field(build_config, field_name, field_value)

        return build_config

    async def process(self) -> dict[str, Message]:
        """Process the input through guardrails validation."""
        logger.info("Starting guardrails validation process")

        # Prepare input
        input_text = ""
        if self.system_message:
            input_text += f"{self.system_message}\n\n"
        if self.input_value:
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
            # Validate using guardrail.checks.create
            client = self.get_nemo_client()

            logger.debug("Making API call to guardrail.checks.create for validation")

            # Determine message role based on validation mode
            role = "user" if validation_mode == "input" else "assistant"

            validation_check = await client.guardrail.checks.create(
                messages=[{"role": role, "content": input_text}],
                guardrails={"config_id": self.guardrails_config},
                extra_headers=self.get_auth_headers(),
            )

            logger.debug(f"Validation check result: {validation_check}")

            if validation_check.status == "blocked":
                logger.info(f"{validation_mode.capitalize()} blocked by guardrails")
                self.status = f"{validation_mode.capitalize()} blocked by guardrails"
                return {"validation_error": Message(text=f"I cannot process that {validation_mode}.")}

            # If validation passes, return the original input
            logger.info(f"{validation_mode.capitalize()} passed guardrails validation")
            self.status = f"{validation_mode.capitalize()} validated successfully"
            return {"validated_output": Message(text=input_text)}

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

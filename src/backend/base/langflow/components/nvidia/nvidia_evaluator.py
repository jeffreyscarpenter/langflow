import asyncio
import json
import time
from dataclasses import asdict, dataclass, field
from typing import Any

import httpx
from loguru import logger
from nemo_microservices import AsyncNeMoMicroservices, NeMoMicroservicesError

from langflow.custom import Component
from langflow.field_typing.range_spec import RangeSpec
from langflow.io import (
    BoolInput,
    DropdownInput,
    FloatInput,
    HandleInput,
    IntInput,
    MultiselectInput,
    Output,
    SecretStrInput,
    SliderInput,
    StrInput,
)
from langflow.schema import Data


@dataclass
class NewConfigInput:
    """Input structure for creating new evaluation configurations with conditional logic."""

    functionality: str = "create"
    fields: dict[str, dict] = field(
        default_factory=lambda: {
            "data": {
                "node": {
                    "name": "create_config",
                    "description": "Create a new evaluation configuration for the selected target",
                    "display_name": "Create New Config",
                    "field_order": [
                        "01_config_name",
                        "02_evaluation_type",
                        "03_task_name",
                        "04_hf_token",
                        "05_few_shot_examples",
                        "06_batch_size",
                        "07_bootstrap_iterations",
                        "08_limit",
                        "09_top_p",
                        "10_top_k",
                        "11_temperature",
                        "12_tokens_to_generate",
                        "13_scorers",
                        "14_num_samples",
                    ],
                    "template": {
                        "01_config_name": StrInput(
                            name="config_name",
                            display_name="Config Name",
                            info="Name for the new evaluation configuration (e.g., my-eval-config@v1.0.0)",
                            required=True,
                        ),
                        "02_evaluation_type": DropdownInput(
                            name="evaluation_type",
                            display_name="Evaluation Type",
                            options=["LM Evaluation Harness", "Similarity Metrics"],
                            value="LM Evaluation Harness",
                            required=True,
                            real_time_refresh=True,
                        ),
                        "03_task_name": StrInput(
                            name="task_name",
                            display_name="Task Name",
                            info="Task from https://github.com/EleutherAI/lm-evaluation-harness/tree/v0.4.3/lm_eval/tasks#tasks",
                            value="gsm8k",
                            required=False,
                        ),
                        "04_hf_token": SecretStrInput(
                            name="hf_token",
                            display_name="HuggingFace Token",
                            info="Token for accessing HuggingFace to fetch the evaluation dataset.",
                            required=False,
                        ),
                        "05_few_shot_examples": IntInput(
                            name="few_shot_examples",
                            display_name="Few-shot Examples",
                            info="The number of few-shot examples before the input.",
                            value=5,
                            required=False,
                        ),
                        "06_batch_size": IntInput(
                            name="batch_size",
                            display_name="Batch Size",
                            info="The batch size used for evaluation.",
                            value=16,
                            required=False,
                        ),
                        "07_bootstrap_iterations": IntInput(
                            name="bootstrap_iterations",
                            display_name="Bootstrap Iterations",
                            info="The number of iterations for bootstrap statistics.",
                            value=100000,
                            required=False,
                        ),
                        "08_limit": IntInput(
                            name="limit",
                            display_name="Limit",
                            info=(
                                "Limits the number of documents to evaluate for debugging, "
                                "or limits to X% of documents."
                            ),
                            value=-1,
                            required=False,
                        ),
                        "09_top_p": FloatInput(
                            name="top_p",
                            display_name="Top_p",
                            info=(
                                "Threshold to select from most probable tokens until "
                                "cumulative probability exceeds this value"
                            ),
                            value=0.0,
                            required=False,
                        ),
                        "10_top_k": IntInput(
                            name="top_k",
                            display_name="Top_k",
                            info="The top_k value to be used during generation sampling.",
                            value=1,
                            required=False,
                        ),
                        "11_temperature": SliderInput(
                            name="temperature",
                            display_name="Temperature",
                            range_spec=RangeSpec(min=0.0, max=1.0, step=0.01),
                            value=0.1,
                            info="The temperature to be used during generation sampling (0.0 to 1.0).",
                            required=False,
                        ),
                        "12_tokens_to_generate": IntInput(
                            name="tokens_to_generate",
                            display_name="Max Tokens",
                            info="The maximum number of tokens to generate. Set to 0 for unlimited tokens.",
                            value=1024,
                            required=False,
                        ),
                        "13_scorers": MultiselectInput(
                            name="scorers",
                            display_name="Scorers",
                            info="List of Scorers for evaluation.",
                            options=["accuracy", "bleu", "rouge", "em", "bert", "f1"],
                            value=["accuracy", "bleu", "rouge", "em", "bert", "f1"],
                            required=False,
                        ),
                        "14_num_samples": IntInput(
                            name="num_samples",
                            display_name="Number of Samples",
                            info="Number of samples to run inference on from the input_file.",
                            value=-1,
                            required=False,
                        ),
                    },
                }
            }
        }
    )


class NvidiaEvaluatorComponent(Component):
    display_name = "NeMo Evaluator"
    description = "Evaluate models using NeMo evaluator microservice."
    icon = "NVIDIA"
    name = "NVIDIANeMoEvaluator"
    beta = True

    def __init__(self, *args, **kwargs):
        """Initialize the component with defensive attribute setup."""
        try:
            super().__init__(*args, **kwargs)
            # Initialize any attributes that might be accessed during component setup
            if not hasattr(self, "auth_token"):
                self.auth_token = None
            if not hasattr(self, "base_url"):
                self.base_url = None
            if not hasattr(self, "namespace"):
                self.namespace = None
        except Exception:
            logger.exception("Error during NvidiaEvaluatorComponent initialization")
            # Re-raise the exception to prevent silent failures
            raise

    def get_auth_headers(self):
        """Get authentication headers for API requests."""
        # Defensive check - if auth_token is not set, return basic headers
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

    def convert_datetime_to_string(self, obj):
        """Convert datetime objects to strings for JSON serialization."""
        if isinstance(obj, dict):
            return {k: self.convert_datetime_to_string(v) for k, v in obj.items()}
        if isinstance(obj, list):
            return [self.convert_datetime_to_string(item) for item in obj]
        if hasattr(obj, "isoformat"):  # datetime objects
            return obj.isoformat()
        return obj

    def extract_customized_model_info(self, customized_model_data):
        """Extract model information from customized model Data object.

        Args:
            customized_model_data: Data object from customizer component

        Returns:
            tuple: (model_name, namespace) or (None, None) if extraction fails
        """
        try:
            if not isinstance(customized_model_data, Data):
                self.log("Customized model input must be a Data object")
                return None, None

            data = customized_model_data.data if hasattr(customized_model_data, "data") else customized_model_data

            if not isinstance(data, dict):
                self.log("Customized model data must be a dictionary")
                return None, None

            # Extract output_model from customizer result
            output_model = data.get("output_model")
            if not output_model:
                self.log("Customized model data does not contain 'output_model' field")
                return None, None

            # Parse namespace/name format
            if "/" in output_model:
                namespace, model_name = output_model.split("/", 1)
                self.log(f"Extracted model name: {model_name}, namespace: {namespace}")
                return model_name, namespace

            # If no namespace in output_model, use the model name as-is
            self.log(f"Using output_model as model name: {output_model}")
            return output_model, None  # noqa: TRY300

        except (ValueError, TypeError, AttributeError) as exc:
            self.log(f"Error extracting customized model info: {exc}")
            return None, None

    def normalize_nim_url(self, base_url: str) -> str:
        """Normalize NIM inference URL to avoid duplicate path components.

        Handles cases where the base URL already contains /v1/completions or /v1/chat/completions
        to prevent double-appending of paths.

        Args:
            base_url (str): The base NIM inference URL

        Returns:
            str: Normalized URL ending with /v1/completions
        """
        if not base_url:
            return ""

        # Remove trailing slash
        url = base_url.rstrip("/")

        # If URL already ends with the correct path, return as-is
        if url.endswith("/v1/completions"):
            return url

        # If URL ends with /v1/chat/completions, change it to /v1/completions
        if url.endswith("/v1/chat/completions"):
            return url.replace("/v1/chat/completions", "/v1/completions")

        # If URL contains /v1/chat/completions/v1/completions (the broken case), fix it
        if "/v1/chat/completions/v1/completions" in url:
            return url.replace("/v1/chat/completions/v1/completions", "/v1/completions")

        # Otherwise, append /v1/completions
        return f"{url}/v1/completions"

    # Define initial static inputs - consistent with customizer component
    inputs = [
        SecretStrInput(
            name="auth_token",
            display_name="Authentication Token",
            info="Bearer token for firewall authentication",
            required=True,
        ),
        StrInput(
            name="base_url",
            display_name="Base API URL",
            info="Base URL for the NeMo services (e.g., https://us-west-2.api-dev.ai.datastax.com/nvidia/nemo)",
            required=True,
            value="https://us-west-2.api-dev.ai.datastax.com/nvidia/nemo",
        ),
        StrInput(
            name="inference_model_url",
            display_name="Inference URL",
            info="Base URL for the NIM to run evaluation inference.",
            required=True,
        ),
        StrInput(
            name="namespace",
            display_name="Namespace",
            info="Namespace for the dataset and evaluation model",
            advanced=True,
            value="default",
            required=True,
        ),
        HandleInput(
            name="dataset",
            display_name="Dataset",
            info="Dataset from NeMo Dataset Creator (optional - if not provided, will use Existing Dataset)",
            required=False,
            input_types=["Data"],
        ),
        HandleInput(
            name="customized_model",
            display_name="Customized Model",
            info="Customized model from NeMo Customizer (optional - if not provided, will use Target dropdown)",
            required=False,
            input_types=["Data"],
        ),
        DropdownInput(
            name="target",
            display_name="Evaluation Target",
            info="Select an evaluation target (pre-configured by administrators)",
            options=[],
            refresh_button=True,
            required=True,
            combobox=True,
        ),
        DropdownInput(
            name="config",
            display_name="Evaluation Configuration",
            info="Select an evaluation configuration or create a new one",
            options=[],
            refresh_button=True,
            required=True,
            combobox=True,
            dialog_inputs=asdict(NewConfigInput()),
        ),
        StrInput(
            name="001_tag",
            display_name="Tag name",
            info="Any user-provided value. Generated results are stored in the NeMo Data Store with this name.",
            value="default",
            required=True,
        ),
        DropdownInput(
            name="002_evaluation_type",
            display_name="Evaluation Type",
            info="Select the type of evaluation",
            options=["LM Evaluation Harness", "Similarity Metrics"],
            real_time_refresh=True,  # Ensure dropdown triggers update on change
            required=True,
        ),
        DropdownInput(
            name="existing_dataset",
            display_name="Existing Dataset",
            info="Select an existing dataset from NeMo Data Store to use for evaluation",
            options=[],
            refresh_button=True,
            required=False,
        ),
        BoolInput(
            name="wait_for_completion",
            display_name="Wait for Job Completion",
            info="If True, the component will wait for the job to complete before returning.",
            required=False,
            value=True,
            advanced=True,
        ),
        IntInput(
            name="max_wait_time_minutes",
            display_name="Maximum Wait Time (minutes)",
            info="Maximum time in minutes to wait for job completion. Only applicable if wait_for_completion is True.",
            required=False,
            value=30,
            advanced=True,
        ),
    ]

    outputs = [
        Output(display_name="Evaluation Data", name="job_info", method="evaluate"),
    ]

    # Inputs for LM Evaluation
    lm_evaluation_inputs = [
        SecretStrInput(
            name="100_huggingface_token",
            display_name="HuggingFace Token",
            info="Token for accessing HuggingFace to fet the evaluation dataset.",
            required=True,
        ),
        StrInput(
            name="110_task_name",
            display_name="Task Name",
            info="Task from https://github.com/EleutherAI/lm-evaluation-harness/tree/v0.4.3/lm_eval/tasks#tasks",
            value="gsm8k",
            required=True,
        ),
        IntInput(
            name="112_few_shot_examples",
            display_name="Few-shot Examples",
            info="The number of few-shot examples before the input.",
            advanced=True,
            value=5,
        ),
        IntInput(
            name="113_batch_size",
            display_name="Batch Size",
            info="The batch size used for evaluation.",
            value=16,
        ),
        IntInput(
            name="114_bootstrap_iterations",
            display_name="Bootstrap Iterations",
            info="The number of iterations for bootstrap statistics.",
            advanced=True,
            value=100000,
        ),
        IntInput(
            name="115_limit",
            display_name="Limit",
            info="Limits the number of documents to evaluate for debugging, or limits to X% of documents.",
            advanced=True,
            value=-1,
        ),
        FloatInput(
            name="151_top_p",
            display_name="Top_p",
            info="Threshold to select from most probable tokens until cumulative probability exceeds this value",
            advanced=True,
            value=0.0,
        ),
        IntInput(
            name="152_top_k",
            display_name="Top_k",
            info="The top_k value to be used during generation sampling.",
            value=1,
        ),
        SliderInput(
            name="153_temperature",
            display_name="Temperature",
            range_spec=RangeSpec(min=0.0, max=1.0, step=0.01),
            value=0.1,
            info="The temperature to be used during generation sampling (0.0 to 2.0).",
        ),
        IntInput(
            name="154_tokens_to_generate",
            display_name="Max Tokens",
            advanced=True,
            info="The maximum number of tokens to generate. Set to 0 for unlimited tokens.",
            value=1024,
        ),
    ]

    # Inputs for Similarity Metrics
    custom_evaluation_inputs = [
        IntInput(
            name="350_num_of_samples",
            display_name="Number of Samples",
            info="Number of samples to run inference on from the input_file.",
            value=-1,
            advanced=True,
        ),
        MultiselectInput(
            name="351_scorers",
            display_name="Scorers",
            info="List of Scorers for evaluation.",
            options=["accuracy", "bleu", "rouge", "em", "bert", "f1"],
            value=["accuracy", "bleu", "rouge", "em", "bert", "f1"],
            required=True,
        ),
        DropdownInput(
            name="310_run_inference",
            display_name="Run Inference",
            info="Select 'True' to run inference or 'False' to use a `response` field in the dataset.",
            options=["True", "False"],
            value="True",
            real_time_refresh=True,
            advanced=True,
        ),
        IntInput(
            name="311_tokens_to_generate",
            display_name="Max Tokens",
            advanced=True,
            info="The maximum number of tokens to generate. Set to 0 for unlimited tokens.",
            value=1024,
        ),
        SliderInput(
            name="312_temperature",
            display_name="Temperature",
            range_spec=RangeSpec(min=0.0, max=1.0, step=0.01),
            value=0.1,
            info="The temperature to be used during generation sampling (0.0 to 2.0).",
        ),
        IntInput(
            name="313_top_k",
            display_name="Top_k",
            info="Top_k value for generation sampling.",
            value=1,
            advanced=True,
        ),
    ]

    async def fetch_models(self, base_url: str):
        """Fetch models from the /nemo/v1/models endpoint and return a list of model names."""
        # Defensive checks
        if not base_url:
            if hasattr(self, "log"):
                self.log("Base URL not provided for fetching models")
            return []

        if not hasattr(self, "auth_token") or not self.auth_token:
            if hasattr(self, "log"):
                self.log("Authentication token not set for fetching models")
            return []

        try:
            # Use direct HTTP call to /nemo/v1/models endpoint
            # The base_url already contains the nvidia/nemo part, so we just add /v1/models
            models_url = f"{base_url}/v1/models"
            async with httpx.AsyncClient(timeout=10.0) as client:
                response = await client.get(models_url, headers=self.get_auth_headers())
                response.raise_for_status()

                models_data = response.json()

                # Extract model names from the response
                model_names = []
                for model in models_data.get("data", []):
                    if "name" in model:
                        # Include namespace/name format for clarity
                        namespace = model.get("namespace", "")
                        name = model["name"]
                        if namespace:
                            model_names.append(f"{namespace}/{name}")
                        else:
                            model_names.append(name)

                if hasattr(self, "log"):
                    self.log(f"Successfully fetched {len(model_names)} models from {models_url}")

                return model_names

        except httpx.HTTPStatusError as exc:
            if hasattr(self, "log"):
                self.log(f"HTTP error {exc.response.status_code} while fetching models: {exc.response.text}")
            return []
        except httpx.RequestError as exc:
            if hasattr(self, "log"):
                self.log(f"Request error while fetching models: {exc}")
            return []
        except (ValueError, TypeError) as exc:
            if hasattr(self, "log"):
                self.log(f"Unexpected error while fetching models: {exc}")
            return []

    async def fetch_available_evaluation_targets(self) -> tuple[list[str], list[dict[str, Any]]]:
        """Fetch available evaluation targets with metadata."""
        try:
            nemo_client = self.get_nemo_client()
            response = await nemo_client.evaluation.targets.list(extra_headers=self.get_auth_headers())
            targets = []
            targets_metadata = []

            if hasattr(response, "data") and response.data:
                for target in response.data:
                    target_name = getattr(target, "name", "")
                    target_id = getattr(target, "id", "")
                    target_type = getattr(target, "type", "")
                    target_created = getattr(target, "created", None)
                    target_updated = getattr(target, "updated", None)

                    if target_name:
                        targets.append(target_name)
                        # Build metadata for this target
                        metadata = self._build_target_metadata(target_id, target_type, target_created, target_updated)
                        targets_metadata.append(metadata)

                return targets, targets_metadata
            return targets, targets_metadata  # noqa: TRY300
        except (NeMoMicroservicesError, httpx.HTTPStatusError, httpx.RequestError) as exc:
            logger.warning("Failed to fetch evaluation targets: %s", exc)
            return [], []

    def _build_target_metadata(
        self, target_id: str, target_type: str, created: str | None = None, updated: str | None = None
    ) -> dict[str, Any]:
        """Build metadata dictionary for an evaluation target."""
        metadata = {}

        # Add target ID (shortened for display)
        if target_id:
            # Extract just the last part of the ID for display
            short_id = target_id.split("-")[-1] if "-" in target_id else target_id[:8]
            metadata["id"] = short_id

        # Add target type
        if target_type:
            metadata["type"] = target_type.upper()

        # Add timestamps if available (but don't display them in dropdown)
        if created:
            metadata["created"] = created
        if updated:
            metadata["updated"] = updated

        # Add icon for targets
        metadata["icon"] = "NVIDIA"

        return metadata

    async def fetch_available_evaluation_configs(
        self, target_name: str | None = None
    ) -> tuple[list[str], list[dict[str, Any]]]:
        """Fetch available evaluation configurations with metadata."""
        try:
            nemo_client = self.get_nemo_client()
            response = await nemo_client.evaluation.configs.list(extra_headers=self.get_auth_headers())
            configs = []
            configs_metadata = []

            if hasattr(response, "data") and response.data:
                for config in response.data:
                    config_name = getattr(config, "name", "")
                    config_type = getattr(config, "type", "")
                    config_params = getattr(config, "params", {})
                    config_created = getattr(config, "created", None)
                    config_updated = getattr(config, "updated", None)

                    # If target_name is provided, filter by target (if configs are target-specific)
                    if target_name:
                        # For now, include all configs since evaluation configs may not be target-specific
                        # This can be updated if the API supports target filtering
                        pass

                    if config_name:
                        configs.append(config_name)
                        # Build metadata for this config
                        metadata = self._build_config_metadata(
                            config_type, config_params, config_created, config_updated
                        )
                        configs_metadata.append(metadata)

                return configs, configs_metadata
            return configs, configs_metadata  # noqa: TRY300
        except (NeMoMicroservicesError, httpx.HTTPStatusError, httpx.RequestError) as exc:
            logger.warning("Failed to fetch evaluation configs: %s", exc)
            return [], []

    def _build_config_metadata(
        self, config_type: str, config_params: dict, created: str | None = None, updated: str | None = None
    ) -> dict[str, Any]:
        """Build metadata dictionary for an evaluation configuration."""
        metadata = {}

        # Add evaluation type
        if config_type:
            metadata["type"] = config_type.replace("_", " ").title()  # "Lm Eval Harness", "Similarity Metrics"

        # Add parameters if available - handle both dict and Pydantic model
        if config_params:
            # Convert Pydantic model to dict if needed
            if hasattr(config_params, "model_dump"):
                params_dict = config_params.model_dump()
            elif hasattr(config_params, "dict"):
                params_dict = config_params.dict()
            else:
                params_dict = config_params if isinstance(config_params, dict) else {}

            # Add task name for LM Evaluation Harness
            if config_type == "lm_eval_harness":
                tasks = params_dict.get("tasks", {})
                if tasks:
                    # Get the first task name
                    first_task = next(iter(tasks.keys()), None)
                    if first_task:
                        metadata["task"] = first_task

            # Add scorers for Similarity Metrics
            elif config_type == "custom":
                tasks = params_dict.get("tasks", {})
                if tasks:
                    first_task = next(iter(tasks.values()), {})
                    metrics = first_task.get("metrics", [])
                    if metrics:
                        scorer_names = [metric.get("name", "") for metric in metrics if metric.get("name")]
                        if scorer_names:
                            metadata["scorers"] = ", ".join(scorer_names[:3])  # Show first 3 scorers

        # Add timestamps if available (but don't display them in dropdown)
        if created:
            metadata["created"] = created
        if updated:
            metadata["updated"] = updated

        # Add icon for configurations
        metadata["icon"] = "Settings"

        # Debug logging
        logger.debug("Built config metadata: %s", metadata)

        return metadata

    def clear_dynamic_inputs(self, build_config, saved_values):
        """Clears dynamically added fields by referring to a special marker in build_config."""
        dynamic_fields = build_config.get("_dynamic_fields", [])
        length_dynamic_fields = len(dynamic_fields)
        message = f"Clearing dynamic inputs. Number of fields to remove: {length_dynamic_fields}"
        logger.info(message)

        for field_name in dynamic_fields:
            if field_name in build_config:
                message = f"Removing dynamic field: {field_name}"
                logger.info(message)
                saved_values[field_name] = build_config[field_name].get("value", None)
                del build_config[field_name]

        build_config["_dynamic_fields"] = []

    def add_inputs_with_saved_values(self, build_config, input_definitions, saved_values):
        """Adds inputs to build_config and restores any saved values."""
        for input_def in input_definitions:
            # Check if input_def is already a dict or needs conversion
            input_dict = input_def if isinstance(input_def, dict) else input_def.to_dict()
            input_name = input_dict["name"]
            input_dict["value"] = saved_values.get(input_name, input_dict.get("value"))
            build_config[input_name] = input_dict
            build_config.setdefault("_dynamic_fields", []).append(input_name)

    def add_evaluation_inputs(self, build_config, saved_values, evaluation_type):
        """Adds inputs based on the evaluation type (LM Evaluation or Similarity Metrics)."""
        if evaluation_type == "LM Evaluation Harness":
            self.add_inputs_with_saved_values(build_config, self.lm_evaluation_inputs, saved_values)
        elif evaluation_type == "Similarity Metrics":
            self.add_inputs_with_saved_values(build_config, self.custom_evaluation_inputs, saved_values)

    async def update_build_config(self, build_config, field_value, field_name=None):
        """Updates the component's configuration based on the selected option."""
        try:
            message = f"Updating build config: field_name={field_name}, field_value={field_value}"
            logger.info(message)
            saved_values = {}

            # Defensive check - if auth_token is not set, don't try to fetch data
            if not hasattr(self, "auth_token") or not self.auth_token:
                logger.info("Auth token not set, skipping data fetching")
                if hasattr(self, "log"):
                    self.log("Authentication token not set, please configure component before refreshing data")
                return build_config

            # Handle base_url changes
            if field_name == "base_url":
                # Defensive check for base_url
                if not hasattr(self, "base_url") or not self.base_url:
                    logger.warning("Base URL not set, cannot fetch data")
                    if hasattr(self, "log"):
                        self.log("Base URL not configured, please set Base API URL before refreshing data")
                    return build_config

                base_url = self.base_url.rstrip("/")

                # Fetch evaluation targets
                targets, targets_metadata = await self.fetch_available_evaluation_targets()
                build_config["target"]["options"] = targets
                build_config["target"]["options_metadata"] = targets_metadata

                # Fetch evaluation configs
                configs, configs_metadata = await self.fetch_available_evaluation_configs()
                build_config["config"]["options"] = configs
                build_config["config"]["options_metadata"] = configs_metadata

                # Fetch existing datasets
                existing_datasets = await self.fetch_existing_datasets(base_url)
                build_config["existing_dataset"]["options"] = existing_datasets

                # Debug logging
                logger.debug("Updated build_config for base_url change:")
                logger.debug("  Targets: %s options, %s metadata", len(targets), len(targets_metadata))
                logger.debug("  Configs: %s options, %s metadata", len(configs), len(configs_metadata))
                logger.debug("  Datasets: %s options", len(existing_datasets))

            # Handle target refresh
            elif field_name == "target":
                # Defensive check for base_url
                if not hasattr(self, "base_url") or not self.base_url:
                    logger.warning("Base URL not set, cannot fetch targets")
                    if hasattr(self, "log"):
                        self.log("Base URL not configured, please set Base API URL before refreshing targets")
                    return build_config

                # Fetch evaluation targets
                targets, targets_metadata = await self.fetch_available_evaluation_targets()
                build_config["target"]["options"] = targets
                build_config["target"]["options_metadata"] = targets_metadata

                # Update configs when target changes (if configs are target-specific)
                if field_value:
                    configs, configs_metadata = await self.fetch_available_evaluation_configs(field_value)
                    build_config["config"]["options"] = configs
                    build_config["config"]["options_metadata"] = configs_metadata

                    # Debug logging
                    logger.debug("Updated build_config for target change '%s':", field_value)
                    logger.debug("  Configs: %s options, %s metadata", len(configs), len(configs_metadata))

            # Handle config refresh and dialog input changes
            elif field_name == "config":
                # Handle dialog input changes - follow customizer pattern
                if isinstance(field_value, dict):
                    # Case 1: New config creation
                    if "01_config_name" in field_value:
                        # Handle new config creation if needed
                        pass

                    # Case 2: Update evaluation type options
                    if "02_evaluation_type" in field_value:
                        return self._update_evaluation_type_options(build_config, field_value)
                else:
                    # Fetch configs when config field is refreshed
                    target_value = getattr(self, "target", None)
                    configs, configs_metadata = await self.fetch_available_evaluation_configs(target_value)
                    build_config["config"]["options"] = configs
                    build_config["config"]["options_metadata"] = configs_metadata

                    # Debug logging
                    logger.debug("Updated build_config for config refresh:")
                    logger.debug("  Configs: %s options, %s metadata", len(configs), len(configs_metadata))

            # Handle existing dataset refresh
            elif field_name == "existing_dataset":
                # Defensive check for base_url
                if not hasattr(self, "base_url") or not self.base_url:
                    logger.warning("Base URL not set, cannot fetch datasets")
                    if hasattr(self, "log"):
                        self.log("Base URL not configured, please set Base API URL before refreshing datasets")
                    return build_config

                base_url = self.base_url.rstrip("/")
                # Refresh dataset options for existing dataset dropdown
                logger.info("Refreshing datasets for field: %s", field_name)
                build_config["existing_dataset"]["options"] = await self.fetch_existing_datasets(base_url)
                dataset_options = build_config["existing_dataset"]["options"]
                msg = f"Updated dataset options: {dataset_options}"
                logger.info(msg)
                if hasattr(self, "log"):
                    self.log(f"Refreshed {len(dataset_options)} datasets for evaluation")

            # Handle evaluation type changes (for backward compatibility)
            elif field_name == "002_evaluation_type":
                if hasattr(self, "clear_dynamic_inputs") and hasattr(self, "add_evaluation_inputs"):
                    self.clear_dynamic_inputs(build_config, saved_values)
                    self.add_evaluation_inputs(build_config, saved_values, field_value)
                else:
                    logger.warning("Dynamic input methods not available yet")

            # Handle run_inference changes (for backward compatibility)
            elif field_name == "310_run_inference":
                if hasattr(self, "custom_evaluation_inputs") and hasattr(self, "clear_dynamic_inputs"):
                    run_inference = field_value == "True"
                    # Always include inputs 1, 2, 3, 7, and 8
                    always_included_inputs = self.custom_evaluation_inputs[:3] + self.custom_evaluation_inputs[6:8]
                    self.clear_dynamic_inputs(build_config, saved_values)
                    self.add_inputs_with_saved_values(build_config, always_included_inputs, saved_values)
                    # Conditionally add fields 4 to 6 if Run Inference is True
                    if run_inference:
                        conditional_inputs = self.custom_evaluation_inputs[3:6]
                        self.add_inputs_with_saved_values(build_config, conditional_inputs, saved_values)
                else:
                    logger.warning("Custom evaluation input methods not available yet")

            logger.info("Build config update completed successfully.")
        except (ValueError, AttributeError, ImportError, RuntimeError) as exc:
            # Catch specific exceptions to prevent UI crashes
            error_msg = f"Error during build config update: {exc}"
            logger.exception(error_msg)
            # Instead of raising, just log the error and return the original config
            if hasattr(self, "log"):
                self.log(f"Build config update failed: {error_msg}")
            return build_config
        return build_config

    def _update_evaluation_type_options(self, build_config: dict, field_value: dict) -> dict:
        """Update evaluation type options based on evaluation type selection."""
        # Extract template path for cleaner access
        template = build_config["config"]["dialog_inputs"]["fields"]["data"]["node"]["template"]

        evaluation_type = field_value["02_evaluation_type"]

        if evaluation_type == "LM Evaluation Harness":
            # Enable LM Evaluation Harness fields
            lm_fields = [
                "03_task_name",
                "04_hf_token",
                "05_few_shot_examples",
                "06_batch_size",
                "07_bootstrap_iterations",
                "08_limit",
                "09_top_p",
                "10_top_k",
                "11_temperature",
                "12_tokens_to_generate",
            ]

            for field_key in lm_fields:
                if field_key in template:
                    field_config = template[field_key]
                    field_config.update(
                        {
                            "readonly": False,
                            "required": field_key in ["03_task_name", "04_hf_token"],
                            "placeholder": "",
                        }
                    )

            # Disable Similarity Metrics fields
            sm_fields = ["13_scorers", "14_num_samples"]
            for field_key in sm_fields:
                if field_key in template:
                    field_config = template[field_key]
                    field_config.update(
                        {
                            "readonly": True,
                            "required": False,
                            "placeholder": "Only available for Similarity Metrics",
                            "value": None,
                        }
                    )

        elif evaluation_type == "Similarity Metrics":
            # Enable Similarity Metrics fields
            sm_fields = ["13_scorers", "14_num_samples"]
            for field_key in sm_fields:
                if field_key in template:
                    field_config = template[field_key]
                    field_config.update(
                        {
                            "readonly": False,
                            "required": field_key == "13_scorers",
                            "placeholder": "",
                        }
                    )

            # Disable LM Evaluation Harness fields
            lm_fields = [
                "03_task_name",
                "04_hf_token",
                "05_few_shot_examples",
                "06_batch_size",
                "07_bootstrap_iterations",
                "08_limit",
                "09_top_p",
                "10_top_k",
                "11_temperature",
                "12_tokens_to_generate",
            ]
            for field_key in lm_fields:
                if field_key in template:
                    field_config = template[field_key]
                    field_config.update(
                        {
                            "readonly": True,
                            "required": False,
                            "placeholder": "Only available for LM Evaluation Harness",
                            "value": None,
                        }
                    )

        return build_config

    async def evaluate(self) -> Data:
        evaluation_type = getattr(self, "002_evaluation_type", "LM Evaluation Harness")

        if not self.namespace:
            error_msg = "Missing namespace"
            raise ValueError(error_msg)

        # Check if we have target and config selections (new approach)
        has_target = hasattr(self, "target") and self.target
        has_config = hasattr(self, "config") and self.config

        if has_target and has_config:
            # Use new target/config approach
            self.log("Using target and config selection approach")

            try:
                nemo_client = self.get_nemo_client()

                # Handle configuration (existing or new)
                if isinstance(self.config, dict):
                    # Create new configuration
                    config_id = await self._create_new_evaluation_config(self.config)
                else:
                    # Use existing configuration
                    config_id = await self._get_config_id(self.config)

                # Handle target (existing only - no creation)
                target_id = await self._get_target_id(self.target)

                # Validate compatibility
                await self._validate_target_config_compatibility(target_id, config_id)

                # Create job with existing target and config IDs
                response = await nemo_client.evaluation.jobs.create(
                    namespace=self.namespace,
                    config=config_id,
                    target=target_id,
                    extra_headers=self.get_auth_headers(),
                )

                # Process the response
                result_dict = response.model_dump()
                result_dict = self.convert_datetime_to_string(result_dict)

                # Log the successful response
                formatted_result = json.dumps(result_dict, indent=2)
                msg = f"Received successful evaluation response: {formatted_result}"
                self.log(msg)

                # Extract job ID for wait-for-completion logic
                id_value = result_dict["id"]
                self.log(f"Evaluation job created successfully with ID: {id_value}")

                # Handle wait for completion
                wait_for_completion = getattr(self, "wait_for_completion", False)
                if wait_for_completion:
                    try:
                        max_wait_time = getattr(self, "max_wait_time_minutes", 30)
                        final_job_result = await self.wait_for_job_completion(
                            job_id=id_value, max_wait_time_minutes=max_wait_time
                        )
                        result_dict.update(final_job_result)
                        self.log(f"Evaluation job {id_value} completed successfully!")
                    except TimeoutError:
                        self.log(f"Evaluation job {id_value} did not complete within {max_wait_time} minutes timeout")
                    except ValueError as exc:
                        error_msg = f"Evaluation job {id_value} failed: {exc}"
                        raise ValueError(error_msg) from exc

                return Data(data=result_dict)

            except (NeMoMicroservicesError, httpx.HTTPStatusError, httpx.RequestError) as exc:
                error_msg = f"Error during evaluation job creation: {exc}"
                self.log(error_msg)
                raise ValueError(error_msg) from exc

        else:
            # Fallback to existing dynamic creation approach
            self.log("Using fallback dynamic creation approach")

            # Prioritize customized_model input over target dropdown
            customized_model_input = getattr(self, "customized_model", None)
            effective_namespace = self.namespace
            if customized_model_input is not None and isinstance(customized_model_input, Data):
                model_name, customized_namespace = self.extract_customized_model_info(customized_model_input)
                if model_name:
                    self.log(f"Using customized model: {model_name}")
                    if customized_namespace:
                        effective_namespace = customized_namespace
                        self.log(f"Using customized model namespace: {effective_namespace}")
                else:
                    self.log("Failed to extract model name from customized model input")
                    model_name = getattr(self, "000_llm_name", "")
                    if not model_name:
                        error_msg = "Refresh and select the model name to be evaluated"
                        raise ValueError(error_msg)
                    self.log(f"Using model from dropdown: {model_name}")
            else:
                model_name = getattr(self, "000_llm_name", "")
                if not model_name:
                    error_msg = "Refresh and select the model name to be evaluated"
                    raise ValueError(error_msg)
                self.log(f"Using model from dropdown: {model_name}")

            if not self.base_url:
                error_msg = "Missing base URL"
                raise ValueError(error_msg)

            base_url = self.base_url.rstrip("/")

            # Create the evaluation using SDK pattern like customizer
            try:
                nemo_client = self.get_nemo_client()

                if evaluation_type == "LM Evaluation Harness":
                    # Create LM evaluation config and target, then create job with IDs
                    config_data, target_data = await self._prepare_lm_evaluation_data(
                        base_url, effective_namespace, model_name
                    )

                    # Create config first
                    config_response = await nemo_client.evaluation.configs.create(
                        type=config_data["type"],
                        namespace=config_data["namespace"],
                        tasks=config_data["tasks"],
                        params=config_data["params"],
                        extra_headers={"Authorization": f"Bearer {self.auth_token}"},
                    )
                    config_id = config_response.id
                    self.log(f"Created evaluation config with ID: {config_id}")

                    # Debug log the config structure
                    formatted_config = json.dumps(config_data, indent=2, default=str)
                    self.log(f"Config data sent: {formatted_config}")

                    # Create target
                    target_response = await nemo_client.evaluation.targets.create(
                        type=target_data["type"],
                        namespace=target_data["namespace"],
                        model=target_data["model"],
                        extra_headers={"Authorization": f"Bearer {self.auth_token}"},
                    )
                    target_id = target_response.id
                    self.log(f"Created evaluation target with ID: {target_id}")

                    # Create job with config and target IDs
                    response = await nemo_client.evaluation.jobs.create(
                        namespace=config_data["namespace"],
                        config=config_id,
                        target=target_id,
                        extra_headers={"Authorization": f"Bearer {self.auth_token}"},
                    )

                elif evaluation_type == "Similarity Metrics":
                    # Create custom evaluation config and target, then create job with IDs
                    config_data, target_data = await self._prepare_custom_evaluation_data(
                        base_url, effective_namespace, model_name
                    )

                    # Create config first
                    config_response = await nemo_client.evaluation.configs.create(
                        type=config_data["type"],
                        namespace=config_data["namespace"],
                        tasks=config_data["tasks"],
                        params=config_data["params"],
                        extra_headers={"Authorization": f"Bearer {self.auth_token}"},
                    )
                    config_id = config_response.id
                    self.log(f"Created evaluation config with ID: {config_id}")

                    # Debug log the config structure
                    formatted_config = json.dumps(config_data, indent=2, default=str)
                    self.log(f"Config data sent: {formatted_config}")

                    # Create target
                    target_response = await nemo_client.evaluation.targets.create(
                        type=target_data["type"],
                        namespace=target_data["namespace"],
                        model=target_data["model"],
                        extra_headers={"Authorization": f"Bearer {self.auth_token}"},
                    )
                    target_id = target_response.id
                    self.log(f"Created evaluation target with ID: {target_id}")

                    # Create job with config and target IDs
                    response = await nemo_client.evaluation.jobs.create(
                        namespace=config_data["namespace"],
                        config=config_id,
                        target=target_id,
                        extra_headers={"Authorization": f"Bearer {self.auth_token}"},
                    )
                else:
                    error_msg = f"Unsupported evaluation type: {evaluation_type}"
                    raise ValueError(error_msg)

                # Process the response
                result_dict = response.model_dump()

                # Convert datetime objects to strings for JSON serialization
                result_dict = self.convert_datetime_to_string(result_dict)

                # Log the successful response
                formatted_result = json.dumps(result_dict, indent=2)
                msg = f"Received successful evaluation response: {formatted_result}"
                self.log(msg)

                # Extract job ID for wait-for-completion logic
                id_value = result_dict["id"]
                self.log(f"Evaluation job created successfully with ID: {id_value}")

                # Check if we should wait for job completion
                wait_for_completion = getattr(self, "wait_for_completion", False)
                logger.info("Wait for completion setting: %s", wait_for_completion)
                if wait_for_completion:
                    logger.info("Wait for completion enabled. Waiting for evaluation job %s to complete...", id_value)
                    try:
                        max_wait_time = getattr(self, "max_wait_time_minutes", 30)
                        logger.info("Starting wait_for_job_completion with max_wait_time: %s", max_wait_time)
                        # Wait for job completion
                        final_job_result = await self.wait_for_job_completion(
                            job_id=id_value, max_wait_time_minutes=max_wait_time
                        )
                        # Update result_dict with final job status
                        result_dict.update(final_job_result)
                        logger.info("Evaluation job %s completed successfully!", id_value)
                        self.log(f"Evaluation job {id_value} completed successfully!")
                    except TimeoutError as exc:
                        logger.warning("Evaluation job %s did not complete within timeout: %s", id_value, exc)
                        self.log(f"Evaluation job {id_value} did not complete within {max_wait_time} minutes timeout")
                        # Continue with the original result (job created but not completed)
                    except ValueError as exc:
                        logger.exception("Evaluation job %s failed", id_value)
                        self.log(f"Evaluation job {id_value} failed: {exc}")
                        # Re-raise the ValueError to indicate job failure
                        error_msg = f"Evaluation job {id_value} failed: {exc}"
                        raise ValueError(error_msg) from exc
                    except (asyncio.CancelledError, RuntimeError, OSError):
                        logger.exception("Unexpected error while waiting for evaluation job completion")
                        self.log(f"Unexpected error while waiting for evaluation job {id_value} completion")
                        # Continue with the original result
                else:
                    logger.info("Wait for completion disabled. Evaluation job %s created successfully.", id_value)

                return Data(data=result_dict)

            except NeMoMicroservicesError as exc:
                error_msg = f"NeMo microservices error during evaluation job creation: {exc}"
                self.log(error_msg, name="NeMoEvaluatorComponent")
                raise ValueError(error_msg) from exc

            except (httpx.HTTPStatusError, httpx.RequestError) as exc:
                error_msg = f"HTTP error during evaluation job creation: {exc}"
                self.log(error_msg, name="NeMoEvaluatorComponent")
                raise ValueError(error_msg) from exc

            except Exception as exc:
                error_msg = f"Unexpected error during evaluation job creation: {exc}"
                self.log(error_msg)
                raise ValueError(error_msg) from exc

    async def _prepare_lm_evaluation_data(self, base_url: str, namespace: str, model_name: str) -> tuple:
        """Prepare LM evaluation config and target data for SDK."""
        # Check if we have a dataset input to determine effective namespace
        dataset_input = getattr(self, "dataset", None)
        effective_namespace = namespace

        if dataset_input is not None and isinstance(dataset_input, Data):
            # Extract dataset information from the provided dataset
            dataset_data = dataset_input.data if hasattr(dataset_input, "data") else dataset_input
            if isinstance(dataset_data, dict):
                dataset_namespace = dataset_data.get("namespace")
                if dataset_namespace:
                    effective_namespace = dataset_namespace
                    self.log(f"Using dataset namespace for LM evaluation: {effective_namespace}")

        # Create target first
        target_data = await self._create_evaluation_target(None, base_url, effective_namespace, model_name)

        # Get required parameters
        hf_token = getattr(self, "100_huggingface_token", None)
        if not hf_token:
            error_msg = "Missing hf token"
            raise ValueError(error_msg)

        # Create config data in SDK format
        config_data = {
            "type": "lm_eval_harness",
            "namespace": effective_namespace,
            "tasks": {
                getattr(self, "110_task_name", "gsm8k"): {
                    "params": {
                        "num_fewshot": getattr(self, "112_few_shot_examples", 5),
                        "batch_size": getattr(self, "113_batch_size", 16),
                        "bootstrap_iters": getattr(self, "114_bootstrap_iterations", 100000),
                        "limit": getattr(self, "115_limit", -1),
                    },
                }
            },
            "params": {
                "hf_token": hf_token,
                "use_greedy": True,
                "top_p": getattr(self, "151_top_p", 0.0),
                "top_k": getattr(self, "152_top_k", 1),
                "temperature": getattr(self, "153_temperature", 0.0),
                "stop": [],
                "tokens_to_generate": getattr(self, "154_tokens_to_generate", 1024),
            },
        }

        return config_data, target_data

    async def _prepare_custom_evaluation_data(self, base_url: str, namespace: str, model_name: str) -> tuple:
        """Prepare custom evaluation config and target data for SDK."""
        # Check if we have a dataset input or existing dataset selection
        dataset_input = getattr(self, "dataset", None)
        existing_dataset = getattr(self, "existing_dataset", None)

        # Priority: 1. Dataset connection, 2. Existing dataset selection
        if dataset_input is not None:
            # Extract dataset information from the provided dataset
            if not isinstance(dataset_input, Data):
                error_msg = "Dataset input must be a Data object"
                raise ValueError(error_msg)

            dataset_data = dataset_input.data if hasattr(dataset_input, "data") else dataset_input

            if not isinstance(dataset_data, dict):
                error_msg = "Dataset data must be a dictionary"
                raise ValueError(error_msg)

            # Extract required fields from dataset
            dataset_name = dataset_data.get("name")
            dataset_namespace = dataset_data.get("namespace")

            if not dataset_name:
                error_msg = "Dataset must contain 'name' field"
                raise ValueError(error_msg)

            if not dataset_namespace:
                error_msg = "Dataset must contain 'namespace' field"
                raise ValueError(error_msg)

            # Use dataset namespace if different from component namespace
            effective_namespace = dataset_namespace
            repo_id = f"{effective_namespace}/{dataset_name}"
            self.log(f"Using dataset connection: {dataset_name} from namespace: {effective_namespace}")

        elif existing_dataset:
            # Use selected existing dataset
            dataset_name = existing_dataset
            effective_namespace = namespace
            repo_id = f"{effective_namespace}/{dataset_name}"
            self.log(f"Using existing dataset: {dataset_name} from namespace: {effective_namespace}")

        else:
            # No dataset provided - require either connection or existing dataset
            error_msg = "Either provide a dataset connection or select an existing dataset " "to run evaluation"
            raise ValueError(error_msg)

        # Use the file path with the repo ID
        input_file = f"hf://datasets/{repo_id}/input.jsonl"

        # Handle run_inference
        run_inference = getattr(self, "310_run_inference", "True").lower() == "true"
        output_file = None if run_inference else f"hf://datasets/{repo_id}/output.jsonl"

        # Create target
        target_data = await self._create_evaluation_target(output_file, base_url, effective_namespace, model_name)

        # Create metrics in SDK format - use string-check for all to ensure consistent return types
        scores = getattr(self, "351_scorers", ["accuracy", "bleu", "rouge", "em", "bert", "f1"])
        metrics_dict = {}
        for score in scores:
            metric_name = score.lower()
            # Use string-check for all metrics to ensure consistent result format
            # This should avoid the model_dump() error while maintaining the correct check parameter format
            if metric_name == "em":
                metrics_dict["exact_match"] = {
                    "type": "string-check",
                    "params": {"check": ["{{response}}", "==", "{{item.ideal_response}}"]},
                }
            else:
                # Use string-check for all other metrics with proper 3-element check array
                metrics_dict[metric_name] = {
                    "type": "string-check",
                    "params": {"check": ["{{response}}", "==", "{{item.ideal_response}}"]},
                }

        # Create config data in SDK format
        config_data = {
            "type": "custom",
            "namespace": effective_namespace,
            "params": {"parallelism": 8},
            "tasks": {
                "default_task": {
                    "type": "completion",
                    "params": {"template": {"prompt": "{{item.prompt}}"}},
                    "metrics": metrics_dict,
                    "dataset": {"files_url": input_file},
                }
            },
        }

        return config_data, target_data

    async def _create_new_evaluation_config(self, config_data: dict) -> str:
        """Create a new evaluation configuration and return its ID."""
        try:
            nemo_client = self.get_nemo_client()

            # Extract config parameters from dialog data
            evaluation_type = config_data.get("02_evaluation_type", "LM Evaluation Harness")

            if evaluation_type == "LM Evaluation Harness":
                # Build LM Evaluation Harness config
                task_name = config_data.get("03_task_name", "gsm8k")
                hf_token = config_data.get("04_hf_token")

                if not hf_token:
                    error_msg = "HuggingFace token is required for LM Evaluation Harness"
                    raise ValueError(error_msg)

                config_params = {
                    "hf_token": hf_token,
                    "use_greedy": True,
                    "top_p": config_data.get("09_top_p", 0.0),
                    "top_k": config_data.get("10_top_k", 1),
                    "temperature": config_data.get("11_temperature", 0.0),
                    "stop": [],
                    "tokens_to_generate": config_data.get("12_tokens_to_generate", 1024),
                }

                # Create config
                response = await nemo_client.evaluation.configs.create(
                    type="lm_eval_harness",
                    namespace=getattr(self, "namespace", "default"),
                    tasks={
                        task_name: {
                            "params": {
                                "num_fewshot": config_data.get("05_few_shot_examples", 5),
                                "batch_size": config_data.get("06_batch_size", 16),
                                "bootstrap_iters": config_data.get("07_bootstrap_iterations", 100000),
                                "limit": config_data.get("08_limit", -1),
                            },
                        }
                    },
                    params=config_params,
                    extra_headers=self.get_auth_headers(),
                )

            elif evaluation_type == "Similarity Metrics":
                # Build Similarity Metrics config
                scorers = config_data.get("13_scorers", ["accuracy", "bleu", "rouge", "em", "bert", "f1"])

                # Create metrics in SDK format
                metrics_dict = {}
                for score in scorers:
                    metric_name = score.lower()
                    if metric_name == "em":
                        metrics_dict["exact_match"] = {
                            "type": "string-check",
                            "params": {"check": ["{{response}}", "==", "{{item.ideal_response}}"]},
                        }
                    else:
                        metrics_dict[metric_name] = {
                            "type": "string-check",
                            "params": {"check": ["{{response}}", "==", "{{item.ideal_response}}"]},
                        }

                # Create config
                response = await nemo_client.evaluation.configs.create(
                    type="custom",
                    namespace=getattr(self, "namespace", "default"),
                    params={"parallelism": 8},
                    tasks={
                        "default_task": {
                            "type": "completion",
                            "params": {"template": {"prompt": "{{item.prompt}}"}},
                            "metrics": metrics_dict,
                            "dataset": {"files_url": "placeholder"},  # Will be updated when used
                        }
                    },
                    extra_headers=self.get_auth_headers(),
                )

            else:
                error_msg = f"Unsupported evaluation type: {evaluation_type}"
                raise ValueError(error_msg)

            config_id = response.id
            self.log(f"Created new evaluation config with ID: {config_id}")
            return config_id

        except (NeMoMicroservicesError, httpx.HTTPStatusError, httpx.RequestError) as exc:
            error_msg = f"Error creating evaluation config: {exc}"
            self.log(error_msg)
            raise ValueError(error_msg) from exc

    async def _get_target_id(self, target_name: str) -> str:
        """Get target ID from target name."""
        try:
            nemo_client = self.get_nemo_client()
            response = await nemo_client.evaluation.targets.list(extra_headers=self.get_auth_headers())
            if hasattr(response, "data") and response.data:
                for target in response.data:
                    if getattr(target, "name", "") == target_name:
                        return getattr(target, "id", target_name)
                # Not found, assume already an ID
                return target_name
            # No data, assume already an ID
            return target_name
        except (NeMoMicroservicesError, httpx.HTTPStatusError, httpx.RequestError) as exc:
            logger.warning("Failed to get target ID for %s: %s", target_name, exc)
            return target_name

    async def _get_config_id(self, config_name: str) -> str:
        """Get config ID from config name."""
        try:
            nemo_client = self.get_nemo_client()
            response = await nemo_client.evaluation.configs.list(extra_headers=self.get_auth_headers())
            if hasattr(response, "data") and response.data:
                for config in response.data:
                    if getattr(config, "name", "") == config_name:
                        return getattr(config, "id", config_name)
                # Not found, assume already an ID
                return config_name
            # No data, assume already an ID
            return config_name
        except (NeMoMicroservicesError, httpx.HTTPStatusError, httpx.RequestError) as exc:
            logger.warning("Failed to get config ID for %s: %s", config_name, exc)
            return config_name

    async def _validate_target_config_compatibility(self, target_id: str, config_id: str):
        """Validate that the selected config is compatible with the selected target."""
        try:
            nemo_client = self.get_nemo_client()
            # Get target details
            target_response = await nemo_client.evaluation.targets.retrieve(
                target_id=target_id, extra_headers=self.get_auth_headers()
            )
            target_type = getattr(target_response, "type", "unknown")

            # Get config details
            config_response = await nemo_client.evaluation.configs.retrieve(
                config_id=config_id, extra_headers=self.get_auth_headers()
            )
            config_type = getattr(config_response, "type", "unknown")

            # Basic compatibility check (can be enhanced based on specific requirements)
            if target_type == "model" and config_type in ["lm_eval_harness", "custom"]:
                # Model targets should work with both evaluation types
                pass
            else:
                logger.warning("Target type %s and config type %s compatibility not verified", target_type, config_type)

        except ValueError:
            raise
        except (NeMoMicroservicesError, httpx.HTTPStatusError, httpx.RequestError) as exc:
            logger.warning("Could not validate target-config compatibility: %s", exc)

    async def _create_evaluation_target(self, output_file, base_url: str, namespace: str, model_name: str):  # noqa: ARG002
        """Create evaluation target using SDK and return the data for job creation."""
        try:
            if output_file:
                # Target with cached outputs
                model_data = {"cached_outputs": {"files_url": output_file}}
            else:
                # Target with API endpoint
                if not self.inference_model_url:
                    error_msg = "Provide the nim url for evaluation inference to be processed"
                    raise ValueError(error_msg)
                model_data = {
                    "api_endpoint": {
                        "url": self.normalize_nim_url(self.inference_model_url),
                        "model_id": model_name,
                    }
                }

            # Create target data for SDK
            target_data = {
                "type": "model",
                "namespace": namespace,
                "model": model_data,
            }

        except Exception as exc:
            error_msg = f"Error creating evaluation target: {exc}"
            self.log(error_msg)
            raise ValueError(error_msg) from exc
        else:
            return target_data

    async def _get_target_object(self, target_id: str, base_url: str):  # noqa: ARG002
        """Get the target object for SDK use."""
        try:
            nemo_client = self.get_nemo_client()
            target_obj = await nemo_client.evaluation.targets.retrieve(
                target_id=target_id,
                namespace=self.namespace,
                extra_headers={"Authorization": f"Bearer {self.auth_token}"},
            )
        except Exception as exc:  # noqa: BLE001
            self.log(f"Warning: Could not retrieve target object {target_id}, using string: {exc}")
            # Fallback to string format if object retrieval fails
            return f"{self.namespace}/{target_id}"
        else:
            return target_obj

    async def create_eval_target(self, output_file, base_url: str) -> str:  # noqa: ARG002
        namespace = self.namespace
        try:
            # Use NeMo client for evaluation target creation
            nemo_client = self.get_nemo_client()

            if output_file:
                # Target with cached outputs
                response = await nemo_client.evaluation.targets.create(
                    type="model",
                    namespace=namespace,
                    model={"cached_outputs": {"files_url": output_file}},
                    extra_headers={"Authorization": f"Bearer {self.auth_token}"},
                )
            else:
                # Target with API endpoint
                response = await nemo_client.evaluation.targets.create(
                    type="model",
                    namespace=namespace,
                    model={
                        "api_endpoint": {
                            "url": self.normalize_nim_url(self.inference_model_url),
                            "model_id": getattr(self, "000_llm_name", ""),
                        }
                    },
                    extra_headers={"Authorization": f"Bearer {self.auth_token}"},
                )

            # Process the response
            result_dict = response.model_dump()

            # Convert datetime objects to strings for JSON serialization
            result_dict = self.convert_datetime_to_string(result_dict)
            formatted_result = json.dumps(result_dict, indent=2)
            self.log(f"Received successful response: {formatted_result}")

            return result_dict.get("id")

        except NeMoMicroservicesError as exc:
            error_msg = f"NeMo microservices error during evaluation target creation: {exc}"
            self.log(error_msg)
            raise ValueError(error_msg) from exc

        except (httpx.TimeoutException, httpx.HTTPStatusError, httpx.RequestError) as exc:
            error_msg = f"HTTP error during evaluation target creation: {exc}"
            self.log(error_msg)
            raise ValueError(error_msg) from exc

        except Exception as exc:
            error_msg = f"Unexpected error during evaluation target creation: {exc}"
            self.log(error_msg)
            raise ValueError(error_msg) from exc

    async def fetch_existing_datasets(self, base_url: str) -> list[str]:  # noqa: ARG002
        """Fetch existing datasets from the NeMo Data Store.

        Args:
            base_url (str): Base URL for the NeMo services

        Returns:
            List of dataset names available for evaluation
        """
        # Defensive checks
        if not hasattr(self, "namespace") or not self.namespace:
            if hasattr(self, "log"):
                self.log("Namespace not set for fetching datasets")
            return []

        if not hasattr(self, "auth_token") or not self.auth_token:
            if hasattr(self, "log"):
                self.log("Authentication token not set for fetching datasets")
            return []

        try:
            # Use NeMo client for dataset fetching
            nemo_client = self.get_nemo_client()
            response = await nemo_client.datasets.list(
                namespace=self.namespace, extra_headers={"Authorization": f"Bearer {self.auth_token}"}
            )

            # Extract dataset names from the response
            return [dataset.name for dataset in response.data if hasattr(dataset, "name") and dataset.name]

        except NeMoMicroservicesError as exc:
            if hasattr(self, "log"):
                self.log(f"NeMo microservices error while fetching datasets: {exc}")
            return []
        except (httpx.RequestError, httpx.HTTPStatusError) as exc:
            if hasattr(self, "log"):
                self.log(f"Error response while requesting datasets: {exc}")
            return []
        except (ValueError, TypeError) as exc:
            if hasattr(self, "log"):
                self.log(f"Unexpected error while fetching datasets: {exc}")
            return []

    async def wait_for_job_completion(
        self, job_id: str, max_wait_time_minutes: int = 30, poll_interval_seconds: int = 15
    ) -> dict:
        """Wait for evaluation job completion with timeout."""
        start_time = time.time()
        max_wait_time_seconds = max_wait_time_minutes * 60

        while True:
            # Check if we've exceeded the maximum wait time
            elapsed_time = time.time() - start_time
            if elapsed_time > max_wait_time_seconds:
                timeout_msg = f"Evaluation job {job_id} did not complete within {max_wait_time_minutes} minutes"
                logger.warning(timeout_msg)
                raise TimeoutError(timeout_msg)

            try:
                # Get job status using NeMo SDK - use the status() method
                nemo_client = self.get_nemo_client()
                response = await nemo_client.evaluation.jobs.status(
                    job_id=job_id, extra_headers=self.get_auth_headers()
                )
                job_status = response.model_dump()

                # Check job status using the correct uppercase values
                status = job_status.get("status", "UNKNOWN")
                logger.info("Evaluation job %s status: %s", job_id, status)

                if status == "COMPLETED":
                    logger.info("Evaluation job %s completed successfully!", job_id)
                    return job_status
                if status == "FAILED":
                    error_msg = f"Evaluation job {job_id} failed with status: {status}"
                    logger.error(error_msg)
                    raise ValueError(error_msg)
                if status == "CANCELLED":
                    error_msg = f"Evaluation job {job_id} was cancelled"
                    logger.error(error_msg)
                    raise ValueError(error_msg)
                if status in ["CREATED", "PENDING", "RUNNING"]:
                    logger.info(
                        "Evaluation job %s is still %s. Waiting %s seconds...",
                        job_id,
                        status,
                        poll_interval_seconds,
                    )
                    await asyncio.sleep(poll_interval_seconds)
                else:
                    logger.warning(
                        "Unknown evaluation job status: %s. Waiting %s seconds...",
                        status,
                        poll_interval_seconds,
                    )
                    await asyncio.sleep(poll_interval_seconds)

            except NeMoMicroservicesError:
                logger.exception("NeMo microservices error while checking evaluation job status")
                await asyncio.sleep(poll_interval_seconds)
            except (asyncio.CancelledError, RuntimeError, OSError):
                logger.exception("Unexpected error while checking evaluation job status")
                await asyncio.sleep(poll_interval_seconds)

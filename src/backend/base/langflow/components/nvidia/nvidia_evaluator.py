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
class LMEvalHarnessConfigInput:
    """Input structure for LM Evaluation Harness configuration."""

    functionality: str = "create"
    fields: dict[str, dict] = field(
        default_factory=lambda: {
            "data": {
                "node": {
                    "name": "create_lm_eval_config",
                    "description": "Create a new LM Evaluation Harness configuration",
                    "display_name": "Create LM Evaluation Config",
                    "field_order": [
                        "01_evaluation_type",
                        "02_config_name",
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
                    ],
                    "template": {
                        "01_evaluation_type": DropdownInput(
                            name="evaluation_type",
                            display_name="Evaluation Type",
                            options=["LM Evaluation Harness", "Similarity Metrics", "Custom Evaluation"],
                            value="LM Evaluation Harness",
                            required=True,
                            real_time_refresh=True,
                        ),
                        "02_config_name": StrInput(
                            name="config_name",
                            display_name="Config Name",
                            info="Name for the new evaluation configuration (e.g., my-eval-config@v1.0.0)",
                            required=True,
                        ),
                        "03_task_name": StrInput(
                            name="task_name",
                            display_name="Task Name",
                            info="Task from https://github.com/EleutherAI/lm-evaluation-harness/tree/v0.4.3/lm_eval/tasks#tasks",
                            value="gsm8k",
                            required=False,
                        ),
                        "04_hf_token": StrInput(
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
                    },
                }
            }
        }
    )


@dataclass
class SimilarityMetricsConfigInput:
    """Input structure for Similarity Metrics configuration."""

    functionality: str = "create"
    fields: dict[str, dict] = field(
        default_factory=lambda: {
            "data": {
                "node": {
                    "name": "create_similarity_config",
                    "description": "Create a new Similarity Metrics configuration",
                    "display_name": "Create Similarity Metrics Config",
                    "field_order": [
                        "01_evaluation_type",
                        "02_config_name",
                        "03_scorers",
                        "04_num_samples",
                    ],
                    "template": {
                        "01_evaluation_type": DropdownInput(
                            name="evaluation_type",
                            display_name="Evaluation Type",
                            options=["LM Evaluation Harness", "Similarity Metrics", "Custom Evaluation"],
                            value="Similarity Metrics",
                            required=True,
                            real_time_refresh=True,
                        ),
                        "02_config_name": StrInput(
                            name="config_name",
                            display_name="Config Name",
                            info="Name for the new evaluation configuration (e.g., my-eval-config@v1.0.0)",
                            required=True,
                        ),
                        "03_scorers": MultiselectInput(
                            name="scorers",
                            display_name="Scorers",
                            info="List of Scorers for evaluation.",
                            options=["accuracy", "bleu", "rouge", "em", "bert", "f1"],
                            value=["accuracy", "bleu", "rouge", "em", "bert", "f1"],
                            required=True,
                        ),
                        "04_num_samples": IntInput(
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


@dataclass
class CustomEvaluationConfigInput:
    """Input structure for Custom Evaluation configuration."""

    functionality: str = "create"
    fields: dict[str, dict] = field(
        default_factory=lambda: {
            "data": {
                "node": {
                    "name": "create_custom_config",
                    "description": "Create a new Custom Evaluation configuration",
                    "display_name": "Create Custom Evaluation Config",
                    "field_order": [
                        "01_evaluation_type",
                        "02_config_name",
                        "03_evaluation_prompt",
                        "04_metrics",
                        "05_batch_size",
                    ],
                    "template": {
                        "01_evaluation_type": DropdownInput(
                            name="evaluation_type",
                            display_name="Evaluation Type",
                            options=["LM Evaluation Harness", "Similarity Metrics", "Custom Evaluation"],
                            value="Custom Evaluation",
                            required=True,
                            real_time_refresh=True,
                        ),
                        "02_config_name": StrInput(
                            name="config_name",
                            display_name="Config Name",
                            info="Name for the new evaluation configuration (e.g., my-eval-config@v1.0.0)",
                            required=True,
                        ),
                        "03_evaluation_prompt": StrInput(
                            name="evaluation_prompt",
                            display_name="Evaluation Prompt",
                            info=(
                                "The prompt template for evaluation (use {{item.prompt}} for input, "
                                "{{response}} for model output)"
                            ),
                            value="Evaluate the following response: {{response}}",
                            required=True,
                        ),
                        "04_metrics": MultiselectInput(
                            name="metrics",
                            display_name="Metrics",
                            info="List of metrics to evaluate.",
                            options=["accuracy", "bleu", "rouge", "em", "bert", "f1", "custom"],
                            value=["accuracy"],
                            required=True,
                        ),
                        "05_batch_size": IntInput(
                            name="batch_size",
                            display_name="Batch Size",
                            info="The batch size used for evaluation.",
                            value=16,
                            required=False,
                        ),
                    },
                }
            }
        }
    )


# Default dialog for evaluation type selection
@dataclass
class EvaluationTypeSelectionInput:
    """Input structure for evaluation type selection."""

    functionality: str = "create"
    fields: dict[str, dict] = field(
        default_factory=lambda: {
            "data": {
                "node": {
                    "name": "select_evaluation_type",
                    "description": "Select the type of evaluation configuration to create",
                    "display_name": "Select Evaluation Type",
                    "field_order": [
                        "01_evaluation_type",
                    ],
                    "template": {
                        "01_evaluation_type": DropdownInput(
                            name="evaluation_type",
                            display_name="Evaluation Type",
                            options=["LM Evaluation Harness", "Similarity Metrics", "Custom Evaluation"],
                            value="LM Evaluation Harness",
                            required=True,
                            real_time_refresh=True,
                        ),
                    },
                }
            }
        }
    )


@dataclass
class DynamicEvaluationConfigInput:
    """Input structure for dynamic evaluation configuration with always-visible type selector."""

    functionality: str = "create"
    fields: dict[str, dict] = field(
        default_factory=lambda: {
            "data": {
                "node": {
                    "name": "create_dynamic_evaluation_config",
                    "description": "Create a new evaluation configuration with dynamic fields",
                    "display_name": "Create Evaluation Config",
                    "field_order": [
                        "01_evaluation_type",
                        "02_config_name",
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
                        "15_evaluation_prompt",
                        "16_metrics",
                    ],
                    "template": {
                        "01_evaluation_type": DropdownInput(
                            name="evaluation_type",
                            display_name="Evaluation Type",
                            options=["LM Evaluation Harness", "Similarity Metrics", "Custom Evaluation"],
                            value="LM Evaluation Harness",
                            required=True,
                            real_time_refresh=True,
                        ),
                        "02_config_name": StrInput(
                            name="config_name",
                            display_name="Config Name",
                            info="Name for the new evaluation configuration (e.g., my-eval-config@v1.0.0)",
                            required=True,
                        ),
                        # LM Evaluation Harness fields
                        "03_task_name": StrInput(
                            name="task_name",
                            display_name="Task Name",
                            info="Task from https://github.com/EleutherAI/lm-evaluation-harness/tree/v0.4.3/lm_eval/tasks#tasks",
                            value="gsm8k",
                            required=False,
                        ),
                        "04_hf_token": StrInput(
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
                        # Similarity Metrics fields
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
                        # Custom Evaluation fields
                        "15_evaluation_prompt": StrInput(
                            name="evaluation_prompt",
                            display_name="Evaluation Prompt",
                            info=(
                                "The prompt template for evaluation (use {{item.prompt}} for input, "
                                "{{response}} for model output)"
                            ),
                            value="Evaluate the following response: {{response}}",
                            required=False,
                        ),
                        "16_metrics": MultiselectInput(
                            name="metrics",
                            display_name="Metrics",
                            info="List of metrics to evaluate.",
                            options=["accuracy", "bleu", "rouge", "em", "bert", "f1", "custom"],
                            value=["accuracy"],
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
            # Initialize dialog state
            self._reset_dialog_state()
        except Exception:
            logger.exception("Error during NvidiaEvaluatorComponent initialization")
            # Re-raise the exception to prevent silent failures
            raise

    def __getattribute__(self, name):
        """Override __getattribute__ to always reset dialog state when component is accessed."""
        if name == "inputs":
            logger.debug("Component inputs accessed - resetting dialog state")
            self._reset_dialog_state()
        return super().__getattribute__(name)

    def build(self, *args, **kwargs):
        """Override build to always reset dialog state when component is built."""
        logger.debug("Component build called - resetting dialog state")
        self._reset_dialog_state()
        return super().build(*args, **kwargs)

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

            # First try to extract from output_model field (primary method)
            output_model = data.get("output_model")
            if output_model:
                # Parse namespace/name format
                if "/" in output_model:
                    namespace, model_name = output_model.split("/", 1)
                    self.log(f"Extracted model name: {model_name}, namespace: {namespace}")
                    return model_name, namespace

                # If no namespace in output_model, use the model name as-is
                self.log(f"Using output_model as model name: {output_model}")
                return output_model, None

            # Fallback: try to extract from individual fields
            model_name = data.get("model_name")
            namespace = data.get("namespace")

            if model_name:
                self.log(f"Extracted model name: {model_name}, namespace: {namespace}")
                return model_name, namespace
            self.log("Customized model data does not contain 'output_model' or 'model_name' field")
            return None, None  # noqa: TRY300

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
            dialog_inputs=asdict(DynamicEvaluationConfigInput()),
        ),
        DropdownInput(
            name="existing_dataset",
            display_name="Existing Dataset",
            info="Select an existing dataset from NeMo Data Store to use for evaluation",
            options=[],
            refresh_button=True,
            required=False,
            combobox=True,
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

    async def update_build_config(self, build_config, field_value, field_name=None):
        """Update the build configuration based on field changes."""
        try:
            logger.debug(
                f"update_build_config called - field_name: {field_name}, "
                f"field_value: {field_value}, type: {type(field_value)}"
            )

            # Handle target refresh
            if field_name == "target":
                # Defensive check for base_url
                if not hasattr(self, "base_url") or not self.base_url:
                    logger.warning("Base URL not set, cannot fetch targets")
                    if hasattr(self, "log"):
                        self.log("Base URL not configured, please set Base API URL before refreshing targets")
                    return build_config

                base_url = self.base_url.rstrip("/")
                # Refresh target options
                logger.info("Refreshing targets for field: %s", field_name)
                targets, targets_metadata = await self.fetch_available_evaluation_targets()
                build_config["target"]["options"] = targets
                build_config["target"]["options_metadata"] = targets_metadata
                target_options = build_config["target"]["options"]
                msg = f"Updated target options: {target_options}"
                logger.info(msg)
                if hasattr(self, "log"):
                    self.log(f"Refreshed {len(target_options)} targets for evaluation")

            # Handle config refresh and dialog input changes
            elif field_name == "config":
                logger.debug(f"Config field update - field_value: {field_value}, type: {type(field_value)}")
                logger.debug(f"Current dialog state: {getattr(self, '_dialog_state', 'unknown')}")
                logger.debug(f"Current selected type: {getattr(self, '_selected_evaluation_type', 'none')}")
                logger.debug(f"Build config keys: {list(build_config.get('config', {}).keys())}")
                if "dialog_inputs" in build_config.get("config", {}):
                    logger.debug(f"Current dialog inputs: {build_config['config']['dialog_inputs']}")

                # Handle dialog input changes - follow customizer pattern
                if isinstance(field_value, dict):
                    logger.debug("Field value is a dict - checking for evaluation type selection")
                    logger.debug(f"Field value keys: {list(field_value.keys())}")
                    logger.debug(f"Field value content: {field_value}")
                    # Case 1: Evaluation type selection - update field visibility
                    if "01_evaluation_type" in field_value:
                        evaluation_type = field_value["01_evaluation_type"]
                        logger.debug(f"Found evaluation type selection: {evaluation_type}")
                        return self._update_dynamic_dialog_fields(build_config, evaluation_type)
                    logger.debug("No '01_evaluation_type' key found in field_value")

                    # Case 2: New config creation - handle based on current dialog type
                    if "01_config_name" in field_value:
                        logger.debug("Found config name in field_value - handling new config creation")
                        # Handle new config creation if needed

                    # Case 3: Empty dict or dict without expected keys - might be a "Create New Config" request
                    if not field_value or (not any(key.startswith("01_") for key in field_value)):
                        logger.debug(
                            "Field value is empty dict or doesn't contain expected keys - "
                            "treating as 'Create New Config' request"
                        )
                        # Reset dialog state to type selection
                        self._reset_dialog_state()

                        # Set dialog inputs to type selection
                        build_config["config"]["dialog_inputs"] = asdict(EvaluationTypeSelectionInput())

                        logger.debug("Reset to type selection dialog for 'Create New Config' (empty dict case)")

                        # Fetch available configs for the current target
                        if build_config.get("target", {}).get("value"):
                            configs, configs_metadata = await self.fetch_available_evaluation_configs(
                                build_config["target"]["value"]
                            )
                            build_config["config"]["options"] = configs
                            build_config["config"]["options_metadata"] = configs_metadata
                            logger.debug(f"Fetched {len(configs)} configs for target {build_config['target']['value']}")

                        return build_config
                else:
                    logger.debug("Field value is NOT a dict - this should trigger reset to type selection")
                    # SIMPLE APPROACH: Always reset to type selection when user selects "Create New Config"
                    # This handles all cases: empty value, string value, cancelled dialogs, etc.

                    logger.debug("About to reset dialog state and set dialog inputs to type selection")

                    # Reset dialog state to type selection
                    self._reset_dialog_state()

                    # Set dialog inputs to type selection
                    build_config["config"]["dialog_inputs"] = asdict(EvaluationTypeSelectionInput())

                    logger.debug("Reset to type selection dialog for 'Create New Config'")
                    logger.debug(f"New dialog state: {self._dialog_state}")
                    logger.debug(f"New selected type: {self._selected_evaluation_type}")

                    # Fetch available configs for the current target
                    if build_config.get("target", {}).get("value"):
                        configs, configs_metadata = await self.fetch_available_evaluation_configs(
                            build_config["target"]["value"]
                        )
                        build_config["config"]["options"] = configs
                        build_config["config"]["options_metadata"] = configs_metadata
                        logger.debug(f"Fetched {len(configs)} configs for target {build_config['target']['value']}")

            # Handle dialog field changes directly (when field_name is the dialog field name)
            elif field_name == "01_evaluation_type":
                logger.debug(f"Dialog field change - evaluation type: {field_value}")
                # Update the dialog fields based on the selected evaluation type
                return self._update_dynamic_dialog_fields(build_config, field_value)

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
                existing_datasets, existing_datasets_metadata = await self.fetch_existing_datasets(base_url)
                build_config["existing_dataset"]["options"] = existing_datasets
                build_config["existing_dataset"]["options_metadata"] = existing_datasets_metadata
                dataset_options = build_config["existing_dataset"]["options"]
                msg = f"Updated dataset options: {dataset_options}"
                logger.info(msg)
                if hasattr(self, "log"):
                    self.log(f"Refreshed {len(dataset_options)} datasets for evaluation")

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

    def get_template(self):
        """Override get_template to always reset dialog state when component is accessed."""
        logger.debug("Component template accessed - resetting dialog state")
        self._reset_dialog_state()
        return super().get_template()

    @property
    def dialog_inputs(self):
        """Override dialog_inputs to always return fresh type selection dialog."""
        logger.debug("Dialog inputs accessed - returning fresh type selection dialog")
        self._reset_dialog_state()
        return asdict(EvaluationTypeSelectionInput())

    def get_dialog_inputs(self):
        """Override to always return fresh type selection dialog."""
        logger.debug("Getting dialog inputs - always returning fresh type selection dialog")
        self._reset_dialog_state()
        return asdict(EvaluationTypeSelectionInput())

    def _force_dialog_reset(self, build_config: dict) -> dict:
        """Force reset the dialog to type selection regardless of current state."""
        logger.debug("Force resetting dialog to type selection")

        # Reset dialog state to type selection
        self._reset_dialog_state()

        # Set dialog inputs to type selection
        build_config["config"]["dialog_inputs"] = asdict(EvaluationTypeSelectionInput())

        logger.debug("Forced dialog reset completed")
        return build_config

    def _reset_dialog_state(self):
        """Reset the dialog state to initial type selection."""
        self._selected_evaluation_type = None
        self._dialog_state = "type_selection"  # "type_selection", "config_creation"
        logger.debug("Reset dialog state to type_selection - cleared selected type and set state to type_selection")

    def _update_dynamic_dialog_fields(self, build_config: dict, evaluation_type: str) -> dict:
        """Update the dynamic dialog fields based on the selected evaluation type."""
        logger.debug(f"Updating dynamic dialog fields for evaluation type: {evaluation_type}")

        # Store the selected evaluation type
        self._selected_evaluation_type = evaluation_type
        self._dialog_state = "config_creation"

        # Switch to the appropriate dialog based on evaluation type
        if evaluation_type == "LM Evaluation Harness":
            logger.debug("Switching to LM Evaluation Harness dialog")
            build_config["config"]["dialog_inputs"] = asdict(LMEvalHarnessConfigInput())
        elif evaluation_type == "Similarity Metrics":
            logger.debug("Switching to Similarity Metrics dialog")
            build_config["config"]["dialog_inputs"] = asdict(SimilarityMetricsConfigInput())
        elif evaluation_type == "Custom Evaluation":
            logger.debug("Switching to Custom Evaluation dialog")
            build_config["config"]["dialog_inputs"] = asdict(CustomEvaluationConfigInput())
        else:
            logger.warning(f"Unknown evaluation type: {evaluation_type}")
            # Default to evaluation type selection dialog
            build_config["config"]["dialog_inputs"] = asdict(EvaluationTypeSelectionInput())
            self._dialog_state = "type_selection"

        return build_config

    def _switch_evaluation_dialog(self, build_config: dict, field_value: dict) -> dict:
        """Switch to the appropriate evaluation dialog based on evaluation type selection."""
        evaluation_type = field_value["01_evaluation_type"]

        logger.debug(f"Switching evaluation dialog - selected type: {evaluation_type}")
        logger.debug(f"Previous dialog state: {getattr(self, '_dialog_state', 'unknown')}")

        # Store the selected evaluation type for later use
        self._selected_evaluation_type = evaluation_type
        self._dialog_state = "config_creation"

        # Switch to the appropriate dialog based on evaluation type
        if evaluation_type == "LM Evaluation Harness":
            build_config["config"]["dialog_inputs"] = asdict(LMEvalHarnessConfigInput())
        elif evaluation_type == "Similarity Metrics":
            build_config["config"]["dialog_inputs"] = asdict(SimilarityMetricsConfigInput())
        elif evaluation_type == "Custom Evaluation":
            build_config["config"]["dialog_inputs"] = asdict(CustomEvaluationConfigInput())
        else:
            # Default to evaluation type selection dialog
            build_config["config"]["dialog_inputs"] = asdict(EvaluationTypeSelectionInput())
            self._dialog_state = "type_selection"

        logger.info(f"Switched to {evaluation_type} dialog (state: {self._dialog_state})")
        return build_config

    async def evaluate(self) -> Data:
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
            # Both target and config are required
            error_msg = "Both target and config must be selected for evaluation"
            self.log(error_msg)
            raise ValueError(error_msg)

    async def _create_new_evaluation_config(self, config_data: dict) -> str:
        """Create a new evaluation configuration and return its ID."""
        try:
            nemo_client = self.get_nemo_client()

            # Determine evaluation type from stored selection or config data
            evaluation_type = getattr(self, "_selected_evaluation_type", None)
            if not evaluation_type:
                # Fallback to config data if not stored
                evaluation_type = config_data.get("01_evaluation_type", "LM Evaluation Harness")

            if evaluation_type == "LM Evaluation Harness":
                # Build LM Evaluation Harness config
                task_name = config_data.get("02_task_name", "gsm8k")
                hf_token = config_data.get("03_hf_token")

                if not hf_token:
                    error_msg = "HuggingFace token is required for LM Evaluation Harness"
                    raise ValueError(error_msg)

                config_params = {
                    "hf_token": hf_token,
                    "use_greedy": True,
                    "top_p": config_data.get("08_top_p", 0.0),
                    "top_k": config_data.get("09_top_k", 1),
                    "temperature": config_data.get("10_temperature", 0.0),
                    "stop": [],
                    "tokens_to_generate": config_data.get("11_tokens_to_generate", 1024),
                }

                # Create config
                response = await nemo_client.evaluation.configs.create(
                    type="lm_eval_harness",
                    namespace=getattr(self, "namespace", "default"),
                    tasks={
                        task_name: {
                            "params": {
                                "num_fewshot": config_data.get("04_few_shot_examples", 5),
                                "batch_size": config_data.get("05_batch_size", 16),
                                "bootstrap_iters": config_data.get("06_bootstrap_iterations", 100000),
                                "limit": config_data.get("07_limit", -1),
                            },
                        }
                    },
                    params=config_params,
                    extra_headers=self.get_auth_headers(),
                )

            elif evaluation_type == "Similarity Metrics":
                # Build Similarity Metrics config
                scorers = config_data.get("02_scorers", ["accuracy", "bleu", "rouge", "em", "bert", "f1"])

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

            elif evaluation_type == "Custom Evaluation":
                # Build Custom Evaluation config
                evaluation_prompt = config_data.get(
                    "02_evaluation_prompt", "Evaluate the following response: {{response}}"
                )
                metrics = config_data.get("03_metrics", ["accuracy"])
                batch_size = config_data.get("04_batch_size", 16)

                # Create metrics in SDK format
                metrics_dict = {}
                for metric in metrics:
                    metric_name = metric.lower()
                    if metric_name == "custom":
                        # For custom metrics, we'll use a simple string check
                        metrics_dict["custom_evaluation"] = {
                            "type": "string-check",
                            "params": {"check": ["{{response}}", "!=", ""]},
                        }
                    elif metric_name == "em":
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
                    params={"parallelism": batch_size},
                    tasks={
                        "custom_task": {
                            "type": "completion",
                            "params": {"template": {"prompt": evaluation_prompt}},
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
            return config_id  # noqa: TRY300

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
            return target_name  # noqa: TRY300
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
            return config_name  # noqa: TRY300
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

    async def fetch_existing_datasets(self, base_url: str | None = None) -> tuple[list[str], list[dict[str, Any]]]:  # noqa: ARG002
        """Fetch existing datasets from the NeMo Data Store with metadata.

        Args:
            base_url (str): Base URL for the NeMo services (unused, kept for compatibility)

        Returns:
            Tuple of (dataset names, dataset metadata) available for evaluation
        """
        # Defensive checks
        if not hasattr(self, "namespace") or not self.namespace:
            if hasattr(self, "log"):
                self.log("Namespace not set for fetching datasets")
            return [], []

        if not hasattr(self, "auth_token") or not self.auth_token:
            if hasattr(self, "log"):
                self.log("Authentication token not set for fetching datasets")
            return [], []

        try:
            # Use NeMo client for dataset fetching
            nemo_client = self.get_nemo_client()
            response = await nemo_client.datasets.list(extra_headers=self.get_auth_headers())

            datasets = []
            datasets_metadata = []

            if hasattr(response, "data") and response.data:
                for dataset in response.data:
                    dataset_name = getattr(dataset, "name", "")
                    dataset_created = getattr(dataset, "created", None)
                    dataset_updated = getattr(dataset, "updated", None)
                    dataset_size = getattr(dataset, "size", None)
                    dataset_records = getattr(dataset, "records", None)

                    if dataset_name:
                        datasets.append(dataset_name)
                        # Build metadata for this dataset
                        metadata = self._build_dataset_metadata(
                            dataset_created, dataset_updated, dataset_size, dataset_records
                        )
                        datasets_metadata.append(metadata)

                # Debug logging
                logger.debug("Fetched %s datasets with metadata", len(datasets))
                if datasets_metadata:
                    logger.debug("Sample dataset metadata: %s", datasets_metadata[0])

                return datasets, datasets_metadata
            return datasets, datasets_metadata  # noqa: TRY300
        except NeMoMicroservicesError as exc:
            if hasattr(self, "log"):
                self.log(f"NeMo microservices error while fetching datasets: {exc}")
            return [], []
        except (httpx.RequestError, httpx.HTTPStatusError) as exc:
            if hasattr(self, "log"):
                self.log(f"Error response while requesting datasets: {exc}")
            return [], []
        except (ValueError, TypeError) as exc:
            if hasattr(self, "log"):
                self.log(f"Unexpected error while fetching datasets: {exc}")
            return [], []

    def _build_dataset_metadata(
        self,
        created: str | None = None,
        updated: str | None = None,
        size: int | None = None,
        records: int | None = None,
    ) -> dict[str, Any]:
        """Build metadata dictionary for a dataset."""
        metadata = {}

        # Add dataset size and records if available
        if size is not None:
            metadata["size"] = f"{size:,}"  # Format with commas
        if records is not None:
            metadata["records"] = f"{records:,}"  # Format with commas

        # Add timestamps if available (but don't display them in dropdown)
        if created:
            metadata["created"] = created
        if updated:
            metadata["updated"] = updated

        # Add icon for datasets
        metadata["icon"] = "Database"

        # Debug logging
        logger.debug("Built dataset metadata: %s", metadata)

        return metadata

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

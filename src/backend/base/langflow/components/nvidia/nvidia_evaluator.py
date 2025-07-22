import io
import json
import logging
import uuid
from datetime import datetime, timezone
from unittest.mock import patch

import httpx
import requests
from huggingface_hub import HfApi
from nemo_microservices import AsyncNeMoMicroservices, NeMoMicroservicesError

from langflow.custom import Component
from langflow.field_typing.range_spec import RangeSpec
from langflow.io import (
    DataInput,
    DropdownInput,
    FloatInput,
    IntInput,
    MultiselectInput,
    Output,
    SecretStrInput,
    SliderInput,
    StrInput,
)
from langflow.schema import Data

logger = logging.getLogger(__name__)


def create_auth_interceptor(auth_token, namespace):
    """Create a function to intercept HTTP requests and add auth headers for namespace URLs"""
    original_request = requests.Session.request

    def patched_request(self, method, url, *args, **kwargs):
        # Check if URL contains the namespace path or if it's an LFS request
        is_namespace_url = url and namespace and f"/{namespace}/" in url
        is_lfs_url = url and "/lfs/" in url

        if is_namespace_url or is_lfs_url:
            # Add authorization header for namespace requests and LFS uploads
            headers = kwargs.get("headers", {})
            if "Authorization" not in headers:
                headers["Authorization"] = f"Bearer {auth_token}"
                kwargs["headers"] = headers
                logger.info(
                    f"Intercepted and added Authorization header for URL: {url} (namespace: {is_namespace_url}, lfs: {is_lfs_url})"
                )

        return original_request(self, method, url, *args, **kwargs)

    return patched_request


class AuthenticatedHfApi(HfApi):
    """Custom HuggingFace API client that adds authentication headers for firewall."""

    def __init__(self, endpoint, auth_token, namespace=None, **kwargs):
        super().__init__(endpoint=endpoint, **kwargs)
        self.auth_token = auth_token
        self.namespace = namespace

    def _build_hf_headers(self, token=None, library_name=None, library_version=None, user_agent=None):
        """Override to add custom authentication headers."""
        # Call parent method with only the parameters it accepts
        headers = super()._build_hf_headers(
            token=token, library_name=library_name, library_version=library_version, user_agent=user_agent
        )

        # Always add our custom authentication header for all requests
        headers["Authorization"] = f"Bearer {self.auth_token}"

        return headers

    def _request_wrapper(self, method, url, *args, **kwargs):
        """Override to intercept requests and add auth headers for namespace URLs."""
        # Check if URL contains the namespace path OR if it's an LFS request
        is_namespace_url = url and self.namespace and f"/{self.namespace}/" in url
        is_lfs_url = url and "/lfs/" in url

        if is_namespace_url or is_lfs_url:
            # Add authorization header for namespace requests and LFS uploads
            headers = kwargs.get("headers", {})
            if "Authorization" not in headers:
                headers["Authorization"] = f"Bearer {self.auth_token}"
                kwargs["headers"] = headers
                logger.info(
                    "Added Authorization header for URL: %s (namespace: %s, lfs: %s)", url, is_namespace_url, is_lfs_url
                )

        return super()._request_wrapper(method, url, *args, **kwargs)


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
            logger.info("NvidiaEvaluatorComponent initialized successfully")
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

    def get_entity_client(self) -> AsyncNeMoMicroservices:
        """Get an authenticated NeMo entity client."""
        return AsyncNeMoMicroservices(
            base_url=self.base_url,
        )

    def get_datastore_client(self) -> AsyncNeMoMicroservices:
        """Get an authenticated NeMo datastore client."""
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
        DropdownInput(
            name="000_llm_name",
            display_name="Model to be evaluated",
            info="Select the model for evaluation (fetched from /nemo/v1/models endpoint)",
            options=[],  # Dynamically populated from /nemo/v1/models
            refresh_button=True,
            required=True,
            combobox=True,
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
    ]

    outputs = [
        Output(display_name="Job Info", name="job_info", method="evaluate"),
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
        DataInput(
            name="evaluation_data",
            display_name="Evaluation Data",
            info="Dataset for evaluation, expecting 2 fields `prompt` and `ideal_response` in dataset",
            is_list=True,
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

    def clear_dynamic_inputs(self, build_config, saved_values):
        """Clears dynamically added fields by referring to a special marker in build_config."""
        dynamic_fields = build_config.get("_dynamic_fields", [])
        length_dynamic_fields = len(dynamic_fields)
        message = f"Clearing dynamic inputs. Number of fields to remove: {length_dynamic_fields}"
        logger.info(message)

        for field in dynamic_fields:
            if field in build_config:
                message = f"Removing dynamic field: {field}"
                logger.info(message)
                saved_values[field] = build_config[field].get("value", None)
                del build_config[field]

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

            # Defensive check - if auth_token is not set, don't try to fetch models
            if not hasattr(self, "auth_token") or not self.auth_token:
                logger.info("Auth token not set, skipping model fetching")
                if hasattr(self, "log"):
                    self.log("Authentication token not set, please configure component before refreshing models")
                return build_config

            # Handle model refresh - check both field name and None (for initial load)
            if field_name == "000_llm_name" or field_name is None:
                # Defensive check for base_url
                if not hasattr(self, "base_url") or not self.base_url:
                    logger.warning("Base URL not set, cannot fetch models")
                    if hasattr(self, "log"):
                        self.log("Base URL not configured, please set Base API URL before refreshing models")
                    return build_config

                base_url = self.base_url.rstrip("/")
                # Refresh model options for LLM Name dropdown using /nemo/v1/models endpoint
                logger.info("Refreshing models for field: %s from %s/nemo/v1/models", field_name, base_url)
                build_config["000_llm_name"]["options"] = await self.fetch_models(base_url)
                options = build_config["000_llm_name"]["options"]
                max_options_display = 5
                msg = (
                    f"Updated LLM Name options: {options[:max_options_display]}..."
                    if len(options) > max_options_display
                    else f"Updated LLM Name options: {options}"
                )
                logger.info(msg)
                if hasattr(self, "log"):
                    self.log(f"Refreshed {len(options)} models for evaluation from /nemo/v1/models endpoint")

            # Handle dataset refresh
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

            elif field_name == "002_evaluation_type":
                if hasattr(self, "clear_dynamic_inputs") and hasattr(self, "add_evaluation_inputs"):
                    self.clear_dynamic_inputs(build_config, saved_values)
                    self.add_evaluation_inputs(build_config, saved_values, field_value)
                else:
                    logger.warning("Dynamic input methods not available yet")
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
            # Note: existing_dataset is now a DropdownInput that gets populated via API call
            logger.info("Build config update completed successfully.")
        except Exception as exc:
            # Catch all exceptions to prevent UI crashes
            error_msg = f"Error during build config update: {exc}"
            logger.exception(error_msg)
            # Instead of raising, just log the error and return the original config
            if hasattr(self, "log"):
                self.log(f"Build config update failed: {error_msg}")
            return build_config
        return build_config

    async def evaluate(self) -> dict:
        evaluation_type = getattr(self, "002_evaluation_type", "LM Evaluation Harness")

        if not self.namespace:
            error_msg = "Missing namespace"
            raise ValueError(error_msg)

        model_name = getattr(self, "000_llm_name", "")
        if not model_name:
            error_msg = "Refresh and select the model name to be evaluated"
            raise ValueError(error_msg)

        if not self.base_url:
            error_msg = "Missing base URL"
            raise ValueError(error_msg)

        base_url = self.base_url.rstrip("/")

        # Create the evaluation using SDK pattern like customizer
        try:
            nemo_client = self.get_nemo_client()

            if evaluation_type == "LM Evaluation Harness":
                # Create LM evaluation config and target, then create job with IDs
                config_data, target_data = await self._prepare_lm_evaluation_data(base_url)

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
                    namespace=self.namespace,
                    config=config_id,
                    target=target_id,
                    extra_headers={"Authorization": f"Bearer {self.auth_token}"},
                )

            elif evaluation_type == "Similarity Metrics":
                # Create custom evaluation config and target, then create job with IDs
                config_data, target_data = await self._prepare_custom_evaluation_data(base_url)

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
                    namespace=self.namespace,
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

            return result_dict

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

    async def _prepare_lm_evaluation_data(self, base_url: str) -> tuple:
        """Prepare LM evaluation config and target data for SDK."""
        # Create target first
        target_data = await self._create_evaluation_target(None, base_url)

        # Get required parameters
        hf_token = getattr(self, "100_huggingface_token", None)
        if not hf_token:
            error_msg = "Missing hf token"
            raise ValueError(error_msg)

        # Create config data in SDK format
        config_data = {
            "type": "lm_eval_harness",
            "namespace": self.namespace,
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

    async def _prepare_custom_evaluation_data(self, base_url: str) -> tuple:
        """Prepare custom evaluation config and target data for SDK."""
        # Process dataset
        existing_dataset = getattr(self, "existing_dataset", None)
        if existing_dataset:
            self.log(f"Using existing dataset: {existing_dataset}")
            repo_id = f"{self.namespace}/{existing_dataset}"
        else:
            if self.evaluation_data is None or len(self.evaluation_data) == 0:
                error_msg = "Either select an existing dataset or provide evaluation data to run evaluation"
                raise ValueError(error_msg)
            repo_id = await self.process_eval_dataset(base_url)

        # Use the file path with the repo ID
        input_file = f"nds:{repo_id}/input.jsonl"

        # Handle run_inference
        run_inference = getattr(self, "310_run_inference", "True").lower() == "true"
        output_file = None if run_inference else f"nds:{repo_id}/output.jsonl"

        # Create target
        target_data = await self._create_evaluation_target(output_file, base_url)

        # Create metrics in SDK format
        scores = getattr(self, "351_scorers", ["accuracy", "bleu", "rouge", "em", "bert", "f1"])
        metrics_dict = {}
        for score in scores:
            metric_name = score.lower()
            if metric_name == "bleu":
                metrics_dict[metric_name] = {"type": "bleu", "params": {"references": ["{{item.ideal_response}}"]}}
            elif metric_name == "rouge":
                metrics_dict[metric_name] = {"type": "rouge", "params": {"ground_truth": "{{item.ideal_response}}"}}
            elif metric_name == "em":
                metrics_dict["exact_match"] = {
                    "type": "string-check",
                    "params": {"ground_truth": "{{item.ideal_response}}"},
                }
            elif metric_name == "f1":
                metrics_dict[metric_name] = {"type": "f1", "params": {"ground_truth": "{{item.ideal_response}}"}}
            else:
                metrics_dict[metric_name] = {
                    "type": "string-check",
                    "params": {"ground_truth": "{{item.ideal_response}}"},
                }

        # Create config data in SDK format
        config_data = {
            "type": "custom",
            "namespace": self.namespace,
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

    async def _create_evaluation_target(self, output_file, base_url: str):
        """Create evaluation target using SDK and return the data for job creation."""
        try:
            nemo_client = self.get_nemo_client()

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
                        "url": f"{self.inference_model_url}/v1/completions",
                        "model_id": getattr(self, "000_llm_name", ""),
                    }
                }

            # Create target data for SDK
            target_data = {
                "type": "model",
                "namespace": self.namespace,
                "model": model_data,
            }

            return target_data

        except Exception as exc:
            error_msg = f"Error creating evaluation target: {exc}"
            self.log(error_msg)
            raise ValueError(error_msg) from exc

    async def _generate_lm_evaluation_body(self, base_url: str) -> dict:
        target_id = await self.create_eval_target(None, base_url)
        hf_token = getattr(self, "100_huggingface_token", None)
        if not hf_token:
            error_msg = "Missing hf token"
            raise ValueError(error_msg)
        namespace = self.namespace
        config_data = {
            "type": "lm_eval_harness",
            "namespace": namespace,
            "tasks": [
                {
                    "type": getattr(self, "110_task_name", ""),
                    "params": {
                        "num_fewshot": getattr(self, "112_few_shot_examples", 5),
                        "batch_size": getattr(self, "113_batch_size", 16),
                        "bootstrap_iters": getattr(self, "114_bootstrap_iterations", 100000),
                        "limit": getattr(self, "115_limit", -1),
                    },
                }
            ],
            "params": {
                "hf_token": hf_token or None,
                "use_greedy": True,
                "top_p": getattr(self, "151_top_p", 0.0),
                "top_k": getattr(self, "152_top_k", 1),
                "temperature": getattr(self, "153_temperature", 0.0),
                "stop": [],  # not exposing this for now, would be 154_stop
                "tokens_to_generate": getattr(self, "154_tokens_to_generate", 1024),
            },
        }

        try:
            # Use NeMo client for evaluation config creation
            nemo_client = self.get_nemo_client()
            response = await nemo_client.evaluation.configs.create(
                type=config_data["type"],
                namespace=config_data["namespace"],
                tasks=config_data["tasks"],
                params=config_data["params"],
                extra_headers={"Authorization": f"Bearer {self.auth_token}"},
            )

            # Process the response
            result_dict = response.model_dump()
            formatted_result = json.dumps(result_dict, indent=2)
            self.log(f"Received successful response: {formatted_result}")

            config_id = result_dict.get("id")
            if not config_id:
                err_msg = f"Missing 'id' in response: {result_dict}"
                raise ValueError(err_msg)

            return {
                "tags": [getattr(self, "001_tag", "")],
                "namespace": namespace,
                "target": f"{namespace}/{target_id}",
                "config": f"{namespace}/{config_id}",
                "config_obj": response,  # Pass the actual config object
                "target_obj": await self._get_target_object(target_id, base_url),  # Get target object
            }

        except NeMoMicroservicesError as exc:
            error_msg = f"NeMo microservices error during evaluation config creation: {exc}"
            self.log(error_msg)
            raise ValueError(error_msg) from exc

        except (httpx.TimeoutException, httpx.HTTPStatusError, httpx.RequestError) as exc:
            error_msg = f"HTTP error during evaluation config creation: {exc}"
            self.log(error_msg)
            raise ValueError(error_msg) from exc

        except Exception as exc:
            error_msg = f"Unexpected error during evaluation config creation: {exc}"
            self.log(error_msg)
            raise ValueError(error_msg) from exc

    async def _generate_custom_evaluation_body(self, base_url: str) -> dict:
        # Check if user selected an existing dataset
        existing_dataset = getattr(self, "existing_dataset", None)
        if existing_dataset:
            # Use existing dataset
            self.log(f"Using existing dataset: {existing_dataset}")
            repo_id = f"{self.namespace}/{existing_dataset}"
        else:
            # Process and upload the dataset if evaluation_data is provided
            if self.evaluation_data is None or len(self.evaluation_data) == 0:
                error_msg = "Either select an existing dataset or provide evaluation data to run evaluation"
                raise ValueError(error_msg)
            repo_id = await self.process_eval_dataset(base_url)

        # Use the file path with the repo ID
        input_file = f"nds:{repo_id}/input.jsonl"
        # Handle run_inference as a boolean
        run_inference = getattr(self, "310_run_inference", "True").lower() == "true"

        if run_inference and not self.inference_model_url:
            error_msg = "Provide the nim url for evaluation inference to be processed"
            raise ValueError(error_msg)

        namespace = self.namespace
        # Set output_file based on run_inference
        output_file = None
        if not run_inference:  # Only set output_file if run_inference is False
            output_file = f"nds:{repo_id}/output.jsonl"
        self.log(f"input_file: {input_file}, output_file: {output_file}")

        target_id = await self.create_eval_target(output_file, base_url)
        scores = getattr(self, "351_scorers", ["accuracy", "bleu", "rouge", "em", "bert", "f1"])

        # Transform the list into the correct custom evaluation format
        metrics_dict = {}
        for score in scores:
            metric_name = score.lower()
            if metric_name == "bleu":
                metrics_dict[metric_name] = {"type": "bleu", "params": {"references": ["{{item.ideal_response}}"]}}
            elif metric_name == "rouge":
                metrics_dict[metric_name] = {"type": "rouge", "params": {"ground_truth": "{{item.ideal_response}}"}}
            elif metric_name == "em":
                metrics_dict["exact_match"] = {
                    "type": "string-check",
                    "params": {"ground_truth": "{{item.ideal_response}}"},
                }
            elif metric_name == "f1":
                metrics_dict[metric_name] = {"type": "f1", "params": {"ground_truth": "{{item.ideal_response}}"}}
            else:
                # For other metrics like accuracy, bert - use string-check
                metrics_dict[metric_name] = {
                    "type": "string-check",
                    "params": {"ground_truth": "{{item.ideal_response}}"},
                }

        config_data = {
            "type": "custom",
            "namespace": namespace,
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

        try:
            # Use NeMo client for evaluation config creation
            nemo_client = self.get_nemo_client()
            response = await nemo_client.evaluation.configs.create(
                type=config_data["type"],
                namespace=config_data["namespace"],
                params=config_data["params"],
                tasks=config_data["tasks"],
                extra_headers={"Authorization": f"Bearer {self.auth_token}"},
            )

            # Process the response
            result_dict = response.model_dump()

            # Convert datetime objects to strings for JSON serialization
            result_dict = self.convert_datetime_to_string(result_dict)
            formatted_result = json.dumps(result_dict, indent=2)
            self.log(f"Received successful response: {formatted_result}")

            config_id = result_dict.get("id")
            return {
                "namespace": namespace,
                "target": f"{namespace}/{target_id}",
                "config": f"{namespace}/{config_id}",
                "tags": [getattr(self, "001_tag", "")],
                "config_obj": response,  # Pass the actual config object
                "target_obj": await self._get_target_object(target_id, base_url),  # Get target object
            }

        except NeMoMicroservicesError as exc:
            error_msg = f"NeMo microservices error during evaluation config creation: {exc}"
            self.log(error_msg)
            raise ValueError(error_msg) from exc

        except (httpx.TimeoutException, httpx.HTTPStatusError, httpx.RequestError) as exc:
            error_msg = f"HTTP error during evaluation config creation: {exc}"
            self.log(error_msg)
            raise ValueError(error_msg) from exc

        except Exception as exc:
            error_msg = f"Unexpected error during evaluation config creation: {exc}"
            self.log(error_msg)
            raise ValueError(error_msg) from exc

    async def _get_target_object(self, target_id: str, base_url: str):
        """Get the target object for SDK use."""
        try:
            nemo_client = self.get_nemo_client()
            target_obj = await nemo_client.evaluation.targets.retrieve(
                target_id=target_id,
                namespace=self.namespace,
                extra_headers={"Authorization": f"Bearer {self.auth_token}"},
            )
            return target_obj
        except Exception as exc:
            self.log(f"Warning: Could not retrieve target object {target_id}, using string: {exc}")
            # Fallback to string format if object retrieval fails
            return f"{self.namespace}/{target_id}"

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
                            "url": f"{self.inference_model_url}/v1/completions",
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

    async def process_eval_dataset(self, base_url: str) -> str:
        """Asynchronously processes and uploads the dataset to the API in chunks.

        Returns the upload status.

        Args:
            base_url (str): Base URL for the NeMo services.
        """
        try:
            # Inputs and repo setup
            dataset_name = str(uuid.uuid4())

            # Initialize clients for dataset operations
            entity_client = self.get_entity_client()
            datastore_client = self.get_datastore_client()

            # Create namespaces using appropriate clients
            await self.create_namespace_with_nemo_client(entity_client, self.namespace)
            await self.create_datastore_namespace_with_nemo_client(datastore_client, self.namespace)

            # Create dataset repository using authenticated HuggingFace API
            hf_endpoint = f"{base_url}/v1/hf"
            token_preview_length = 20
            token_preview = (
                self.auth_token[:token_preview_length] + "..."
                if self.auth_token and len(self.auth_token) > token_preview_length
                else self.auth_token
            )
            self.log(f"Creating HuggingFace API with endpoint: {hf_endpoint}")
            self.log(f"Using namespace: {self.namespace}")
            self.log(f"Using auth token: {token_preview}")

            hf_api = AuthenticatedHfApi(
                endpoint=hf_endpoint,
                auth_token=self.auth_token,
                namespace=self.namespace,
                token=self.auth_token,
            )
            repo_id = f"{self.namespace}/{dataset_name}"
            repo_type = "dataset"
            hf_api.create_repo(repo_id, repo_type=repo_type, exist_ok=True)
            self.log(f"repo_id : {repo_id}")
            generate_output_file = getattr(self, "310_run_inference", None) == "False"

            # Initialize lists for the two JSON structures
            input_file_data = []
            output_file_data = []

            # Ensure DataFrame is iterable correctly
            for data_obj in self.evaluation_data or []:
                # Check if the object is an instance of Data
                if not isinstance(data_obj, Data):
                    self.log(f"Skipping non-Data object in training data, but got: {data_obj}")
                    continue

                # Extract and transform fields
                filtered_data = {
                    "prompt": getattr(data_obj, "prompt", None) or "",
                    "ideal_response": getattr(data_obj, "ideal_response", None) or "",
                    "category": getattr(data_obj, "category", "Generation") or "Generation",
                    "source": getattr(data_obj, "source", None) or "",
                    "response": getattr(data_obj, "response", None) or "",
                    "llm_name": getattr(data_obj, "llm_name", None) or "",
                }
                # Check if both fields are present
                if filtered_data["prompt"] is not None and filtered_data["ideal_response"] is not None:
                    # Create data for the first file
                    input_file_data.append(
                        {
                            "prompt": filtered_data["prompt"],
                            "ideal_response": filtered_data["ideal_response"],
                            "category": filtered_data["category"],
                            "source": filtered_data["source"],
                        }
                    )
                    if generate_output_file:
                        # Create data for the second file
                        output_file_data.append(
                            {
                                "input": {
                                    "prompt": filtered_data["prompt"],
                                    "ideal_response": filtered_data["ideal_response"],
                                    "category": filtered_data["category"],
                                    "source": filtered_data["source"],
                                },
                                "response": filtered_data["response"],
                                "llm_name": filtered_data["llm_name"],
                            }
                        )
            # Create in-memory JSONL files (compact format like customizer to avoid LFS)
            input_jsonl_data = "\n".join(json.dumps(record) for record in input_file_data)
            input_file_buffer = io.BytesIO(input_jsonl_data.encode("utf-8"))
            input_file_name = "input.jsonl"

            # Patch requests to intercept namespace URLs
            patched_request = create_auth_interceptor(self.auth_token, self.namespace)

            try:
                with patch.object(requests.Session, "request", patched_request):
                    hf_api.upload_file(
                        path_or_fileobj=input_file_buffer,
                        path_in_repo=input_file_name,
                        repo_id=repo_id,
                        repo_type="dataset",
                        commit_message=f"Input file at time: {datetime.now(timezone.utc).strftime('%Y%m%d%H%M%S')}",
                    )
            finally:
                input_file_buffer.close()

            if generate_output_file:
                output_jsonl_data = "\n".join(json.dumps(record) for record in output_file_data)
                output_file_buffer = io.BytesIO(output_jsonl_data.encode("utf-8"))
                output_file_name = "output.jsonl"
                try:
                    with patch.object(requests.Session, "request", patched_request):
                        hf_api.upload_file(
                            path_or_fileobj=output_file_buffer,
                            path_in_repo=output_file_name,
                            repo_id=repo_id,
                            repo_type="dataset",
                            commit_message=f"Output file at time: {datetime.now(timezone.utc).strftime('%Y%m%d%H%M%S')}",
                        )
                finally:
                    output_file_buffer.close()

            logger.info("All data has been processed and uploaded successfully.")
        except NeMoMicroservicesError as exc:
            error_msg = f"NeMo microservices error during processing/upload: {exc}"
            self.log(error_msg)
            raise ValueError(error_msg) from exc
        except Exception as exc:
            exception_str = str(exc)
            error_msg = f"An unexpected error occurred during processing/upload: {exception_str}"
            self.log(error_msg)
            raise ValueError(error_msg) from exc

        # Register dataset with entity store using NeMo client
        try:
            file_url = f"hf://datasets/{repo_id}"
            description = f"Dataset loaded using the input data {dataset_name}"

            # Use NeMo client for entity registry operations
            nemo_client = self.get_nemo_client()
            response = await nemo_client.datasets.create(
                name=dataset_name,
                namespace=self.namespace,
                description=description,
                files_url=file_url,
                format="jsonl",
                project=dataset_name,
                extra_headers={"Authorization": f"Bearer {self.auth_token}"},
            )

            # Log the entity store dataset creation
            entity_dataset_id = response.id if hasattr(response, "id") else dataset_name
            self.log(f"Created dataset in entity store with ID: {entity_dataset_id}")

            logger.info("Dataset registered successfully with entity store")
            logger.info("All data has been processed and uploaded successfully.")
        except NeMoMicroservicesError as exc:
            error_msg = f"NeMo microservices error while registering dataset: {exc}"
            self.log(error_msg)
            raise ValueError(error_msg) from exc
        except Exception as exc:
            exception_str = str(exc)
            error_msg = f"An unexpected error occurred while registering dataset: {exception_str}"
            self.log(error_msg)
            raise ValueError(error_msg) from exc

        # Return the repo_id (used for HF repo) for file path construction
        return repo_id

    async def create_namespace_with_nemo_client(self, nemo_client: AsyncNeMoMicroservices, namespace: str):
        """Create entity namespace using NeMo client."""
        try:
            # First check if namespace exists
            try:
                await nemo_client.namespaces.retrieve(
                    namespace_id=namespace, extra_headers={"Authorization": f"Bearer {self.auth_token}"}
                )
            except NeMoMicroservicesError:
                # Namespace doesn't exist, create it
                pass
            else:
                self.log(f"Entity namespace already exists: {namespace}")
                return

            await nemo_client.namespaces.create(
                id=namespace,
                description=f"Entity namespace for {namespace} resources",
                extra_headers={"Authorization": f"Bearer {self.auth_token}"},
            )
            self.log(f"Created entity namespace: {namespace}")
        except NeMoMicroservicesError as exc:
            if "already exists" in str(exc).lower():
                self.log(f"Entity namespace already exists: {namespace}")
            else:
                error_msg = f"Failed to create entity namespace: {exc}"
                raise ValueError(error_msg) from exc

    async def create_datastore_namespace_with_nemo_client(self, nemo_client: AsyncNeMoMicroservices, namespace: str):  # noqa: ARG002
        """Create datastore namespace using direct HTTP request."""
        try:
            # Use the correct endpoint from the customizer pattern
            nds_url = f"{self.base_url}/v1/datastore/namespaces"

            headers = {"Authorization": f"Bearer {self.auth_token}"}
            data = {"namespace": namespace}

            async with httpx.AsyncClient() as client:
                # First check if namespace exists
                try:
                    response = await client.get(f"{nds_url}/{namespace}", headers=headers)
                    if response.status_code == 200:  # noqa: PLR2004
                        self.log(f"Datastore namespace already exists: {namespace}")
                        return
                except httpx.HTTPError:
                    # Namespace doesn't exist, create it
                    pass

                # Create the namespace
                self.log(f"Creating datastore namespace at URL: {nds_url}")
                response = await client.post(nds_url, headers=headers, json=data)

                self.log(f"Response status: {response.status_code}")

                if response.status_code in (200, 201):
                    self.log(f"Created datastore namespace: {namespace}")
                    return
                if response.status_code in (409, 422):
                    self.log(f"Datastore namespace already exists: {namespace}")
                    return
                error_msg = f"Failed to create datastore namespace: {response.status_code} - {response.text}"
                self.log(error_msg)
                raise ValueError(error_msg)

        except httpx.HTTPError as exc:
            error_msg = f"HTTP error creating datastore namespace: {exc}"
            self.log(error_msg)
            raise ValueError(error_msg) from exc
        except Exception as exc:
            error_msg = f"Unexpected error creating datastore namespace: {exc}"
            self.log(error_msg)
            raise ValueError(error_msg) from exc

    async def create_namespace(self, namespace: str, base_url: str):  # noqa: ARG002
        """Checks and creates namespace in entity-store with authentication using NeMo client."""
        entity_client = self.get_entity_client()
        await self.create_namespace_with_nemo_client(entity_client, namespace)

    async def create_datastore_namespace(self, namespace: str, base_url: str):  # noqa: ARG002
        """Checks and creates namespace in datastore with authentication using direct HTTP request."""
        datastore_client = self.get_datastore_client()
        await self.create_datastore_namespace_with_nemo_client(datastore_client, namespace)

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

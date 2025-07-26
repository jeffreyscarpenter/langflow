import json
import asyncio
import time

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

            # Parse namespace/model_name format
            if "/" in output_model:
                namespace, model_name = output_model.split("/", 1)
                self.log(f"Extracted model name: {model_name}, namespace: {namespace}")
                return model_name, namespace
            # If no namespace in output_model, use the model name as-is
            self.log(f"Using output_model as model name: {output_model}")
            return output_model, None

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
            info="Customized model from NeMo Customizer (optional - if not provided, will use Model dropdown)",
            required=False,
            input_types=["Data"],
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
        except (ValueError, AttributeError, ImportError, RuntimeError) as exc:
            # Catch specific exceptions to prevent UI crashes
            error_msg = f"Error during build config update: {exc}"
            logger.exception(error_msg)
            # Instead of raising, just log the error and return the original config
            if hasattr(self, "log"):
                self.log(f"Build config update failed: {error_msg}")
            return build_config
        return build_config

    async def evaluate(self) -> Data:
        evaluation_type = getattr(self, "002_evaluation_type", "LM Evaluation Harness")

        if not self.namespace:
            error_msg = "Missing namespace"
            raise ValueError(error_msg)

        # Prioritize customized_model input over 000_llm_name dropdown
        customized_model_input = getattr(self, "customized_model", None)
        effective_namespace = self.namespace
        if customized_model_input is not None and isinstance(customized_model_input, Data):
            model_name, customized_namespace = self.extract_customized_model_info(customized_model_input)
            if model_name:
                self.log(f"Using customized model: {model_name}")
                # Use customized model namespace if available, otherwise use component namespace
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
                    logger.info("Evaluation job %s is still %s. Waiting %s seconds...", job_id, status, poll_interval_seconds)
                    await asyncio.sleep(poll_interval_seconds)
                else:
                    logger.warning("Unknown evaluation job status: %s. Waiting %s seconds...", status, poll_interval_seconds)
                    await asyncio.sleep(poll_interval_seconds)

            except NeMoMicroservicesError:
                logger.exception("NeMo microservices error while checking evaluation job status")
                await asyncio.sleep(poll_interval_seconds)
            except (asyncio.CancelledError, RuntimeError, OSError):
                logger.exception("Unexpected error while checking evaluation job status")
                await asyncio.sleep(poll_interval_seconds)

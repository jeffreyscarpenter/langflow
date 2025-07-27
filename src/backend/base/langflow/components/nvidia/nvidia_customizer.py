import asyncio
import logging
import time
from dataclasses import asdict, dataclass, field

import httpx
import nemo_microservices

from langflow.custom import Component
from langflow.io import (
    BoolInput,
    DropdownInput,
    FloatInput,
    HandleInput,
    IntInput,
    Output,
    SecretStrInput,
    StrInput,
)
from langflow.schema import Data

logger = logging.getLogger(__name__)


@dataclass
class NewConfigInput:
    """Input structure for creating new customization configs with conditional logic."""

    functionality: str = "create"
    fields: dict[str, dict] = field(
        default_factory=lambda: {
            "data": {
                "node": {
                    "name": "create_config",
                    "description": "Create a new training configuration for the selected target",
                    "display_name": "Create New Config",
                    "field_order": [
                        "01_config_name",
                        "02_training_type",
                        "03_finetuning_type",
                        "04_max_seq_length",
                        "05_prompt_template",
                        "06_training_precision",
                        "07_lora_adapter_dim",
                        "08_lora_alpha",
                        "09_lora_target_modules",
                    ],
                    "template": {
                        "01_config_name": StrInput(
                            name="config_name",
                            display_name="Config Name",
                            info="Name for the new configuration (e.g., my-custom-config@v1.0.0+L40)",
                            required=True,
                        ),
                        "02_training_type": DropdownInput(
                            name="training_type",
                            display_name="Training Type",
                            options=["sft"],
                            value="sft",
                            required=True,
                            refresh_button=True,
                        ),
                        "03_finetuning_type": DropdownInput(
                            name="finetuning_type",
                            display_name="Fine-tuning Type",
                            options=["all_weights", "lora"],
                            value="lora",
                            required=True,
                        ),
                        "04_max_seq_length": IntInput(
                            name="max_seq_length",
                            display_name="Max Sequence Length",
                            info="Maximum sequence length for training",
                            value=4096,
                            required=True,
                        ),
                        "05_prompt_template": StrInput(
                            name="prompt_template",
                            display_name="Prompt Template",
                            info="Template for formatting prompts and completions",
                            value="{prompt} {completion}",
                            required=True,
                        ),
                        "06_training_precision": DropdownInput(
                            name="training_precision",
                            display_name="Training Precision",
                            options=["bf16-mixed", "fp16-mixed", "fp32"],
                            value="bf16-mixed",
                            required=True,
                        ),
                        "07_lora_adapter_dim": IntInput(
                            name="lora_adapter_dim",
                            display_name="LoRA Adapter Dimension",
                            info="LoRA adapter dimension (only shown for LoRA fine-tuning)",
                            value=32,
                            required=False,
                            advanced=True,
                        ),
                        "08_lora_alpha": IntInput(
                            name="lora_alpha",
                            display_name="LoRA Alpha",
                            info="LoRA alpha parameter (only shown for LoRA fine-tuning)",
                            value=16,
                            required=False,
                            advanced=True,
                        ),
                        "09_lora_target_modules": StrInput(
                            name="lora_target_modules",
                            display_name="LoRA Target Modules",
                            info="Comma-separated list of target modules for LoRA (e.g., q_proj,v_proj)",
                            value="q_proj,v_proj",
                            required=False,
                            advanced=True,
                        ),
                    },
                }
            }
        }
    )


class NvidiaCustomizerComponent(Component):
    """NeMo Customizer component for LLM fine-tuning.

    This component provides:
    1. **Target Selection**: Fetches available base models for customization
    2. **Config Selection**: Shows existing training configurations or allows creating new ones
    3. **Job Creation**: Creates customization jobs using selected targets and configurations

    Features:
    - Reusable training configurations
    - Enhanced LoRA support with conditional parameters
    - Dataset integration (connected or existing)
    - Job monitoring with completion tracking
    """

    display_name = "NeMo Customizer"
    description = "LLM fine-tuning using NeMo customizer microservice"
    icon = "NVIDIA"
    name = "NVIDIANeMoCustomizer"
    beta = True

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
            name="namespace",
            display_name="Namespace",
            info="Namespace for the dataset and output model",
            advanced=True,
            value="default",
            required=True,
        ),
        DropdownInput(
            name="target",
            display_name="Base Model",
            info="Select a base model for customization",
            options=[],
            refresh_button=True,
            required=True,
            combobox=True,
        ),
        DropdownInput(
            name="config",
            display_name="Training Configuration",
            info="Select an existing configuration or create a new one",
            options=[],
            refresh_button=True,
            required=True,
            combobox=True,
            dialog_inputs=asdict(NewConfigInput()),
        ),
        StrInput(
            name="fine_tuned_model_name",
            display_name="Output Model Name",
            info="Enter the name to reference the output fine tuned model, ex: `imdb-data@v1`",
            required=True,
        ),
        HandleInput(
            name="dataset",
            display_name="Dataset",
            info="Dataset from NeMo Dataset Creator (optional - if not provided, will use Existing Dataset)",
            required=False,
            input_types=["Data"],
        ),
        DropdownInput(
            name="existing_dataset",
            display_name="Existing Dataset",
            info="Select an existing dataset from NeMo Data Store to use for training",
            options=[],
            refresh_button=True,
            combobox=True,
            required=False,
        ),
        IntInput(
            name="epochs",
            display_name="Fine tuning cycles",
            info="Number of cycle to run through the training data.",
            value=5,
        ),
        IntInput(
            name="batch_size",
            display_name="Batch size",
            info="The number of samples used in each training iteration",
            value=16,
            advanced=True,
        ),
        FloatInput(
            name="learning_rate",
            display_name="Learning Rate",
            info="The learning rate for training",
            value=0.0001,
            advanced=True,
        ),
        IntInput(
            name="adapter_dim",
            display_name="LoRA Adapter Dimension",
            info="LoRA adapter dimension (only for LoRA fine-tuning)",
            value=32,
            advanced=True,
        ),
        IntInput(
            name="alpha",
            display_name="LoRA Alpha",
            info="LoRA alpha parameter (only for LoRA fine-tuning)",
            value=16,
            advanced=True,
        ),
        BoolInput(
            name="sequence_packing_enabled",
            display_name="Enable Sequence Packing",
            info="Enable sequence packing for training",
            value=False,
            advanced=True,
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
        Output(display_name="Job Details", name="job_info", method="customize"),
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

    def get_nemo_client(self) -> nemo_microservices.AsyncNeMoMicroservices:
        """Get an authenticated NeMo microservices client."""
        return nemo_microservices.AsyncNeMoMicroservices(
            base_url=self.base_url,
        )

    async def update_build_config(self, build_config, field_value=None, field_name=None):
        """Update build config to fetch available options."""
        if not hasattr(self, "auth_token") or not self.auth_token:
            return build_config

        try:
            if field_name == "base_url":
                # Fetch targets when base_url changes
                targets = await self.fetch_available_targets()
                build_config["target"]["options"] = targets

                # Fetch existing datasets
                existing_datasets = await self.fetch_existing_datasets()
                build_config["existing_dataset"]["options"] = existing_datasets

            elif field_name == "target":
                # Fetch targets when target field is refreshed
                targets = await self.fetch_available_targets()
                build_config["target"]["options"] = targets

                # Fetch configs when target changes
                if field_value:
                    configs = await self.fetch_available_configs(field_value)
                    build_config["config"]["options"] = configs

            elif field_name == "config":
                # Fetch configs when config field is refreshed
                # If a target is selected, filter configs by that target
                target_value = getattr(self, "target", None)
                configs = await self.fetch_available_configs(target_value)
                build_config["config"]["options"] = configs

            elif field_name == "existing_dataset":
                # Fetch existing datasets when existing_dataset field is refreshed
                existing_datasets = await self.fetch_existing_datasets()
                build_config["existing_dataset"]["options"] = existing_datasets

        except (nemo_microservices.APIError, httpx.HTTPStatusError, httpx.RequestError) as exc:
            error_msg = f"Error updating build config: {exc}"
            logger.exception(error_msg)
            self.log(error_msg)

        return build_config

    async def update_dialog_config(self, dialog_config, field_value=None, field_name=None):
        """Update dialog config for conditional field visibility in popup."""
        try:
            if field_name == "03_finetuning_type":
                # Show/hide LoRA-specific fields based on fine-tuning type
                if field_value == "lora":
                    # Show LoRA fields
                    dialog_config["template"]["07_lora_adapter_dim"]["required"] = True
                    dialog_config["template"]["08_lora_alpha"]["required"] = True
                    dialog_config["template"]["09_lora_target_modules"]["required"] = True
                else:
                    # Hide LoRA fields
                    dialog_config["template"]["07_lora_adapter_dim"]["required"] = False
                    dialog_config["template"]["08_lora_alpha"]["required"] = False
                    dialog_config["template"]["09_lora_target_modules"]["required"] = False

            elif field_name == "02_training_type":
                # Update fine-tuning type options based on training type
                if field_value == "sft":
                    dialog_config["template"]["03_finetuning_type"]["options"] = ["all_weights", "lora"]
                elif field_value == "dpo":
                    dialog_config["template"]["03_finetuning_type"]["options"] = ["all_weights"]
                # Add more training types as needed

            elif field_name == "init":
                # Initialize training type options from API
                try:
                    training_types = await self.fetch_available_training_types()
                    if training_types:
                        dialog_config["template"]["02_training_type"]["options"] = training_types
                except (nemo_microservices.APIError, httpx.HTTPStatusError, httpx.RequestError) as exc:
                    logger.warning("Failed to fetch training types for dialog: %s", exc)

        except (nemo_microservices.APIError, httpx.HTTPStatusError, httpx.RequestError) as exc:
            error_msg = f"Error updating dialog config: {exc}"
            logger.exception(error_msg)

        return dialog_config

    async def fetch_available_targets(self) -> list[str]:
        """Fetch available base models for customization."""
        try:
            nemo_client = self.get_nemo_client()
            response = await nemo_client.customization.targets.list(extra_headers=self.get_auth_headers())
            targets = []
            if hasattr(response, "data") and response.data:
                for target in response.data:
                    # Format: "target_name@version" or just "target_name"
                    target_name = getattr(target, "name", "")
                    if target_name:
                        targets.append(target_name)
                return targets
            return targets  # noqa: TRY300
        except (nemo_microservices.APIError, httpx.HTTPStatusError, httpx.RequestError) as exc:
            logger.warning("Failed to fetch targets: %s", exc)
            return []

    async def fetch_available_configs(self, target_name: str | None = None) -> list[str]:
        """Fetch available configurations for customization."""
        try:
            nemo_client = self.get_nemo_client()
            response = await nemo_client.customization.configs.list(extra_headers=self.get_auth_headers())
            configs = []
            if hasattr(response, "data") and response.data:
                for config in response.data:
                    config_name = getattr(config, "name", "")
                    config_target = getattr(config, "target", None)

                    # If target_name is provided, filter by target
                    if target_name and config_target:
                        target_id = getattr(config_target, "id", "")
                        # Get the target ID for the selected target name
                        target_id_for_selection = await self._get_target_id(target_name)
                        if target_id == target_id_for_selection:
                            configs.append(config_name)
                    # If no target_name provided, return all configs
                    elif config_name:
                        configs.append(config_name)
                return configs
            return configs  # noqa: TRY300
        except (nemo_microservices.APIError, httpx.HTTPStatusError, httpx.RequestError) as exc:
            logger.warning("Failed to fetch configs: %s", exc)
            return []

    async def fetch_available_training_types(self) -> list[str]:
        """Fetch available training types from NeMo service."""
        try:
            nemo_client = self.get_nemo_client()
            response = await nemo_client.customization.training_types.list(extra_headers=self.get_auth_headers())
            return [tt.name for tt in response.data] if hasattr(response, "data") else []
        except (nemo_microservices.APIError, httpx.HTTPStatusError, httpx.RequestError) as exc:
            logger.warning("Failed to fetch training types: %s", exc)
            return []

    async def fetch_existing_datasets(self) -> list[str]:
        """Fetch existing datasets from NeMo service."""
        try:
            nemo_client = self.get_nemo_client()
            response = await nemo_client.datasets.list(extra_headers=self.get_auth_headers())
            return [dataset.name for dataset in response.data] if hasattr(response, "data") else []
        except (nemo_microservices.APIError, httpx.HTTPStatusError, httpx.RequestError) as exc:
            logger.warning("Failed to fetch existing datasets: %s", exc)
            return []

    async def _validate_config_target_compatibility(self, config_id: str, target_id: str):
        """Validate that the selected config is compatible with the selected target."""
        try:
            nemo_client = self.get_nemo_client()
            # NOTE: The use of _get here is a workaround due to the lack of a public get method
            # on AsyncConfigsResource in the current nemo_microservices SDK. If a public get method
            # becomes available in a future SDK version, this should be updated to use it instead.
            response = await nemo_client.customization.configs._get(
                config_id=config_id, extra_headers=self.get_auth_headers()
            )
            config_target_id = getattr(response, "target", {}).get("id", "")

            if config_target_id != target_id:
                error_msg = (
                    f"Selected configuration is not compatible with the selected target. "
                    f"Config target ID: {config_target_id}, Selected target ID: {target_id}"
                )
                raise ValueError(error_msg)

        except ValueError:
            raise
        except (nemo_microservices.APIError, httpx.HTTPStatusError, httpx.RequestError) as exc:
            logger.warning("Could not validate config-target compatibility: %s", exc)

    async def _create_new_config(self, config_data: dict) -> str:
        """Create a new training configuration and return its ID."""
        try:
            nemo_client = self.get_nemo_client()

            # Extract config parameters from dialog data
            config_name = config_data.get("01_config_name", "new-config")
            training_type = config_data.get("02_training_type", "sft")
            finetuning_type = config_data.get("03_finetuning_type", "lora")
            max_seq_length = config_data.get("04_max_seq_length", 4096)
            prompt_template = config_data.get("05_prompt_template", "{prompt} {completion}")
            training_precision = config_data.get("06_training_precision", "bf16-mixed")

            # Build config parameters
            config_params = {
                "training_type": training_type,
                "finetuning_type": finetuning_type,
                "max_seq_length": max_seq_length,
                "prompt_template": prompt_template,
                "training_precision": training_precision,
            }

            # Add LoRA parameters if using LoRA
            if finetuning_type == "lora":
                lora_adapter_dim = config_data.get("07_lora_adapter_dim", 32)
                lora_alpha = config_data.get("08_lora_alpha", 16)
                lora_target_modules = config_data.get("09_lora_target_modules", "q_proj,v_proj")

                config_params["lora"] = {
                    "adapter_dim": lora_adapter_dim,
                    "alpha": lora_alpha,
                    "target_modules": lora_target_modules.split(",") if lora_target_modules else ["q_proj", "v_proj"],
                }

            # Create the config
            response = await nemo_client.customization.configs.create(
                name=config_name,
                namespace=getattr(self, "namespace", "default"),
                params=config_params,
                extra_headers=self.get_auth_headers(),
            )

            config_id = response.id
            self.log(f"Created new config with ID: {config_id}")
            return config_id  # noqa: TRY300

        except (nemo_microservices.APIError, httpx.HTTPStatusError, httpx.RequestError) as exc:
            error_msg = f"Error creating config: {exc}"
            self.log(error_msg)
            raise ValueError(error_msg) from exc

    def extract_dataset_info(self, dataset_input):
        """Extract dataset information from the provided dataset."""
        dataset_type_error = "Dataset input must be a Data object"
        dataset_dict_error = "Dataset data must be a dictionary"
        dataset_name_error = "Dataset must contain 'dataset_name' field"
        dataset_namespace_error = "Dataset must contain 'namespace' field"

        if not isinstance(dataset_input, Data):
            raise TypeError(dataset_type_error)

        dataset_data = dataset_input.data if hasattr(dataset_input, "data") else dataset_input

        if not isinstance(dataset_data, dict):
            raise TypeError(dataset_dict_error)

        # Extract required fields from dataset
        dataset_name = dataset_data.get("dataset_name")
        dataset_namespace = dataset_data.get("namespace")

        if not dataset_name:
            raise ValueError(dataset_name_error)

        if not dataset_namespace:
            raise ValueError(dataset_namespace_error)

        return dataset_name, dataset_namespace

    async def customize(self) -> Data:
        """Create a customization job using the selected target and configuration."""
        # Validate required fields
        auth_token_error = "Authentication token is required"  # noqa: S105
        base_url_error = "Base URL is required"
        target_error = "Target selection is required"
        config_error = "Configuration selection is required"
        output_model_error = "Output model name is required"
        dataset_error = "Either a connected dataset or existing dataset must be provided"
        existing_dataset_error = "Existing dataset selection is required"

        if not hasattr(self, "auth_token") or not self.auth_token:
            raise ValueError(auth_token_error)

        if not hasattr(self, "base_url") or not self.base_url:
            raise ValueError(base_url_error)

        if not hasattr(self, "target") or not self.target:
            raise ValueError(target_error)

        if not hasattr(self, "config") or not self.config:
            raise ValueError(config_error)

        if not hasattr(self, "fine_tuned_model_name") or not self.fine_tuned_model_name:
            raise ValueError(output_model_error)

        # Check if we have a dataset (either connected or existing)
        if not hasattr(self, "dataset") and not hasattr(self, "existing_dataset"):
            raise ValueError(dataset_error)

        try:
            nemo_client = self.get_nemo_client()
            # Get target ID
            target_id = await self._get_target_id(self.target)

            # Handle configuration (existing or new)
            if isinstance(self.config, dict):
                # Create new configuration
                config_id = await self._create_new_config(self.config)
            else:
                # Use existing configuration
                config_id = await self._get_target_id(self.config)
                # Validate that the selected config is compatible with the selected target
                await self._validate_config_target_compatibility(config_id, target_id)

            # Prepare dataset information
            if hasattr(self, "dataset") and self.dataset:
                # Use connected dataset
                dataset_name, dataset_namespace = self.extract_dataset_info(self.dataset)
            else:
                # Use existing dataset
                if not hasattr(self, "existing_dataset") or not self.existing_dataset:
                    raise ValueError(existing_dataset_error)
                dataset_name = self.existing_dataset
                dataset_namespace = getattr(self, "namespace", "default")

            # Build job data
            job_data = {
                "name": f"customization-{int(time.time())}",
                "description": f"Customization job for {self.target} using {self.config}",
                "target_id": target_id,
                "config_id": config_id,
                "dataset_name": dataset_name,
                "dataset_namespace": dataset_namespace,
                "output_model_name": self.fine_tuned_model_name,
                "epochs": getattr(self, "epochs", 5),
                "batch_size": getattr(self, "batch_size", 16),
                "learning_rate": getattr(self, "learning_rate", 0.0001),
                "adapter_dim": getattr(self, "adapter_dim", 32),
                "alpha": getattr(self, "alpha", 16),
                "sequence_packing_enabled": getattr(self, "sequence_packing_enabled", False),
            }

            # Create the job using base model and configuration IDs
            response = await nemo_client.customization.jobs.create(
                name=job_data["name"],
                description=job_data["description"],
                target_id=job_data["target_id"],
                config_id=job_data["config_id"],
                dataset_name=job_data["dataset_name"],
                dataset_namespace=job_data["dataset_namespace"],
                output_model_name=job_data["output_model_name"],
                epochs=job_data["epochs"],
                batch_size=job_data["batch_size"],
                learning_rate=job_data["learning_rate"],
                adapter_dim=job_data["adapter_dim"],
                alpha=job_data["alpha"],
                sequence_packing_enabled=job_data["sequence_packing_enabled"],
                extra_headers=self.get_auth_headers(),
            )

            job_id = response.id
            self.log(f"Created customization job with ID: {job_id}")

            # Wait for completion if requested
            if getattr(self, "wait_for_completion", True):
                max_wait_time = getattr(self, "max_wait_time_minutes", 30)
                job_result = await self.wait_for_job_completion(job_id, max_wait_time)
                result_dict = {
                    "job_id": job_id,
                    "status": "completed",
                    "result": job_result,
                }
            else:
                result_dict = {
                    "job_id": job_id,
                    "status": "created",
                    "message": "Job created successfully. Use wait_for_completion=True to wait for completion.",
                }

            # Convert datetime objects to strings for JSON serialization
            def convert_datetime_to_string(obj):
                if hasattr(obj, "isoformat"):
                    return obj.isoformat()
                return str(obj)

            # Recursively convert datetime objects in the result
            if isinstance(result_dict, dict):
                for key, value in result_dict.items():
                    if isinstance(value, dict):
                        result_dict[key] = convert_datetime_to_string(value)
                    elif hasattr(value, "isoformat"):
                        result_dict[key] = value.isoformat()

            return Data(data=result_dict)

        except nemo_microservices.APIError as exc:
            # Check if the error is due to a 409 Conflict (model name already exists)
            if "409" in str(exc) or "conflict" in str(exc).lower() or "already exists" in str(exc).lower():
                conflict_msg = (
                    "Received conflict error. Output model name already exists. "
                    "Retry with a different output model name"
                )
                logger.warning(conflict_msg)
                error_msg = (
                    f"There is already a fined tuned model with name {self.fine_tuned_model_name}. "
                    f"Please use a different name for the output model."
                )
                raise ValueError(error_msg) from exc
            # Handle other API errors
            error_msg = f"NeMo microservices error during job creation: {exc}"
            logger.exception(error_msg)
            raise ValueError(error_msg) from exc

        except httpx.HTTPStatusError as exc:
            # Handle HTTP errors
            error_msg = f"HTTP error during job creation: {exc}"
            logger.exception(error_msg)
            raise ValueError(error_msg) from exc

        except httpx.RequestError as exc:
            # Catch any other unexpected errors
            error_msg = f"Unexpected error during job creation: {exc}"
            logger.exception(error_msg)
            raise ValueError(error_msg) from exc

    async def _get_target_id(self, target_name: str) -> str:
        """Get target ID from target name."""
        try:
            nemo_client = self.get_nemo_client()
            response = await nemo_client.customization.targets.list(extra_headers=self.get_auth_headers())
            if hasattr(response, "data") and response.data:
                for target in response.data:
                    if getattr(target, "name", "") == target_name:
                        return getattr(target, "id", target_name)
                return target_name  # Not found, assume already an ID
            return target_name  # No data, assume already an ID  # noqa: TRY300
        except (nemo_microservices.APIError, httpx.HTTPStatusError, httpx.RequestError) as exc:
            logger.warning("Failed to get target ID for %s: %s", target_name, exc)
            return target_name

    async def wait_for_job_completion(
        self, job_id: str, max_wait_time_minutes: int = 30, poll_interval_seconds: int = 15
    ) -> dict:
        """Wait for job completion with timeout."""
        start_time = time.time()
        max_wait_time_seconds = max_wait_time_minutes * 60

        while True:
            # Check if we've exceeded the maximum wait time
            elapsed_time = time.time() - start_time
            if elapsed_time > max_wait_time_seconds:
                timeout_msg = f"Job {job_id} did not complete within {max_wait_time_minutes} minutes"
                logger.warning(timeout_msg)
                raise TimeoutError(timeout_msg)

            try:
                # Get job status
                nemo_client = self.get_nemo_client()
                response = await nemo_client.customization.jobs.get(
                    job_id=job_id, extra_headers=self.get_auth_headers()
                )
                job_status = response.model_dump()

                # Check job status
                status = job_status.get("status", "unknown")
                logger.info("Job %s status: %s", job_id, status)

                if status == "completed":
                    logger.info("Job %s completed successfully!", job_id)
                    return job_status
                if status == "failed":
                    error_msg = f"Job {job_id} failed with status: {status}"
                    logger.error(error_msg)
                    raise ValueError(error_msg)
                if status in ["running", "pending", "queued"]:
                    logger.info("Job %s is still %s. Waiting %s seconds...", job_id, status, poll_interval_seconds)
                    await asyncio.sleep(poll_interval_seconds)
                else:
                    logger.warning("Unknown job status: %s. Waiting %s seconds...", status, poll_interval_seconds)
                    await asyncio.sleep(poll_interval_seconds)

            except nemo_microservices.APIError:
                logger.exception("NeMo microservices error while checking job status")
                await asyncio.sleep(poll_interval_seconds)
            except (asyncio.CancelledError, RuntimeError, OSError):
                logger.exception("Unexpected error while checking job status")
                await asyncio.sleep(poll_interval_seconds)

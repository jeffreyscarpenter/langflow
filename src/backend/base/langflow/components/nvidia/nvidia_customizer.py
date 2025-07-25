import asyncio
import json
import time

import httpx
import nemo_microservices
from huggingface_hub import HfApi
from loguru import logger
from nemo_microservices import AsyncNeMoMicroservices

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

        # Always add our custom authentication header
        headers["Authorization"] = f"Bearer {self.auth_token}"

        return headers

    def _request_wrapper(self, method, url, *args, **kwargs):
        """Override to add authentication headers to all requests."""
        headers = kwargs.get("headers", {})
        headers["Authorization"] = f"Bearer {self.auth_token}"
        kwargs["headers"] = headers
        return super()._request_wrapper(method, url, *args, **kwargs)


class NvidiaCustomizerComponent(Component):
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
        DropdownInput(
            name="model_name",
            display_name="Base Model Name",
            info="Base model to fine tune",
            refresh_button=True,
            required=True,
            options=[],
            combobox=True,
        ),
        DropdownInput(
            name="training_type",
            display_name="Training Type",
            info="Select the type of training to use",
            refresh_button=True,
            required=True,
            options=[],
            combobox=True,
        ),
        DropdownInput(
            name="fine_tuning_type",
            display_name="Fine Tuning Type",
            info="Select the fine tuning type to use",
            required=True,
            options=[],
            combobox=True,
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
            info="The number of samples used in each training iteration",
            value=0.0001,
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
        """Get headers with authentication token."""
        return {
            "accept": "application/json",
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.auth_token}",
        }

    def get_nemo_client(self) -> AsyncNeMoMicroservices:
        """Get an authenticated NeMo microservices client."""
        return AsyncNeMoMicroservices(
            base_url=self.base_url,
            # No inference_base_url needed for dataset operations
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

    async def create_namespace_with_nemo_client(self, nemo_client: AsyncNeMoMicroservices, namespace: str):
        """Create namespace using NeMo client."""
        try:
            await nemo_client.datastore.namespaces.create(namespace=namespace)
            logger.info("Created namespace: %s", namespace)
        except (nemo_microservices.APIError, httpx.HTTPStatusError, httpx.RequestError) as exc:
            # Namespace might already exist, which is fine
            logger.info("Namespace %s might already exist: %s", namespace, exc)

    async def create_datastore_namespace_with_nemo_client(self, nemo_client: AsyncNeMoMicroservices, namespace: str):
        """Create datastore namespace using NeMo client."""
        try:
            await nemo_client.datastore.namespaces.create(namespace=namespace)
            logger.info("Created datastore namespace: %s", namespace)
        except (nemo_microservices.APIError, httpx.HTTPStatusError, httpx.RequestError) as exc:
            # Namespace might already exist, which is fine
            logger.info("Datastore namespace %s might already exist: %s", namespace, exc)

    async def update_build_config(self, build_config, field_value=None, field_name=None):  # noqa: ARG002
        """Update build config to fetch available options."""
        if field_name == "base_url":
            # Fetch available models when base_url changes
            try:
                models = await self.fetch_available_models()
                build_config["model_name"]["options"] = models
            except (nemo_microservices.APIError, httpx.HTTPStatusError, httpx.RequestError) as exc:
                logger.warning("Failed to fetch models: %s", exc)

            # Fetch available training types
            try:
                training_types = await self.fetch_available_training_types()
                build_config["training_type"]["options"] = training_types
            except (nemo_microservices.APIError, httpx.HTTPStatusError, httpx.RequestError) as exc:
                logger.warning("Failed to fetch training types: %s", exc)

            # Fetch available fine tuning types
            try:
                fine_tuning_types = await self.fetch_available_fine_tuning_types()
                build_config["fine_tuning_type"]["options"] = fine_tuning_types
            except (nemo_microservices.APIError, httpx.HTTPStatusError, httpx.RequestError) as exc:
                logger.warning("Failed to fetch fine tuning types: %s", exc)

            # Fetch existing datasets
            try:
                existing_datasets = await self.fetch_existing_datasets()
                build_config["existing_dataset"]["options"] = existing_datasets
            except (nemo_microservices.APIError, httpx.HTTPStatusError, httpx.RequestError) as exc:
                logger.warning("Failed to fetch existing datasets: %s", exc)

        return build_config

    async def customize(self) -> Data:
        if not self.auth_token:
            error_msg = "Missing authentication token"
            raise ValueError(error_msg)

        base_url = self.base_url.rstrip("/")

        fine_tuned_model_name = self.fine_tuned_model_name

        if not fine_tuned_model_name:
            error_msg = "Missing Output Model Name"
            raise ValueError(error_msg)

        namespace = self.namespace
        if not self.namespace:
            error_msg = "Missing Namespace"
            raise ValueError(error_msg)

        if not self.model_name:
            error_msg = "Missing Base Model Name"
            raise ValueError(error_msg)

        if not (self.training_type and self.fine_tuning_type):
            error_msg = "Refresh and select the training type and fine tuning type"
            raise ValueError(error_msg)

        # Check if we have a dataset input or existing dataset selection
        dataset_input = getattr(self, "dataset", None)
        existing_dataset = getattr(self, "existing_dataset", None)

        if dataset_input is None and existing_dataset is None:
            error_msg = "Either dataset connection or existing dataset selection must be provided"
            raise ValueError(error_msg)

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
            dataset_name = dataset_data.get("dataset_name")
            dataset_namespace = dataset_data.get("namespace")

            if not dataset_name:
                error_msg = "Dataset must contain 'dataset_name' field"
                raise ValueError(error_msg)

            if not dataset_namespace:
                error_msg = "Dataset must contain 'namespace' field"
                raise ValueError(error_msg)

            # Use dataset namespace if different from component namespace
            effective_namespace = dataset_namespace
            logger.info("Using dataset connection: %s from namespace: %s", dataset_name, effective_namespace)

        else:
            # Use selected existing dataset
            dataset_name = existing_dataset
            effective_namespace = namespace
            logger.info("Using existing dataset: %s from namespace: %s", dataset_name, effective_namespace)

        output_model = f"{effective_namespace}/{fine_tuned_model_name}"

        description = f"Fine tuning base model {self.model_name} using dataset {dataset_name}"
        # Build the data payload following API spec
        data = {
            "name": f"customization-{fine_tuned_model_name}",
            "description": description,
            "config": f"meta/{self.model_name}",
            "dataset": {"name": dataset_name, "namespace": effective_namespace},
            "hyperparameters": {
                "training_type": self.training_type,
                "finetuning_type": self.fine_tuning_type,
                "epochs": int(self.epochs),
                "batch_size": int(self.batch_size),
                "learning_rate": float(self.learning_rate),
            },
            "output_model": output_model,
        }

        # Add `adapter_dim` if fine tuning type is "lora"
        if self.fine_tuning_type == "lora":
            data["hyperparameters"]["lora"] = {"adapter_dim": 16}  # type: ignore[index]
        try:
            formatted_data = json.dumps(data, indent=2)
            logger.info("Sending customization request using NeMo client")
            logger.info("Request payload: %s", formatted_data)

            # Use NeMo client for job creation
            nemo_client = self.get_nemo_client()
            response = await nemo_client.customization.jobs.create(
                name=data["name"],
                description=data["description"],
                config=data["config"],
                dataset=data["dataset"],
                hyperparameters=data["hyperparameters"],
                output_model=data["output_model"],
                extra_headers={"Authorization": f"Bearer {self.auth_token}"},
            )

            # Process a successful response
            result_dict = response.model_dump()

            # Convert datetime objects to strings for JSON serialization
            def convert_datetime_to_string(obj):
                if isinstance(obj, dict):
                    return {k: convert_datetime_to_string(v) for k, v in obj.items()}
                if isinstance(obj, list):
                    return [convert_datetime_to_string(item) for item in obj]
                if hasattr(obj, "isoformat"):  # datetime objects
                    return obj.isoformat()
                return obj

            result_dict = convert_datetime_to_string(result_dict)

            id_value = result_dict["id"]
            result_dict["url"] = f"{base_url}/v1/customization/jobs/{id_value}/status"

            # Check if we should wait for job completion
            wait_for_completion = getattr(self, "wait_for_completion", False)
            logger.info("Wait for completion setting: %s", wait_for_completion)
            if wait_for_completion:
                logger.info("Wait for completion enabled. Waiting for job %s to complete...", id_value)
                try:
                    max_wait_time = getattr(self, "max_wait_time_minutes", 30)
                    logger.info("Starting wait_for_job_completion with max_wait_time: %s", max_wait_time)
                    # Wait for job completion
                    final_job_result = await self.wait_for_job_completion(
                        job_id=id_value, max_wait_time_minutes=max_wait_time
                    )
                    # Update result_dict with final job status
                    result_dict.update(final_job_result)
                    logger.info("Job %s completed successfully!", id_value)
                except TimeoutError as exc:
                    logger.warning("Job %s did not complete within timeout: %s", id_value, exc)
                    # Continue with the original result (job created but not completed)
                except ValueError as exc:
                    logger.exception("Job %s failed", id_value)
                    # Re-raise the ValueError to indicate job failure
                    error_msg = f"Job {id_value} failed: {exc}"
                    raise ValueError(error_msg) from exc
                except (asyncio.CancelledError, RuntimeError, OSError):
                    logger.exception("Unexpected error while waiting for job completion")
                    # Continue with the original result
            else:
                logger.info("Wait for completion disabled. Job %s created successfully.", id_value)

        except nemo_microservices.APIError as exc:
            # Log the request details for debugging
            logger.exception("NeMo microservices error occurred during job creation")
            logger.exception("Request payload was: %s", formatted_data)

            # Check if the error is due to a 409 Conflict (model name already exists)
            if "409" in str(exc) or "conflict" in str(exc).lower() or "already exists" in str(exc).lower():
                conflict_msg = (
                    "Received conflict error. Output model name already exists. "
                    "Retry with a different output model name"
                )
                logger.warning(conflict_msg)
                error_msg = (
                    f"There is already a fined tuned model with name {output_model}. "
                    f"Please choose a different Output Model Name."
                )
                raise ValueError(error_msg) from exc

            error_msg = f"NeMo microservices error during job creation: {exc}"
            logger.exception(error_msg)
            raise ValueError(error_msg) from exc

        except (httpx.TimeoutException, httpx.HTTPStatusError, httpx.RequestError) as exc:
            # Keep httpx error handling for backward compatibility
            error_msg = f"HTTP error during job creation: {exc}"
            logger.exception(error_msg)
            raise ValueError(error_msg) from exc

        except Exception as exc:
            # Catch any other unexpected errors
            error_msg = f"Unexpected error during job creation: {exc}"
            logger.exception(error_msg)
            raise ValueError(error_msg) from exc

        return Data(data=result_dict)

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
                    job_id=job_id, extra_headers={"Authorization": f"Bearer {self.auth_token}"}
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

    async def fetch_available_models(self) -> list[str]:
        """Fetch available models from NeMo service."""
        try:
            nemo_client = self.get_nemo_client()
            response = await nemo_client.models.list(extra_headers={"Authorization": f"Bearer {self.auth_token}"})
            return [model.name for model in response.data] if hasattr(response, "data") else []
        except (nemo_microservices.APIError, httpx.HTTPStatusError, httpx.RequestError) as exc:
            logger.warning("Failed to fetch models: %s", exc)
            return []

    async def fetch_available_training_types(self) -> list[str]:
        """Fetch available training types from NeMo service."""
        try:
            nemo_client = self.get_nemo_client()
            response = await nemo_client.customization.training_types.list(
                extra_headers={"Authorization": f"Bearer {self.auth_token}"}
            )
            return [tt.name for tt in response.data] if hasattr(response, "data") else []
        except (nemo_microservices.APIError, httpx.HTTPStatusError, httpx.RequestError) as exc:
            logger.warning("Failed to fetch training types: %s", exc)
            return []

    async def fetch_available_fine_tuning_types(self) -> list[str]:
        """Fetch available fine tuning types from NeMo service."""
        try:
            nemo_client = self.get_nemo_client()
            response = await nemo_client.customization.finetuning_types.list(
                extra_headers={"Authorization": f"Bearer {self.auth_token}"}
            )
            return [ftt.name for ftt in response.data] if hasattr(response, "data") else []
        except (nemo_microservices.APIError, httpx.HTTPStatusError, httpx.RequestError) as exc:
            logger.warning("Failed to fetch fine tuning types: %s", exc)
            return []

    async def fetch_existing_datasets(self) -> list[str]:
        """Fetch existing datasets from NeMo service."""
        try:
            nemo_client = self.get_nemo_client()
            response = await nemo_client.datasets.list(extra_headers={"Authorization": f"Bearer {self.auth_token}"})
            return [dataset.name for dataset in response.data] if hasattr(response, "data") else []
        except (nemo_microservices.APIError, httpx.HTTPStatusError, httpx.RequestError) as exc:
            logger.warning("Failed to fetch existing datasets: %s", exc)
            return []

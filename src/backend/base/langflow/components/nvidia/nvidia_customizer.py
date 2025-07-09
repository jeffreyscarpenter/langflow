import asyncio
import json
import logging
import uuid
from datetime import datetime, timezone
from io import BytesIO
from unittest.mock import patch

import httpx
import pandas as pd
import requests
from huggingface_hub import HfApi

from langflow.custom import Component
from langflow.io import (
    DataInput,
    DatasetInput,
    DropdownInput,
    FloatInput,
    IntInput,
    Output,
    SecretStrInput,
    StrInput,
)
from langflow.schema import Data
from langflow.services.deps import get_settings_service
from langflow.services.nemo_microservices_factory import get_nemo_service

logger = logging.getLogger(__name__)


def create_auth_interceptor(auth_token, namespace):
    """Create a function to intercept HTTP requests and add auth headers for namespace URLs."""
    original_request = requests.Session.request

    def patched_request(self, method, url, *args, **kwargs):
        # Check if URL contains the namespace path
        if url and namespace and f"/{namespace}/" in url:
            headers = kwargs.get("headers", {})
            if auth_token:
                headers["Authorization"] = f"Bearer {auth_token}"
                kwargs["headers"] = headers
                logger.info("Intercepted and added Authorization header for namespace URL: %s", url)

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
        return super()._build_hf_headers(
            token=token,
            library_name=library_name,
            library_version=library_version,
            user_agent=user_agent,
        )

    def _request_wrapper(self, method, url, *args, **kwargs):
        """Override to intercept requests and add auth headers for namespace URLs."""
        # Check if URL contains the namespace path
        if url and self.namespace and f"/{self.namespace}/" in url:
            headers = kwargs.get("headers", {})
            if self.auth_token:
                headers["Authorization"] = f"Bearer {self.auth_token}"
                kwargs["headers"] = headers
                logger.info("Added Authorization header for namespace URL: %s", url)

        return super()._request_wrapper(method, url, *args, **kwargs)


class NvidiaCustomizerComponent(Component):
    display_name = "NeMo Customizer"
    description = "LLM fine-tuning using NeMo customizer microservice"
    icon = "NVIDIA"
    name = "NVIDIANeMoCustomizer"
    beta = True

    # Use settings to determine mock vs real API
    def _get_use_mock(self):
        """Get whether to use mock service from settings."""
        settings_service = get_settings_service()
        return settings_service.settings.nemo_use_mock

    chunk_number = 1

    inputs = [
        SecretStrInput(
            name="api_key",
            display_name="API Key",
            info="API key for NeMo services authentication (only used when USE_MOCK=False)",
            required=False,
        ),
        StrInput(
            name="base_url",
            display_name="Base API URL",
            info="Base URL for the NeMo services (only used when USE_MOCK=False)",
            required=False,
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
        DatasetInput(
            name="existing_dataset",
            display_name="Existing Dataset",
            info="Select an existing dataset from NeMo Data Store to use instead of uploading new training data",
            dataset_types=["fileset"],
        ),
        DataInput(
            name="training_data",
            display_name="Training Data",
            info="Provide training data to create a new dataset, or leave empty if using an existing dataset",
            is_list=True,
            required=False,
        ),
        DropdownInput(
            name="model_name",
            display_name="Base Model Name",
            info="Base model to fine tune (click refresh to load options)",
            refresh_button=True,
            required=True,
        ),
        DropdownInput(
            name="training_type",
            display_name="Training Type",
            info="Select the type of training to use (click refresh to load options)",
            refresh_button=True,
            required=True,
        ),
        DropdownInput(
            name="fine_tuning_type",
            display_name="Fine Tuning Type",
            info="Select the fine tuning type to use (click refresh to load options)",
            required=True,
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
    ]

    outputs = [
        Output(display_name="Job Info", name="job_info", method="customize"),
    ]

    def get_auth_headers(self):
        """Get headers with authentication token if using real API."""
        headers = {"accept": "application/json", "Content-Type": "application/json"}
        if not self._get_use_mock() and hasattr(self, "api_key") and self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        return headers

    def get_base_url(self):
        """Get the appropriate base URL based on mock setting."""
        if self._get_use_mock():
            return "mock-url"  # Use mock service
        return (
            self.base_url
            if hasattr(self, "base_url") and self.base_url
            else "https://us-west-2.api-dev.ai.datastax.com/nvidia/nemo"
        )

    async def update_build_config(self, build_config, field_value, field_name=None):
        """Updates the component's configuration based on the selected option or refresh button."""
        self.log(f"update_build_config called with field_name={field_name}, field_value={field_value}")
        self.log(f"USE_MOCK={self._get_use_mock()}")

        if self._get_use_mock():
            # Use mock service for configuration updates
            try:
                if field_name == "model_name":
                    self.log("Refreshing model names from mock service")
                    nemo_service = get_nemo_service()
                    models_data = await nemo_service.get_customization_configs()
                    self.log(f"Mock service returned {len(models_data.get('data', []))} models")
                    model_names = [model["base_model"] for model in models_data.get("data", [])]
                    build_config["model_name"]["options"] = model_names
                    self.log(f"Updated model_name dropdown options: {model_names}")

                    # Also update training_type and fine_tuning_type options if a model is selected
                    if field_value:
                        selected_model = next(
                            (model for model in models_data.get("data", []) if model["base_model"] == field_value),
                            None,
                        )
                        if selected_model:
                            training_types = selected_model.get("training_types", [])
                            build_config["training_type"]["options"] = training_types
                            self.log(f"Updated training_type dropdown options: {training_types}")

                            fine_tuning_types = selected_model.get("finetuning_types", [])
                            build_config["fine_tuning_type"]["options"] = fine_tuning_types
                            self.log(f"Updated fine_tuning_type dropdown options: {fine_tuning_types}")

                elif field_name == "training_type":
                    self.log("Refreshing training types from mock service")
                    nemo_service = get_nemo_service()
                    models_data = await nemo_service.get_customization_configs()
                    selected_model_name = getattr(self, "model_name", None)
                    self.log(f"Selected model name: {selected_model_name}")
                    if selected_model_name:
                        selected_model = next(
                            (
                                model
                                for model in models_data.get("data", [])
                                if model["base_model"] == selected_model_name
                            ),
                            None,
                        )
                        if selected_model:
                            training_types = selected_model.get("training_types", [])
                            build_config["training_type"]["options"] = training_types
                            self.log(f"Updated training_type dropdown options: {training_types}")

                            fine_tuning_types = selected_model.get("finetuning_types", [])
                            build_config["fine_tuning_type"]["options"] = fine_tuning_types
                            self.log(f"Updated fine_tuning_type dropdown options: {fine_tuning_types}")
                        else:
                            self.log(f"Model {selected_model_name} not found in available models")
                    else:
                        self.log("No model name selected")

            except Exception as exc:
                error_msg = f"Error refreshing model names from mock service: {exc}"
                self.log(error_msg)
                raise ValueError(error_msg) from exc

        else:
            # Use real API
            self.log("Using real API for configuration updates")
            if not hasattr(self, "api_key") or not self.api_key:
                self.log("No API key provided for real API")
                return build_config

            base_url = self.get_base_url()
            if not base_url:
                self.log("No base URL provided for real API")
                return build_config

            models_url = f"{base_url}/v1/customization/configs"

            try:
                if field_name == "model_name":
                    self.log(f"Refreshing model names from endpoint {models_url}, value: {field_value}")

                    # Use a synchronous HTTP client with authentication
                    with httpx.Client(timeout=5.0) as client:
                        response = client.get(models_url, headers=self.get_auth_headers())
                        response.raise_for_status()

                        models_data = response.json()
                        # Use the config name which includes version and GPU type
                        # (e.g., "llama-3.1-8b-instruct@v1.0.0+A100")
                        model_names = [model["name"] for model in models_data.get("data", []) if "name" in model]

                        build_config["model_name"]["options"] = model_names

                    self.log(f"Updated model_name dropdown options: {model_names}")

                elif field_name == "training_type":
                    self.log(f"Refreshing training types from endpoint {models_url}")
                    # Use a synchronous HTTP client with authentication
                    with httpx.Client(timeout=5.0) as client:
                        response = client.get(models_url, headers=self.get_auth_headers())
                        response.raise_for_status()

                        models_data = response.json()

                        # Logic to update `training_type` dropdown based on selected model
                        selected_model_name = getattr(self, "model_name", None)
                        self.log(f"Selected model name: {selected_model_name}")
                        if selected_model_name:
                            # Find the selected model in the response
                            # Find model by config name (which includes version and GPU type)
                            selected_model = next(
                                (
                                    model
                                    for model in models_data.get("data", [])
                                    if model.get("name") == selected_model_name
                                ),
                                None,
                            )

                            if selected_model:
                                # Extract training types and fine-tuning types from training_options
                                training_options = selected_model.get("training_options", [])
                                training_types = list(
                                    {opt.get("training_type") for opt in training_options if opt.get("training_type")}
                                )
                                finetuning_types = list(
                                    {
                                        opt.get("finetuning_type")
                                        for opt in training_options
                                        if opt.get("finetuning_type")
                                    }
                                )

                                build_config["training_type"]["options"] = training_types
                                self.log(f"Updated training_type dropdown options: {training_types}")
                                build_config["fine_tuning_type"]["options"] = finetuning_types
                                self.log(f"Updated fine_tuning_type dropdown options: {finetuning_types}")
                            else:
                                self.log(f"Model {selected_model_name} not found in available models")
                        else:
                            self.log("No model name selected")

            except httpx.HTTPStatusError as exc:
                error_msg = f"HTTP error {exc.response.status_code} on {models_url}"
                self.log(error_msg)
                raise ValueError(error_msg) from exc
            except (httpx.RequestError, ValueError) as exc:
                exception_str = str(exc)
                error_msg = f"Error refreshing model names: {exception_str}"
                self.log(error_msg)
                raise ValueError(error_msg) from exc

        self.log(f"Final build_config: {build_config}")
        return build_config

    async def customize(self) -> dict:
        if self._get_use_mock():
            # Use mock service
            settings_service = get_settings_service()
            nemo_customizer_url = settings_service.settings.nemo_customizer_url
            nemo_data_store_url = settings_service.settings.nemo_data_store_url
            nemo_entity_store_url = settings_service.settings.nemo_entity_store_url
        else:
            # Use real API
            if not hasattr(self, "api_key") or not self.api_key:
                error_msg = "Missing API key for real API mode"
                raise ValueError(error_msg)

            base_url = self.get_base_url()
            if not base_url:
                error_msg = "Missing base URL for real API mode"
                raise ValueError(error_msg)

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

        # Check if user selected an existing dataset or provided training data
        existing_dataset = getattr(self, "existing_dataset", None)
        if existing_dataset:
            # Use existing dataset
            self.log(f"Using existing dataset: {existing_dataset}")
            dataset_name = existing_dataset
        elif self.training_data is not None and len(self.training_data) > 0:
            # Process and upload new dataset
            if self.USE_MOCK:
                dataset_name = await self.process_dataset(nemo_data_store_url, nemo_entity_store_url)
            else:
                dataset_name = await self.process_dataset(base_url)
        else:
            error_msg = "Either select an existing dataset or provide training data to create a new dataset"
            raise ValueError(error_msg)

        if self.USE_MOCK:
            customizations_url = f"{nemo_customizer_url}/v1/customization/jobs"
        else:
            customizations_url = f"{base_url}/v1/customization/jobs"

        error_code_already_present = 409
        output_model = f"{namespace}/{fine_tuned_model_name}"

        description = f"Fine tuning base model {self.model_name} using dataset {dataset_name}"

        # Build the data payload (using real API format for both mock and real)
        data = {
            "name": f"customization-{fine_tuned_model_name}",
            "description": description,
            "config": f"meta/{self.model_name}",
            "dataset": {"name": dataset_name, "namespace": namespace},
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
            data["hyperparameters"]["lora"] = {"adapter_dim": 16}

        try:
            formatted_data = json.dumps(data, indent=2)
            self.log(f"Sending customization request to URL: {customizations_url}")
            self.log(f"Request payload: {formatted_data}")

            async with httpx.AsyncClient(timeout=5.0) as client:
                response = await client.post(customizations_url, headers=self.get_auth_headers(), json=data)

            # For non-409 responses, raise for any HTTP error status
            response.raise_for_status()

            # Process a successful response
            result = response.json()
            formatted_result = json.dumps(result, indent=2)
            self.log(f"Received successful response: {formatted_result}")

            result_dict = {**result}
            id_value = result_dict["id"]
            result_dict["url"] = f"{customizations_url}/{id_value}/status"

            # Track the job for monitoring in Langflow dashboard (only for mock mode)
            if self.USE_MOCK:
                try:
                    job_metadata = {
                        "source": "nvidia_customizer_component",
                        "config": self.model_name,
                        "dataset": dataset_name,
                        "training_type": self.training_type,
                        "fine_tuning_type": self.fine_tuning_type,
                        "epochs": int(self.epochs),
                        "batch_size": int(self.batch_size),
                        "learning_rate": float(self.learning_rate),
                        "output_model": output_model,
                        "namespace": namespace,
                        "description": description,
                    }

                    nemo_service = get_nemo_service()
                    await nemo_service.track_customizer_job(id_value, job_metadata)
                    self.log(f"Successfully tracked job {id_value} for monitoring")

                    # Add tracking info to the result
                    result_dict["tracking_enabled"] = True
                    result_dict["monitoring_url"] = "/nemo?tab=jobs"

                except (httpx.RequestError, httpx.HTTPStatusError, ValueError) as tracking_error:
                    # Don't fail the job creation if tracking fails
                    self.log(f"Warning: Failed to track job for monitoring: {tracking_error}")
                    result_dict["tracking_enabled"] = False

        except httpx.TimeoutException as exc:
            error_msg = f"Request to {customizations_url} timed out"
            self.log(error_msg)
            raise ValueError(error_msg) from exc

        except httpx.HTTPStatusError as exc:
            # Log the request details for debugging
            if not self._get_use_mock():
                logger.exception("HTTP error occurred. Request was sent to: %s", customizations_url)
                logger.exception("Request payload was: %s", formatted_data)
                logger.exception("Request headers were: %s", json.dumps(self.get_auth_headers(), indent=2))

            # Check if the error is due to a 409 Conflict
            if exc.response.status_code == error_code_already_present:
                if not self._get_use_mock():
                    logger.exception(
                        "Received HTTP 409. Conflict output model name. " "Retry with a different output model name"
                    )
                self.log("Received HTTP 409. Conflict output model name. Retry with a different output model name")
                error_msg = (
                    f"There is already a fined tuned model with name {fine_tuned_model_name} "
                    f"Please choose a different Output Model Name."
                )
                raise ValueError(error_msg) from exc
            status_code = exc.response.status_code
            response_content = exc.response.text
            error_msg = f"HTTP error {status_code} on URL: {customizations_url}. Response content: {response_content}"
            if not self._get_use_mock():
                logger.exception(error_msg)
            self.log(error_msg)
            raise ValueError(error_msg) from exc

        except (httpx.RequestError, ValueError) as exc:
            exception_str = str(exc)
            error_msg = f"An unexpected error occurred on URL {customizations_url}: {exception_str}"
            self.log(error_msg)
            raise ValueError(error_msg) from exc
        else:
            return result_dict

    async def create_namespace(self, namespace: str, base_url: str):
        """Checks and creates namespace in entity-store with authentication."""
        url = f"{base_url}/v1/namespaces"

        try:
            async with httpx.AsyncClient() as client:
                response = await client.get(f"{url}/{namespace}", headers=self.get_auth_headers())
                http_status_code_non_found = 404
                if response.status_code == http_status_code_non_found:
                    self.log(f"Namespace not found, creating namespace:  {namespace}")
                    create_payload = {"namespace": namespace}
                    create_response = await client.post(url, json=create_payload, headers=self.get_auth_headers())
                    create_response.raise_for_status()
                else:
                    response.raise_for_status()

        except httpx.HTTPStatusError as e:
            exception_str = str(e)
            error_msg = f"Error processing namespace: {exception_str}"
            self.log(error_msg)
            raise ValueError(error_msg) from e

    async def create_datastore_namespace(self, namespace: str, base_url: str):
        """Checks and creates namespace in datastore with authentication."""
        url = f"{base_url}/v1/datastore/namespaces"

        try:
            async with httpx.AsyncClient() as client:
                response = await client.get(f"{url}/{namespace}", headers=self.get_auth_headers())
                http_status_code_non_found = 404
                if response.status_code == http_status_code_non_found:
                    self.log(f"Datastore namespace not found, creating namespace: {namespace}")
                    create_payload = {"namespace": namespace}
                    create_response = await client.post(url, json=create_payload, headers=self.get_auth_headers())
                    create_response.raise_for_status()
                else:
                    response.raise_for_status()

        except httpx.HTTPStatusError as e:
            exception_str = str(e)
            error_msg = f"Error processing datastore namespace: {exception_str}"
            self.log(error_msg)
            raise ValueError(error_msg) from e

    async def process_dataset(self, base_url_or_data_store_url: str, entity_store_url: str | None = None) -> str:
        """Asynchronously processes and uploads the dataset for training(90%) and validation(10%).

        If the total valid record count is less than 10, at least one record is added to validation.

        Args:
            base_url_or_data_store_url (str): Base URL for real API or Data store URL for mock
            entity_store_url (str): Entity store URL (only used for mock mode)
        """
        try:
            # Inputs and repo setup
            dataset_name = str(uuid.uuid4())

            if self._get_use_mock():
                # Use mock service
                hf_api = HfApi(endpoint=f"{base_url_or_data_store_url}/v1/hf", token="")
                await self.create_namespace(self.namespace, base_url_or_data_store_url)
            else:
                # Use real API with authentication
                hf_api = AuthenticatedHfApi(
                    endpoint=f"{base_url_or_data_store_url}/v1/hf",
                    auth_token=self.api_key,
                    namespace=self.namespace,
                    token=self.api_key,
                )
                await self.create_namespace(self.namespace, base_url_or_data_store_url)
                await self.create_datastore_namespace(self.namespace, base_url_or_data_store_url)

            repo_id = f"{self.namespace}/{dataset_name}"
            repo_type = "dataset"
            hf_api.create_repo(repo_id, repo_type=repo_type, exist_ok=True)
        except Exception as exc:
            exception_str = str(exc)
            error_msg = f"An unexpected error occurred while creating repo: {exception_str}"
            self.log(error_msg)
            raise ValueError(error_msg) from exc

        try:
            chunk_size = 100000  # Ensure chunk_size is an integer
            self.log(f"repo_id : {repo_id}")

            tasks = []

            # =====================================================
            # STEP 1: Build a list of valid records from training_data
            # =====================================================
            valid_records = []
            for data_obj in self.training_data or []:
                # Skip non-Data objects
                if not isinstance(data_obj, Data):
                    self.log(f"Skipping non-Data object in training data, but got: {data_obj}")
                    continue

                # Extract only "prompt" and "completion" fields if present
                filtered_data = {
                    "prompt": getattr(data_obj, "prompt", None),
                    "completion": getattr(data_obj, "completion", None),
                }
                if filtered_data["prompt"] is not None and filtered_data["completion"] is not None:
                    valid_records.append(filtered_data)

            total_records = len(valid_records)
            min_records_process = 2
            min_records_validation = 10
            if total_records < min_records_process:
                error_msg = f"Not enough records for processing. Record count : {total_records}"
                raise ValueError(error_msg)

            # =====================================================
            # STEP 2: Split into validation (10%) and training (90%)
            # =====================================================
            # If the total size is less than 10, force at least one record into validation.
            validation_count = 1 if total_records < min_records_validation else max(1, int(round(total_records * 0.1)))

            # For simplicity, we take the first validation_count records for validation.
            # (You could also randomize the order if needed.)
            validation_records = valid_records[:validation_count]
            training_records = valid_records[validation_count:]

            # =====================================================
            # STEP 3: Process training data in chunks (90%)
            # =====================================================
            chunk = []
            is_validation = False
            for record in training_records:
                chunk.append(record)
                if len(chunk) == chunk_size:
                    chunk_df = pd.DataFrame(chunk)
                    task = self.upload_chunk(
                        chunk_df,
                        self.chunk_number,
                        dataset_name,
                        repo_id,
                        hf_api,
                        is_validation,
                    )
                    tasks.append(task)
                    chunk = []  # Reset the chunk
                    self.chunk_number += 1

            # Process any remaining training records
            if chunk:
                chunk_df = pd.DataFrame(chunk)
                task = self.upload_chunk(
                    chunk_df,
                    self.chunk_number,
                    dataset_name,
                    repo_id,
                    hf_api,
                    is_validation,
                )
                tasks.append(task)

            # Await all training upload tasks
            await asyncio.gather(*tasks)

            # =====================================================
            # STEP 4: Upload validation data (without chunking)
            # =====================================================
            if validation_records:
                is_validation = True
                validation_df = pd.DataFrame(validation_records)
                await self.upload_chunk(validation_df, 1, dataset_name, repo_id, hf_api, is_validation)

        except Exception as exc:
            exception_str = str(exc)
            error_msg = f"An unexpected error occurred during processing/upload: {exception_str}"
            self.log(error_msg)
            raise ValueError(error_msg) from exc

        # =====================================================
        # STEP 5: Post dataset info to the entity registry
        # =====================================================
        try:
            file_url = f"hf://datasets/{repo_id}"
            description = f"Dataset loaded using the input data {dataset_name}"

            if self._get_use_mock():
                entity_registry_url = f"{entity_store_url}/v1/datasets"
            else:
                entity_registry_url = f"{base_url_or_data_store_url}/v1/datasets"

            create_payload = {
                "name": dataset_name,
                "namespace": self.namespace,
                "description": description,
                "files_url": file_url,
                "format": "jsonl",
                "project": dataset_name,
            }

            async with httpx.AsyncClient() as client:
                response = await client.post(entity_registry_url, json=create_payload, headers=self.get_auth_headers())

            success_status_code = 200
            if response.status_code == success_status_code:
                logger.info("Dataset uploaded successfully in %s chunks", self.chunk_number)
            else:
                logger.warning("Failed to upload files. Status code: %s", response.status_code)
                response.raise_for_status()

            logger.info("All data has been processed and uploaded successfully.")
        except Exception as exc:
            exception_str = str(exc)
            error_msg = f"An unexpected error occurred while posting to entity service: {exception_str}"
            self.log(error_msg)
            raise ValueError(error_msg) from exc

        return dataset_name

    async def upload_chunk(self, chunk_df, chunk_number, file_name_prefix, repo_id, hf_api, is_validation):
        """Asynchronously uploads a chunk of data to the REST API."""
        try:
            json_data = chunk_df.to_json(orient="records", lines=True)

            # Build file paths
            if is_validation:
                file_name_training = f"validation/{file_name_prefix}_validation.jsonl"
            else:
                file_name_training = f"training/{file_name_prefix}_chunk_{chunk_number}.jsonl"

            # Prepare BytesIO objects
            training_file_obj = BytesIO(json_data.encode("utf-8"))
            commit_message = f"Updated training file at time: {datetime.now(timezone.utc).strftime('%Y%m%d%H%M%S')}"

            if self._get_use_mock():
                # Use standard HF API for mock
                try:
                    hf_api.upload_file(
                        path_or_fileobj=training_file_obj,
                        path_in_repo=file_name_training,
                        repo_id=repo_id,
                        repo_type="dataset",
                        commit_message=commit_message,
                    )
                finally:
                    training_file_obj.close()
            else:
                # Use authenticated HF API with request patching for real API
                try:
                    # Patch requests to intercept namespace URLs
                    patched_request = create_auth_interceptor(self.api_key, self.namespace)

                    with patch.object(requests.Session, "request", patched_request):
                        hf_api.upload_file(
                            path_or_fileobj=training_file_obj,
                            path_in_repo=file_name_training,
                            repo_id=repo_id,
                            repo_type="dataset",
                            commit_message=commit_message,
                        )
                finally:
                    training_file_obj.close()

        except Exception:
            logger.exception("An error occurred while uploading chunk %s", chunk_number)

    async def fetch_existing_datasets(self, nemo_data_store_url: str) -> list[str]:
        """Fetch existing datasets from the NeMo Data Store.

        Args:
            nemo_data_store_url (str): Base URL for the NeMo Data Store API

        Returns:
            List of dataset names available for training
        """
        try:
            # Use the proper service factory to get datasets

            # Get the current user ID and session (this would need to be passed in)
            # For now, we'll use the direct API approach
            datasets_url = f"{nemo_data_store_url}/datasets"
            async with httpx.AsyncClient(timeout=5.0) as client:
                response = await client.get(datasets_url, headers=self.get_auth_headers())
                response.raise_for_status()

                datasets_data = response.json()
                return [dataset["name"] for dataset in datasets_data if "name" in dataset]

        except httpx.RequestError as exc:
            self.log(f"An error occurred while requesting datasets: {exc}")
            return []
        except httpx.HTTPStatusError as exc:
            self.log(f"Error response {exc.response.status_code} while requesting datasets: {exc}")
            return []

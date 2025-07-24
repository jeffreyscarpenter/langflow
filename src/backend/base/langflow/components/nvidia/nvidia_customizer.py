import asyncio
import json
import logging
import uuid
from datetime import datetime, timezone
from io import BytesIO

import httpx
import pandas as pd
from huggingface_hub import HfApi
from nemo_microservices import AsyncNeMoMicroservices, NeMoMicroservicesError

from langflow.custom import Component
from langflow.io import (
    DataInput,
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
        """Override to intercept requests and add auth headers for namespace URLs."""
        # Check if URL contains the namespace path
        if url and self.namespace and f"/{self.namespace}/" in url:
            # Add authorization header for namespace requests
            headers = kwargs.get("headers", {})
            if "Authorization" not in headers:
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

    chunk_number = 1

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
            info="New Dataset from NeMo Dataset Creator (optional - if not provided, will use Existing Dataset)",
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
        DataInput(
            name="training_data",
            display_name="Training Data",
            info="Raw training data (prompt/completion pairs) - for backward compatibility",
            is_list=True,
            required=False,
            advanced=True,
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
    ]

    outputs = [
        Output(display_name="Customization Data", name="job_info", method="customize"),
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
                logger.info("Entity namespace already exists: %s", namespace)
                return

            await nemo_client.namespaces.create(
                id=namespace,
                description=f"Entity namespace for {namespace} resources",
                extra_headers={"Authorization": f"Bearer {self.auth_token}"},
            )
            logger.info("Created entity namespace: %s", namespace)
        except NeMoMicroservicesError as exc:
            if "already exists" in str(exc).lower():
                logger.info("Entity namespace already exists: %s", namespace)
            else:
                error_msg = f"Failed to create entity namespace: {exc}"
                raise ValueError(error_msg) from exc

    async def create_datastore_namespace_with_nemo_client(self, nemo_client: AsyncNeMoMicroservices, namespace: str):  # noqa: ARG002
        """Create datastore namespace using direct HTTP request."""
        try:
            # Use the correct endpoint from the last commit
            nds_url = f"{self.base_url}/v1/datastore/namespaces"

            headers = {"Authorization": f"Bearer {self.auth_token}"}
            data = {"namespace": namespace}

            async with httpx.AsyncClient() as client:
                # First check if namespace exists
                try:
                    response = await client.get(f"{nds_url}/{namespace}", headers=headers)
                    if response.status_code == 200:  # noqa: PLR2004
                        logger.info("Datastore namespace already exists: %s", namespace)
                        return
                except httpx.HTTPError:
                    # Namespace doesn't exist, create it
                    pass

                # Create the namespace
                logger.info("Creating datastore namespace at URL: %s", nds_url)
                response = await client.post(nds_url, headers=headers, json=data)

                logger.info("Response status: %s", response.status_code)

                if response.status_code in (200, 201):
                    logger.info("Created datastore namespace: %s", namespace)
                    return
                if response.status_code in (409, 422):
                    logger.info("Datastore namespace already exists: %s", namespace)
                    return
                error_msg = f"Failed to create datastore namespace: {response.status_code} - {response.text}"
                logger.error(error_msg)
                raise ValueError(error_msg)

        except httpx.HTTPError as exc:
            error_msg = f"HTTP error creating datastore namespace: {exc}"
            logger.exception(error_msg)
            raise ValueError(error_msg) from exc
        except Exception as exc:
            error_msg = f"Unexpected error creating datastore namespace: {exc}"
            logger.exception(error_msg)
            raise ValueError(error_msg) from exc

    async def update_build_config(self, build_config, field_value=None, field_name=None):  # noqa: ARG002
        """Updates the component's configuration based on the selected option or refresh button."""
        try:
            if field_name == "existing_dataset":
                # Defensive check for auth_token
                if not hasattr(self, "auth_token") or not self.auth_token:
                    logger.warning("Authentication token not set, cannot fetch datasets")
                    return build_config

                # Defensive check for namespace
                if not hasattr(self, "namespace") or not self.namespace:
                    logger.warning("Namespace not set, cannot fetch datasets")
                    return build_config

                # Refresh dataset options for existing dataset dropdown
                logger.info("Refreshing datasets for field: %s", field_name)
                try:
                    dataset_options = await self.fetch_existing_datasets()
                    build_config["existing_dataset"]["options"] = dataset_options
                    msg = f"Updated dataset options: {dataset_options}"
                    logger.info(msg)
                except Exception:
                    logger.exception("Error fetching datasets")
                    # Return empty options instead of failing completely
                    build_config["existing_dataset"]["options"] = []

            elif field_name == "model_name" or field_name is None:
                # Defensive check for auth_token
                if not hasattr(self, "auth_token") or not self.auth_token:
                    logger.warning("Authentication token not set, cannot fetch models")
                    return build_config

                # Use NeMo client for model fetching
                nemo_client = self.get_nemo_client()
                response = await nemo_client.customization.configs.list(
                    extra_headers={"Authorization": f"Bearer {self.auth_token}"}
                )

                # Use the config name which includes version and GPU type
                # (e.g., "llama-3.1-8b-instruct@v1.0.0+A100")
                model_names = [model.name for model in response.data if hasattr(model, "name") and model.name]

                build_config["model_name"]["options"] = model_names

            elif field_name == "training_type":
                # Use NeMo client for model fetching
                nemo_client = self.get_nemo_client()
                response = await nemo_client.customization.configs.list(
                    extra_headers={"Authorization": f"Bearer {self.auth_token}"}
                )

                # Logic to update `training_type` dropdown based on selected model
                selected_model_name = getattr(self, "model_name", None)
                if selected_model_name:
                    # Find the selected model in the response
                    # Find model by config name (which includes version and GPU type)
                    selected_model = next(
                        (
                            model
                            for model in response.data
                            if hasattr(model, "name") and model.name == selected_model_name
                        ),
                        None,
                    )

                    if selected_model:
                        # Extract training types and fine-tuning types from training_options
                        training_options = getattr(selected_model, "training_options", [])
                        training_types = list(
                            {
                                getattr(opt, "training_type", None)
                                for opt in training_options
                                if hasattr(opt, "training_type") and opt.training_type
                            }
                        )
                        finetuning_types = list(
                            {
                                getattr(opt, "finetuning_type", None)
                                for opt in training_options
                                if hasattr(opt, "finetuning_type") and opt.finetuning_type
                            }
                        )

                        build_config["training_type"]["options"] = training_types
                        build_config["fine_tuning_type"]["options"] = finetuning_types

        except NeMoMicroservicesError as exc:
            error_msg = f"NeMo microservices error while fetching models: {exc}"
            logger.exception(error_msg)
            raise ValueError(error_msg) from exc
        except (httpx.HTTPStatusError, httpx.RequestError) as exc:
            # Keep httpx error handling for any remaining httpx calls
            error_msg = f"HTTP error while fetching models: {exc}"
            logger.exception(error_msg)
            raise ValueError(error_msg) from exc
        except ValueError as exc:
            error_msg = f"Error refreshing model names: {exc}"
            logger.exception(error_msg)
            raise ValueError(error_msg) from exc
        except Exception as exc:
            error_msg = f"Unexpected error during build config update: {exc}"
            logger.exception(error_msg)
            return build_config

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

            # Check if we have a dataset input, existing dataset selection, or training data
        dataset_input = getattr(self, "dataset", None)
        existing_dataset = getattr(self, "existing_dataset", None)
        training_data = getattr(self, "training_data", None)

        if dataset_input is None and existing_dataset is None and training_data is None:
            error_msg = "Either dataset connection, existing dataset selection, or training_data must be provided"
            raise ValueError(error_msg)

        # Priority: 1. Dataset connection, 2. Existing dataset selection, 3. Training data
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

        elif existing_dataset is not None:
            # Use selected existing dataset
            dataset_name = existing_dataset
            effective_namespace = namespace
            logger.info("Using existing dataset: %s from namespace: %s", dataset_name, effective_namespace)

        else:
            # Process and upload the dataset if training_data is provided
            if training_data is None:
                error_msg = "Training data is empty, cannot customize the model"
                raise ValueError(error_msg)

            dataset_name = await self.process_dataset(base_url)
            effective_namespace = namespace

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

            # Also use self.log for component logging
            # self.log("Sending customization request using NeMo client") # Removed
            # self.log(f"Request payload: {formatted_data}") # Removed

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

            # self.log(f"Received successful response: {formatted_result}") # Removed

            id_value = result_dict["id"]
            result_dict["url"] = f"{base_url}/v1/customization/jobs/{id_value}/status"

        except NeMoMicroservicesError as exc:
            # Log the request details for debugging
            logger.exception("NeMo microservices error occurred during job creation")
            logger.exception("Request payload was: %s", formatted_data)

            # Check if the error is due to a 409 Conflict (model name already exists)
            if "409" in str(exc) or "conflict" in str(exc).lower() or "already exists" in str(exc).lower():
                conflict_msg = (
                    "Received conflict error. Output model name already exists. "
                    "Retry with a different output model name"
                )
                logger.exception(conflict_msg)
                # self.log(conflict_msg) # Removed
                error_msg = (
                    f"There is already a fined tuned model with name {output_model}. "
                    f"Please choose a different Output Model Name."
                )
                raise ValueError(error_msg) from exc

            error_msg = f"NeMo microservices error during job creation: {exc}"
            logger.exception(error_msg)
            # self.log(error_msg) # Removed
            raise ValueError(error_msg) from exc

        except (httpx.TimeoutException, httpx.HTTPStatusError, httpx.RequestError) as exc:
            # Keep httpx error handling for backward compatibility
            error_msg = f"HTTP error during job creation: {exc}"
            logger.exception(error_msg)
            # self.log(error_msg) # Removed
            raise ValueError(error_msg) from exc

        except ValueError as exc:
            exception_str = str(exc)
            error_msg = f"An unexpected error occurred during job creation: {exception_str}"
            # self.log(error_msg) # Removed
            raise ValueError(error_msg) from exc
        else:
            return Data(data=result_dict)

    async def fetch_existing_datasets(self) -> list[str]:
        """Fetch existing datasets from the NeMo Data Store.

        Returns:
            List of dataset names available for training
        """
        # Defensive checks
        if not hasattr(self, "namespace") or not self.namespace:
            error_msg = "Namespace not set for fetching datasets"
            logger.warning(error_msg)
            return []

        if not hasattr(self, "auth_token") or not self.auth_token:
            error_msg = "Authentication token not set for fetching datasets"
            logger.warning(error_msg)
            return []

        try:
            logger.info("Fetching datasets from namespace: %s", self.namespace)
            # Use NeMo client for dataset fetching
            nemo_client = self.get_nemo_client()

            # Build filter for namespace
            filter_params = {"namespace": self.namespace}

            response = await nemo_client.datasets.list(
                filter=filter_params, extra_headers={"Authorization": f"Bearer {self.auth_token}"}
            )

            # Extract dataset names from the response
            dataset_names = [dataset.name for dataset in response.data if hasattr(dataset, "name") and dataset.name]
            logger.info("Successfully fetched %d datasets: %s", len(dataset_names), dataset_names)

        except NeMoMicroservicesError as exc:
            error_msg = f"NeMo microservices error while fetching datasets: {exc}"
            logger.exception(error_msg)
            return []
        except (httpx.RequestError, httpx.HTTPStatusError) as exc:
            error_msg = f"HTTP error while fetching datasets: {exc}"
            logger.exception(error_msg)
            return []
        except (ValueError, TypeError) as exc:
            error_msg = f"Unexpected error while fetching datasets: {exc}"
            logger.exception(error_msg)
            return []
        except Exception as exc:
            error_msg = f"Unknown error while fetching datasets: {exc}"
            logger.exception(error_msg)
            return []
        else:
            return dataset_names

    async def create_namespace(self, namespace: str, base_url: str):  # noqa: ARG002
        """Checks and creates namespace in entity-store with authentication using NeMo client."""
        entity_client = self.get_entity_client()
        await self.create_namespace_with_nemo_client(entity_client, namespace)

    async def create_datastore_namespace(self, namespace: str, base_url: str):  # noqa: ARG002
        """Checks and creates namespace in datastore with authentication using direct HTTP request."""
        datastore_client = self.get_datastore_client()
        await self.create_datastore_namespace_with_nemo_client(datastore_client, namespace)

    async def process_dataset(self, base_url: str) -> str:
        """Asynchronously processes and uploads the dataset with authentication."""
        try:
            # Inputs and repo setup
            dataset_name = str(uuid.uuid4())

            # Initialize clients for dataset operations
            entity_client = self.get_entity_client()
            datastore_client = self.get_datastore_client()

            # Create namespaces using appropriate clients
            await self.create_namespace_with_nemo_client(entity_client, self.namespace)
            await self.create_datastore_namespace_with_nemo_client(datastore_client, self.namespace)

            # Create dataset repository using NeMo client
            repo_id = f"{self.namespace}/{dataset_name}"
            # Note: We'll create the dataset in the entity registry later with the HuggingFace URL
            # For now, we just use the HuggingFace API to create the repo
            logger.info("Will create dataset repository: %s", repo_id)

            # Still use HuggingFace API for file uploads (for now)
            hf_api = AuthenticatedHfApi(
                endpoint=f"{base_url}/v1/hf",
                auth_token=self.auth_token,
                namespace=self.namespace,
                token=self.auth_token,
            )
            hf_api.create_repo(repo_id, repo_type="dataset", exist_ok=True)
        except NeMoMicroservicesError as exc:
            error_msg = f"NeMo microservices error while creating repo: {exc}"
            logger.exception(error_msg)
            raise ValueError(error_msg) from exc
        except Exception as exc:
            exception_str = str(exc)
            error_msg = f"An unexpected error occurred while creating repo: {exception_str}"
            logger.exception(error_msg)
            raise ValueError(error_msg) from exc

        try:
            chunk_size = 100000  # Ensure chunk_size is an integer
            logger.info("repo_id : %s", repo_id)

            tasks = []

            # =====================================================
            # STEP 1: Build a list of valid records from training_data
            # =====================================================
            valid_records = []
            for data_obj in self.training_data or []:
                # Skip non-Data objects
                if not isinstance(data_obj, Data):
                    logger.warning("Skipping non-Data object in training data, but got: %s", data_obj)
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

        except NeMoMicroservicesError as exc:
            error_msg = f"NeMo microservices error during processing/upload: {exc}"
            logger.exception(error_msg)
            raise ValueError(error_msg) from exc
        except Exception as exc:
            exception_str = str(exc)
            error_msg = f"An unexpected error occurred during processing/upload: {exception_str}"
            logger.exception(error_msg)
            raise ValueError(error_msg) from exc

        # =====================================================
        # STEP 5: Post dataset info to the entity registry
        # =====================================================
        try:
            file_url = f"hf://datasets/{repo_id}"
            description = f"Dataset loaded using the input data {dataset_name}"

            # Use NeMo client for entity registry operations
            nemo_client = self.get_nemo_client()
            _response = await nemo_client.datasets.create(
                name=dataset_name,
                namespace=self.namespace,
                description=description,
                files_url=file_url,
                format="jsonl",
                project=dataset_name,
                extra_headers={"Authorization": f"Bearer {self.auth_token}"},
            )

            logger.info("Dataset uploaded successfully in %s chunks", self.chunk_number)
            logger.info("All data has been processed and uploaded successfully.")
        except NeMoMicroservicesError as exc:
            error_msg = f"NeMo microservices error while posting to entity service: {exc}"
            logger.exception(error_msg)
            raise ValueError(error_msg) from exc
        except Exception as exc:
            exception_str = str(exc)
            error_msg = f"An unexpected error occurred while posting to entity service: {exception_str}"
            logger.exception(error_msg)
            raise ValueError(error_msg) from exc

        return dataset_name

    async def upload_chunk(self, chunk_df, chunk_number, file_name_prefix, repo_id, hf_api, is_validation):
        """Asynchronously uploads a chunk of data using authenticated HF API."""
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

            # Use authenticated HuggingFace API directly (no more request patching)
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

        except Exception:
            logger.exception("An error occurred while uploading chunk %s", chunk_number)

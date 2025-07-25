import json
from datetime import datetime, timezone
from io import BytesIO

import nemo_microservices
import pandas as pd
from huggingface_hub import HfApi
from loguru import logger
from nemo_microservices import AsyncNeMoMicroservices

from langflow.custom import Component
from langflow.io import (
    DataInput,
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


class NvidiaDatasetCreatorComponent(Component):
    display_name = "NeMo Dataset Creator"
    description = "Create datasets in NeMo Data Store for use with NeMo Customizer and Evaluator components"
    icon = "NVIDIA"
    name = "NVIDIANeMoDatasetCreator"
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
            info="Namespace for the dataset",
            advanced=True,
            value="default",
            required=True,
        ),
        StrInput(
            name="dataset_name",
            display_name="Dataset Name",
            info="Name for the dataset (will be created if it doesn't exist)",
            required=True,
        ),
        StrInput(
            name="description",
            display_name="Description",
            info="Description of the dataset",
            value="Dataset created via Langflow NeMo Dataset Creator",
        ),
        DataInput(
            name="training_data",
            display_name="Training Data",
            info="Data for training (prompt/completion pairs)",
            is_list=True,
        ),
        DataInput(
            name="evaluation_data",
            display_name="Evaluation Data",
            info="Data for evaluation (prompt/ideal_response pairs)",
            is_list=True,
        ),
    ]

    outputs = [
        Output(display_name="Dataset Data", name="dataset_info", method="create_dataset"),
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
        )

    async def create_namespace_with_nemo_client(self, nemo_client: AsyncNeMoMicroservices, namespace: str):
        """Create entity namespace using NeMo client."""
        try:
            # First check if namespace exists
            try:
                await nemo_client.namespaces.retrieve(
                    namespace_id=namespace, extra_headers={"Authorization": f"Bearer {self.auth_token}"}
                )
            except nemo_microservices.APIError:
                # Namespace doesn't exist, create it
                pass
            else:
                logger.info(f"Entity namespace already exists: {namespace}")
                return

            await nemo_client.namespaces.create(
                id=namespace,
                description=f"Entity namespace for {namespace} resources",
                extra_headers={"Authorization": f"Bearer {self.auth_token}"},
            )
            logger.info(f"Created entity namespace: {namespace}")
        except nemo_microservices.APIError as exc:
            if "already exists" in str(exc).lower():
                logger.info(f"Entity namespace already exists: {namespace}")
            else:
                error_msg = f"Failed to create entity namespace: {exc}"
                raise ValueError(error_msg) from exc

    async def create_dataset(self) -> Data:
        """Create a dataset in NeMo Data Store."""
        if not self.auth_token:
            error_msg = "Missing authentication token"
            raise ValueError(error_msg)

        base_url = self.base_url.rstrip("/")
        namespace = self.namespace
        dataset_name = self.dataset_name
        description = getattr(self, "description", "Dataset created via Langflow NeMo Dataset Creator")

        if not dataset_name:
            error_msg = "Dataset name is required"
            raise ValueError(error_msg)

        # Check if we have any data to process
        training_data = getattr(self, "training_data", None)
        evaluation_data = getattr(self, "evaluation_data", None)

        if not training_data and not evaluation_data:
            error_msg = "Either training_data or evaluation_data must be provided"
            raise ValueError(error_msg)

        try:
            # Initialize client for dataset operations
            nemo_client = self.get_nemo_client()

            # Create namespaces using the client
            await self.create_namespace_with_nemo_client(nemo_client, namespace)

            # Create HuggingFace repo
            hf_api = AuthenticatedHfApi(
                endpoint=f"{base_url}/v1/hf",
                auth_token=self.auth_token,
                namespace=namespace,
                token=self.auth_token,
            )
            repo_id = f"{namespace}/{dataset_name}"
            repo_type = "dataset"
            hf_api.create_repo(repo_id, repo_type=repo_type, exist_ok=True)

            logger.info(f"Created/accessed repo: {repo_id}")

            # Process and upload data
            if training_data:
                await self.process_training_data(hf_api, repo_id, training_data)
                logger.info("Uploaded training data")

            if evaluation_data:
                await self.upload_evaluation_data(hf_api, repo_id, evaluation_data)
                logger.info("Uploaded evaluation data")

            # Register dataset in entity store using NeMo client
            file_url = f"hf://datasets/{repo_id}"

            response = await nemo_client.datasets.create(
                name=dataset_name,
                namespace=namespace,
                description=description,
                files_url=file_url,
                format="jsonl",
                project=dataset_name,
                extra_headers={"Authorization": f"Bearer {self.auth_token}"},
            )

            # Convert response to dictionary and handle datetime objects
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

            # Add additional metadata
            result_dict.update(
                {
                    "has_training_data": bool(training_data),
                    "has_evaluation_data": bool(evaluation_data),
                }
            )

            logger.info(f"Successfully created dataset: {dataset_name}")

            return Data(data=result_dict)

        except Exception as exc:
            exception_str = str(exc)
            error_msg = f"Error creating dataset: {exception_str}"
            logger.error(error_msg)
            raise ValueError(error_msg) from exc

    async def process_training_data(self, hf_api, repo_id: str, training_data):
        """Process and upload training data with chunking support."""
        import asyncio

        try:
            chunk_size = 100000  # Ensure chunk_size is an integer
            logger.info(f"Processing training data for repo: {repo_id}")

            tasks = []

            # =====================================================
            # STEP 1: Build a list of valid records from training_data
            # =====================================================
            valid_records = []
            for data_obj in training_data or []:
                # Skip non-Data objects
                if not isinstance(data_obj, Data):
                    logger.warning(f"Skipping non-Data object in training data: {data_obj}")
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
                        self.dataset_name,
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
                    self.dataset_name,
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
                await self.upload_chunk(validation_df, 1, self.dataset_name, repo_id, hf_api, is_validation)

        except Exception as exc:
            exception_str = str(exc)
            error_msg = f"An unexpected error occurred during processing/upload: {exception_str}"
            logger.error(error_msg)
            raise ValueError(error_msg) from exc

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

            # Use authenticated HuggingFace API directly
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

        except Exception:  # noqa: BLE001
            logger.error(f"An error occurred while uploading chunk {chunk_number}")

    async def upload_evaluation_data(self, hf_api, repo_id: str, evaluation_data):
        """Upload evaluation data to the dataset."""
        input_records = []
        output_records = []

        for data_obj in evaluation_data or []:
            if not isinstance(data_obj, Data):
                logger.warning(f"Skipping non-Data object in evaluation data: {data_obj}")
                continue

            filtered_data = {
                "prompt": getattr(data_obj, "prompt", None) or "",
                "ideal_response": getattr(data_obj, "ideal_response", None) or "",
                "category": getattr(data_obj, "category", "Generation") or "Generation",
                "source": getattr(data_obj, "source", None) or "",
                "response": getattr(data_obj, "response", None) or "",
                "llm_name": getattr(data_obj, "llm_name", None) or "",
            }

            if filtered_data["prompt"] and filtered_data["ideal_response"]:
                # Input file data
                input_records.append(
                    {
                        "prompt": filtered_data["prompt"],
                        "ideal_response": filtered_data["ideal_response"],
                        "category": filtered_data["category"],
                        "source": filtered_data["source"],
                    }
                )

                # Output file data (if response is provided)
                if filtered_data["response"]:
                    output_records.append(
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

        if input_records:
            # Upload input file
            input_file_buffer = BytesIO(json.dumps(input_records, indent=4).encode("utf-8"))
            try:
                hf_api.upload_file(
                    path_or_fileobj=input_file_buffer,
                    path_in_repo="input.json",
                    repo_id=repo_id,
                    repo_type="dataset",
                    commit_message=f"Evaluation input data at {datetime.now(timezone.utc).strftime('%Y%m%d%H%M%S')}",
                )
            finally:
                input_file_buffer.close()

        if output_records:
            # Upload output file
            output_file_buffer = BytesIO(json.dumps(output_records, indent=4).encode("utf-8"))
            try:
                hf_api.upload_file(
                    path_or_fileobj=output_file_buffer,
                    path_in_repo="output.json",
                    repo_id=repo_id,
                    repo_type="dataset",
                    commit_message=f"Evaluation output data at {datetime.now(timezone.utc).strftime('%Y%m%d%H%M%S')}",
                )
            finally:
                output_file_buffer.close()

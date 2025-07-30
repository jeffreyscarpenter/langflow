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


class NvidiaDatasetCreatorComponent(Component):
    display_name = "NeMo Dataset Creator"
    description = "Create datasets in NeMo Data Store for use with NeMo Customizer and Evaluator components"
    icon = "NVIDIA"
    name = "NVIDIANeMoDatasetCreator"
    beta = True

    chunk_number = 1

    inputs = [
        # Authentication and Configuration
        SecretStrInput(
            name="auth_token",
            display_name="Authentication Token",
            info="Bearer token for firewall authentication",
            required=True,
        ),
        StrInput(
            name="base_url",
            display_name="Base API URL",
            info="Base URL for the NeMo services",
            required=True,
            value="https://us-west-2.api-dev.ai.datastax.com/nvidia/nemo",
        ),
        StrInput(
            name="namespace",
            display_name="Namespace",
            info="Namespace for the dataset",
            required=True,
            value="default",
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
        # Data Input (single input with smart classification)
        HandleInput(
            name="data",
            display_name="Data",
            info="Data to upload (Data or DataFrame) - will be automatically classified",
            required=True,
            input_types=["Data", "DataFrame"],
        ),
        # Model Type Configuration
        DropdownInput(
            name="model_type",
            display_name="Model Type",
            info="Type of model this dataset is intended for (affects upload format)",
            options=["auto", "chat", "completion"],
            value="auto",
            required=False,
            advanced=True,
        ),
        # Split Configuration
        FloatInput(
            name="validation_split_ratio",
            display_name="Validation Split Ratio",
            info="Ratio of training data to use for validation (0.0-1.0). Set to 0 for no validation split",
            value=0.1,
            required=False,
            advanced=True,
        ),
        # Upload Configuration
        BoolInput(
            name="partial_success",
            display_name="Allow Partial Success",
            info="If True, upload data even if some records are invalid or unclassified",
            value=True,
            required=False,
            advanced=True,
        ),
        BoolInput(
            name="preserve_all_fields",
            display_name="Preserve All Fields",
            info="If True, include all fields from input data in dataset files. If False, only include essential fields (prompt/completion/messages)",
            value=True,
            required=False,
            advanced=True,
        ),
        IntInput(
            name="chunk_size",
            display_name="Upload Chunk Size",
            info="Number of records per upload chunk",
            value=100000,
            required=False,
            advanced=True,
        ),
    ]

    outputs = [
        Output(
            display_name="Dataset Info",
            name="dataset_info",
            method="create_dataset",
            info="Information about the created dataset",
        ),
    ]

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.upload_statistics = {
            "total_input_records": 0,
            "classified_records": {"training": 0, "evaluation": 0, "unknown": 0},
            "uploaded_records": {"training": 0, "validation": 0, "evaluation": 0},
            "discarded_records": {"training": 0, "evaluation": 0, "unknown": 0},
            "file_info": {},
            "split_config": {},
            "error_summary": {},
            "processing_time": 0.0,
            "model_type": None,
        }

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

    def convert_to_dataframe(self, input_data) -> pd.DataFrame:
        """Convert input data to DataFrame for processing."""
        if isinstance(input_data, pd.DataFrame):
            return input_data.copy()

        if isinstance(input_data, list):
            # Check if all items are Data objects
            if all(isinstance(item, Data) for item in input_data):
                # Convert list[Data] to DataFrame
                records = []
                for data_item in input_data:
                    if hasattr(data_item, "data"):
                        if isinstance(data_item.data, dict):
                            records.append(data_item.data)
                        elif isinstance(data_item.data, list):
                            records.extend(data_item.data)
                        else:
                            records.append({"data": data_item.data})
                    else:
                        records.append({"data": data_item})

                return pd.DataFrame(records)
            # Assume it's already a list of dictionaries
            return pd.DataFrame(input_data)

        # Handle single Data object
        if isinstance(input_data, Data):
            if hasattr(input_data, "data"):
                if isinstance(input_data.data, dict):
                    return pd.DataFrame([input_data.data])
                if isinstance(input_data.data, list):
                    return pd.DataFrame(input_data.data)
                return pd.DataFrame([{"data": input_data.data}])
            return pd.DataFrame([{"data": input_data}])

        error_msg = "Input data must be DataFrame, Data, or list[Data]"
        raise ValueError(error_msg)

    def detect_model_type(self, data: pd.DataFrame) -> str:
        """Detect whether data is intended for chat or completion models."""
        chat_indicators = 0
        completion_indicators = 0

        records = data.to_dict("records")
        for record in records:
            # Check for messages structure (chat model)
            if record.get("messages") or (
                isinstance(record.get("data"), dict) and "messages" in record.get("data", {})
            ):
                chat_indicators += 1

            # Check for prompt/completion structure (completion model)
            if ("prompt" in record and "completion" in record) or (
                isinstance(record.get("data"), dict)
                and "prompt" in record.get("data", {})
                and "completion" in record.get("data", {})
            ):
                completion_indicators += 1

        # Determine based on majority
        if chat_indicators > completion_indicators:
            return "chat"
        if completion_indicators > chat_indicators:
            return "completion"
        # Default to completion for backward compatibility
        return "completion"

    def determine_model_type(self, data: pd.DataFrame) -> str:
        """Determine model type with fallback to configuration."""
        if self.model_type == "auto":
            return self.detect_model_type(data)
        return self.model_type

    def classify_data(self, data: pd.DataFrame) -> dict:
        """Classify input data into training, evaluation, and unknown categories."""
        training_records = []
        evaluation_records = []
        unknown_records = []

        records = data.to_dict("records")
        for record in records:
            record_type = self._classify_record_type(record)

            if record_type == "training":
                training_records.append(record)
            elif record_type == "evaluation":
                evaluation_records.append(record)
            else:
                unknown_records.append(record)

        return {
            "training": training_records,
            "evaluation": evaluation_records,
            "unknown": unknown_records,
            "total": len(records),
        }

    def _classify_record_type(self, record: dict) -> str:
        """Classify a single record as training, evaluation, or unknown."""
        # Check for evaluation indicators
        if record.get("ideal_response") or record.get("expected_answer") or record.get("ground_truth"):
            return "evaluation"

        # Check for training indicators
        if record.get("prompt") and record.get("completion"):
            return "training"

        # Check for messages format (chat model)
        if record.get("messages"):
            return "training"

        return "unknown"

    def apply_training_split(self, training_records: list[dict]) -> tuple[list[dict], list[dict]]:
        """Apply training/validation split based on validation ratio."""
        total_records = len(training_records)

        min_records = 2
        if total_records < min_records:
            error_msg = f"Insufficient records for split: {total_records} < {min_records}"
            raise ValueError(error_msg)

        # Calculate split based on validation ratio
        validation_count = int(total_records * self.validation_split_ratio)

        # Ensure at least 1 record for validation and training
        validation_count = max(1, min(validation_count, total_records - 1))

        # Split the records
        validation_records = training_records[:validation_count]
        actual_training_records = training_records[validation_count:]

        return actual_training_records, validation_records

    def filter_record_fields(self, record: dict, model_type: str) -> dict:
        """Filter record fields based on preserve_all_fields setting."""
        if self.preserve_all_fields:
            return record

        # Only keep essential fields based on model type
        essential_fields = ["messages"] if model_type == "chat" else ["prompt", "completion", "ideal_response"]

        filtered_record = {}
        for field in essential_fields:
            if field in record:
                filtered_record[field] = record[field]

        return filtered_record

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
        """Create a dataset in NeMo Data Store with intelligent classification."""
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

        # Get input data
        data = getattr(self, "data", None)
        if not data:
            error_msg = "Data input is required"
            raise ValueError(error_msg)

        try:
            # Convert input data to DataFrame
            df = self.convert_to_dataframe(data)

            # Initialize statistics
            self.upload_statistics["total_input_records"] = len(df)

            # Determine model type
            model_type = self.determine_model_type(df)
            self.upload_statistics["model_type"] = model_type

            # Classify data
            classified_data = self.classify_data(df)

            # Update classification statistics
            self.upload_statistics["classified_records"] = {
                "training": len(classified_data["training"]),
                "evaluation": len(classified_data["evaluation"]),
                "unknown": len(classified_data["unknown"]),
            }

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

            # Process training data with split
            training_records = classified_data["training"]
            if training_records:
                if self.validation_split_ratio > 0.0:
                    actual_training, validation_records = self.apply_training_split(training_records)
                    await self.process_training_data(hf_api, repo_id, actual_training, validation_records)
                else:
                    await self.process_training_data(hf_api, repo_id, training_records, [])
                logger.info("Uploaded training data")

            # Process evaluation data
            evaluation_records = classified_data["evaluation"]
            if evaluation_records:
                await self.upload_evaluation_data(hf_api, repo_id, evaluation_records)
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
                    "has_training_data": bool(training_records),
                    "has_evaluation_data": bool(evaluation_records),
                    "model_type": model_type,
                    "upload_statistics": self.upload_statistics,
                }
            )

            logger.info(f"Successfully created dataset: {dataset_name}")

            # Log upload statistics
            self.log("Upload Statistics:")
            self.log(f"  Total input records: {self.upload_statistics['total_input_records']}")
            self.log(
                f"  Classified records - Training: {self.upload_statistics['classified_records']['training']}, Evaluation: {self.upload_statistics['classified_records']['evaluation']}, Unknown: {self.upload_statistics['classified_records']['unknown']}"
            )
            self.log(
                f"  Uploaded records - Training: {self.upload_statistics['uploaded_records']['training']}, Validation: {self.upload_statistics['uploaded_records']['validation']}, Evaluation: {self.upload_statistics['uploaded_records']['evaluation']}"
            )
            self.log(f"  Model type: {self.upload_statistics['model_type']}")
            if self.preserve_all_fields:
                self.log("  Field preservation: All fields preserved in dataset files")
            else:
                self.log("  Field preservation: Only essential fields included in dataset files")
            if self.upload_statistics["file_info"]:
                self.log(f"  Files created: {self.upload_statistics['file_info']}")

            return Data(data=result_dict)

        except Exception as exc:
            exception_str = str(exc)
            error_msg = f"Error creating dataset: {exception_str}"
            logger.error(error_msg)
            raise ValueError(error_msg) from exc

    async def process_training_data(
        self, hf_api, repo_id: str, training_records: list[dict], validation_records: list[dict]
    ):
        """Process and upload training data with chunking support."""
        import asyncio

        try:
            chunk_size = getattr(self, "chunk_size", 100000)
            logger.info(f"Processing training data for repo: {repo_id}")

            tasks = []

            # Process training records
            if training_records:
                chunk = []
                for record in training_records:
                    # Filter fields based on preserve_all_fields setting
                    filtered_record = self.filter_record_fields(record, self.upload_statistics["model_type"])
                    chunk.append(filtered_record)
                    if len(chunk) == chunk_size:
                        chunk_df = pd.DataFrame(chunk)
                        task = self.upload_chunk(
                            chunk_df,
                            self.chunk_number,
                            self.dataset_name,
                            repo_id,
                            hf_api,
                            is_validation=False,
                        )
                        tasks.append(task)
                        chunk = []
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
                        is_validation=False,
                    )
                    tasks.append(task)

                # Await all training upload tasks
                await asyncio.gather(*tasks)
                self.upload_statistics["uploaded_records"]["training"] = len(training_records)

            # Process validation records
            if validation_records:
                filtered_validation_records = []
                for record in validation_records:
                    # Filter fields based on preserve_all_fields setting
                    filtered_record = self.filter_record_fields(record, self.upload_statistics["model_type"])
                    filtered_validation_records.append(filtered_record)

                validation_df = pd.DataFrame(filtered_validation_records)
                await self.upload_chunk(validation_df, 1, self.dataset_name, repo_id, hf_api, is_validation=True)
                self.upload_statistics["uploaded_records"]["validation"] = len(validation_records)

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

                # Update file info
                if is_validation:
                    if "validation" not in self.upload_statistics["file_info"]:
                        self.upload_statistics["file_info"]["validation"] = []
                    self.upload_statistics["file_info"]["validation"].append(file_name_training)
                else:
                    if "training" not in self.upload_statistics["file_info"]:
                        self.upload_statistics["file_info"]["training"] = []
                    self.upload_statistics["file_info"]["training"].append(file_name_training)

            finally:
                training_file_obj.close()

        except Exception:  # noqa: BLE001
            logger.error(f"An error occurred while uploading chunk {chunk_number}")

    async def upload_evaluation_data(self, hf_api, repo_id: str, evaluation_data: list[dict]):
        """Upload evaluation data to the dataset."""
        input_records = []
        output_records = []

        for record in evaluation_data:
            # Filter fields based on preserve_all_fields setting
            filtered_record = self.filter_record_fields(record, "completion")  # Evaluation uses completion format

            prompt = filtered_record.get("prompt", "")
            ideal_response = (
                filtered_record.get("ideal_response", "")
                or filtered_record.get("expected_answer", "")
                or filtered_record.get("ground_truth", "")
            )
            category = filtered_record.get("category", "Generation")
            source = filtered_record.get("source", "")
            response = filtered_record.get("response", "")
            llm_name = filtered_record.get("llm_name", "")

            if prompt and ideal_response:
                # Input file data
                input_record = {
                    "prompt": prompt,
                    "ideal_response": ideal_response,
                }

                # Add optional fields if preserve_all_fields is enabled
                if self.preserve_all_fields:
                    if category:
                        input_record["category"] = category
                    if source:
                        input_record["source"] = source

                input_records.append(input_record)

                # Output file data (if response is provided)
                if response:
                    output_record = {
                        "input": input_record.copy(),
                        "response": response,
                    }

                    # Add optional fields if preserve_all_fields is enabled
                    if self.preserve_all_fields and llm_name:
                        output_record["llm_name"] = llm_name

                    output_records.append(output_record)

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
                self.upload_statistics["uploaded_records"]["evaluation"] = len(input_records)
                if "evaluation" not in self.upload_statistics["file_info"]:
                    self.upload_statistics["file_info"]["evaluation"] = []
                self.upload_statistics["file_info"]["evaluation"].append("input.json")
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
                if "evaluation" not in self.upload_statistics["file_info"]:
                    self.upload_statistics["file_info"]["evaluation"] = []
                self.upload_statistics["file_info"]["evaluation"].append("output.json")
            finally:
                output_file_buffer.close()

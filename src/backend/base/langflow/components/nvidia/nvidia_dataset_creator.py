import json
import logging
from datetime import datetime, timezone
from io import BytesIO

import httpx
from huggingface_hub import HfApi

from langflow.custom import Component
from langflow.io import (
    DataInput,
    Output,
    StrInput,
)
from langflow.schema import Data
from langflow.services.deps import get_settings_service

logger = logging.getLogger(__name__)


class NvidiaDatasetCreatorComponent(Component):
    display_name = "NeMo Dataset Creator"
    description = "Create datasets in NeMo Data Store for use with NeMo Customizer and Evaluator components"
    icon = "NVIDIA"
    name = "NVIDIANeMoDatasetCreator"
    beta = True

    headers = {"accept": "application/json", "Content-Type": "application/json"}

    inputs = [
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

    async def create_dataset(self) -> Data:
        """Create a dataset in NeMo Data Store."""
        settings_service = get_settings_service()
        nemo_data_store_url = settings_service.settings.nemo_data_store_url
        nemo_entity_store_url = settings_service.settings.nemo_entity_store_url

        if not nemo_data_store_url or not nemo_entity_store_url:
            error_msg = "Missing NeMo Data Store or Entity Store URL configuration"
            raise ValueError(error_msg)

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
            # Create namespace if it doesn't exist
            await self.create_namespace(namespace, nemo_data_store_url)

            # Create HuggingFace repo
            hf_api = HfApi(endpoint=f"{nemo_data_store_url}/v1/hf", token="")
            repo_id = f"{namespace}/{dataset_name}"
            repo_type = "dataset"
            hf_api.create_repo(repo_id, repo_type=repo_type, exist_ok=True)

            self.log(f"Created/accessed repo: {repo_id}")

            # Process and upload data
            if training_data:
                await self.upload_training_data(hf_api, repo_id, training_data)
                self.log("Uploaded training data")

            if evaluation_data:
                await self.upload_evaluation_data(hf_api, repo_id, evaluation_data)
                self.log("Uploaded evaluation data")

            # Register dataset in entity store
            file_url = f"hf://datasets/{repo_id}"
            entity_registry_url = f"{nemo_entity_store_url}/v1/datasets"
            create_payload = {
                "name": dataset_name,
                "namespace": namespace,
                "description": description,
                "files_url": file_url,
                "format": "jsonl",
                "project": dataset_name,
            }

            async with httpx.AsyncClient() as client:
                response = await client.post(entity_registry_url, json=create_payload)
                response.raise_for_status()

            self.log(f"Successfully created dataset: {dataset_name}")

            return Data(
                data={
                    "dataset_name": dataset_name,
                    "namespace": namespace,
                    "repo_id": repo_id,
                    "description": description,
                    "file_url": file_url,
                    "has_training_data": bool(training_data),
                    "has_evaluation_data": bool(evaluation_data),
                }
            )

        except Exception as exc:
            exception_str = str(exc)
            error_msg = f"Error creating dataset: {exception_str}"
            self.log(error_msg)
            raise ValueError(error_msg) from exc

    async def upload_training_data(self, hf_api, repo_id: str, training_data):
        """Upload training data to the dataset."""
        valid_records = []
        for data_obj in training_data or []:
            if not isinstance(data_obj, Data):
                self.log(f"Skipping non-Data object in training data: {data_obj}")
                continue

            filtered_data = {
                "prompt": getattr(data_obj, "prompt", None),
                "completion": getattr(data_obj, "completion", None),
            }
            if filtered_data["prompt"] is not None and filtered_data["completion"] is not None:
                valid_records.append(filtered_data)

        if valid_records:
            # Create training file
            training_file_buffer = BytesIO(json.dumps(valid_records, indent=4).encode("utf-8"))
            try:
                hf_api.upload_file(
                    path_or_fileobj=training_file_buffer,
                    path_in_repo="training.json",
                    repo_id=repo_id,
                    repo_type="dataset",
                    commit_message=f"Training data at {datetime.now(timezone.utc).strftime('%Y%m%d%H%M%S')}",
                )
            finally:
                training_file_buffer.close()

    async def upload_evaluation_data(self, hf_api, repo_id: str, evaluation_data):
        """Upload evaluation data to the dataset."""
        input_records = []
        output_records = []

        for data_obj in evaluation_data or []:
            if not isinstance(data_obj, Data):
                self.log(f"Skipping non-Data object in evaluation data: {data_obj}")
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

    async def create_namespace(self, namespace: str, nemo_data_store_url: str):
        """Check and create namespace in datastore."""
        url = f"{nemo_data_store_url}/v1/datastore/namespaces"

        try:
            async with httpx.AsyncClient() as client:
                response = await client.get(f"{url}/{namespace}")
                if response.status_code == 404:  # noqa: PLR2004
                    self.log(f"Namespace not found, creating namespace: {namespace}")
                    create_payload = {"namespace": namespace}
                    create_response = await client.post(url, json=create_payload)
                    create_response.raise_for_status()
                else:
                    response.raise_for_status()

        except httpx.HTTPStatusError as e:
            exception_str = str(e)
            error_msg = f"Error processing namespace: {exception_str}"
            self.log(error_msg)
            raise ValueError(error_msg) from e

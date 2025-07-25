"""NeMo Microservices client for production use.

This module provides an implementation of NeMo Microservices APIs
that makes actual HTTP calls to NeMo services.

Includes real implementations for:
- NeMo Data Store (datasets and files)
- NeMo Entity Store (entities, projects, models)
- NeMo Customizer (job tracking and status)
- NeMo Evaluator (evaluation results)
"""

import logging
from datetime import datetime, timezone
from io import BytesIO
from typing import Any
from unittest.mock import patch

import httpx
import requests
from fastapi import UploadFile
from httpx import codes
from huggingface_hub import HfApi
from nemo_microservices import AsyncNeMoMicroservices, NeMoMicroservicesError
from nemo_microservices._types import NOT_GIVEN

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


class RealNeMoMicroservicesService:
    """Real implementation of NeMo Microservices APIs.

    Makes actual HTTP calls to NeMo services for production use.
    """

    def __init__(self, base_url: str | None = None, api_key: str | None = None):
        self.base_url = base_url or "https://us-west-2.api-dev.ai.datastax.com/nvidia/nemo"
        self.api_key = api_key

        # All endpoints use the same base URL with different paths
        # Based on the actual NeMo API structure
        logger.info("Initialized NeMo service with base URL: %s", self.base_url)

    def _get_auth_headers(self) -> dict[str, str]:
        """Get headers with authentication token."""
        headers = {"accept": "application/json", "Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        return headers

    def get_nemo_client(self) -> AsyncNeMoMicroservices:
        """Get an authenticated NeMo microservices client."""
        return AsyncNeMoMicroservices(
            base_url=self.base_url,
        )

    def _serialize_datetime_objects(self, obj: Any) -> Any:
        """Convert datetime objects to strings for JSON serialization."""
        if isinstance(obj, dict):
            return {k: self._serialize_datetime_objects(v) for k, v in obj.items()}
        if isinstance(obj, list):
            return [self._serialize_datetime_objects(item) for item in obj]
        if hasattr(obj, "isoformat"):  # datetime objects
            return obj.isoformat()
        return obj

    def _force_json_serializable(self, obj: Any) -> Any:
        """Force an object to be JSON serializable by converting complex types."""
        try:
            # Try simple types first
            if obj is None or isinstance(obj, bool | int | float | str):
                return obj

            # Handle lists and tuples
            if isinstance(obj, list | tuple):
                return [self._force_json_serializable(item) for item in obj]

            # Handle dictionaries
            if isinstance(obj, dict):
                return {str(k): self._force_json_serializable(v) for k, v in obj.items()}

            # Handle datetime objects
            if hasattr(obj, "isoformat"):
                return obj.isoformat()

            # Handle objects with model_dump (Pydantic models)
            if hasattr(obj, "model_dump"):
                return self._force_json_serializable(obj.model_dump())

            # Handle objects with __dict__
            if hasattr(obj, "__dict__"):
                # Filter out methods and non-serializable attributes
                result = {}
                for key, value in obj.__dict__.items():
                    if not key.startswith("_") and not callable(value):
                        try:
                            result[key] = self._force_json_serializable(value)
                        except (TypeError, ValueError, AttributeError):
                            result[key] = str(value)
                return result

            # For other objects, convert to string
            return str(obj)

        except (TypeError, ValueError, AttributeError) as e:
            logger.warning("Error serializing object of type %s: %s", type(obj), e)
            return f"<{type(obj).__name__}: {str(obj)[:100]}>"

    # =============================================================================
    # Dataset Management (Data Store)
    # =============================================================================

    async def list_datasets(
        self, page: int = 1, page_size: int = 10, dataset_name: str | None = None, namespace: str | None = None
    ) -> dict[str, Any]:
        """Get list of datasets from NeMo Entity Store with pagination and optional filtering.

        Args:
            page: Page number (1-based)
            page_size: Number of datasets per page
            dataset_name: Optional dataset name to filter by
            namespace: Optional namespace to filter by

        Returns:
            Paginated dataset response with data, pagination info
        """
        try:
            nemo_client = self.get_nemo_client()

            # Build filter for search and namespace
            filter_params = {}
            if namespace:
                filter_params["namespace"] = namespace

            # Note: The SDK doesn't have direct dataset_name filtering,
            # so we'll filter results after getting them if needed

            # Use SDK pagination with entity store
            paginated_response = await nemo_client.datasets.list(
                page=page,
                page_size=page_size,
                filter=filter_params if filter_params else NOT_GIVEN,
                extra_headers={"Authorization": f"Bearer {self.api_key}"},
            )

            # Convert SDK response to current format
            datasets = []
            for dataset in paginated_response.data:
                dataset_dict = dataset.model_dump()
                # Apply datetime serialization
                dataset_dict = self._serialize_datetime_objects(dataset_dict)
                datasets.append(dataset_dict)

            # Apply dataset_name filtering if provided (since SDK doesn't support it directly)
            if dataset_name:
                datasets = [d for d in datasets if dataset_name.lower() in d.get("name", "").lower()]

            # Build response in the format expected by frontend
            current_page = paginated_response.pagination.page if paginated_response.pagination else page
            page_size = paginated_response.pagination.page_size if paginated_response.pagination else page_size
            total_pages = paginated_response.pagination.total_pages if paginated_response.pagination else 1
            total_results = (
                paginated_response.pagination.total_results if paginated_response.pagination else len(datasets)
            )

            response = {
                "data": datasets,
                "page": current_page,
                "page_size": page_size,
                "total": total_results,
                "total_pages": total_pages,
                "has_next": current_page < total_pages,
                "has_prev": current_page > 1,
            }

            logger.info("Retrieved %s datasets from entity store (page %s, size %s)", len(datasets), page, page_size)

        except NeMoMicroservicesError as exc:
            logger.exception("NeMo microservices error while listing datasets")

            # Handle 401 Unauthorized gracefully
            if "401" in str(exc) or "Unauthorized" in str(exc):
                logger.warning("Authentication failed for NeMo microservices - returning empty dataset list")
                return {
                    "data": [],
                    "page": page,
                    "page_size": page_size,
                    "total": 0,
                    "total_pages": 0,
                    "has_next": False,
                    "has_prev": False,
                    "error": "Authentication failed. Please check your NeMo credentials.",
                }

            msg = f"NeMo microservices error while listing datasets: {exc}"
            raise ValueError(msg) from exc
        except (httpx.HTTPError, OSError, ValueError) as e:
            logger.exception("Failed to list datasets")
            msg = f"Failed to list datasets: {e}"
            raise ValueError(msg) from e
        else:
            return response

    async def create_dataset(
        self, name: str, description: str | None = None, _dataset_type: str = "fileset", namespace: str = "default"
    ) -> dict[str, Any]:
        """Create a new dataset in NeMo Entity Store."""
        try:
            nemo_client = self.get_nemo_client()

            # Create a basic dataset without HuggingFace repo - just metadata
            dataset_response = await nemo_client.datasets.create(
                name=name,
                namespace=namespace,
                description=description or f"Dataset {name} created via API",
                files_url=f"hf://datasets/{namespace}/{name}",  # Placeholder URL
                format="jsonl",
                project=name,
                extra_headers={"Authorization": f"Bearer {self.api_key}"},
            )

            # Convert SDK response to dict and serialize datetime objects
            result = dataset_response.model_dump()
            result = self._serialize_datetime_objects(result)

            logger.info("Created dataset %s in namespace %s", name, namespace)

        except NeMoMicroservicesError as exc:
            logger.exception("NeMo microservices error while creating dataset")
            msg = f"NeMo microservices error while creating dataset: {exc}"
            raise ValueError(msg) from exc
        except (httpx.HTTPError, OSError, ValueError) as e:
            logger.exception("Failed to create dataset")
            msg = f"Failed to create dataset: {e}"
            raise ValueError(msg) from e
        else:
            return result

    async def create_dataset_with_namespace(
        self, name: str, namespace: str, description: str | None = None, dataset_type: str = "fileset"
    ) -> dict[str, Any]:
        """Create a new dataset with namespace using HuggingFace API pattern from nvidia_customizer.

        This method follows the same pattern as the nvidia_customizer component:
        1. Create namespace if it doesn't exist
        2. Create datastore namespace if it doesn't exist
        3. Create HuggingFace repository
        4. Register dataset in entity registry

        Args:
            name: Dataset name (user-provided, not UUID)
            namespace: Dataset namespace
            description: Optional description
            dataset_type: Type of dataset (default: fileset)

        Returns:
            Created dataset object
        """
        try:
            # Use API with authentication
            hf_api = AuthenticatedHfApi(
                endpoint=f"{self.base_url}/v1/hf",
                auth_token=self.api_key,
                namespace=namespace,
                token=self.api_key,
            )

            # Create namespace if it doesn't exist
            await self.create_namespace(namespace)

            # Create datastore namespace if it doesn't exist
            await self.create_datastore_namespace(namespace)

            # Create HuggingFace repository
            repo_id = f"{namespace}/{name}"
            repo_type = "dataset"
            hf_api.create_repo(repo_id, repo_type=repo_type, exist_ok=True)

            logger.info("Created HuggingFace repo: %s", repo_id)

            # Register dataset in entity registry using SDK
            file_url = f"hf://datasets/{repo_id}"
            description_text = description or f"Dataset {name} created via NeMo interface"

            nemo_client = self.get_nemo_client()
            dataset_response = await nemo_client.datasets.create(
                name=name,
                namespace=namespace,
                description=description_text,
                files_url=file_url,
                format="jsonl",
                project=name,
                extra_headers={"Authorization": f"Bearer {self.api_key}"},
            )

            # Convert SDK response to dict and serialize datetime objects
            result = dataset_response.model_dump()
            result = self._serialize_datetime_objects(result)

            logger.info("Successfully registered dataset %s in entity registry using SDK", name)

            # Prepare structured response
            response_data = {
                "name": name,
                "namespace": namespace,
                "description": description_text,
                "type": dataset_type,
                "files_url": file_url,
                "repo_id": repo_id,
                "created_at": result.get("created_at", ""),
                "id": result.get("id", repo_id),
                "message": f"Dataset {name} created successfully with HuggingFace repository {repo_id}",
            }

        except NeMoMicroservicesError as exc:
            logger.exception("NeMo microservices error while creating dataset")
            msg = f"NeMo microservices error while creating dataset: {exc}"
            raise ValueError(msg) from exc
        except (httpx.HTTPError, OSError, ValueError) as e:
            logger.exception("Failed to create dataset with namespace")
            msg = f"Failed to create dataset with namespace: {e}"
            raise ValueError(msg) from e
        else:
            return response_data

    async def create_namespace(self, namespace: str):
        """Create namespace in entity-store with authentication."""
        url = f"{self.base_url}/v1/namespaces"

        try:
            async with httpx.AsyncClient(timeout=30.0) as client:
                response = await client.get(f"{url}/{namespace}", headers=self._get_auth_headers())
                if response.status_code == codes.NOT_FOUND:
                    logger.info("Namespace not found, creating namespace: %s", namespace)
                    create_payload = {"namespace": namespace}
                    create_response = await client.post(url, json=create_payload, headers=self._get_auth_headers())
                    create_response.raise_for_status()
                else:
                    response.raise_for_status()

        except httpx.HTTPStatusError as e:
            logger.exception("Error processing namespace")
            msg = f"Error processing namespace: {e}"
            raise ValueError(msg) from e

    async def create_datastore_namespace(self, namespace: str):
        """Create namespace in datastore with authentication."""
        url = f"{self.base_url}/v1/datastore/namespaces"

        try:
            async with httpx.AsyncClient(timeout=30.0) as client:
                response = await client.get(f"{url}/{namespace}", headers=self._get_auth_headers())
                if response.status_code == codes.NOT_FOUND:
                    logger.info("Datastore namespace not found, creating namespace: %s", namespace)
                    create_payload = {"namespace": namespace}
                    create_response = await client.post(url, json=create_payload, headers=self._get_auth_headers())
                    create_response.raise_for_status()
                else:
                    response.raise_for_status()

        except httpx.HTTPStatusError as e:
            logger.exception("Error processing datastore namespace")
            msg = f"Error processing datastore namespace: {e}"
            raise ValueError(msg) from e

    async def get_dataset(self, dataset_name: str, namespace: str | None = None) -> dict[str, Any] | None:
        """Get dataset details from NeMo Entity Store.

        Args:
            dataset_name: Dataset name
            namespace: Dataset namespace (required for SDK)

        Returns:
            Dataset details or None if not found
        """
        try:
            nemo_client = self.get_nemo_client()

            # SDK requires namespace, use 'default' if not provided
            ns = namespace or "default"

            dataset = await nemo_client.datasets.retrieve(
                dataset_name=dataset_name, namespace=ns, extra_headers={"Authorization": f"Bearer {self.api_key}"}
            )

            # Convert SDK response to dict and serialize datetime objects
            dataset_dict = dataset.model_dump()
            dataset_dict = self._serialize_datetime_objects(dataset_dict)

            logger.info("Retrieved dataset %s from namespace %s", dataset_name, ns)

        except NeMoMicroservicesError as exc:
            # Check if it's a 404 (not found) error
            if "404" in str(exc) or "not found" in str(exc).lower():
                logger.info("Dataset %s not found in namespace %s", dataset_name, ns)
                return None
            # Handle 401 Unauthorized gracefully
            if "401" in str(exc) or "Unauthorized" in str(exc):
                logger.warning("Authentication failed for NeMo microservices - cannot get dataset")
                return None
            logger.exception("NeMo microservices error while getting dataset")
            msg = f"NeMo microservices error while getting dataset: {exc}"
            raise ValueError(msg) from exc
        except (httpx.HTTPError, OSError, ValueError) as e:
            logger.exception("Failed to get dataset")
            msg = f"Failed to get dataset: {e}"
            raise ValueError(msg) from e
        else:
            return dataset_dict

    async def delete_dataset(self, dataset_name: str, namespace: str | None = None) -> bool:
        """Delete a dataset from NeMo Entity Store.

        Args:
            dataset_name: Dataset name
            namespace: Dataset namespace (required for SDK)

        Returns:
            True if deleted successfully, False if not found
        """
        try:
            nemo_client = self.get_nemo_client()

            # SDK requires namespace, use 'default' if not provided
            ns = namespace or "default"

            await nemo_client.datasets.delete(
                dataset_name=dataset_name, namespace=ns, extra_headers={"Authorization": f"Bearer {self.api_key}"}
            )

            # SDK returns DeleteResponse object, check if successful
            logger.info("Deleted dataset %s from namespace %s", dataset_name, ns)

        except NeMoMicroservicesError as exc:
            # Check if it's a 404 (not found) error
            if "404" in str(exc) or "not found" in str(exc).lower():
                logger.info("Dataset %s not found in namespace %s for deletion", dataset_name, ns)
                return False
            logger.exception("NeMo microservices error while deleting dataset")
            msg = f"NeMo microservices error while deleting dataset: {exc}"
            raise ValueError(msg) from exc
        except (httpx.HTTPError, OSError, ValueError) as e:
            logger.exception("Failed to delete dataset")
            msg = f"Failed to delete dataset: {e}"
            raise ValueError(msg) from e
        else:
            return True

    async def upload_files(self, dataset_id: str, files: list[UploadFile]) -> dict[str, Any]:
        """Upload files to a NeMo dataset."""
        try:
            form_data = {}
            for _i, file in enumerate(files):
                form_data["files"] = (file.filename, file.file, file.content_type)

            async with httpx.AsyncClient(timeout=60.0) as client:
                response = await client.post(
                    f"{self.base_url}/v1/datasets/{dataset_id}/files",
                    files=form_data,
                    headers={"Authorization": f"Bearer {self.api_key}"} if self.api_key else {},
                )
                response.raise_for_status()
                response_data = response.json()
        except (httpx.HTTPError, OSError, ValueError) as e:
            logger.exception("Failed to upload files")
            msg = f"Failed to upload files: {e}"
            raise ValueError(msg) from e
        else:
            return response_data

    async def upload_dataset_files_with_path(
        self, dataset_name: str, path: str, namespace: str, files: list[UploadFile]
    ) -> dict[str, Any]:
        """Upload files to a NeMo dataset with specified path using HuggingFace API.

        This method mimics the upload functionality from nvidia_customizer component,
        uploading files to a dataset using the HuggingFace API with Git LFS support.
        Files are uploaded to the specified path within the dataset (e.g., training/file.jsonl).

        Args:
            dataset_name: NeMo Data Store dataset name
            path: Path within the dataset (e.g., 'training', 'validation')
            namespace: Dataset namespace
            files: List of files to upload

        Returns:
            Upload result with file information and paths
        """
        try:
            # Use HuggingFace API with authentication
            hf_api = AuthenticatedHfApi(
                endpoint=f"{self.base_url}/v1/hf",
                auth_token=self.api_key,
                namespace=namespace,
                token=self.api_key,
            )

            repo_id = f"{namespace}/{dataset_name}"
            repo_type = "dataset"

            # Ensure repo exists
            try:
                hf_api.create_repo(repo_id, repo_type=repo_type, exist_ok=True)
            except (ValueError, RuntimeError, OSError) as e:
                logger.warning("Failed to create/verify repo %s: %s", repo_id, e)

            uploaded_files = []

            for file in files:
                try:
                    # Read file content
                    content = await file.read()
                    file_obj = BytesIO(content)

                    # Build file path with the specified path prefix
                    file_path_in_repo = f"{path}/{file.filename}"

                    commit_message = (
                        f"Upload {file.filename} to {path}/ at {datetime.now(timezone.utc).strftime('%Y%m%d%H%M%S')}"
                    )

                    # Use authenticated HF API with request patching for Git LFS support
                    patched_request = create_auth_interceptor(self.api_key, namespace)

                    with patch.object(requests.Session, "request", patched_request):
                        hf_api.upload_file(
                            path_or_fileobj=file_obj,
                            path_in_repo=file_path_in_repo,
                            repo_id=repo_id,
                            repo_type=repo_type,
                            commit_message=commit_message,
                        )

                    uploaded_files.append(
                        {
                            "filename": file.filename,
                            "path": file_path_in_repo,
                            "size": len(content),
                            "content_type": file.content_type,
                            "uploaded_at": datetime.now(timezone.utc).isoformat(),
                        }
                    )

                    logger.info("Successfully uploaded %s to %s", file.filename, file_path_in_repo)

                finally:
                    file_obj.close()

            result_data = {
                "message": f"Successfully uploaded {len(uploaded_files)} file(s) to {path}/",
                "dataset_name": dataset_name,
                "namespace": namespace,
                "path": path,
                "files": uploaded_files,
                "repo_id": repo_id,
            }

        except Exception:
            logger.exception("Failed to upload files with path")
            raise
        else:
            return result_data

    async def get_dataset_files(self, dataset_id: str) -> list[dict[str, Any]]:
        """Get list of files in a NeMo dataset."""
        try:
            async with httpx.AsyncClient(timeout=30.0) as client:
                response = await client.get(
                    f"{self.base_url}/v1/datasets/{dataset_id}/files", headers=self._get_auth_headers()
                )
                response.raise_for_status()
                response_data = response.json().get("data", [])
        except Exception:
            logger.exception("Failed to get dataset files")
            raise
        else:
            return response_data

    # =============================================================================
    # Job Management (Customizer)
    # =============================================================================

    async def get_customization_configs(self) -> dict[str, Any]:
        """Get available model configurations for customization."""
        try:
            async with httpx.AsyncClient(timeout=30.0) as client:
                response = await client.get(
                    f"{self.base_url}/v1/customization/configs", headers=self._get_auth_headers()
                )
                response.raise_for_status()
                response_data = response.json()
        except Exception:
            logger.exception("Failed to get customization configs")
            raise
        else:
            return response_data

    async def create_customization_job(self, job_data: dict) -> dict[str, Any]:
        """Create a new customization job."""
        try:
            async with httpx.AsyncClient(timeout=30.0) as client:
                response = await client.post(
                    f"{self.base_url}/v1/customization/jobs", json=job_data, headers=self._get_auth_headers()
                )
                response.raise_for_status()
                response_data = response.json()
        except Exception:
            logger.exception("Failed to create customization job")
            raise
        else:
            return response_data

    async def get_customizer_job_status(self, job_id: str) -> dict[str, Any] | None:
        """Get customization job status."""
        try:
            async with httpx.AsyncClient(timeout=30.0) as client:
                response = await client.get(
                    f"{self.base_url}/v1/customization/jobs/{job_id}/status", headers=self._get_auth_headers()
                )
                response.raise_for_status()
                response_data = response.json()
        except httpx.HTTPStatusError as e:
            if e.response.status_code == codes.NOT_FOUND:
                return None
            raise
        except Exception:
            logger.exception("Failed to get job status")
            raise
        else:
            return response_data

    async def get_customizer_job_details(self, job_id: str) -> dict[str, Any] | None:
        """Get customization job details."""
        try:
            async with httpx.AsyncClient(timeout=30.0) as client:
                response = await client.get(
                    f"{self.base_url}/v1/customization/jobs/{job_id}", headers=self._get_auth_headers()
                )
                response.raise_for_status()
                response_data = response.json()
        except httpx.HTTPStatusError as e:
            if e.response.status_code == codes.NOT_FOUND:
                return None
            raise
        except Exception:
            logger.exception("Failed to get job details")
            raise
        else:
            return response_data

    async def list_customizer_jobs(self, page: int = 1, page_size: int = 10) -> dict[str, Any]:
        """List customization jobs with pagination."""
        try:
            # Get all jobs first since NeMo Customizer API doesn't support pagination
            async with httpx.AsyncClient(timeout=30.0) as client:
                response = await client.get(f"{self.base_url}/v1/customization/jobs", headers=self._get_auth_headers())
                response.raise_for_status()
                all_jobs = response.json().get("data", [])

            # Apply client-side pagination
            total_jobs = len(all_jobs)
            start_index = (page - 1) * page_size
            end_index = start_index + page_size
            paginated_jobs = all_jobs[start_index:end_index]

            total_pages = (total_jobs + page_size - 1) // page_size

            # Return paginated response in the same format as datasets
            return {  # noqa: TRY300
                "data": paginated_jobs,
                "page": page,
                "page_size": page_size,
                "total": total_jobs,
                "total_pages": total_pages,
                "has_next": page < total_pages,
                "has_prev": page > 1,
            }
        except Exception as exc:
            logger.exception("Failed to list customization jobs")

            # Handle authentication errors gracefully
            if "401" in str(exc) or "Unauthorized" in str(exc):
                logger.warning("Authentication failed for NeMo microservices - returning empty jobs list")
                return {
                    "data": [],
                    "page": page,
                    "page_size": page_size,
                    "total": 0,
                    "total_pages": 0,
                    "has_next": False,
                    "has_prev": False,
                    "error": "Authentication failed. Please check your NeMo credentials.",
                }

            msg = f"Failed to list customization jobs: {exc}"
            raise ValueError(msg) from exc

    async def cancel_customization_job(self, job_id: str) -> dict[str, Any]:
        """Cancel a customization job using NeMo Python SDK."""
        try:
            # Use NeMo client for customization job cancellation
            nemo_client = self.get_nemo_client()

            logger.info("Using NeMo SDK to cancel customization job: %s", job_id)
            # Cancel customization job using SDK
            result = await nemo_client.customization.jobs.cancel(
                job_id=job_id,
                extra_headers={"Authorization": f"Bearer {self.api_key}"},
            )

            # Convert SDK response to dict and serialize datetime objects
            result_dict = result.model_dump()
            result_dict = self._serialize_datetime_objects(result_dict)

            logger.info("Successfully cancelled customization job %s", job_id)

        except NeMoMicroservicesError as exc:
            logger.exception("NeMo SDK error cancelling customization job %s", job_id)
            # Check if it's a 404 (not found) error
            if "404" in str(exc) or "not found" in str(exc).lower():
                logger.info("Customization job %s not found for cancellation", job_id)
                return {"message": f"Customization job {job_id} not found"}
            # Handle 401 Unauthorized gracefully
            if "401" in str(exc) or "Unauthorized" in str(exc):
                logger.warning("Authentication failed for NeMo microservices - cannot cancel customization job")
                msg = f"Authentication failed for NeMo microservices: {exc}"
                raise ValueError(msg) from exc
            logger.exception("NeMo microservices error while cancelling customization job")
            msg = f"NeMo microservices error while cancelling customization job: {exc}"
            raise ValueError(msg) from exc
        except Exception:
            logger.exception("Exception cancelling customization job %s", job_id)
            raise
        else:
            return result_dict

    async def delete_customization_job_custom(self, job_id: str) -> dict[str, Any]:
        """Delete a customization job using cancel since customizer jobs don't support direct deletion.

        Note: Customizer jobs don't have a delete endpoint, only cancel.
        This method will always use the cancel endpoint via the NeMo SDK.
        """
        logger.info("Service layer: Attempting to cancel customization job %s (no delete endpoint available)", job_id)
        try:
            # Customizer jobs don't support deletion, only cancellation
            # Use the cancel method directly
            result = await self.cancel_customization_job(job_id)

            # Update the message to reflect that this was a cancellation, not deletion
            if isinstance(result, dict) and "message" not in result:
                result["message"] = (
                    f"Customization job {job_id} cancelled successfully (jobs cannot be deleted, only cancelled)"
                )

            logger.info("Successfully cancelled customization job %s", job_id)

        except Exception:
            logger.exception("Exception cancelling customization job %s", job_id)
            raise
        else:
            return result

    async def get_customization_job_container_logs(self, job_id: str) -> dict[str, Any]:
        """Get container logs for a customization job using NeMo Python SDK."""
        try:
            # Use NeMo client for customization job logs
            nemo_client = self.get_nemo_client()

            logger.info("Using NeMo SDK to get container logs for customization job: %s", job_id)
            # Get customization job logs using SDK
            try:
                result = await nemo_client.customization.jobs.logs(
                    job_id=job_id,
                    extra_headers={"Authorization": f"Bearer {self.api_key}"},
                )

                # Convert SDK response to dict and serialize datetime objects
                if hasattr(result, "model_dump"):
                    result_dict = result.model_dump()
                    return self._serialize_datetime_objects(result_dict)
                else:  # noqa: RET505
                    # If SDK returns raw data, return as is
                    return result

            except AttributeError:
                # Fallback to direct HTTP call if SDK doesn't support logs method
                logger.info("SDK doesn't support logs method, falling back to HTTP call")
                logs_url = f"{self.base_url}/v1/customization/jobs/{job_id}/container-logs"
                logger.info("Getting container logs from: %s", logs_url)

                async with httpx.AsyncClient(timeout=30.0) as client:
                    response = await client.get(logs_url, headers=self._get_auth_headers())
                    response.raise_for_status()
                    try:
                        return response.json()
                    except (UnicodeDecodeError, ValueError):
                        # Handle binary or non-JSON response
                        content = response.content.decode("utf-8", errors="replace") if response.content else ""
                        return {"logs": content}

        except NeMoMicroservicesError as exc:
            logger.exception("NeMo SDK error getting container logs for job %s", job_id)
            # Check if it's a 404 (not found) error
            if "404" in str(exc) or "not found" in str(exc).lower():
                logger.info("Customization job %s not found for logs", job_id)
                return {"message": f"Customization job {job_id} not found"}
            # Handle 401 Unauthorized gracefully
            if "401" in str(exc) or "Unauthorized" in str(exc):
                logger.warning("Authentication failed for NeMo microservices - cannot get customization job logs")
                msg = f"Authentication failed for NeMo microservices: {exc}"
                raise ValueError(msg) from exc
            logger.exception("NeMo microservices error while getting customization job logs")
            msg = f"NeMo microservices error while getting customization job logs: {exc}"
            raise ValueError(msg) from exc
        except httpx.HTTPStatusError as e:
            logger.exception(
                "HTTP error getting container logs for job %s: %s - %s", job_id, e.response.status_code, e.response.text
            )
            raise
        except Exception:
            logger.exception("Exception getting container logs for job %s", job_id)
            raise

    async def get_evaluation_job_logs(self, job_id: str) -> dict[str, Any]:
        """Get logs for an evaluation job using NeMo Python SDK."""
        try:
            # Use NeMo client for evaluation job logs
            nemo_client = self.get_nemo_client()

            logger.info("Using NeMo SDK to get logs for evaluation job: %s", job_id)
            # Get evaluation job logs using SDK
            try:
                result = await nemo_client.evaluation.jobs.logs(
                    job_id=job_id,
                    extra_headers={"Authorization": f"Bearer {self.api_key}"},
                )

                # Convert SDK response to dict and serialize datetime objects
                if hasattr(result, "model_dump"):
                    result_dict = result.model_dump()
                    return self._serialize_datetime_objects(result_dict)
                else:  # noqa: RET505
                    # If SDK returns raw data, return as is
                    return result

            except AttributeError:
                # Fallback to direct HTTP call if SDK doesn't support logs method
                logger.info("SDK doesn't support logs method, falling back to HTTP call")
                logs_url = f"{self.base_url}/v1/evaluation/jobs/{job_id}/logs"
                logger.info("Getting evaluation logs from: %s", logs_url)

                async with httpx.AsyncClient(timeout=30.0) as client:
                    response = await client.get(logs_url, headers=self._get_auth_headers())
                    response.raise_for_status()
                    try:
                        return response.json()
                    except (UnicodeDecodeError, ValueError):
                        # Handle binary or non-JSON response
                        content = response.content.decode("utf-8", errors="replace") if response.content else ""
                        return {"logs": content}

        except NeMoMicroservicesError as exc:
            logger.exception("NeMo SDK error getting evaluation logs for job %s", job_id)
            # Check if it's a 404 (not found) error
            if "404" in str(exc) or "not found" in str(exc).lower():
                logger.info("Evaluation job %s not found for logs", job_id)
                return {"message": f"Evaluation job {job_id} not found"}
            # Handle 401 Unauthorized gracefully
            if "401" in str(exc) or "Unauthorized" in str(exc):
                logger.warning("Authentication failed for NeMo microservices - cannot get evaluation job logs")
                msg = f"Authentication failed for NeMo microservices: {exc}"
                raise ValueError(msg) from exc
            logger.exception("NeMo microservices error while getting evaluation job logs")
            msg = f"NeMo microservices error while getting evaluation job logs: {exc}"
            raise ValueError(msg) from exc
        except httpx.HTTPStatusError as e:
            logger.exception(
                "HTTP error getting evaluation logs for job %s: %s - %s",
                job_id,
                e.response.status_code,
                e.response.text,
            )
            raise
        except Exception:
            logger.exception("Exception getting evaluation logs for job %s", job_id)
            raise

    async def get_evaluation_job_results(self, job_id: str) -> dict[str, Any]:
        """Get results for a completed evaluation job using NeMo Python SDK."""
        try:
            # Use NeMo client for evaluation job results
            nemo_client = self.get_nemo_client()

            logger.info("Using NeMo SDK to get results for evaluation job: %s", job_id)
            # Get evaluation job results using SDK
            try:
                result = await nemo_client.evaluation.jobs.results(
                    job_id=job_id,
                    extra_headers={"Authorization": f"Bearer {self.api_key}"},
                )

                # Convert SDK response to dict and serialize datetime objects
                if hasattr(result, "model_dump"):
                    result_dict = result.model_dump()
                    return self._serialize_datetime_objects(result_dict)
                else:  # noqa: RET505
                    # If SDK returns raw data, return as is
                    return result

            except AttributeError:
                # Fallback to direct HTTP call if SDK doesn't support results method
                logger.info("SDK doesn't support results method, falling back to HTTP call")
                results_url = f"{self.base_url}/v1/evaluation/jobs/{job_id}/results"
                logger.info("Getting evaluation results from: %s", results_url)

                async with httpx.AsyncClient(timeout=30.0) as client:
                    response = await client.get(results_url, headers=self._get_auth_headers())
                    response.raise_for_status()
                    return response.json()

        except NeMoMicroservicesError as exc:
            logger.exception("NeMo SDK error getting evaluation results for job %s", job_id)
            # Check if it's a 404 (not found) error
            if "404" in str(exc) or "not found" in str(exc).lower():
                logger.info("Evaluation job %s not found for results", job_id)
                return {"message": f"Evaluation job {job_id} not found"}
            # Handle 401 Unauthorized gracefully
            if "401" in str(exc) or "Unauthorized" in str(exc):
                logger.warning("Authentication failed for NeMo microservices - cannot get evaluation job results")
                msg = f"Authentication failed for NeMo microservices: {exc}"
                raise ValueError(msg) from exc
            logger.exception("NeMo microservices error while getting evaluation job results")
            msg = f"NeMo microservices error while getting evaluation job results: {exc}"
            raise ValueError(msg) from exc
        except httpx.HTTPStatusError as e:
            logger.exception(
                "HTTP error getting evaluation results for job %s: %s - %s",
                job_id,
                e.response.status_code,
                e.response.text,
            )
            raise
        except Exception:
            logger.exception("Exception getting evaluation results for job %s", job_id)
            raise

    async def download_evaluation_job_results(self, job_id: str) -> dict[str, Any]:
        """Download results for a completed evaluation job using NeMo Python SDK."""
        try:
            # Use NeMo client for evaluation job results download
            nemo_client = self.get_nemo_client()

            logger.info("Using NeMo SDK to download results for evaluation job: %s", job_id)
            # Download evaluation job results using SDK
            try:
                result = await nemo_client.evaluation.jobs.download_results(
                    job_id=job_id,
                    extra_headers={"Authorization": f"Bearer {self.api_key}"},
                )

                logger.info("Download result type: %s", type(result))
                logger.info("Download result has model_dump: %s", hasattr(result, "model_dump"))
                logger.info(
                    "Download result attributes: %s", dir(result) if hasattr(result, "__dict__") else "No __dict__"
                )

                # Check if result is an HTTP response-like object
                if hasattr(result, "http_response") and hasattr(result.http_response, "content"):
                    logger.info("Found http_response with content, extracting content")
                    content = result.http_response.content
                    if isinstance(content, bytes):
                        # Check if it's a ZIP file (starts with PK signature)
                        if content.startswith(b"PK"):
                            logger.info("Detected ZIP file content")
                            import base64

                            return {
                                "content": base64.b64encode(content).decode("utf-8"),
                                "content_type": "application/zip",
                                "encoding": "base64",
                                "filename": f"evaluation_results_{job_id}.zip",
                            }

                        try:
                            # Try to decode as JSON
                            import json

                            content_str = content.decode("utf-8")
                            return json.loads(content_str)
                        except (UnicodeDecodeError, json.JSONDecodeError):
                            # If not JSON, return as base64 encoded string
                            import base64

                            return {
                                "content": base64.b64encode(content).decode("utf-8"),
                                "content_type": "application/octet-stream",
                                "encoding": "base64",
                            }
                    elif isinstance(content, str):
                        try:
                            import json

                            return json.loads(content)
                        except json.JSONDecodeError:
                            return {"content": content, "content_type": "text"}
                    else:
                        return self._force_json_serializable(content)

                # Check if result has response content directly
                elif hasattr(result, "content"):
                    logger.info("Found content attribute directly on result")
                    content = result.content
                    if isinstance(content, bytes):
                        # Check if it's a ZIP file (starts with PK signature)
                        if content.startswith(b"PK"):
                            logger.info("Detected ZIP file content (direct access)")
                            import base64

                            return {
                                "content": base64.b64encode(content).decode("utf-8"),
                                "content_type": "application/zip",
                                "encoding": "base64",
                                "filename": f"evaluation_results_{job_id}.zip",
                            }

                        try:
                            import json

                            content_str = content.decode("utf-8")
                            return json.loads(content_str)
                        except (UnicodeDecodeError, json.JSONDecodeError):
                            import base64

                            return {
                                "content": base64.b64encode(content).decode("utf-8"),
                                "content_type": "application/octet-stream",
                                "encoding": "base64",
                            }
                    else:
                        return self._force_json_serializable(content)

                # Convert SDK response to dict and serialize datetime objects
                elif hasattr(result, "model_dump"):
                    result_dict = result.model_dump()
                    return self._serialize_datetime_objects(result_dict)
                elif isinstance(result, dict | list | str | int | float | bool) or result is None:
                    # If result is already a JSON-serializable type, serialize datetime objects
                    return self._serialize_datetime_objects(result)
                else:
                    # Use the robust JSON serializer for complex objects
                    logger.info("Using robust serialization for object of type: %s", type(result))
                    # For HTTP response objects, try to extract useful information
                    if hasattr(result, "status_code"):
                        logger.warning(
                            "Received HTTP response object with status %s but no accessible content", result.status_code
                        )
                        return {
                            "error": "Download endpoint returned HTTP response metadata instead of file content",
                            "status_code": getattr(result, "status_code", None),
                            "message": "The NeMo API may not support direct file downloads through the SDK",
                        }
                    return self._force_json_serializable(result)

            except AttributeError:
                # Fallback to direct HTTP call if SDK doesn't support download_results method
                logger.info("SDK doesn't support download_results method, falling back to HTTP call")
                download_url = f"{self.base_url}/v1/evaluation/jobs/{job_id}/download-results"
                logger.info("Downloading evaluation results from: %s", download_url)

                async with httpx.AsyncClient(timeout=30.0) as client:
                    response = await client.get(download_url, headers=self._get_auth_headers())
                    response.raise_for_status()

                    # Handle different content types
                    content_type = response.headers.get("content-type", "").lower()
                    if "application/json" in content_type:
                        return response.json()
                    if content_type.startswith(("application/", "text/")):
                        # For binary or text files, return content with metadata
                        content = response.content
                        if isinstance(content, bytes):
                            try:
                                # Try to decode as text first
                                content_str = content.decode("utf-8")
                            except UnicodeDecodeError:
                                # Return as base64 for binary content
                                import base64

                                return {
                                    "content": base64.b64encode(content).decode("utf-8"),
                                    "content_type": content_type,
                                    "encoding": "base64",
                                }
                            else:
                                return {"content": content_str, "content_type": content_type, "encoding": "text"}
                        else:
                            return {"content": str(content), "content_type": content_type}
                    else:
                        # Default to JSON parsing
                        return response.json()

        except NeMoMicroservicesError as exc:
            logger.exception("NeMo SDK error downloading evaluation results for job %s", job_id)
            # Check if it's a 404 (not found) error
            if "404" in str(exc) or "not found" in str(exc).lower():
                logger.info("Evaluation job %s not found for download", job_id)
                return {"message": f"Evaluation job {job_id} not found"}
            # Handle 401 Unauthorized gracefully
            if "401" in str(exc) or "Unauthorized" in str(exc):
                logger.warning("Authentication failed for NeMo microservices - cannot download evaluation job results")
                msg = f"Authentication failed for NeMo microservices: {exc}"
                raise ValueError(msg) from exc
            logger.exception("NeMo microservices error while downloading evaluation job results")
            msg = f"NeMo microservices error while downloading evaluation job results: {exc}"
            raise ValueError(msg) from exc
        except httpx.HTTPStatusError as e:
            logger.exception(
                "HTTP error downloading evaluation results for job %s: %s - %s",
                job_id,
                e.response.status_code,
                e.response.text,
            )
            raise
        except Exception:
            logger.exception("Exception downloading evaluation results for job %s", job_id)
            raise

    # =============================================================================
    # Job Tracking for Langflow Dashboard
    # =============================================================================

    async def track_customizer_job(self, job_id: str, metadata: dict | None = None) -> dict[str, Any]:
        """Track a customization job for dashboard monitoring."""
        # For API, we just return success since tracking is handled by the service
        return {
            "job_id": job_id,
            "tracked_at": "2024-01-01T00:00:00Z",
            "metadata": metadata or {},
            "message": f"Now tracking job {job_id}",
        }

    async def get_tracked_jobs(self) -> list[dict[str, Any]]:
        """Get tracked jobs for dashboard monitoring."""
        # For API, return actual jobs from the service
        try:
            jobs_response = await self.list_customizer_jobs(page=1, page_size=1000)  # Get all jobs for tracking
            jobs = jobs_response["data"]
        except Exception:
            logger.exception("Failed to get tracked jobs")
            raise
        else:
            tracked = []
            for job in jobs:
                config = job.get("config")
                if isinstance(config, dict):
                    config_name = config.get("name", "Unknown Model")
                elif isinstance(config, str):
                    config_name = config
                else:
                    config_name = "Unknown Model"
                tracked.append(
                    {
                        "job_id": job["id"],
                        "status": job["status"],
                        "created_at": job["created_at"],
                        "updated_at": job["updated_at"],
                        "config": config_name,
                        "dataset": job.get("dataset", "Unknown Dataset"),
                        "progress": job.get("status_details", {}).get("percentage_done", 0),
                        "output_model": job.get("output_model"),
                        "hyperparameters": job.get("hyperparameters"),
                        "custom_fields": job.get("custom_fields"),
                    }
                )
            return tracked

    async def stop_tracking_job(self, job_id: str) -> dict[str, Any]:
        """Stop tracking a job."""
        return {"message": f"Stopped tracking job {job_id}"}

    # =============================================================================
    # Evaluation Management (Evaluator)
    # =============================================================================

    async def create_evaluation_job(self, job_data: dict) -> dict[str, Any]:
        """Create a new evaluation job using NeMo Python SDK."""
        try:
            # Use NeMo client for evaluation job creation
            nemo_client = self.get_nemo_client()

            # Create evaluation job using SDK
            job = await nemo_client.evaluation.jobs.create(
                namespace=job_data.get("namespace", "default"),
                config=job_data["config"],
                target=job_data["target"],
                tags=job_data.get("tags", []),
                extra_headers={"Authorization": f"Bearer {self.api_key}"},
            )

            # Convert SDK response to dict and serialize datetime objects
            job_dict = job.model_dump()
            job_dict = self._serialize_datetime_objects(job_dict)

            logger.info("Created evaluation job %s", job_dict.get("id"))

        except NeMoMicroservicesError as exc:
            logger.exception("NeMo microservices error while creating evaluation job")
            msg = f"NeMo microservices error while creating evaluation job: {exc}"
            raise ValueError(msg) from exc
        except Exception:
            logger.exception("Failed to create evaluation job")
            raise
        else:
            return job_dict

    async def create_evaluation_config(self, config_data: dict) -> dict[str, Any]:
        """Create a new evaluation config using NeMo Python SDK."""
        try:
            # Use NeMo client for evaluation config creation
            nemo_client = self.get_nemo_client()

            # Create evaluation config using SDK
            config = await nemo_client.evaluation.configs.create(
                type=config_data["type"],
                namespace=config_data.get("namespace", "default"),
                tasks=config_data["tasks"],
                params=config_data.get("params", {}),
                extra_headers={"Authorization": f"Bearer {self.api_key}"},
            )

            # Convert SDK response to dict and serialize datetime objects
            config_dict = config.model_dump()
            config_dict = self._serialize_datetime_objects(config_dict)

            logger.info("Created evaluation config %s", config_dict.get("id"))

        except NeMoMicroservicesError as exc:
            logger.exception("NeMo microservices error while creating evaluation config")
            msg = f"NeMo microservices error while creating evaluation config: {exc}"
            raise ValueError(msg) from exc
        except Exception:
            logger.exception("Failed to create evaluation config")
            raise
        else:
            return config_dict

    async def create_evaluation_target(self, target_data: dict) -> dict[str, Any]:
        """Create a new evaluation target using NeMo Python SDK."""
        try:
            # Use NeMo client for evaluation target creation
            nemo_client = self.get_nemo_client()

            # Create evaluation target using SDK
            target = await nemo_client.evaluation.targets.create(
                type=target_data["type"],
                namespace=target_data.get("namespace", "default"),
                model=target_data["model"],
                extra_headers={"Authorization": f"Bearer {self.api_key}"},
            )

            # Convert SDK response to dict and serialize datetime objects
            target_dict = target.model_dump()
            target_dict = self._serialize_datetime_objects(target_dict)

            logger.info("Created evaluation target %s", target_dict.get("id"))

        except NeMoMicroservicesError as exc:
            logger.exception("NeMo microservices error while creating evaluation target")
            msg = f"NeMo microservices error while creating evaluation target: {exc}"
            raise ValueError(msg) from exc
        except Exception:
            logger.exception("Failed to create evaluation target")
            raise
        else:
            return target_dict

    async def get_evaluation_job(self, job_id: str) -> dict[str, Any] | None:
        """Get evaluation job details using NeMo Python SDK."""
        try:
            # Use NeMo client for evaluation job retrieval
            nemo_client = self.get_nemo_client()

            # Get evaluation job using SDK
            job = await nemo_client.evaluation.jobs.retrieve(
                job_id=job_id,
                extra_headers={"Authorization": f"Bearer {self.api_key}"},
            )

            # Convert SDK response to dict and serialize datetime objects
            job_dict = job.model_dump()
            job_dict = self._serialize_datetime_objects(job_dict)

            logger.info("Retrieved evaluation job %s", job_id)

        except NeMoMicroservicesError as exc:
            # Check if it's a 404 (not found) error
            if "404" in str(exc) or "not found" in str(exc).lower():
                logger.info("Evaluation job %s not found", job_id)
                return None
            # Handle 401 Unauthorized gracefully
            if "401" in str(exc) or "Unauthorized" in str(exc):
                logger.warning("Authentication failed for NeMo microservices - cannot get evaluation job")
                return None
            logger.exception("NeMo microservices error while getting evaluation job")
            msg = f"NeMo microservices error while getting evaluation job: {exc}"
            raise ValueError(msg) from exc
        except Exception:
            logger.exception("Failed to get evaluation job")
            raise
        else:
            return job_dict

    async def list_evaluation_jobs(
        self, page: int = 1, page_size: int = 10, namespace: str | None = None
    ) -> dict[str, Any]:
        """List evaluation jobs using NeMo Python SDK."""
        logger.info(
            "Starting list_evaluation_jobs with params: page=%s, page_size=%s, namespace=%s", page, page_size, namespace
        )

        try:
            # Use NeMo client for evaluation jobs
            nemo_client = self.get_nemo_client()
            logger.info("Created NeMo client with base_url=%s", self.base_url)

            # Build headers for SDK call with authentication
            headers = {"Authorization": f"Bearer {self.api_key}"}

            # Add namespace to headers if provided
            if namespace:
                headers["X-Namespace"] = namespace
                headers["X-NeMo-Namespace"] = namespace
                logger.info("Using namespace filter in headers: %s", namespace)
            else:
                logger.info("No namespace filter provided")

            # Debug logging to check auth token
            api_key_preview_length = 10
            api_key_preview = (
                f"{self.api_key[:api_key_preview_length]}..."
                if self.api_key and len(self.api_key) > api_key_preview_length
                else "None"
            )
            logger.info("Calling SDK evaluation.jobs.list with auth token: %s", api_key_preview)
            logger.info("Headers being sent to SDK: %s", list(headers.keys()))

            # Get evaluation jobs using SDK with proper auth headers
            try:
                # Pass authentication headers to SDK
                response = await nemo_client.evaluation.jobs.list(extra_headers=headers)
                logger.info("SDK call successful")
            except Exception as e:
                logger.exception("SDK call failed with error")
                # Fallback to direct HTTP call if SDK fails
                logger.info("Falling back to direct HTTP call for evaluation jobs")
                try:
                    jobs_url = f"{self.base_url}/v1/evaluation/jobs"
                    logger.info("Getting evaluation jobs from: %s", jobs_url)

                    async with httpx.AsyncClient(timeout=30.0) as client:
                        response = await client.get(jobs_url, headers=self._get_auth_headers())
                        response.raise_for_status()
                        response_data = response.json()

                        # Handle different response formats from direct API
                        if isinstance(response_data, dict) and "data" in response_data:
                            all_jobs_raw = response_data["data"]
                        elif isinstance(response_data, list):
                            all_jobs_raw = response_data
                        else:
                            all_jobs_raw = []

                        # Convert to consistent format
                        all_jobs = []
                        for job in all_jobs_raw:
                            if isinstance(job, dict):
                                # Apply datetime serialization
                                job_dict = self._serialize_datetime_objects(job)

                                # Filter by namespace if provided
                                if namespace and job_dict.get("namespace") != namespace:
                                    continue

                                all_jobs.append(job_dict)

                        # Apply client-side pagination
                        total_jobs = len(all_jobs)
                        start_index = (page - 1) * page_size
                        end_index = start_index + page_size
                        paginated_jobs = all_jobs[start_index:end_index]

                        # Calculate pagination metadata
                        total_pages = (total_jobs + page_size - 1) // page_size if total_jobs > 0 else 1

                        # Return paginated response
                        result = {
                            "data": paginated_jobs,
                            "page": page,
                            "page_size": page_size,
                            "total": total_jobs,
                            "total_pages": total_pages,
                            "has_next": page < total_pages,
                            "has_prev": page > 1,
                        }

                        logger.info(
                            "HTTP fallback: Successfully retrieved %s total evaluation jobs, returning %s for page %s",
                            len(all_jobs),
                            len(paginated_jobs),
                            page,
                        )
                        return result

                except httpx.HTTPStatusError as http_e:
                    logger.exception(
                        "HTTP fallback also failed: %s - %s", http_e.response.status_code, http_e.response.text
                    )
                    return {
                        "data": [],
                        "page": page,
                        "page_size": page_size,
                        "total": 0,
                        "total_pages": 0,
                        "has_next": False,
                        "has_prev": False,
                        "error": f"Both SDK and HTTP calls failed. SDK: {e}, HTTP: {http_e}",
                    }
                except Exception as http_e:
                    logger.exception("HTTP fallback failed with exception")
                    return {
                        "data": [],
                        "page": page,
                        "page_size": page_size,
                        "total": 0,
                        "total_pages": 0,
                        "has_next": False,
                        "has_prev": False,
                        "error": f"Both SDK and HTTP calls failed. SDK: {e}, HTTP: {http_e}",
                    }
            logger.info("SDK returned response with %s jobs", len(response.data) if response.data else 0)

            # Convert SDK response to list format for frontend
            all_jobs = []
            for job in response.data:
                job_dict = job.model_dump()
                # Apply datetime serialization
                job_dict = self._serialize_datetime_objects(job_dict)

                # Filter by namespace if provided (client-side filtering)
                if namespace and job_dict.get("namespace") != namespace:
                    logger.debug("Skipping job %s - namespace mismatch", job_dict.get("id", "unknown"))
                    continue

                all_jobs.append(job_dict)
                logger.debug(
                    "Processed job: %s with status: %s",
                    job_dict.get("id", "unknown"),
                    job_dict.get("status", "unknown"),
                )

            # Apply client-side pagination since SDK might not support it
            start_index = (page - 1) * page_size
            end_index = start_index + page_size
            paginated_jobs = all_jobs[start_index:end_index]

            # Calculate pagination metadata
            total_jobs = len(all_jobs)
            total_pages = (total_jobs + page_size - 1) // page_size if total_jobs > 0 else 1

            # Return paginated response in the same format as customizer jobs
            result = {
                "data": paginated_jobs,
                "page": page,
                "page_size": page_size,
                "total": total_jobs,
                "total_pages": total_pages,
                "has_next": page < total_pages,
                "has_prev": page > 1,
            }

            logger.info(
                "Successfully retrieved %s total evaluation jobs, returning %s for page %s (namespace=%s)",
                len(all_jobs),
                len(paginated_jobs),
                page,
                namespace,
            )

        except NeMoMicroservicesError as exc:
            logger.exception("NeMo microservices error while listing evaluation jobs")

            # Handle 401 Unauthorized gracefully
            if "401" in str(exc) or "Unauthorized" in str(exc):
                logger.warning("Authentication failed for NeMo microservices - returning empty jobs list")
                return {
                    "data": [],
                    "page": page,
                    "page_size": page_size,
                    "total": 0,
                    "total_pages": 0,
                    "has_next": False,
                    "has_prev": False,
                    "error": "Authentication failed. Please check your NeMo credentials.",
                }

            msg = f"NeMo microservices error while listing evaluation jobs: {exc}"
            raise ValueError(msg) from exc
        except Exception:
            logger.exception("Failed to list evaluation jobs")
            raise
        else:
            return result

    async def delete_evaluation_job(self, job_id: str) -> dict[str, Any]:
        """Delete an evaluation job using NeMo Python SDK."""
        logger.info("Service layer: Attempting to delete evaluation job %s", job_id)
        try:
            # Use NeMo client for evaluation job deletion
            nemo_client = self.get_nemo_client()

            logger.info("Using NeMo SDK to delete evaluation job: %s", job_id)
            # Delete evaluation job using SDK
            result = await nemo_client.evaluation.jobs.delete(
                job_id=job_id,
                extra_headers={"Authorization": f"Bearer {self.api_key}"},
            )

            logger.info("SDK delete result for evaluation job %s: %s", job_id, result)
            logger.info("Successfully deleted evaluation job %s", job_id)
            delete_result = {"message": f"Evaluation job {job_id} deleted successfully"}

        except NeMoMicroservicesError as exc:
            logger.exception("NeMo SDK error deleting evaluation job %s", job_id)
            # Check if it's a 404 (not found) error
            if "404" in str(exc) or "not found" in str(exc).lower():
                logger.info("Evaluation job %s not found for deletion", job_id)
                return {"message": f"Evaluation job {job_id} not found"}
            # Handle 401 Unauthorized gracefully
            if "401" in str(exc) or "Unauthorized" in str(exc):
                logger.warning("Authentication failed for NeMo microservices - cannot delete evaluation job")
                msg = f"Authentication failed for NeMo microservices: {exc}"
                raise ValueError(msg) from exc
            logger.exception("NeMo microservices error while deleting evaluation job")
            msg = f"NeMo microservices error while deleting evaluation job: {exc}"
            raise ValueError(msg) from exc
        except Exception:
            logger.exception("Exception deleting evaluation job %s", job_id)
            raise
        else:
            return delete_result

    async def get_dataset_details(self, dataset_name: str, namespace: str | None = None) -> dict[str, Any]:
        """Get detailed dataset information including files from HuggingFace API.

        Args:
            dataset_name: Dataset name
            namespace: Dataset namespace (optional, defaults to 'default')

        Returns:
            Dataset details with files from siblings section
        """
        try:
            # Use default namespace if not provided
            ns = namespace or "default"

            # Use HuggingFace API endpoint for detailed dataset info
            url = f"{self.base_url}/v1/hf/api/datasets/{ns}/{dataset_name}"

            async with httpx.AsyncClient(timeout=30.0) as client:
                response = await client.get(url, headers=self._get_auth_headers())
                response.raise_for_status()
                response_data = response.json()
        except Exception:
            logger.exception("Failed to get dataset details")
            raise
        else:
            return response_data

    def cleanup(self):
        """Cleanup resources."""

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
        elif isinstance(obj, list):
            return [self._serialize_datetime_objects(item) for item in obj]
        elif hasattr(obj, 'isoformat'):  # datetime objects
            return obj.isoformat()
        else:
            return obj

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
                extra_headers={"Authorization": f"Bearer {self.api_key}"}
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
            total_results = paginated_response.pagination.total_results if paginated_response.pagination else len(datasets)
            
            response = {
                "data": datasets,
                "page": current_page,
                "page_size": page_size,
                "total": total_results,
                "total_pages": total_pages,
                "has_next": current_page < total_pages,
                "has_prev": current_page > 1,
            }
            
            logger.info(f"Retrieved {len(datasets)} datasets from entity store (page {page}, size {page_size})")
            return response
            
        except NeMoMicroservicesError as exc:
            logger.exception("NeMo microservices error while listing datasets: %s", exc)
            
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
                    "error": "Authentication failed. Please check your NeMo credentials."
                }
            
            raise ValueError(f"NeMo microservices error while listing datasets: {exc}") from exc
        except Exception:
            logger.exception("Failed to list datasets")
            raise

    async def create_dataset(
        self, name: str, description: str | None = None, dataset_type: str = "fileset", namespace: str = "default"
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
                extra_headers={"Authorization": f"Bearer {self.api_key}"}
            )
            
            # Convert SDK response to dict and serialize datetime objects
            result = dataset_response.model_dump()
            result = self._serialize_datetime_objects(result)
            
            logger.info(f"Created dataset {name} in namespace {namespace}")
            return result
            
        except NeMoMicroservicesError as exc:
            logger.exception("NeMo microservices error while creating dataset: %s", exc)
            raise ValueError(f"NeMo microservices error while creating dataset: {exc}") from exc
        except Exception:
            logger.exception("Failed to create dataset")
            raise

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

            logger.info(f"Created HuggingFace repo: {repo_id}")

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
                extra_headers={"Authorization": f"Bearer {self.api_key}"}
            )

            # Convert SDK response to dict and serialize datetime objects
            result = dataset_response.model_dump()
            result = self._serialize_datetime_objects(result)

            logger.info(f"Successfully registered dataset {name} in entity registry using SDK")

            # Return structured response
            return {
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
            logger.exception("NeMo microservices error while creating dataset: %s", exc)
            raise ValueError(f"NeMo microservices error while creating dataset: {exc}") from exc
        except Exception:
            logger.exception("Failed to create dataset with namespace")
            raise

    async def create_namespace(self, namespace: str):
        """Create namespace in entity-store with authentication."""
        url = f"{self.base_url}/v1/namespaces"

        try:
            async with httpx.AsyncClient(timeout=30.0) as client:
                response = await client.get(f"{url}/{namespace}", headers=self._get_auth_headers())
                if response.status_code == codes.NOT_FOUND:
                    logger.info(f"Namespace not found, creating namespace: {namespace}")
                    create_payload = {"namespace": namespace}
                    create_response = await client.post(url, json=create_payload, headers=self._get_auth_headers())
                    create_response.raise_for_status()
                else:
                    response.raise_for_status()

        except httpx.HTTPStatusError as e:
            logger.error(f"Error processing namespace: {e}")
            raise ValueError(f"Error processing namespace: {e}") from e

    async def create_datastore_namespace(self, namespace: str):
        """Create namespace in datastore with authentication."""
        url = f"{self.base_url}/v1/datastore/namespaces"

        try:
            async with httpx.AsyncClient(timeout=30.0) as client:
                response = await client.get(f"{url}/{namespace}", headers=self._get_auth_headers())
                if response.status_code == codes.NOT_FOUND:
                    logger.info(f"Datastore namespace not found, creating namespace: {namespace}")
                    create_payload = {"namespace": namespace}
                    create_response = await client.post(url, json=create_payload, headers=self._get_auth_headers())
                    create_response.raise_for_status()
                else:
                    response.raise_for_status()

        except httpx.HTTPStatusError as e:
            logger.error(f"Error processing datastore namespace: {e}")
            raise ValueError(f"Error processing datastore namespace: {e}") from e

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
                dataset_name=dataset_name,
                namespace=ns,
                extra_headers={"Authorization": f"Bearer {self.api_key}"}
            )
            
            # Convert SDK response to dict and serialize datetime objects
            dataset_dict = dataset.model_dump()
            dataset_dict = self._serialize_datetime_objects(dataset_dict)
            
            logger.info(f"Retrieved dataset {dataset_name} from namespace {ns}")
            return dataset_dict
            
        except NeMoMicroservicesError as exc:
            # Check if it's a 404 (not found) error
            if "404" in str(exc) or "not found" in str(exc).lower():
                logger.info(f"Dataset {dataset_name} not found in namespace {ns}")
                return None
            # Handle 401 Unauthorized gracefully
            if "401" in str(exc) or "Unauthorized" in str(exc):
                logger.warning("Authentication failed for NeMo microservices - cannot get dataset")
                return None
            logger.exception("NeMo microservices error while getting dataset: %s", exc)
            raise ValueError(f"NeMo microservices error while getting dataset: {exc}") from exc
        except Exception:
            logger.exception("Failed to get dataset")
            raise

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
            
            delete_response = await nemo_client.datasets.delete(
                dataset_name=dataset_name,
                namespace=ns,
                extra_headers={"Authorization": f"Bearer {self.api_key}"}
            )
            
            # SDK returns DeleteResponse object, check if successful
            logger.info(f"Deleted dataset {dataset_name} from namespace {ns}")
            return True
            
        except NeMoMicroservicesError as exc:
            # Check if it's a 404 (not found) error
            if "404" in str(exc) or "not found" in str(exc).lower():
                logger.info(f"Dataset {dataset_name} not found in namespace {ns} for deletion")
                return False
            logger.exception("NeMo microservices error while deleting dataset: %s", exc)
            raise ValueError(f"NeMo microservices error while deleting dataset: {exc}") from exc
        except Exception:
            logger.exception("Failed to delete dataset")
            raise

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
                return response.json()
        except Exception:
            logger.exception("Failed to upload files")
            raise

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
            except Exception as e:
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

            return {
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

    async def get_dataset_files(self, dataset_id: str) -> list[dict[str, Any]]:
        """Get list of files in a NeMo dataset."""
        try:
            async with httpx.AsyncClient(timeout=30.0) as client:
                response = await client.get(
                    f"{self.base_url}/v1/datasets/{dataset_id}/files", headers=self._get_auth_headers()
                )
                response.raise_for_status()
                return response.json().get("data", [])
        except Exception:
            logger.exception("Failed to get dataset files")
            raise

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
                return response.json()
        except Exception:
            logger.exception("Failed to get customization configs")
            raise

    async def create_customization_job(self, job_data: dict) -> dict[str, Any]:
        """Create a new customization job."""
        try:
            async with httpx.AsyncClient(timeout=30.0) as client:
                response = await client.post(
                    f"{self.base_url}/v1/customization/jobs", json=job_data, headers=self._get_auth_headers()
                )
                response.raise_for_status()
                return response.json()
        except Exception:
            logger.exception("Failed to create customization job")
            raise

    async def get_customizer_job_status(self, job_id: str) -> dict[str, Any] | None:
        """Get customization job status."""
        try:
            async with httpx.AsyncClient(timeout=30.0) as client:
                response = await client.get(
                    f"{self.base_url}/v1/customization/jobs/{job_id}/status", headers=self._get_auth_headers()
                )
                response.raise_for_status()
                return response.json()
        except httpx.HTTPStatusError as e:
            if e.response.status_code == codes.NOT_FOUND:
                return None
            raise
        except Exception:
            logger.exception("Failed to get job status")
            raise

    async def get_customizer_job_details(self, job_id: str) -> dict[str, Any] | None:
        """Get customization job details."""
        try:
            async with httpx.AsyncClient(timeout=30.0) as client:
                response = await client.get(
                    f"{self.base_url}/v1/customization/jobs/{job_id}", headers=self._get_auth_headers()
                )
                response.raise_for_status()
                return response.json()
        except httpx.HTTPStatusError as e:
            if e.response.status_code == codes.NOT_FOUND:
                return None
            raise
        except Exception:
            logger.exception("Failed to get job details")
            raise

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
            return {
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
                    "error": "Authentication failed. Please check your NeMo credentials."
                }
            
            raise

    async def cancel_customization_job(self, job_id: str) -> dict[str, Any]:
        """Cancel a customization job."""
        try:
            async with httpx.AsyncClient(timeout=30.0) as client:
                response = await client.post(
                    f"{self.base_url}/v1/customization/jobs/{job_id}/cancel", headers=self._get_auth_headers()
                )
                response.raise_for_status()
                return response.json()
        except Exception:
            logger.exception("Failed to cancel customization job")
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
        """Create a new evaluation job."""
        try:
            async with httpx.AsyncClient(timeout=30.0) as client:
                response = await client.post(
                    f"{self.base_url}/v1/evaluation/jobs", json=job_data, headers=self._get_auth_headers()
                )
                response.raise_for_status()
                return response.json()
        except Exception:
            logger.exception("Failed to create evaluation job")
            raise

    async def get_evaluation_job(self, job_id: str) -> dict[str, Any] | None:
        """Get evaluation job details."""
        try:
            async with httpx.AsyncClient(timeout=30.0) as client:
                response = await client.get(
                    f"{self.base_url}/v1/evaluation/jobs/{job_id}", headers=self._get_auth_headers()
                )
                response.raise_for_status()
                return response.json()
        except httpx.HTTPStatusError as e:
            if e.response.status_code == codes.NOT_FOUND:
                return None
            raise
        except Exception:
            logger.exception("Failed to get evaluation job")
            raise

    async def list_evaluation_jobs(self, page: int = 1, page_size: int = 10) -> dict[str, Any]:
        """List evaluation jobs with pagination."""
        try:
            # Get all jobs first since NeMo Evaluator API doesn't support pagination
            async with httpx.AsyncClient(timeout=30.0) as client:
                response = await client.get(f"{self.base_url}/v1/evaluation/jobs", headers=self._get_auth_headers())
                response.raise_for_status()
                all_jobs = response.json().get("data", [])
            
            # Apply client-side pagination
            total_jobs = len(all_jobs)
            start_index = (page - 1) * page_size
            end_index = start_index + page_size
            paginated_jobs = all_jobs[start_index:end_index]
            
            total_pages = (total_jobs + page_size - 1) // page_size
            
            # Return paginated response in the same format as datasets
            return {
                "data": paginated_jobs,
                "page": page,
                "page_size": page_size,
                "total": total_jobs,
                "total_pages": total_pages,
                "has_next": page < total_pages,
                "has_prev": page > 1,
            }
        except Exception:
            logger.exception("Failed to list evaluation jobs")
            raise

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
                return response.json()
        except Exception:
            logger.exception("Failed to get dataset details")
            raise

    def cleanup(self):
        """Cleanup resources."""

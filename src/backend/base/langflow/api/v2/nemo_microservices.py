"""NeMo Microservices API endpoints for Langflow.

This module provides API endpoints for managing NeMo Microservices integration
within Langflow, including Data Store, Customizer, and Evaluator operations.

The NeMo Microservices integration allows Langflow users to:
- List and manage datasets from NeMo Data Store
- Upload files to datasets for training/evaluation
- Track and monitor Customizer jobs
- View Evaluator results and metrics
- Integrate with NeMo components without requiring direct NeMo service access

API Endpoints:
- GET /api/v2/nemo/datasets - List all datasets with pagination
- POST /api/v2/nemo/datasets - Create new dataset
- GET /api/v2/nemo/datasets/{id} - Get dataset details
- GET /api/v2/nemo/datasets/{id}/details - Get dataset details with file info
- DELETE /api/v2/nemo/datasets/{id} - Delete dataset
- POST /api/v2/nemo/datasets/{id}/files - Upload files
- GET /api/v2/nemo/datasets/{id}/files - List dataset files
- GET /api/v2/nemo/jobs - List customizer jobs
- GET /api/v2/nemo/jobs/{id} - Get job details
- POST /api/v2/nemo/jobs - Store job for tracking

Note: This implementation requires proper NeMo configuration via global variables or environment variables.
"""

from typing import Annotated

import httpx
from fastapi import APIRouter, File, Form, Header, HTTPException, UploadFile

from langflow.api.utils import CurrentActiveUser, DbSession
from langflow.services.nemo_microservices_factory import get_nemo_service

router = APIRouter(prefix="/nemo", tags=["NeMo Microservices"])


# =============================================================================
# Dataset Management (Data Store)
# =============================================================================


@router.get("/datasets", response_model=dict)
async def list_datasets(
    current_user: CurrentActiveUser,
    session: DbSession,
    page: int = 1,
    page_size: int = 10,
    dataset_name: str | None = None,
    namespace: str | None = None,
    x_nemo_auth_token: Annotated[str | None, Header(alias="X-NeMo-Auth-Token")] = None,
    x_nemo_base_url: Annotated[str | None, Header(alias="X-NeMo-Base-URL")] = None,
):
    """List NeMo datasets with pagination and optional filtering by name and namespace.

    Args:
        current_user: Current authenticated user
        session: Database session
        page: Page number (1-based)
        page_size: Number of datasets per page (default: 10)
        dataset_name: Optional dataset name to filter by
        namespace: Optional namespace to filter by
        x_nemo_auth_token: Optional NeMo authentication token override
        x_nemo_base_url: Optional NeMo base URL override

    Returns:
        Paginated list of dataset objects with metadata
    """
    try:
        nemo_service = await get_nemo_service(
            current_user.id, session, header_api_key=x_nemo_auth_token, header_base_url=x_nemo_base_url
        )
        return await nemo_service.list_datasets(
            page=page, page_size=page_size, dataset_name=dataset_name, namespace=namespace
        )
    except ValueError as e:
        if "configuration is incomplete" in str(e):
            raise HTTPException(status_code=503, detail=f"NeMo service unavailable: {e}") from e
        raise HTTPException(status_code=500, detail=f"Failed to list datasets: {e}") from e
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to list datasets: {e}") from e


@router.post("/datasets", response_model=dict)
async def create_dataset(
    current_user: CurrentActiveUser,
    session: DbSession,
    name: Annotated[str, Form()],
    namespace: Annotated[str, Form()],
    description: Annotated[str | None, Form()] = None,
    dataset_type: Annotated[str, Form()] = "fileset",
    x_nemo_auth_token: Annotated[str | None, Header(alias="X-NeMo-Auth-Token")] = None,
    x_nemo_base_url: Annotated[str | None, Header(alias="X-NeMo-Base-URL")] = None,
):
    """Create a new NeMo dataset.

    Args:
        current_user: Current authenticated user
        session: Database session
        name: Dataset name
        namespace: Dataset namespace
        description: Optional description
        dataset_type: Type of dataset (default: fileset)
        x_nemo_auth_token: Optional NeMo authentication token override
        x_nemo_base_url: Optional NeMo base URL override

    Returns:
        Created dataset object
    """
    try:
        nemo_service = await get_nemo_service(
            current_user.id, session, header_api_key=x_nemo_auth_token, header_base_url=x_nemo_base_url
        )
        return await nemo_service.create_dataset_with_namespace(
            name=name, namespace=namespace, description=description, dataset_type=dataset_type
        )
    except ValueError as e:
        if "configuration is incomplete" in str(e):
            raise HTTPException(status_code=503, detail=f"NeMo service unavailable: {e}") from e
        raise HTTPException(status_code=500, detail=f"Failed to create dataset: {e}") from e
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to create dataset: {e}") from e


@router.get("/datasets/{dataset_name}", response_model=dict)
async def get_dataset(
    current_user: CurrentActiveUser,
    session: DbSession,
    dataset_name: str,
    namespace: str | None = None,
    x_nemo_auth_token: Annotated[str | None, Header(alias="X-NeMo-Auth-Token")] = None,
    x_nemo_base_url: Annotated[str | None, Header(alias="X-NeMo-Base-URL")] = None,
):
    """Get detailed information about a specific dataset by name.

    Args:
        current_user: Current authenticated user
        session: Database session
        dataset_name: NeMo Entity Store dataset name
        namespace: Dataset namespace (optional, defaults to 'default')
        x_nemo_auth_token: Optional NeMo authentication token override
        x_nemo_base_url: Optional NeMo base URL override

    Returns:
        Dataset details including files and metadata
    """
    try:
        nemo_service = await get_nemo_service(
            current_user.id, session, header_api_key=x_nemo_auth_token, header_base_url=x_nemo_base_url
        )
        dataset = await nemo_service.get_dataset(dataset_name, namespace=namespace)
        if dataset:
            return dataset
        raise HTTPException(status_code=404, detail="Dataset not found")
    except HTTPException:
        raise
    except ValueError as e:
        if "configuration is incomplete" in str(e):
            raise HTTPException(status_code=503, detail=f"NeMo service unavailable: {e}") from e
        raise HTTPException(status_code=500, detail=f"Failed to get dataset: {e}") from e
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get dataset: {e}") from e


@router.get("/datasets/{dataset_name}/details", response_model=dict)
async def get_dataset_details_with_files(
    current_user: CurrentActiveUser,
    session: DbSession,
    dataset_name: str,
    namespace: str | None = None,
    x_nemo_auth_token: Annotated[str | None, Header(alias="X-NeMo-Auth-Token")] = None,
    x_nemo_base_url: Annotated[str | None, Header(alias="X-NeMo-Base-URL")] = None,
):
    """Get detailed information about a dataset including file list from HuggingFace API.

    Args:
        current_user: Current authenticated user
        session: Database session
        dataset_name: NeMo Data Store dataset name
        namespace: Dataset namespace (optional)
        x_nemo_auth_token: Optional NeMo authentication token override
        x_nemo_base_url: Optional NeMo base URL override

    Returns:
        Dataset details with files from siblings section
    """
    try:
        nemo_service = await get_nemo_service(
            current_user.id, session, header_api_key=x_nemo_auth_token, header_base_url=x_nemo_base_url
        )
        return await nemo_service.get_dataset_details(dataset_name, namespace=namespace)
    except ValueError as e:
        if "configuration is incomplete" in str(e):
            raise HTTPException(status_code=503, detail=f"NeMo service unavailable: {e}") from e
        raise HTTPException(status_code=500, detail=f"Failed to get dataset details: {e}") from e
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get dataset details: {e}") from e


@router.delete("/datasets/{dataset_name}")
async def delete_dataset(
    current_user: CurrentActiveUser,
    session: DbSession,
    dataset_name: str,
    namespace: str | None = None,
    x_nemo_auth_token: Annotated[str | None, Header(alias="X-NeMo-Auth-Token")] = None,
    x_nemo_base_url: Annotated[str | None, Header(alias="X-NeMo-Base-URL")] = None,
):
    """Delete a dataset from NeMo Entity Store.

    Args:
        current_user: Current authenticated user
        session: Database session
        dataset_name: NeMo Entity Store dataset name
        namespace: Dataset namespace (optional, defaults to 'default')
        x_nemo_auth_token: NeMo API authentication token (optional, from header)
        x_nemo_base_url: NeMo API base URL (optional, from header)

    Returns:
        Success message
    """
    try:
        nemo_service = await get_nemo_service(
            current_user.id, session, header_api_key=x_nemo_auth_token, header_base_url=x_nemo_base_url
        )
        deleted = await nemo_service.delete_dataset(dataset_name, namespace=namespace)
        if deleted:
            return {"message": "Dataset deleted successfully"}
        raise HTTPException(status_code=404, detail="Dataset not found")
    except HTTPException:
        raise
    except ValueError as e:
        if "configuration is incomplete" in str(e):
            raise HTTPException(status_code=503, detail=f"NeMo service unavailable: {e}") from e
        raise HTTPException(status_code=500, detail=f"Failed to delete dataset: {e}") from e
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to delete dataset: {e}") from e


@router.post("/datasets/{dataset_id}/files", response_model=dict)
async def upload_files(
    current_user: CurrentActiveUser,
    session: DbSession,
    dataset_id: str,
    files: Annotated[list[UploadFile], File(...)],
    x_nemo_auth_token: Annotated[str | None, Header(alias="X-NeMo-Auth-Token")] = None,
    x_nemo_base_url: Annotated[str | None, Header(alias="X-NeMo-Base-URL")] = None,
):
    """Upload files to a NeMo dataset.

    Args:
        current_user: Current authenticated user
        session: Database session
        dataset_id: NeMo Data Store dataset ID
        files: List of files to upload
        x_nemo_auth_token: NeMo API authentication token (optional, from header)
        x_nemo_base_url: NeMo API base URL (optional, from header)

    Returns:
        Upload result with file information
    """
    try:
        nemo_service = await get_nemo_service(
            current_user.id, session, header_api_key=x_nemo_auth_token, header_base_url=x_nemo_base_url
        )
        return await nemo_service.upload_files(dataset_id=dataset_id, files=files)
    except (httpx.HTTPError, ValueError, OSError) as e:
        raise HTTPException(status_code=500, detail=f"Failed to upload files: {e}") from e


@router.get("/datasets/{dataset_id}/files", response_model=list[dict])
async def get_dataset_files(
    dataset_id: str,
    current_user: CurrentActiveUser,
    session: DbSession,
    x_nemo_auth_token: Annotated[str | None, Header(alias="X-NeMo-Auth-Token")] = None,
    x_nemo_base_url: Annotated[str | None, Header(alias="X-NeMo-Base-URL")] = None,
):
    """Get list of files in a NeMo dataset.

    Args:
        dataset_id: NeMo Data Store dataset ID
        current_user: Current authenticated user
        session: Database session
        x_nemo_auth_token: NeMo API authentication token (optional, from header)
        x_nemo_base_url: NeMo API base URL (optional, from header)

    Returns:
        List of files in the dataset
    """
    try:
        nemo_service = await get_nemo_service(
            current_user.id, session, header_api_key=x_nemo_auth_token, header_base_url=x_nemo_base_url
        )
        return await nemo_service.get_dataset_files(dataset_id)
    except (httpx.HTTPError, ValueError, OSError) as e:
        raise HTTPException(status_code=500, detail=f"Failed to get dataset files: {e}") from e


@router.post("/datasets/{dataset_name}/upload", response_model=dict)
async def upload_dataset_files(
    current_user: CurrentActiveUser,
    session: DbSession,
    dataset_name: str,
    path: Annotated[str, Form()],
    namespace: Annotated[str, Form()],
    files: Annotated[list[UploadFile], File(...)],
    x_nemo_auth_token: Annotated[str | None, Header(alias="X-NeMo-Auth-Token")] = None,
    x_nemo_base_url: Annotated[str | None, Header(alias="X-NeMo-Base-URL")] = None,
):
    """Upload files to a NeMo dataset with specified path using HuggingFace API.

    This endpoint mimics the upload functionality from nvidia_customizer component,
    uploading files to a dataset using the HuggingFace API with Git LFS support.
    Files are uploaded to the specified path within the dataset (e.g., training/file.jsonl).

    Args:
        current_user: Current authenticated user
        session: Database session
        dataset_name: NeMo Data Store dataset name
        path: Path within the dataset (e.g., 'training', 'validation')
        namespace: Dataset namespace
        files: List of files to upload
        x_nemo_auth_token: NeMo API authentication token (optional, from header)
        x_nemo_base_url: NeMo API base URL (optional, from header)

    Returns:
        Upload result with file information and paths
    """
    try:
        nemo_service = await get_nemo_service(
            current_user.id, session, header_api_key=x_nemo_auth_token, header_base_url=x_nemo_base_url
        )
        return await nemo_service.upload_dataset_files_with_path(
            dataset_name=dataset_name, path=path, namespace=namespace, files=files
        )
    except (httpx.HTTPError, ValueError, OSError) as e:
        raise HTTPException(status_code=500, detail=f"Failed to upload files: {e}") from e


# =============================================================================
# Job Management (Customizer) - NeMo API Structure
# =============================================================================


@router.get("/v1/customization/configs", response_model=dict)
async def get_customization_configs(
    current_user: CurrentActiveUser,
    session: DbSession,
    x_nemo_auth_token: Annotated[str | None, Header(alias="X-NeMo-Auth-Token")] = None,
    x_nemo_base_url: Annotated[str | None, Header(alias="X-NeMo-Base-URL")] = None,
):
    """Get available model configurations for customization.

    This endpoint matches the NeMo Customizer API:
    GET /v1/customization/configs

    Used by the NeMo Customizer component to populate dropdown options
    for model selection, training types, and fine-tuning types.

    Returns:
        Available model configurations with training and fine-tuning types
    """
    try:
        nemo_service = await get_nemo_service(
            current_user.id, session, header_api_key=x_nemo_auth_token, header_base_url=x_nemo_base_url
        )
        return await nemo_service.get_customization_configs()
    except (httpx.HTTPError, ValueError) as e:
        raise HTTPException(status_code=500, detail=f"Failed to get customization configs: {e}") from e


@router.get("/v1/customization/jobs/{job_id}/status", response_model=dict)
async def get_job_status(
    job_id: str,
    current_user: CurrentActiveUser,
    session: DbSession,
    x_nemo_auth_token: Annotated[str | None, Header(alias="X-NeMo-Auth-Token")] = None,
    x_nemo_base_url: Annotated[str | None, Header(alias="X-NeMo-Base-URL")] = None,
):
    """Get customization job status with timestamped training/validation loss.

    This endpoint matches the NeMo Customizer API:
    GET /v1/customization/jobs/{customizationID}/status

    Args:
        job_id: NeMo Customizer job ID
        current_user: Current authenticated user
        session: Database session
        x_nemo_auth_token: NeMo API authentication token (optional, from header)
        x_nemo_base_url: NeMo API base URL (optional, from header)

    Returns:
        Job status with timestamped training and validation loss values
    """
    try:
        nemo_service = await get_nemo_service(
            current_user.id, session, header_api_key=x_nemo_auth_token, header_base_url=x_nemo_base_url
        )
        job_status = await nemo_service.get_customizer_job_status(job_id)
        if job_status:
            return job_status
        raise HTTPException(status_code=404, detail="Job not found")
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get job status: {e}") from e


@router.post("/v1/customization/jobs", response_model=dict)
async def create_customization_job(
    job_data: dict,
    current_user: CurrentActiveUser,
    session: DbSession,
    x_nemo_auth_token: Annotated[str | None, Header(alias="X-NeMo-Auth-Token")] = None,
    x_nemo_base_url: Annotated[str | None, Header(alias="X-NeMo-Base-URL")] = None,
):
    """Create a new customization job.

    This endpoint matches the NeMo Customizer API:
    POST /v1/customization/jobs

    Args:
        job_data: Job configuration data
        current_user: Current authenticated user
        session: Database session
        x_nemo_auth_token: NeMo API authentication token (optional, from header)
        x_nemo_base_url: NeMo API base URL (optional, from header)

    Returns:
        Created job information
    """
    try:
        nemo_service = await get_nemo_service(
            current_user.id, session, header_api_key=x_nemo_auth_token, header_base_url=x_nemo_base_url
        )
        return await nemo_service.create_customization_job(job_data)
    except (httpx.HTTPError, ValueError) as e:
        raise HTTPException(status_code=500, detail=f"Failed to create customization job: {e}") from e


@router.get("/v1/customization/jobs/{job_id}", response_model=dict)
async def get_job_details(
    job_id: str,
    current_user: CurrentActiveUser,
    session: DbSession,
    x_nemo_auth_token: Annotated[str | None, Header(alias="X-NeMo-Auth-Token")] = None,
    x_nemo_base_url: Annotated[str | None, Header(alias="X-NeMo-Base-URL")] = None,
):
    """Get detailed information about a customization job.

    This endpoint matches the NeMo Customizer API:
    GET /v1/customization/jobs/{customizationID}

    Args:
        job_id: NeMo Customizer job ID
        current_user: Current authenticated user
        session: Database session
        x_nemo_auth_token: NeMo API authentication token (optional, from header)
        x_nemo_base_url: NeMo API base URL (optional, from header)

    Returns:
        Detailed job information
    """
    try:
        nemo_service = await get_nemo_service(
            current_user.id, session, header_api_key=x_nemo_auth_token, header_base_url=x_nemo_base_url
        )
        job_details = await nemo_service.get_customizer_job_details(job_id)
        if job_details:
            return job_details
        raise HTTPException(status_code=404, detail="Job not found")
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get job details: {e}") from e


@router.post("/v1/customization/jobs/{job_id}/cancel", response_model=dict)
async def cancel_customization_job(
    job_id: str,
    current_user: CurrentActiveUser,
    session: DbSession,
    x_nemo_auth_token: Annotated[str | None, Header(alias="X-NeMo-Auth-Token")] = None,
    x_nemo_base_url: Annotated[str | None, Header(alias="X-NeMo-Base-URL")] = None,
):
    """Cancel a customization job.

    This endpoint matches the NeMo Customizer API:
    POST /v1/customization/jobs/{job_id}/cancel

    Args:
        job_id: NeMo Customizer job ID to cancel
        current_user: Current authenticated user
        session: Database session
        x_nemo_auth_token: NeMo API authentication token (optional, from header)
        x_nemo_base_url: NeMo API base URL (optional, from header)

    Returns:
        Cancellation confirmation
    """
    try:
        nemo_service = await get_nemo_service(
            current_user.id, session, header_api_key=x_nemo_auth_token, header_base_url=x_nemo_base_url
        )
        return await nemo_service.cancel_customization_job(job_id)
    except HTTPException:
        raise
    except ValueError as e:
        if "configuration is incomplete" in str(e):
            raise HTTPException(status_code=503, detail=f"NeMo service unavailable: {e}") from e
        raise HTTPException(status_code=500, detail=f"Failed to cancel customization job: {e}") from e
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to cancel customization job: {e}") from e


@router.delete("/v1/customization/jobs/{job_id}", response_model=dict)
async def delete_customization_job(
    job_id: str,
    current_user: CurrentActiveUser,
    session: DbSession,
    x_nemo_auth_token: Annotated[str | None, Header(alias="X-NeMo-Auth-Token")] = None,
    x_nemo_base_url: Annotated[str | None, Header(alias="X-NeMo-Base-URL")] = None,
):
    # Add print statement to ensure we see this even if logging is not working
    """Delete a customization job.

    This endpoint provides delete functionality for customization jobs.
    If direct deletion is not supported, it will fall back to cancellation.

    Args:
        job_id: NeMo Customizer job ID to delete
        current_user: Current authenticated user
        session: Database session
        x_nemo_auth_token: NeMo API authentication token (optional, from header)
        x_nemo_base_url: NeMo API base URL (optional, from header)

    Returns:
        Deletion confirmation
    """
    import logging

    logger = logging.getLogger(__name__)

    logger.info("Delete customization job request - Job ID: %s", job_id)

    try:
        nemo_service = await get_nemo_service(
            current_user.id, session, header_api_key=x_nemo_auth_token, header_base_url=x_nemo_base_url
        )
        logger.info("Attempting to delete customization job: %s", job_id)
        result = await nemo_service.cancel_customization_job(job_id)
        logger.info("Successfully deleted customization job: %s, Result: %s", job_id, result)
        if True:  # Always true since we got here
            return result
    except HTTPException:
        raise
    except ValueError as e:
        logger.exception("ValueError deleting customization job %s", job_id)
        if "configuration is incomplete" in str(e):
            raise HTTPException(status_code=503, detail=f"NeMo service unavailable: {e}") from e
        raise HTTPException(status_code=500, detail=f"Failed to delete customization job: {e}") from e
    except Exception as e:
        logger.exception("Exception deleting customization job %s", job_id)
        raise HTTPException(status_code=500, detail=f"Failed to delete customization job: {e}") from e


@router.get("/v1/customization/jobs/{job_id}/container-logs", response_model=dict)
async def get_customization_job_container_logs(
    job_id: str,
    current_user: CurrentActiveUser,
    session: DbSession,
    x_nemo_auth_token: Annotated[str | None, Header(alias="X-NeMo-Auth-Token")] = None,
    x_nemo_base_url: Annotated[str | None, Header(alias="X-NeMo-Base-URL")] = None,
):
    """Get container logs for a customization job."""
    try:
        nemo_service = await get_nemo_service(
            current_user.id, session, header_api_key=x_nemo_auth_token, header_base_url=x_nemo_base_url
        )
        return await nemo_service.get_customization_job_container_logs(job_id)
    except HTTPException:
        raise
    except ValueError as e:
        if "configuration is incomplete" in str(e):
            raise HTTPException(status_code=503, detail=f"NeMo service unavailable: {e}") from e
        raise HTTPException(status_code=500, detail=f"Failed to get customization job container logs: {e}") from e
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get customization job container logs: {e}") from e


@router.get("/v1/customization/jobs", response_model=dict)
async def list_all_customizer_jobs(
    current_user: CurrentActiveUser,
    session: DbSession,
    page: int = 1,
    page_size: int = 10,
    x_nemo_auth_token: Annotated[str | None, Header(alias="X-NeMo-Auth-Token")] = None,
    x_nemo_base_url: Annotated[str | None, Header(alias="X-NeMo-Base-URL")] = None,
):
    """List customization jobs with pagination.

    This endpoint matches the NeMo Customizer API:
    GET /v1/customization/jobs

    Args:
        current_user: Current authenticated user
        session: Database session
        page: Page number (1-based)
        page_size: Number of jobs per page (default: 10)
        x_nemo_auth_token: NeMo API authentication token (optional, from header)
        x_nemo_base_url: NeMo API base URL (optional, from header)

    Returns:
        Paginated list of customization jobs
    """
    try:
        nemo_service = await get_nemo_service(
            current_user.id, session, header_api_key=x_nemo_auth_token, header_base_url=x_nemo_base_url
        )
        return await nemo_service.list_customizer_jobs(page=page, page_size=page_size)
    except (httpx.HTTPError, ValueError) as e:
        raise HTTPException(status_code=500, detail=f"Failed to list customization jobs: {e}") from e


# =============================================================================
# Evaluation Management (Evaluator) - NeMo API Structure
# =============================================================================


@router.post("/v1/evaluation/jobs", response_model=dict)
async def create_evaluation_job(
    job_data: dict,
    current_user: CurrentActiveUser,
    session: DbSession,
    x_nemo_auth_token: Annotated[str | None, Header(alias="X-NeMo-Auth-Token")] = None,
    x_nemo_base_url: Annotated[str | None, Header(alias="X-NeMo-Base-URL")] = None,
):
    """Create a new evaluation job.

    This endpoint matches the NeMo Evaluator API:
    POST /v1/evaluation/jobs

    Args:
        job_data: Evaluation job configuration data
        current_user: Current authenticated user
        session: Database session
        x_nemo_auth_token: NeMo API authentication token (optional, from header)
        x_nemo_base_url: NeMo API base URL (optional, from header)

    Returns:
        Created evaluation job information
    """
    try:
        nemo_service = await get_nemo_service(
            current_user.id, session, header_api_key=x_nemo_auth_token, header_base_url=x_nemo_base_url
        )
        return await nemo_service.create_evaluation_job(job_data)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to create evaluation job: {e}") from e


@router.post("/v1/evaluation/configs", response_model=dict)
async def create_evaluation_config(
    config_data: dict,
    current_user: CurrentActiveUser,
    session: DbSession,
    x_nemo_auth_token: Annotated[str | None, Header(alias="X-NeMo-Auth-Token")] = None,
    x_nemo_base_url: Annotated[str | None, Header(alias="X-NeMo-Base-URL")] = None,
):
    """Create an evaluation configuration.

    This endpoint matches the NeMo Evaluator API:
    POST /v1/evaluation/configs

    Args:
        config_data: Evaluation configuration data
        current_user: Current authenticated user
        session: Database session
        x_nemo_auth_token: NeMo API authentication token (optional, from header)
        x_nemo_base_url: NeMo API base URL (optional, from header)

    Returns:
        Created evaluation configuration information
    """
    try:
        nemo_service = await get_nemo_service(
            current_user.id, session, header_api_key=x_nemo_auth_token, header_base_url=x_nemo_base_url
        )
        return await nemo_service.create_evaluation_config(config_data)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to create evaluation config: {e}") from e


@router.post("/v1/evaluation/targets", response_model=dict)
async def create_evaluation_target(
    target_data: dict,
    current_user: CurrentActiveUser,
    session: DbSession,
    x_nemo_auth_token: Annotated[str | None, Header(alias="X-NeMo-Auth-Token")] = None,
    x_nemo_base_url: Annotated[str | None, Header(alias="X-NeMo-Base-URL")] = None,
):
    """Create an evaluation target.

    This endpoint matches the NeMo Evaluator API:
    POST /v1/evaluation/targets

    Args:
        target_data: Evaluation target configuration data
        current_user: Current authenticated user
        session: Database session
        x_nemo_auth_token: NeMo API authentication token (optional, from header)
        x_nemo_base_url: NeMo API base URL (optional, from header)

    Returns:
        Created evaluation target information
    """
    try:
        nemo_service = await get_nemo_service(
            current_user.id, session, header_api_key=x_nemo_auth_token, header_base_url=x_nemo_base_url
        )
        return await nemo_service.create_evaluation_target(target_data)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to create evaluation target: {e}") from e


@router.get("/v1/evaluation/jobs/{job_id}", response_model=dict)
async def get_evaluation_job(
    job_id: str,
    current_user: CurrentActiveUser,
    session: DbSession,
    x_nemo_auth_token: Annotated[str | None, Header(alias="X-NeMo-Auth-Token")] = None,
    x_nemo_base_url: Annotated[str | None, Header(alias="X-NeMo-Base-URL")] = None,
):
    """Get evaluation job details.

    This endpoint matches the NeMo Evaluator API:
    GET /v1/evaluation/jobs/{job_id}

    Args:
        job_id: NeMo Evaluator job ID
        current_user: Current authenticated user
        session: Database session
        x_nemo_auth_token: NeMo API authentication token (optional, from header)
        x_nemo_base_url: NeMo API base URL (optional, from header)

    Returns:
        Evaluation job details
    """
    try:
        nemo_service = await get_nemo_service(
            current_user.id, session, header_api_key=x_nemo_auth_token, header_base_url=x_nemo_base_url
        )
        job_details = await nemo_service.get_evaluation_job(job_id)
        if job_details:
            return job_details
        raise HTTPException(status_code=404, detail="Evaluation job not found")
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get evaluation job: {e}") from e


@router.get("/v1/evaluation/jobs", response_model=dict)
async def list_evaluation_jobs(
    current_user: CurrentActiveUser,
    session: DbSession,
    page: int = 1,
    page_size: int = 10,
    x_nemo_auth_token: Annotated[str | None, Header(alias="X-NeMo-Auth-Token")] = None,
    x_nemo_base_url: Annotated[str | None, Header(alias="X-NeMo-Base-URL")] = None,
    x_nemo_namespace: Annotated[str | None, Header(alias="X-NeMo-Namespace")] = None,
):
    """List evaluation jobs with pagination.

    This endpoint matches the NeMo Evaluator API:
    GET /v1/evaluation/jobs

    Args:
        current_user: Current authenticated user
        session: Database session
        page: Page number (1-based)
        page_size: Number of jobs per page (default: 10)
        x_nemo_auth_token: NeMo API authentication token (optional, from header)
        x_nemo_base_url: NeMo API base URL (optional, from header)
        x_nemo_namespace: NeMo namespace (optional, from header)

    Returns:
        Paginated response with evaluation jobs data and metadata
    """
    try:
        nemo_service = await get_nemo_service(
            current_user.id, session, header_api_key=x_nemo_auth_token, header_base_url=x_nemo_base_url
        )
        # Pass namespace from headers to service
        return await nemo_service.list_evaluation_jobs(page=page, page_size=page_size, namespace=x_nemo_namespace)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to list evaluation jobs: {e}") from e


@router.delete("/v1/evaluation/jobs/{job_id}", response_model=dict)
async def delete_evaluation_job(
    job_id: str,
    current_user: CurrentActiveUser,
    session: DbSession,
    x_nemo_auth_token: Annotated[str | None, Header(alias="X-NeMo-Auth-Token")] = None,
    x_nemo_base_url: Annotated[str | None, Header(alias="X-NeMo-Base-URL")] = None,
    x_nemo_namespace: Annotated[str | None, Header(alias="X-NeMo-Namespace")] = None,
):
    """Delete an evaluation job.

    This endpoint provides delete functionality for evaluation jobs using the NeMo SDK.

    Args:
        job_id: NeMo Evaluator job ID to delete
        current_user: Current authenticated user
        session: Database session
        x_nemo_auth_token: NeMo API authentication token (optional, from header)
        x_nemo_base_url: NeMo API base URL (optional, from header)
        x_nemo_namespace: NeMo namespace (optional, from header)

    Returns:
        Deletion confirmation
    """
    import logging

    logger = logging.getLogger(__name__)

    logger.info("Delete evaluation job request - Job ID: %s, Namespace: %s", job_id, x_nemo_namespace)

    try:
        nemo_service = await get_nemo_service(
            current_user.id, session, header_api_key=x_nemo_auth_token, header_base_url=x_nemo_base_url
        )
        logger.info("Attempting to delete evaluation job: %s", job_id)
        result = await nemo_service.delete_evaluation_job(job_id)
        logger.info("Successfully deleted evaluation job: %s, Result: %s", job_id, result)
        if True:  # Always true since we got here
            return result
    except HTTPException:
        raise
    except ValueError as e:
        logger.exception("ValueError deleting evaluation job %s", job_id)
        if "configuration is incomplete" in str(e):
            raise HTTPException(status_code=503, detail=f"NeMo service unavailable: {e}") from e
        raise HTTPException(status_code=500, detail=f"Failed to delete evaluation job: {e}") from e
    except Exception as e:
        logger.exception("Exception deleting evaluation job %s", job_id)
        raise HTTPException(status_code=500, detail=f"Failed to delete evaluation job: {e}") from e


@router.get("/v1/evaluation/jobs/{job_id}/logs", response_model=dict)
async def get_evaluation_job_logs(
    job_id: str,
    current_user: CurrentActiveUser,
    session: DbSession,
    x_nemo_auth_token: Annotated[str | None, Header(alias="X-NeMo-Auth-Token")] = None,
    x_nemo_base_url: Annotated[str | None, Header(alias="X-NeMo-Base-URL")] = None,
):
    """Get logs for an evaluation job."""
    try:
        nemo_service = await get_nemo_service(
            current_user.id, session, header_api_key=x_nemo_auth_token, header_base_url=x_nemo_base_url
        )
        return await nemo_service.get_evaluation_job_logs(job_id)
    except HTTPException:
        raise
    except ValueError as e:
        if "configuration is incomplete" in str(e):
            raise HTTPException(status_code=503, detail=f"NeMo service unavailable: {e}") from e
        raise HTTPException(status_code=500, detail=f"Failed to get evaluation job logs: {e}") from e
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get evaluation job logs: {e}") from e


@router.get("/v1/evaluation/jobs/{job_id}/results", response_model=dict)
async def get_evaluation_job_results(
    job_id: str,
    current_user: CurrentActiveUser,
    session: DbSession,
    x_nemo_auth_token: Annotated[str | None, Header(alias="X-NeMo-Auth-Token")] = None,
    x_nemo_base_url: Annotated[str | None, Header(alias="X-NeMo-Base-URL")] = None,
):
    """Get results for a completed evaluation job."""
    try:
        nemo_service = await get_nemo_service(
            current_user.id, session, header_api_key=x_nemo_auth_token, header_base_url=x_nemo_base_url
        )
        return await nemo_service.get_evaluation_job_results(job_id)
    except HTTPException:
        raise
    except ValueError as e:
        if "configuration is incomplete" in str(e):
            raise HTTPException(status_code=503, detail=f"NeMo service unavailable: {e}") from e
        raise HTTPException(status_code=500, detail=f"Failed to get evaluation job results: {e}") from e
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get evaluation job results: {e}") from e


@router.get("/v1/evaluation/jobs/{job_id}/download-results")
async def download_evaluation_job_results(
    job_id: str,
    current_user: CurrentActiveUser,
    session: DbSession,
    x_nemo_auth_token: Annotated[str | None, Header(alias="X-NeMo-Auth-Token")] = None,
    x_nemo_base_url: Annotated[str | None, Header(alias="X-NeMo-Base-URL")] = None,
):
    """Download results for a completed evaluation job."""
    try:
        nemo_service = await get_nemo_service(
            current_user.id, session, header_api_key=x_nemo_auth_token, header_base_url=x_nemo_base_url
        )
        return await nemo_service.download_evaluation_job_results(job_id)
    except HTTPException:
        raise
    except ValueError as e:
        if "configuration is incomplete" in str(e):
            raise HTTPException(status_code=503, detail=f"NeMo service unavailable: {e}") from e
        raise HTTPException(status_code=500, detail=f"Failed to download evaluation job results: {e}") from e
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to download evaluation job results: {e}") from e


# =============================================================================
# Job Tracking for Langflow Dashboard
# =============================================================================


@router.post("/jobs/track", response_model=dict)
async def track_job(
    job_id: str,
    current_user: CurrentActiveUser,
    session: DbSession,
    metadata: dict | None = None,
    x_nemo_auth_token: Annotated[str | None, Header(alias="X-NeMo-Auth-Token")] = None,
    x_nemo_base_url: Annotated[str | None, Header(alias="X-NeMo-Base-URL")] = None,
):
    """Start tracking a NeMo Customizer job for dashboard monitoring.

    Args:
        job_id: NeMo Customizer job ID to track
        current_user: Current authenticated user
        session: Database session
        metadata: Optional metadata for tracking
        x_nemo_auth_token: NeMo API authentication token (optional, from header)
        x_nemo_base_url: NeMo API base URL (optional, from header)

    Returns:
        Tracking confirmation
    """
    try:
        nemo_service = await get_nemo_service(
            current_user.id, session, header_api_key=x_nemo_auth_token, header_base_url=x_nemo_base_url
        )
        return await nemo_service.track_customizer_job(job_id, metadata)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to track job: {e}") from e


@router.get("/jobs/tracked", response_model=list[dict])
async def get_tracked_jobs(
    current_user: CurrentActiveUser,
    session: DbSession,
    x_nemo_auth_token: Annotated[str | None, Header(alias="X-NeMo-Auth-Token")] = None,
    x_nemo_base_url: Annotated[str | None, Header(alias="X-NeMo-Base-URL")] = None,
):
    """Get all jobs being tracked for dashboard monitoring.

    Returns:
        List of tracked job IDs with their current status
    """
    try:
        nemo_service = await get_nemo_service(
            current_user.id, session, header_api_key=x_nemo_auth_token, header_base_url=x_nemo_base_url
        )
        return await nemo_service.get_tracked_jobs()
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get tracked jobs: {e}") from e


@router.delete("/jobs/track/{job_id}")
async def stop_tracking_job(
    job_id: str,
    current_user: CurrentActiveUser,
    session: DbSession,
    x_nemo_auth_token: Annotated[str | None, Header(alias="X-NeMo-Auth-Token")] = None,
    x_nemo_base_url: Annotated[str | None, Header(alias="X-NeMo-Base-URL")] = None,
):
    """Stop tracking a job for dashboard monitoring.

    Args:
        job_id: Job ID to stop tracking
        current_user: Current authenticated user
        session: Database session
        x_nemo_auth_token: NeMo API authentication token (optional, from header)
        x_nemo_base_url: NeMo API base URL (optional, from header)

    Returns:
        Confirmation message
    """
    try:
        nemo_service = await get_nemo_service(
            current_user.id, session, header_api_key=x_nemo_auth_token, header_base_url=x_nemo_base_url
        )
        return await nemo_service.stop_tracking_job(job_id)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to stop tracking job: {e}") from e


# =============================================================================
# Legacy Endpoints (Deprecated - Use NeMo API Endpoints Above)
# =============================================================================


@router.post("/jobs", response_model=dict)
async def store_job_for_tracking_legacy(
    job_data: dict,
    current_user: CurrentActiveUser,
    session: DbSession,
    x_nemo_auth_token: Annotated[str | None, Header(alias="X-NeMo-Auth-Token")] = None,
    x_nemo_base_url: Annotated[str | None, Header(alias="X-NeMo-Base-URL")] = None,
):
    """Legacy endpoint: Store job info from NeMo component for tracking.

    DEPRECATED: Use POST /jobs/track instead

    Args:
        job_data: Job information from NeMo Customizer component
        current_user: Current authenticated user
        session: Database session
        x_nemo_auth_token: NeMo API authentication token (optional, from header)
        x_nemo_base_url: NeMo API base URL (optional, from header)

    Returns:
        Stored job data
    """
    try:
        job_id = job_data.get("job_info", {}).get("id")
        if not job_id:
            raise HTTPException(status_code=400, detail="Missing job ID in job_data")
        metadata = {"legacy_data": job_data, "source": "legacy_endpoint"}
        nemo_service = await get_nemo_service(
            current_user.id, session, header_api_key=x_nemo_auth_token, header_base_url=x_nemo_base_url
        )
        return await nemo_service.track_customizer_job(job_id, metadata)
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to store job: {e}") from e


@router.get("/jobs", response_model=list[dict])
async def list_customizer_jobs_legacy(
    current_user: CurrentActiveUser,
    session: DbSession,
    x_nemo_auth_token: Annotated[str | None, Header(alias="X-NeMo-Auth-Token")] = None,
    x_nemo_base_url: Annotated[str | None, Header(alias="X-NeMo-Base-URL")] = None,
):
    """Legacy endpoint: List all tracked customizer jobs.

    DEPRECATED: Use GET /jobs/tracked or GET /v1/customization/jobs

    Returns:
        List of tracked jobs
    """
    try:
        nemo_service = await get_nemo_service(
            current_user.id, session, header_api_key=x_nemo_auth_token, header_base_url=x_nemo_base_url
        )
        return await nemo_service.get_tracked_jobs()
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to list jobs: {e}") from e


@router.get("/jobs/{job_id}", response_model=dict)
async def get_customizer_job_legacy(
    job_id: str,
    current_user: CurrentActiveUser,
    session: DbSession,
    x_nemo_auth_token: Annotated[str | None, Header(alias="X-NeMo-Auth-Token")] = None,
    x_nemo_base_url: Annotated[str | None, Header(alias="X-NeMo-Base-URL")] = None,
):
    """Get customizer job details (legacy endpoint).

    Args:
        job_id: NeMo Customizer job ID
        current_user: Current authenticated user
        session: Database session
        x_nemo_auth_token: NeMo API authentication token (optional, from header)
        x_nemo_base_url: NeMo API base URL (optional, from header)

    Returns:
        Job details
    """
    try:
        nemo_service = await get_nemo_service(
            current_user.id, session, header_api_key=x_nemo_auth_token, header_base_url=x_nemo_base_url
        )
        job = await nemo_service.get_customizer_job_details(job_id)
        if job:
            return job
        raise HTTPException(status_code=404, detail="Job not found")
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get job: {e}") from e

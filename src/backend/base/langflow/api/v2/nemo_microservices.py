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
            raise HTTPException(status_code=503, detail=f"NeMo service unavailable: {e!s}") from e
        raise HTTPException(status_code=500, detail=f"Failed to list datasets: {e!s}") from e
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to list datasets: {e!s}") from e


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
            raise HTTPException(status_code=503, detail=f"NeMo service unavailable: {e!s}") from e
        raise HTTPException(status_code=500, detail=f"Failed to create dataset: {e!s}") from e
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to create dataset: {e!s}") from e


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

    Returns:
        Dataset details including files and metadata
    """
    try:
        nemo_service = await get_nemo_service(
            current_user.id, session, header_api_key=x_nemo_auth_token, header_base_url=x_nemo_base_url
        )
        dataset = await nemo_service.get_dataset(dataset_name, namespace=namespace)
        if not dataset:
            raise HTTPException(status_code=404, detail="Dataset not found")
        return dataset
    except HTTPException:
        raise
    except ValueError as e:
        if "configuration is incomplete" in str(e):
            raise HTTPException(status_code=503, detail=f"NeMo service unavailable: {e!s}") from e
        raise HTTPException(status_code=500, detail=f"Failed to get dataset: {e!s}") from e
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get dataset: {e!s}") from e


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
            raise HTTPException(status_code=503, detail=f"NeMo service unavailable: {e!s}") from e
        raise HTTPException(status_code=500, detail=f"Failed to get dataset details: {e!s}") from e
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get dataset details: {e!s}") from e


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

    Returns:
        Success message
    """
    try:
        nemo_service = await get_nemo_service(
            current_user.id, session, header_api_key=x_nemo_auth_token, header_base_url=x_nemo_base_url
        )
        deleted = await nemo_service.delete_dataset(dataset_name, namespace=namespace)
        if not deleted:
            raise HTTPException(status_code=404, detail="Dataset not found")
        return {"message": "Dataset deleted successfully"}
    except HTTPException:
        raise
    except ValueError as e:
        if "configuration is incomplete" in str(e):
            raise HTTPException(status_code=503, detail=f"NeMo service unavailable: {e!s}") from e
        raise HTTPException(status_code=500, detail=f"Failed to delete dataset: {e!s}") from e
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to delete dataset: {e!s}") from e


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

    Returns:
        Upload result with file information
    """
    try:
        nemo_service = await get_nemo_service(
            current_user.id, session, header_api_key=x_nemo_auth_token, header_base_url=x_nemo_base_url
        )
        return await nemo_service.upload_files(dataset_id=dataset_id, files=files)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to upload files: {e!s}") from e


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

    Returns:
        List of files in the dataset
    """
    try:
        nemo_service = await get_nemo_service(
            current_user.id, session, header_api_key=x_nemo_auth_token, header_base_url=x_nemo_base_url
        )
        return await nemo_service.get_dataset_files(dataset_id)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get dataset files: {e!s}") from e


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
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to upload files: {e!s}") from e


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
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get customization configs: {e!s}") from e


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

    Returns:
        Job status with timestamped training and validation loss values
    """
    try:
        nemo_service = await get_nemo_service(
            current_user.id, session, header_api_key=x_nemo_auth_token, header_base_url=x_nemo_base_url
        )
        job_status = await nemo_service.get_customizer_job_status(job_id)
        if not job_status:
            raise HTTPException(status_code=404, detail="Job not found")
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get job status: {e!s}") from e
    else:
        return job_status


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

    Returns:
        Created job information
    """
    try:
        nemo_service = await get_nemo_service(
            current_user.id, session, header_api_key=x_nemo_auth_token, header_base_url=x_nemo_base_url
        )
        return await nemo_service.create_customization_job(job_data)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to create customization job: {e!s}") from e


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

    Returns:
        Detailed job information
    """
    try:
        nemo_service = await get_nemo_service(
            current_user.id, session, header_api_key=x_nemo_auth_token, header_base_url=x_nemo_base_url
        )
        job_details = await nemo_service.get_customizer_job_details(job_id)
        if not job_details:
            raise HTTPException(status_code=404, detail="Job not found")
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get job details: {e!s}") from e
    else:
        return job_details


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
            raise HTTPException(status_code=503, detail=f"NeMo service unavailable: {e!s}") from e
        raise HTTPException(status_code=500, detail=f"Failed to cancel customization job: {e!s}") from e
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to cancel customization job: {e!s}") from e


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
        page: Page number (1-based)
        page_size: Number of jobs per page (default: 10)

    Returns:
        Paginated list of customization jobs
    """
    try:
        nemo_service = await get_nemo_service(
            current_user.id, session, header_api_key=x_nemo_auth_token, header_base_url=x_nemo_base_url
        )
        return await nemo_service.list_customizer_jobs(page=page, page_size=page_size)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to list customization jobs: {e!s}") from e


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

    Returns:
        Created evaluation job information
    """
    try:
        nemo_service = await get_nemo_service(
            current_user.id, session, header_api_key=x_nemo_auth_token, header_base_url=x_nemo_base_url
        )
        return await nemo_service.create_evaluation_job(job_data)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to create evaluation job: {e!s}") from e


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

    Returns:
        Created evaluation configuration information
    """
    try:
        nemo_service = await get_nemo_service(
            current_user.id, session, header_api_key=x_nemo_auth_token, header_base_url=x_nemo_base_url
        )
        return await nemo_service.create_evaluation_config(config_data)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to create evaluation config: {e!s}") from e


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

    Returns:
        Created evaluation target information
    """
    try:
        nemo_service = await get_nemo_service(
            current_user.id, session, header_api_key=x_nemo_auth_token, header_base_url=x_nemo_base_url
        )
        return await nemo_service.create_evaluation_target(target_data)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to create evaluation target: {e!s}") from e


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

    Returns:
        Evaluation job details
    """
    try:
        nemo_service = await get_nemo_service(
            current_user.id, session, header_api_key=x_nemo_auth_token, header_base_url=x_nemo_base_url
        )
        job_details = await nemo_service.get_evaluation_job(job_id)
        if not job_details:
            raise HTTPException(status_code=404, detail="Evaluation job not found")
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get evaluation job: {e!s}") from e
    else:
        return job_details


@router.get("/v1/evaluation/jobs", response_model=list[dict])
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
        page: Page number (1-based)
        page_size: Number of jobs per page (default: 10)
        x_nemo_namespace: Optional namespace from frontend headers

    Returns:
        Paginated list of evaluation jobs
    """
    try:
        nemo_service = await get_nemo_service(
            current_user.id, session, header_api_key=x_nemo_auth_token, header_base_url=x_nemo_base_url
        )
        # Pass namespace from headers to service
        return await nemo_service.list_evaluation_jobs(page=page, page_size=page_size, namespace=x_nemo_namespace)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to list evaluation jobs: {e!s}") from e


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

    Returns:
        Tracking confirmation
    """
    try:
        nemo_service = await get_nemo_service(
            current_user.id, session, header_api_key=x_nemo_auth_token, header_base_url=x_nemo_base_url
        )
        return await nemo_service.track_customizer_job(job_id, metadata)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to track job: {e!s}") from e


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
        raise HTTPException(status_code=500, detail=f"Failed to get tracked jobs: {e!s}") from e


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

    Returns:
        Confirmation message
    """
    try:
        nemo_service = await get_nemo_service(
            current_user.id, session, header_api_key=x_nemo_auth_token, header_base_url=x_nemo_base_url
        )
        return await nemo_service.stop_tracking_job(job_id)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to stop tracking job: {e!s}") from e


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
        raise HTTPException(status_code=500, detail=f"Failed to store job: {e!s}") from e


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
        raise HTTPException(status_code=500, detail=f"Failed to list jobs: {e!s}") from e


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

    Returns:
        Job details
    """
    try:
        nemo_service = await get_nemo_service(
            current_user.id, session, header_api_key=x_nemo_auth_token, header_base_url=x_nemo_base_url
        )
        job = await nemo_service.get_customizer_job_details(job_id)
        if not job:
            raise HTTPException(status_code=404, detail="Job not found")
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get job: {e!s}") from e
    else:
        return job

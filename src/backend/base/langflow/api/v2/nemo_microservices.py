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
- GET /api/v2/nemo/datasets - List all datasets
- POST /api/v2/nemo/datasets - Create new dataset
- GET /api/v2/nemo/datasets/{id} - Get dataset details
- DELETE /api/v2/nemo/datasets/{id} - Delete dataset
- POST /api/v2/nemo/datasets/{id}/files - Upload files
- GET /api/v2/nemo/datasets/{id}/files - List dataset files
- GET /api/v2/nemo/jobs - List customizer jobs
- GET /api/v2/nemo/jobs/{id} - Get job details
- POST /api/v2/nemo/jobs - Store job for tracking

Note: This implementation requires proper NeMo configuration via global variables or environment variables.
"""

from typing import Annotated

from fastapi import APIRouter, File, HTTPException, UploadFile

from langflow.api.utils import CurrentActiveUser, DbSession
from langflow.services.nemo_microservices_factory import get_nemo_service

router = APIRouter(prefix="/nemo", tags=["NeMo Microservices"])


# =============================================================================
# Dataset Management (Data Store)
# =============================================================================


@router.get("/datasets", response_model=list[dict])
async def list_datasets(
    current_user: CurrentActiveUser,
    session: DbSession,
):
    """List all NeMo datasets.

    Returns:
        List of dataset objects from NeMo Data Store
    """
    try:
        nemo_service = await get_nemo_service(current_user.id, session)
        return await nemo_service.list_datasets()
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
    name: str,
    description: str | None = None,
    dataset_type: str = "fileset",
):
    """Create a new NeMo dataset.

    Args:
        current_user: Current authenticated user
        session: Database session
        name: Dataset name
        description: Optional description
        dataset_type: Type of dataset (default: fileset)

    Returns:
        Created dataset object
    """
    try:
        nemo_service = await get_nemo_service(current_user.id, session)
        return await nemo_service.create_dataset(name=name, description=description, dataset_type=dataset_type)
    except ValueError as e:
        if "configuration is incomplete" in str(e):
            raise HTTPException(status_code=503, detail=f"NeMo service unavailable: {e!s}") from e
        raise HTTPException(status_code=500, detail=f"Failed to create dataset: {e!s}") from e
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to create dataset: {e!s}") from e


@router.get("/datasets/{dataset_id}", response_model=dict)
async def get_dataset(
    current_user: CurrentActiveUser,
    session: DbSession,
    dataset_id: str,
):
    """Get detailed information about a specific dataset.

    Args:
        current_user: Current authenticated user
        session: Database session
        dataset_id: NeMo Data Store dataset ID

    Returns:
        Dataset details including files and metadata
    """
    try:
        nemo_service = await get_nemo_service(current_user.id, session)
        dataset = await nemo_service.get_dataset(dataset_id)
        if not dataset:
            raise HTTPException(status_code=404, detail="Dataset not found")
    except HTTPException:
        raise
    except ValueError as e:
        if "configuration is incomplete" in str(e):
            raise HTTPException(status_code=503, detail=f"NeMo service unavailable: {e!s}") from e
        raise HTTPException(status_code=500, detail=f"Failed to get dataset: {e!s}") from e
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get dataset: {e!s}") from e
    else:
        return dataset


@router.delete("/datasets/{dataset_id}")
async def delete_dataset(
    current_user: CurrentActiveUser,
    session: DbSession,
    dataset_id: str,
):
    """Delete a dataset from NeMo Data Store.

    Args:
        current_user: Current authenticated user
        session: Database session
        dataset_id: NeMo Data Store dataset ID

    Returns:
        Success message
    """
    try:
        nemo_service = await get_nemo_service(current_user.id, session)
        deleted = await nemo_service.delete_dataset(dataset_id)
        if not deleted:
            raise HTTPException(status_code=404, detail="Dataset not found")
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to delete dataset: {e!s}") from e
    else:
        return {"message": "Dataset deleted successfully"}


@router.post("/datasets/{dataset_id}/files", response_model=dict)
async def upload_files(
    current_user: CurrentActiveUser,
    session: DbSession,
    dataset_id: str,
    files: Annotated[list[UploadFile], File(...)],
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
        nemo_service = await get_nemo_service(current_user.id, session)
        return await nemo_service.upload_files(dataset_id=dataset_id, files=files)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to upload files: {e!s}") from e


@router.get("/datasets/{dataset_id}/files", response_model=list[dict])
async def get_dataset_files(
    dataset_id: str,
    current_user: CurrentActiveUser,
    session: DbSession,
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
        nemo_service = await get_nemo_service(current_user.id, session)
        return await nemo_service.get_dataset_files(dataset_id)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get dataset files: {e!s}") from e


# =============================================================================
# Job Management (Customizer) - NeMo API Structure
# =============================================================================


@router.get("/v1/customization/configs", response_model=dict)
async def get_customization_configs(
    current_user: CurrentActiveUser,
    session: DbSession,
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
        nemo_service = await get_nemo_service(current_user.id, session)
        return await nemo_service.get_customization_configs()
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get customization configs: {e!s}") from e


@router.get("/v1/customization/jobs/{job_id}/status", response_model=dict)
async def get_job_status(
    job_id: str,
    current_user: CurrentActiveUser,
    session: DbSession,
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
        nemo_service = await get_nemo_service(current_user.id, session)
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
        nemo_service = await get_nemo_service(current_user.id, session)
        return await nemo_service.create_customization_job(job_data)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to create customization job: {e!s}") from e


@router.get("/v1/customization/jobs/{job_id}", response_model=dict)
async def get_job_details(
    job_id: str,
    current_user: CurrentActiveUser,
    session: DbSession,
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
        nemo_service = await get_nemo_service(current_user.id, session)
        job_details = await nemo_service.get_customizer_job_details(job_id)
        if not job_details:
            raise HTTPException(status_code=404, detail="Job not found")
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get job details: {e!s}") from e
    else:
        return job_details


@router.get("/v1/customization/jobs", response_model=list[dict])
async def list_all_customizer_jobs(
    current_user: CurrentActiveUser,
    session: DbSession,
):
    """List all customization jobs.

    This endpoint matches the NeMo Customizer API:
    GET /v1/customization/jobs

    Returns:
        List of all customization jobs
    """
    try:
        nemo_service = await get_nemo_service(current_user.id, session)
        return await nemo_service.list_customizer_jobs()
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
        nemo_service = await get_nemo_service(current_user.id, session)
        return await nemo_service.create_evaluation_job(job_data)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to create evaluation job: {e!s}") from e


@router.post("/v1/evaluation/configs", response_model=dict)
async def create_evaluation_config(
    config_data: dict,
    current_user: CurrentActiveUser,
    session: DbSession,
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
        nemo_service = await get_nemo_service(current_user.id, session)
        return await nemo_service.create_evaluation_config(config_data)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to create evaluation config: {e!s}") from e


@router.post("/v1/evaluation/targets", response_model=dict)
async def create_evaluation_target(
    target_data: dict,
    current_user: CurrentActiveUser,
    session: DbSession,
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
        nemo_service = await get_nemo_service(current_user.id, session)
        return await nemo_service.create_evaluation_target(target_data)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to create evaluation target: {e!s}") from e


@router.get("/v1/evaluation/jobs/{job_id}", response_model=dict)
async def get_evaluation_job(
    job_id: str,
    current_user: CurrentActiveUser,
    session: DbSession,
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
        nemo_service = await get_nemo_service(current_user.id, session)
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
):
    """List all evaluation jobs.

    This endpoint matches the NeMo Evaluator API:
    GET /v1/evaluation/jobs

    Returns:
        List of all evaluation jobs
    """
    try:
        nemo_service = await get_nemo_service(current_user.id, session)
        return await nemo_service.list_evaluation_jobs()
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
        nemo_service = await get_nemo_service(current_user.id, session)
        return await nemo_service.track_customizer_job(job_id, metadata)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to track job: {e!s}") from e


@router.get("/jobs/tracked", response_model=list[dict])
async def get_tracked_jobs(
    current_user: CurrentActiveUser,
    session: DbSession,
):
    """Get all jobs being tracked for dashboard monitoring.

    Returns:
        List of tracked job IDs with their current status
    """
    try:
        nemo_service = await get_nemo_service(current_user.id, session)
        return await nemo_service.get_tracked_jobs()
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get tracked jobs: {e!s}") from e


@router.delete("/jobs/track/{job_id}")
async def stop_tracking_job(
    job_id: str,
    current_user: CurrentActiveUser,
    session: DbSession,
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
        nemo_service = await get_nemo_service(current_user.id, session)
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
        nemo_service = await get_nemo_service(current_user.id, session)
        return await nemo_service.track_customizer_job(job_id, metadata)
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to store job: {e!s}") from e


@router.get("/jobs", response_model=list[dict])
async def list_customizer_jobs_legacy(
    current_user: CurrentActiveUser,
    session: DbSession,
):
    """Legacy endpoint: List all tracked customizer jobs.

    DEPRECATED: Use GET /jobs/tracked or GET /v1/customization/jobs

    Returns:
        List of tracked jobs
    """
    try:
        nemo_service = await get_nemo_service(current_user.id, session)
        return await nemo_service.get_tracked_jobs()
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to list jobs: {e!s}") from e


@router.get("/jobs/{job_id}", response_model=dict)
async def get_customizer_job_legacy(
    job_id: str,
    current_user: CurrentActiveUser,
    session: DbSession,
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
        nemo_service = await get_nemo_service(current_user.id, session)
        job = await nemo_service.get_customizer_job_details(job_id)
        if not job:
            raise HTTPException(status_code=404, detail="Job not found")
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get job: {e!s}") from e
    else:
        return job

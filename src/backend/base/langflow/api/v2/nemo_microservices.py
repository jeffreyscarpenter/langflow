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

Note: This implementation uses a mock service for development/testing.
In production, replace mock_nemo_service with actual NeMo Microservices clients.
"""

from typing import Annotated

from fastapi import APIRouter, File, HTTPException, UploadFile

from langflow.services.nemo_microservices_mock import mock_nemo_service

router = APIRouter(prefix="/nemo", tags=["NeMo Microservices"])


# =============================================================================
# Dataset Management (Data Store)
# =============================================================================


@router.get("/datasets", response_model=list[dict])
async def list_datasets():
    """List all NeMo datasets.

    Returns:
        List of dataset objects from NeMo Data Store
    """
    try:
        return await mock_nemo_service.list_datasets()
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to list datasets: {e!s}") from e


@router.post("/datasets", response_model=dict)
async def create_dataset(
    name: str,
    description: str | None = None,
    dataset_type: str = "fileset",
):
    """Create a new NeMo dataset.

    Args:
        name: Dataset name
        description: Optional description
        dataset_type: Type of dataset (default: fileset)

    Returns:
        Created dataset object
    """
    try:
        return await mock_nemo_service.create_dataset(name=name, description=description, dataset_type=dataset_type)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to create dataset: {e!s}") from e


@router.get("/datasets/{dataset_id}", response_model=dict)
async def get_dataset(dataset_id: str):
    """Get detailed information about a specific dataset.

    Args:
        dataset_id: NeMo Data Store dataset ID

    Returns:
        Dataset details including files and metadata
    """
    try:
        dataset = await mock_nemo_service.get_dataset(dataset_id)
        if not dataset:
            raise HTTPException(status_code=404, detail="Dataset not found")
        return dataset
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get dataset: {e!s}") from e


@router.delete("/datasets/{dataset_id}")
async def delete_dataset(dataset_id: str):
    """Delete a dataset from NeMo Data Store.

    Args:
        dataset_id: NeMo Data Store dataset ID

    Returns:
        Success message
    """
    try:
        deleted = await mock_nemo_service.delete_dataset(dataset_id)
        if not deleted:
            raise HTTPException(status_code=404, detail="Dataset not found")
        return {"message": "Dataset deleted successfully"}
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to delete dataset: {e!s}") from e


@router.post("/datasets/{dataset_id}/files", response_model=dict)
async def upload_files(
    dataset_id: str,
    files: Annotated[list[UploadFile], File(...)],
):
    """Upload files to a NeMo dataset.

    Args:
        dataset_id: NeMo Data Store dataset ID
        files: List of files to upload

    Returns:
        Upload result with file information
    """
    try:
        return await mock_nemo_service.upload_files(dataset_id=dataset_id, files=files)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to upload files: {e!s}") from e


@router.get("/datasets/{dataset_id}/files", response_model=list[dict])
async def get_dataset_files(dataset_id: str):
    """Get list of files in a NeMo dataset.

    Args:
        dataset_id: NeMo Data Store dataset ID

    Returns:
        List of files in the dataset
    """
    try:
        return await mock_nemo_service.get_dataset_files(dataset_id)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get dataset files: {e!s}") from e


# =============================================================================
# Job Management (Customizer)
# =============================================================================


@router.post("/jobs", response_model=dict)
async def store_job_for_tracking(job_data: dict):
    """Store job info from NeMo component for tracking.

    Args:
        job_data: Job information from NeMo Customizer component

    Returns:
        Stored job data
    """
    try:
        return await mock_nemo_service.store_customizer_job(job_data)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to store job: {e!s}") from e


@router.get("/jobs", response_model=list[dict])
async def list_customizer_jobs():
    """List all tracked customizer jobs.

    Returns:
        List of job objects with status and metadata
    """
    try:
        return await mock_nemo_service.get_customizer_jobs()
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to list jobs: {e!s}") from e


@router.get("/jobs/{job_id}", response_model=dict)
async def get_customizer_job(job_id: str):
    """Get detailed information about a specific customizer job.

    Args:
        job_id: NeMo Customizer job ID

    Returns:
        Job details including status, progress, and metrics
    """
    try:
        job = await mock_nemo_service.get_customizer_job(job_id)
        if not job:
            raise HTTPException(status_code=404, detail="Job not found")
        return job
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get job: {e!s}") from e

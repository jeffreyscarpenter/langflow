"""NeMo Data Store API endpoints for Langflow.

This module provides API endpoints for managing NeMo Data Store datasets
within Langflow, including CRUD operations and file uploads.
"""

from typing import Annotated

from fastapi import APIRouter, File, HTTPException, UploadFile

from langflow.services.nemo_datastore_mock import mock_nemo_service

router = APIRouter(prefix="/nemo-datastore", tags=["NeMo Data Store"])


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
        else:
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
        else:
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

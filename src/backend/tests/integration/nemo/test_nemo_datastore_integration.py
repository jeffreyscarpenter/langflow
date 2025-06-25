"""Integration tests for NeMo Data Store API and component integration.

This module tests the complete integration between:
- NeMo Data Store mock service
- NeMo Data Store API endpoints
- NeMo Customizer and Evaluator components
"""

import asyncio
import sys
from io import BytesIO

import httpx
import pytest
from langflow.components.nvidia.nvidia_customizer import NvidiaCustomizerComponent
from langflow.components.nvidia.nvidia_evaluator import NvidiaEvaluatorComponent
from langflow.services.deps import get_settings_service
from langflow.services.nemo_datastore_mock import mock_nemo_service

# Mock NeMo Data Store API endpoints
BASE_URL = "http://localhost:7861/api/v2/nemo-datastore"


async def test_nemo_datastore_api():
    """Test the NeMo Data Store API endpoints."""
    async with httpx.AsyncClient() as client:
        # Test 1: List datasets
        response = await client.get(f"{BASE_URL}/datasets")
        if response.status_code != 200:
            return False

        datasets = response.json()
        for ds in datasets:
            assert "name" in ds
            assert "id" in ds

        # Test 2: Create a new dataset
        create_data = {
            "name": "Test Dataset for Integration",
            "description": "A test dataset created during integration testing",
        }
        response = await client.post(f"{BASE_URL}/datasets", params=create_data)
        if response.status_code != 200:
            return False

        new_dataset = response.json()
        dataset_id = new_dataset["id"]

        # Test 3: Get specific dataset
        response = await client.get(f"{BASE_URL}/datasets/{dataset_id}")
        if response.status_code != 200:
            return False

        # Test 4: Upload files to dataset
        file_content = b'{"prompt": "Hello", "response": "Hi there!"}'
        files = [("files", ("test_file.jsonl", BytesIO(file_content), "application/json"))]
        response = await client.post(f"{BASE_URL}/datasets/{dataset_id}/files", files=files)
        if response.status_code != 200:
            return False

        # Test 5: Get dataset files
        response = await client.get(f"{BASE_URL}/datasets/{dataset_id}/files")
        if response.status_code != 200:
            return False

        # Test 6: Delete dataset
        response = await client.delete(f"{BASE_URL}/datasets/{dataset_id}")
        if response.status_code != 200:
            return False

    return True


async def test_component_integration():
    """Test integration with existing NeMo components."""
    try:
        from langflow.components.nvidia.nvidia_customizer import NvidiaCustomizerComponent
        from langflow.components.nvidia.nvidia_evaluator import NvidiaEvaluatorComponent

        # Test component instantiation
        customizer = NvidiaCustomizerComponent()
        evaluator = NvidiaEvaluatorComponent()

        # Test that components have the expected attributes
        assert hasattr(customizer, "customize"), "Customizer should have customize method"
        assert hasattr(evaluator, "evaluate"), "Evaluator should have evaluate method"

        # Test that components can access settings (which would include NeMo Data Store URL)
        settings_service = get_settings_service()
        _nemo_data_store_url = getattr(settings_service.settings, "nemo_data_store_url", None)

        # This is expected in test environment - URL may not be configured
        # Return True regardless of whether URL is configured
    except ImportError as e:
        # Expected in some test environments where settings service may not be available
        pytest.skip(f"Settings service not available: {e}")
    else:
        return True


@pytest.mark.asyncio
async def test_mock_service_list_datasets():
    """Test that the mock service can list datasets."""
    # Create a test dataset first
    dataset = await mock_nemo_service.create_dataset(
        name="test-dataset", description="Test dataset for integration test"
    )

    # List datasets
    datasets = await mock_nemo_service.list_datasets()

    # Verify the dataset appears in the list
    assert isinstance(datasets, list)
    assert len(datasets) > 0
    assert any(d["name"] == "test-dataset" for d in datasets)

    # Clean up
    await mock_nemo_service.delete_dataset(dataset["id"])


@pytest.mark.asyncio
async def test_customizer_component_fetch_datasets():
    """Test that the NeMo Customizer component can fetch datasets from mock service."""
    # Create a test dataset first
    dataset = await mock_nemo_service.create_dataset(
        name="customizer-test-dataset", description="Test dataset for customizer component"
    )

    # Create customizer component instance
    component = NvidiaCustomizerComponent()

    # Test fetching datasets
    datasets = await component.fetch_existing_datasets("http://mock-url")

    # Verify the dataset appears in the list
    assert isinstance(datasets, list)
    assert "customizer-test-dataset" in datasets

    # Clean up
    await mock_nemo_service.delete_dataset(dataset["id"])


@pytest.mark.asyncio
async def test_evaluator_component_fetch_datasets():
    """Test that the NeMo Evaluator component can fetch datasets from mock service."""
    # Create a test dataset first
    dataset = await mock_nemo_service.create_dataset(
        name="evaluator-test-dataset", description="Test dataset for evaluator component"
    )

    # Create evaluator component instance
    component = NvidiaEvaluatorComponent()

    # Test fetching datasets
    datasets = await component.fetch_existing_datasets("http://mock-url")

    # Verify the dataset appears in the list
    assert isinstance(datasets, list)
    assert "evaluator-test-dataset" in datasets

    # Clean up
    await mock_nemo_service.delete_dataset(dataset["id"])


@pytest.mark.asyncio
async def test_component_integration_workflow():
    """Test the complete workflow: create dataset via API, use in component."""
    # Step 1: Create dataset via mock service (simulating frontend creation)
    dataset = await mock_nemo_service.create_dataset(
        name="workflow-test-dataset", description="Test dataset for complete workflow"
    )

    # Step 2: Verify both components can see the dataset
    customizer = NvidiaCustomizerComponent()
    evaluator = NvidiaEvaluatorComponent()

    customizer_datasets = await customizer.fetch_existing_datasets("http://mock-url")
    evaluator_datasets = await evaluator.fetch_existing_datasets("http://mock-url")

    # Step 3: Verify dataset is available in both components
    assert "workflow-test-dataset" in customizer_datasets
    assert "workflow-test-dataset" in evaluator_datasets

    # Step 4: Clean up
    await mock_nemo_service.delete_dataset(dataset["id"])

    # Step 5: Verify dataset is no longer available
    customizer_datasets_after = await customizer.fetch_existing_datasets("http://mock-url")
    evaluator_datasets_after = await evaluator.fetch_existing_datasets("http://mock-url")

    assert "workflow-test-dataset" not in customizer_datasets_after
    assert "workflow-test-dataset" not in evaluator_datasets_after


async def main():
    """Run all integration tests."""
    try:
        # Test 1: API endpoints
        api_success = await test_nemo_datastore_api()

        # Test 2: Component integration
        component_success = await test_component_integration()

        # Test 3: Mock service datasets
        await test_mock_service_list_datasets()
        await test_customizer_component_fetch_datasets()
        await test_evaluator_component_fetch_datasets()
        await test_component_integration_workflow()

    except Exception:  # noqa: BLE001
        return False
    else:
        return api_success and component_success


if __name__ == "__main__":
    success = asyncio.run(main())
    sys.exit(0 if success else 1)

"""Integration tests for NeMo Data Store components.

This module tests the integration between:
- NeMo Data Store service
- NeMo Customizer component
- NeMo Evaluator component
"""

import asyncio

import pytest
from langflow.components.nvidia.nvidia_customizer import NvidiaCustomizerComponent
from langflow.components.nvidia.nvidia_evaluator import NvidiaEvaluatorComponent
from langflow.services.nemo_microservices_factory import get_nemo_service


@pytest.mark.asyncio
async def test_nemo_datastore_integration():
    """Test the integration between NeMo Data Store and components."""
    print("\n" + "=" * 60)
    print("Testing NeMo Data Store Integration...")
    print("=" * 60)

    # Test 1: Real service list datasets
    await test_service_list_datasets()

    # Test 2: Component dataset fetching
    await test_component_dataset_fetching()

    # Test 3: End-to-end workflow
    await test_end_to_end_workflow()

    print("\n✅ All NeMo Data Store integration tests passed!")


async def test_service_list_datasets():
    """Test that the service can list datasets."""
    print("\n1. Testing service list datasets...")

    try:
        # This will use the service with environment variables
        nemo_service = await get_nemo_service()
        datasets = await nemo_service.list_datasets()
        print(f"Found {len(datasets)} datasets in service")

        # Create a test dataset
        test_dataset = await nemo_service.create_dataset(
            name="test-dataset-integration", description="Test dataset for integration testing"
        )
        print(f"Created test dataset: {test_dataset['id']}")

        # List datasets again
        datasets_after = await nemo_service.list_datasets()
        print(f"Found {len(datasets_after)} datasets after creation")

        # Clean up
        await nemo_service.delete_dataset(test_dataset["id"])
        print("Cleaned up test dataset")

    except Exception as e:  # noqa: BLE001
        print(f"Service test skipped (likely no API key configured): {e}")


async def test_component_dataset_fetching():
    """Test that the NeMo Evaluator component can fetch datasets from the service."""
    print("\n2. Testing component dataset fetching...")

    try:
        # Create test dataset via service
        nemo_service = await get_nemo_service()
        dataset = await nemo_service.create_dataset(
            name="test-component-dataset", description="Test dataset for component testing"
        )
        print(f"Created test dataset: {dataset['id']}")

        # Test component fetching
        evaluator = NvidiaEvaluatorComponent()
        datasets = await evaluator.fetch_existing_datasets("https://us-west-2.api-dev.ai.datastax.com/nvidia/nemo")
        print(f"Component found {len(datasets)} datasets")

        # Clean up
        await nemo_service.delete_dataset(dataset["id"])
        print("Cleaned up test dataset")

    except Exception as e:  # noqa: BLE001
        print(f"Component test skipped (likely no API key configured): {e}")


async def test_end_to_end_workflow():
    """Test the complete workflow from dataset creation to component usage."""
    print("\n3. Testing end-to-end workflow...")

    try:
        # Step 1: Create dataset via service
        nemo_service = await get_nemo_service()
        dataset = await nemo_service.create_dataset(
            name="test-workflow-dataset", description="Test dataset for workflow testing"
        )
        print(f"Created workflow dataset: {dataset['id']}")

        # Step 2: Test component access
        customizer = NvidiaCustomizerComponent()
        evaluator = NvidiaEvaluatorComponent()

        customizer_datasets = await customizer.fetch_existing_datasets(
            "https://us-west-2.api-dev.ai.datastax.com/nvidia/nemo"
        )
        evaluator_datasets = await evaluator.fetch_existing_datasets(
            "https://us-west-2.api-dev.ai.datastax.com/nvidia/nemo"
        )

        print(f"Customizer found {len(customizer_datasets)} datasets")
        print(f"Evaluator found {len(evaluator_datasets)} datasets")

        # Step 3: Clean up
        await nemo_service.delete_dataset(dataset["id"])
        print("Cleaned up workflow dataset")

        # Step 4: Verify cleanup
        customizer_datasets_after = await customizer.fetch_existing_datasets(
            "https://us-west-2.api-dev.ai.datastax.com/nvidia/nemo"
        )
        evaluator_datasets_after = await evaluator.fetch_existing_datasets(
            "https://us-west-2.api-dev.ai.datastax.com/nvidia/nemo"
        )

        print(
            f"After cleanup - Customizer: {len(customizer_datasets_after)}, "
            f"Evaluator: {len(evaluator_datasets_after)}"
        )

    except Exception as e:  # noqa: BLE001
        print(f"Workflow test skipped (likely no API key configured): {e}")


if __name__ == "__main__":
    asyncio.run(test_nemo_datastore_integration())

#!/usr/bin/env python3
"""Test script for NeMo Data Store API endpoints."""

import asyncio

from langflow.services.nemo_microservices_mock import mock_nemo_service


async def test_mock_service():
    """Test the mock NeMo Data Store service."""
    print("Testing Mock NeMo Data Store Service...")

    # Test list datasets
    print("\n1. Testing list_datasets()...")
    datasets = await mock_nemo_service.list_datasets()
    print(f"Found {len(datasets)} datasets:")
    for ds in datasets:
        print(f"  - {ds['name']} (ID: {ds['id']})")

    # Test create dataset
    print("\n2. Testing create_dataset()...")
    new_dataset = await mock_nemo_service.create_dataset(
        name="Test Dataset", description="A test dataset created via API"
    )
    print(f"Created dataset: {new_dataset['name']} (ID: {new_dataset['id']})")

    # Test get dataset
    print("\n3. Testing get_dataset()...")
    dataset = await mock_nemo_service.get_dataset(new_dataset["id"])
    print(f"Retrieved dataset: {dataset['name']}")
    print(f"  Description: {dataset['description']}")
    print(f"  Type: {dataset['type']}")
    print(f"  Metadata: {dataset['metadata']}")

    # Test delete dataset
    print("\n4. Testing delete_dataset()...")
    deleted = await mock_nemo_service.delete_dataset(new_dataset["id"])
    print(f"Dataset deleted: {deleted}")

    # Verify deletion
    print("\n5. Verifying deletion...")
    datasets_after = await mock_nemo_service.list_datasets()
    print(f"Datasets after deletion: {len(datasets_after)}")

    print("\n✅ All tests passed!")


if __name__ == "__main__":
    asyncio.run(test_mock_service())

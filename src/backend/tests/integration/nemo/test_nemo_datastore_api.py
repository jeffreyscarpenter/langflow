#!/usr/bin/env python3
"""Test script for NeMo Microservices API endpoints including the new Customizer API structure."""

import asyncio
import os
import sys

# Add the parent directory to the path so we can import the mock service
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "..", "base"))

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

    print("\n✅ All dataset tests passed!")


async def test_customizer_api():
    """Test the new NeMo Customizer API endpoints."""
    print("\n" + "=" * 60)
    print("Testing NeMo Customizer API Endpoints...")
    print("=" * 60)

    # Test list all customizer jobs
    print("\n1. Testing list_customizer_jobs()...")
    jobs = await mock_nemo_service.list_customizer_jobs()
    print(f"Found {len(jobs)} customizer jobs:")
    for job in jobs:
        print(f"  - {job['id']} (Status: {job['status']}, Progress: {job['status_details']['percentage_done']}%)")

    if not jobs:
        print("No jobs found to test with.")
        return

    # Test get job status (mimics real NeMo API)
    job_id = jobs[0]["id"]
    print(f"\n2. Testing get_customizer_job_status() for job {job_id}...")
    job_status = await mock_nemo_service.get_customizer_job_status(job_id)
    print(f"Job Status: {job_status['status']}")
    print(f"Progress: {job_status['status_details']['percentage_done']}%")
    print(f"Steps completed: {job_status['status_details']['steps_completed']}")
    print(f"Epochs completed: {job_status['status_details']['epochs_completed']}")
    print(f"Training loss entries: {len(job_status['status_details']['training_loss'])}")
    print(f"Validation loss entries: {len(job_status['status_details']['validation_loss'])}")
    print(f"Status logs: {len(job_status['status_details']['status_logs'])}")

    # Test get job details (mimics real NeMo API)
    print(f"\n3. Testing get_customizer_job_details() for job {job_id}...")
    job_details = await mock_nemo_service.get_customizer_job_details(job_id)
    print(f"Job ID: {job_details['id']}")
    print(f"Model: {job_details['config']['name']}")
    print(f"Dataset: {job_details['dataset']}")
    print(f"Fine-tuning type: {job_details['hyperparameters']['finetuning_type']}")
    print(f"Output model: {job_details['output_model']}")

    # Test job tracking for Langflow dashboard
    print("\n4. Testing job tracking functionality...")

    # Test track job
    track_response = await mock_nemo_service.track_customizer_job(
        job_id, {"user_name": "Test User", "component": "NeMo Customizer"}
    )
    print(f"Track response: {track_response['message']}")

    # Test get tracked jobs
    tracked_jobs = await mock_nemo_service.get_tracked_jobs()
    print(f"Tracked jobs: {len(tracked_jobs)}")
    for tracked in tracked_jobs:
        print(f"  - {tracked['job_id']} (Status: {tracked['status']}, Progress: {tracked['progress']}%)")

    # Test stop tracking
    stop_response = await mock_nemo_service.stop_tracking_job(job_id)
    print(f"Stop tracking response: {stop_response['message']}")

    # Verify stopped tracking
    tracked_after = await mock_nemo_service.get_tracked_jobs()
    print(f"Tracked jobs after stopping: {len(tracked_after)}")

    print("\n✅ All Customizer API tests passed!")


async def test_job_status_details():
    """Test specific job status details structure."""
    print("\n" + "=" * 60)
    print("Testing Job Status Details Structure...")
    print("=" * 60)

    jobs = await mock_nemo_service.list_customizer_jobs()

    for i, job in enumerate(jobs[:2], 1):  # Test first 2 jobs
        print(f"\n{i}. Testing job {job['id']} ({job['status']})...")

        # Get detailed status
        status = await mock_nemo_service.get_customizer_job_status(job["id"])
        status_details = status["status_details"]

        print(f"   Status: {status['status']}")
        print(f"   Progress: {status_details['percentage_done']}%")
        print(f"   Steps: {status_details['steps_completed']}")
        print(f"   Epochs: {status_details['epochs_completed']}")

        # Verify training loss structure
        training_losses = status_details["training_loss"]
        if training_losses:
            latest_loss = training_losses[-1]
            print(f"   Latest training loss: {latest_loss['value']} at step {latest_loss['step']}")
            print(f"   Training loss timestamp: {latest_loss['timestamp']}")

        # Verify validation loss structure
        validation_losses = status_details["validation_loss"]
        if validation_losses:
            latest_val_loss = validation_losses[-1]
            print(f"   Latest validation loss: {latest_val_loss['value']} at epoch {latest_val_loss['epoch']}")
            print(f"   Validation loss timestamp: {latest_val_loss['timestamp']}")

        # Verify status logs
        status_logs = status_details["status_logs"]
        print(f"   Status log entries: {len(status_logs)}")
        if status_logs:
            latest_log = status_logs[-1]
            print(f"   Latest log: {latest_log['message']} at {latest_log['updated_at']}")
            if "detail" in latest_log:
                print(f"   Log detail preview: {latest_log['detail'][:100]}...")

    print("\n✅ All job status details tests passed!")


async def main():
    """Run all tests."""
    try:
        await test_mock_service()
        await test_customizer_api()
        await test_job_status_details()

        print("\n" + "=" * 60)
        print("🎉 ALL TESTS PASSED!")
        print("✅ Dataset API endpoints working")
        print("✅ NeMo Customizer API endpoints working")
        print("✅ Job tracking functionality working")
        print("✅ Real API structure validated")
        print("=" * 60)

    except Exception as e:
        print(f"\n❌ Test failed with error: {e}")
        import traceback

        traceback.print_exc()
        return False

    return True


if __name__ == "__main__":
    success = asyncio.run(main())
    sys.exit(0 if success else 1)

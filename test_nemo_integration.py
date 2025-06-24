#!/usr/bin/env python3
"""
Test script to verify NeMo Data Store integration with existing NeMo components.
"""

import asyncio
import json
import httpx
from typing import List, Dict, Any
from io import BytesIO

# Mock NeMo Data Store API endpoints
BASE_URL = "http://localhost:7861/api/v2/nemo-datastore"

async def test_nemo_datastore_api():
    """Test the NeMo Data Store API endpoints."""
    print("🧪 Testing NeMo Data Store API Integration")
    print("=" * 50)
    
    async with httpx.AsyncClient() as client:
        # Test 1: List datasets
        print("\n1. Testing GET /datasets")
        response = await client.get(f"{BASE_URL}/datasets")
        if response.status_code == 200:
            datasets = response.json()
            print(f"✅ Successfully retrieved {len(datasets)} datasets")
            for ds in datasets:
                print(f"   - {ds['name']} (ID: {ds['id']})")
        else:
            print(f"❌ Failed to list datasets: {response.status_code}")
            return False
        
        # Test 2: Create a new dataset
        print("\n2. Testing POST /datasets")
        create_data = {
            "name": "Test Dataset for Integration",
            "description": "Dataset created during integration testing",
            "dataset_type": "fileset"
        }
        response = await client.post(f"{BASE_URL}/datasets", params=create_data)
        if response.status_code == 200:
            new_dataset = response.json()
            print(f"✅ Successfully created dataset: {new_dataset['name']} (ID: {new_dataset['id']})")
            dataset_id = new_dataset['id']
        else:
            print(f"❌ Failed to create dataset: {response.status_code}")
            return False
        
        # Test 3: Get specific dataset
        print(f"\n3. Testing GET /datasets/{dataset_id}")
        response = await client.get(f"{BASE_URL}/datasets/{dataset_id}")
        if response.status_code == 200:
            dataset = response.json()
            print(f"✅ Successfully retrieved dataset: {dataset['name']}")
        else:
            print(f"❌ Failed to get dataset: {response.status_code}")
            return False
        
        # Test 4: Upload files to dataset
        print(f"\n4. Testing POST /datasets/{dataset_id}/files")
        # Create a mock file using BytesIO
        file_content = b'{"prompt": "Hello", "response": "Hi there!"}'
        files = {
            'files': ('test.json', BytesIO(file_content), 'application/json')
        }
        response = await client.post(f"{BASE_URL}/datasets/{dataset_id}/files", files=files)
        if response.status_code == 200:
            upload_result = response.json()
            print(f"✅ Successfully uploaded files: {upload_result['message']}")
        else:
            print(f"❌ Failed to upload files: {response.status_code}")
            return False
        
        # Test 5: Get dataset files
        print(f"\n5. Testing GET /datasets/{dataset_id}/files")
        response = await client.get(f"{BASE_URL}/datasets/{dataset_id}/files")
        if response.status_code == 200:
            files = response.json()
            print(f"✅ Successfully retrieved {len(files)} files")
        else:
            print(f"❌ Failed to get files: {response.status_code}")
            return False
        
        # Test 6: Delete dataset
        print(f"\n6. Testing DELETE /datasets/{dataset_id}")
        response = await client.delete(f"{BASE_URL}/datasets/{dataset_id}")
        if response.status_code == 200:
            print("✅ Successfully deleted dataset")
        else:
            print(f"❌ Failed to delete dataset: {response.status_code}")
            return False
    
    print("\n🎉 All NeMo Data Store API tests passed!")
    return True

async def test_component_integration():
    """Test integration with existing NeMo components."""
    print("\n🔧 Testing Component Integration")
    print("=" * 50)
    
    # Test that we can import and use the existing NeMo components
    try:
        from langflow.components.nvidia.nvidia_customizer import NvidiaCustomizerComponent
        from langflow.components.nvidia.nvidia_evaluator import NvidiaEvaluatorComponent
        
        print("✅ Successfully imported NeMo components")
        
        # Test component instantiation
        customizer = NvidiaCustomizerComponent()
        evaluator = NvidiaEvaluatorComponent()
        
        print("✅ Successfully instantiated NeMo components")
        
        # Test that components have the expected attributes
        assert hasattr(customizer, 'customize'), "Customizer should have customize method"
        assert hasattr(evaluator, 'evaluate'), "Evaluator should have evaluate method"
        
        print("✅ Components have expected methods")
        
        # Test that components can access settings (which would include NeMo Data Store URL)
        from langflow.services.deps import get_settings_service
        settings_service = get_settings_service()
        
        # Check if NeMo settings are available
        nemo_data_store_url = getattr(settings_service.settings, 'nemo_data_store_url', None)
        if nemo_data_store_url:
            print(f"✅ NeMo Data Store URL configured: {nemo_data_store_url}")
        else:
            print("⚠️  NeMo Data Store URL not configured (this is expected in test environment)")
        
        return True
        
    except ImportError as e:
        print(f"❌ Failed to import NeMo components: {e}")
        return False
    except Exception as e:
        print(f"❌ Error testing component integration: {e}")
        return False

async def main():
    """Run all integration tests."""
    print("🚀 Starting NeMo Data Store Integration Tests")
    print("=" * 60)
    
    # Test 1: API endpoints
    api_success = await test_nemo_datastore_api()
    
    # Test 2: Component integration
    component_success = await test_component_integration()
    
    # Summary
    print("\n" + "=" * 60)
    print("📊 Test Results Summary")
    print("=" * 60)
    print(f"API Integration: {'✅ PASSED' if api_success else '❌ FAILED'}")
    print(f"Component Integration: {'✅ PASSED' if component_success else '❌ FAILED'}")
    
    if api_success and component_success:
        print("\n🎉 All tests passed! NeMo Data Store integration is working correctly.")
        return True
    else:
        print("\n❌ Some tests failed. Please check the output above.")
        return False

if __name__ == "__main__":
    success = asyncio.run(main())
    exit(0 if success else 1) 
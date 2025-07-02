#!/usr/bin/env python3
"""Test script for NeMo Evaluator API endpoints."""

import asyncio

from langflow.services.nemo_microservices_mock import mock_nemo_service


async def test_evaluator_endpoints():
    """Test the evaluator API endpoints."""
    print("🧪 Testing NeMo Evaluator API Endpoints")
    print("=" * 50)

    # Test 1: Create evaluation target
    print("\n1. Testing evaluation target creation...")
    target_data = {
        "type": "model",
        "namespace": "default",
        "model": {"api_endpoint": {"url": "http://localhost:8000/v1/completions", "model_id": "test-model"}},
    }

    target_result = await mock_nemo_service.create_evaluation_target(target_data)
    print(f"✅ Created evaluation target: {target_result['id']}")

    # Test 2: Create evaluation config
    print("\n2. Testing evaluation config creation...")
    config_data = {
        "type": "lm_eval_harness",
        "namespace": "default",
        "tasks": [
            {
                "type": "gsm8k",
                "params": {
                    "num_fewshot": 5,
                    "batch_size": 16,
                    "bootstrap_iters": 100000,
                    "limit": -1,
                },
            }
        ],
        "params": {
            "hf_token": "test-token",
            "use_greedy": True,
            "top_p": 0.0,
            "top_k": 1,
            "temperature": 0.0,
            "stop": [],
            "tokens_to_generate": 1024,
        },
    }

    config_result = await mock_nemo_service.create_evaluation_config(config_data)
    print(f"✅ Created evaluation config: {config_result['id']}")

    # Test 3: Create evaluation job
    print("\n3. Testing evaluation job creation...")
    job_data = {
        "namespace": "default",
        "target": f"default/{target_result['id']}",
        "config": f"default/{config_result['id']}",
        "tags": ["test-evaluation"],
    }

    job_result = await mock_nemo_service.create_evaluation_job(job_data)
    print(f"✅ Created evaluation job: {job_result['id']}")

    # Test 4: Get evaluation job
    print("\n4. Testing evaluation job retrieval...")
    retrieved_job = await mock_nemo_service.get_evaluation_job(job_result["id"])
    print(f"✅ Retrieved evaluation job: {retrieved_job['id']}")

    # Test 5: List evaluation jobs
    print("\n5. Testing evaluation job listing...")
    jobs_list = await mock_nemo_service.list_evaluation_jobs()
    print(f"✅ Listed {len(jobs_list)} evaluation jobs")

    # Test 6: Test similarity metrics config
    print("\n6. Testing similarity metrics config creation...")
    similarity_config_data = {
        "type": "similarity_metrics",
        "namespace": "default",
        "tasks": [
            {
                "type": "default",
                "metrics": [{"name": "accuracy"}, {"name": "bleu"}],
                "dataset": {"files_url": "nds:default/test-dataset/input.json"},
                "params": {
                    "tokens_to_generate": 1024,
                    "temperature": 0.0,
                    "top_k": 1,
                    "n_samples": -1,
                },
            }
        ],
    }

    similarity_config_result = await mock_nemo_service.create_evaluation_config(similarity_config_data)
    print(f"✅ Created similarity metrics config: {similarity_config_result['id']}")

    print("\n🎉 All evaluator tests passed!")


if __name__ == "__main__":
    asyncio.run(test_evaluator_endpoints())

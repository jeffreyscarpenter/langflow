from dataclasses import asdict

import pytest
from langflow.components.nvidia.nvidia_evaluator import (
    CustomEvaluationConfigInput,
    EvaluationTypeSelectionInput,
    LMEvalHarnessConfigInput,
    NvidiaEvaluatorComponent,
    SimilarityMetricsConfigInput,
)


class TestNvidiaEvaluatorDialogs:
    """Test the dynamic dialog switching functionality."""

    @pytest.fixture
    def component(self):
        """Create a component instance for testing."""
        return NvidiaEvaluatorComponent()

    @pytest.fixture
    def mock_build_config(self):
        """Create a mock build config for testing."""
        return {"config": {"dialog_inputs": {"fields": {"data": {"node": {"template": {}}}}}}}

    def test_lm_eval_harness_dialog_structure(self):
        """Test that LM Evaluation Harness dialog has correct structure."""
        dialog = asdict(LMEvalHarnessConfigInput())

        # Check basic structure
        assert "fields" in dialog
        assert "data" in dialog["fields"]
        assert "node" in dialog["fields"]["data"]

        node = dialog["fields"]["data"]["node"]
        assert node["name"] == "create_lm_eval_config"
        assert node["display_name"] == "Create LM Evaluation Config"

        # Check field order
        expected_fields = [
            "01_config_name",
            "02_task_name",
            "03_hf_token",
            "04_few_shot_examples",
            "05_batch_size",
            "06_bootstrap_iterations",
            "07_limit",
            "08_top_p",
            "09_top_k",
            "10_temperature",
            "11_tokens_to_generate",
        ]
        assert node["field_order"] == expected_fields

        # Check template has all expected fields
        template = node["template"]
        for field in expected_fields:
            assert field in template

    def test_similarity_metrics_dialog_structure(self):
        """Test that Similarity Metrics dialog has correct structure."""
        dialog = asdict(SimilarityMetricsConfigInput())

        # Check basic structure
        assert "fields" in dialog
        assert "data" in dialog["fields"]
        assert "node" in dialog["fields"]["data"]

        node = dialog["fields"]["data"]["node"]
        assert node["name"] == "create_similarity_config"
        assert node["display_name"] == "Create Similarity Metrics Config"

        # Check field order
        expected_fields = [
            "01_config_name",
            "02_scorers",
            "03_num_samples",
        ]
        assert node["field_order"] == expected_fields

        # Check template has all expected fields
        template = node["template"]
        for field in expected_fields:
            assert field in template

    def test_custom_evaluation_dialog_structure(self):
        """Test that Custom Evaluation dialog has correct structure."""
        dialog = asdict(CustomEvaluationConfigInput())

        # Check basic structure
        assert "fields" in dialog
        assert "data" in dialog["fields"]
        assert "node" in dialog["fields"]["data"]

        node = dialog["fields"]["data"]["node"]
        assert node["name"] == "create_custom_config"
        assert node["display_name"] == "Create Custom Evaluation Config"

        # Check field order
        expected_fields = [
            "01_config_name",
            "02_evaluation_prompt",
            "03_metrics",
            "04_batch_size",
        ]
        assert node["field_order"] == expected_fields

        # Check template has all expected fields
        template = node["template"]
        for field in expected_fields:
            assert field in template

    def test_evaluation_type_selection_dialog_structure(self):
        """Test that Evaluation Type Selection dialog has correct structure."""
        dialog = asdict(EvaluationTypeSelectionInput())

        # Check basic structure
        assert "fields" in dialog
        assert "data" in dialog["fields"]
        assert "node" in dialog["fields"]["data"]

        node = dialog["fields"]["data"]["node"]
        assert node["name"] == "select_evaluation_type"
        assert node["display_name"] == "Select Evaluation Type"

        # Check field order
        expected_fields = ["01_evaluation_type"]
        assert node["field_order"] == expected_fields

        # Check template has evaluation type field
        template = node["template"]
        assert "01_evaluation_type" in template

        # Check evaluation type options
        eval_type_field = template["01_evaluation_type"]
        expected_options = ["LM Evaluation Harness", "Similarity Metrics", "Custom Evaluation"]
        # Convert DropdownInput to dict to access options
        field_dict = eval_type_field.model_dump() if hasattr(eval_type_field, "model_dump") else eval_type_field.dict()
        assert field_dict["options"] == expected_options

    @pytest.mark.asyncio
    async def test_switch_to_lm_eval_dialog(self, component, mock_build_config):
        """Test switching to LM Evaluation Harness dialog."""
        field_value = {"01_evaluation_type": "LM Evaluation Harness"}

        result = component._switch_evaluation_dialog(mock_build_config, field_value)

        # Check that dialog was switched
        dialog_inputs = result["config"]["dialog_inputs"]
        node = dialog_inputs["fields"]["data"]["node"]
        assert node["name"] == "create_lm_eval_config"
        assert node["display_name"] == "Create LM Evaluation Config"

        # Check that evaluation type was stored
        assert component._selected_evaluation_type == "LM Evaluation Harness"

    @pytest.mark.asyncio
    async def test_switch_to_similarity_metrics_dialog(self, component, mock_build_config):
        """Test switching to Similarity Metrics dialog."""
        field_value = {"01_evaluation_type": "Similarity Metrics"}

        result = component._switch_evaluation_dialog(mock_build_config, field_value)

        # Check that dialog was switched
        dialog_inputs = result["config"]["dialog_inputs"]
        node = dialog_inputs["fields"]["data"]["node"]
        assert node["name"] == "create_similarity_config"
        assert node["display_name"] == "Create Similarity Metrics Config"

        # Check that evaluation type was stored
        assert component._selected_evaluation_type == "Similarity Metrics"

    @pytest.mark.asyncio
    async def test_switch_to_custom_evaluation_dialog(self, component, mock_build_config):
        """Test switching to Custom Evaluation dialog."""
        field_value = {"01_evaluation_type": "Custom Evaluation"}

        result = component._switch_evaluation_dialog(mock_build_config, field_value)

        # Check that dialog was switched
        dialog_inputs = result["config"]["dialog_inputs"]
        node = dialog_inputs["fields"]["data"]["node"]
        assert node["name"] == "create_custom_config"
        assert node["display_name"] == "Create Custom Evaluation Config"

        # Check that evaluation type was stored
        assert component._selected_evaluation_type == "Custom Evaluation"

    @pytest.mark.asyncio
    async def test_switch_to_unknown_evaluation_type(self, component, mock_build_config):
        """Test switching to unknown evaluation type defaults to selection dialog."""
        field_value = {"01_evaluation_type": "Unknown Type"}

        result = component._switch_evaluation_dialog(mock_build_config, field_value)

        # Check that dialog defaults to selection dialog
        dialog_inputs = result["config"]["dialog_inputs"]
        node = dialog_inputs["fields"]["data"]["node"]
        assert node["name"] == "select_evaluation_type"
        assert node["display_name"] == "Select Evaluation Type"

    def test_dialog_field_counts(self):
        """Test that each dialog has the appropriate number of fields."""
        # LM Evaluation Harness should have 11 fields (config name + 10 parameters)
        lm_dialog = asdict(LMEvalHarnessConfigInput())
        lm_template = lm_dialog["fields"]["data"]["node"]["template"]
        assert len(lm_template) == 11

        # Similarity Metrics should have 3 fields (config name + 2 parameters)
        sm_dialog = asdict(SimilarityMetricsConfigInput())
        sm_template = sm_dialog["fields"]["data"]["node"]["template"]
        assert len(sm_template) == 3

        # Custom Evaluation should have 4 fields (config name + 3 parameters)
        custom_dialog = asdict(CustomEvaluationConfigInput())
        custom_template = custom_dialog["fields"]["data"]["node"]["template"]
        assert len(custom_template) == 4

        # Evaluation Type Selection should have 1 field
        selection_dialog = asdict(EvaluationTypeSelectionInput())
        selection_template = selection_dialog["fields"]["data"]["node"]["template"]
        assert len(selection_template) == 1

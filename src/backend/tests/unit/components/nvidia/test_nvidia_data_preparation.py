"""Unit tests for the NeMo Data Preparation component."""

from unittest.mock import Mock

import pandas as pd
import pytest
from langflow.components.nvidia.nvidia_data_preparation import NeMoDataPreparationComponent
from langflow.schema import Data


class TestNeMoDataPreparationComponent:
    """Test the NeMo Data Preparation component."""

    @pytest.fixture
    def component(self):
        """Create a test component instance."""
        component = NeMoDataPreparationComponent()
        component.log = Mock()  # Mock logging
        return component

    @pytest.fixture
    def sample_completion_data(self):
        """Sample completion model data."""
        return [
            {"prompt": "What is 2+2?", "completion": "4"},
            {"prompt": "What is the capital of France?", "completion": "Paris"},
            {"prompt": "What color is the sky?", "completion": "Blue"},
        ]

    @pytest.fixture
    def sample_chat_data(self):
        """Sample chat model data."""
        return [
            {"messages": [{"role": "user", "content": "Hello"}, {"role": "assistant", "content": "Hi there!"}]},
            {
                "messages": [
                    {"role": "user", "content": "How are you?"},
                    {"role": "assistant", "content": "I'm doing well!"},
                ]
            },
        ]

    @pytest.fixture
    def sample_nested_data(self):
        """Sample data with nested fields."""
        return [
            {"request": {"prompt": "What is 2+2?"}, "response": {"text": "4", "expected_answer": "4"}},
            {"request": {"prompt": "What is 3+3?"}, "response": {"text": "6", "expected_answer": "6"}},
        ]

    def test_component_initialization(self, component):
        """Test component initialization."""
        assert component.display_name == "NeMo Data Preparation"
        assert component.name == "NVIDIANeMoDataPreparation"
        assert component.beta is True
        assert component.processed_data is None
        assert component.preparation_stats == {}

    def test_convert_to_dataframe_from_dataframe(self, component):
        """Test converting DataFrame input."""
        df = pd.DataFrame({"col1": [1, 2, 3], "col2": ["a", "b", "c"]})
        result = component.convert_to_dataframe(df)
        assert isinstance(result, pd.DataFrame)
        assert len(result) == 3
        assert list(result.columns) == ["col1", "col2"]

    def test_convert_to_dataframe_from_list_data(self, component):
        """Test converting list[Data] input."""
        data_list = [
            Data(data={"col1": 1, "col2": "a"}),
            Data(data={"col1": 2, "col2": "b"}),
        ]
        result = component.convert_to_dataframe(data_list)
        assert isinstance(result, pd.DataFrame)
        assert len(result) == 2
        assert list(result.columns) == ["col1", "col2"]

    def test_convert_to_dataframe_from_list_data_with_nested(self, component):
        """Test converting list[Data] with nested data."""
        data_list = [
            Data(data={"nested": {"value": 1}}),
            Data(data={"nested": {"value": 2}}),
        ]
        result = component.convert_to_dataframe(data_list)
        assert isinstance(result, pd.DataFrame)
        assert len(result) == 2
        assert "nested" in result.columns

    def test_convert_to_dataframe_invalid_input(self, component):
        """Test converting invalid input raises error."""
        with pytest.raises(ValueError, match="Input data must be DataFrame or list\\[Data\\]"):
            component.convert_to_dataframe("invalid")

    def test_detect_model_type_completion(self, component, sample_completion_data):
        """Test model type detection for completion data."""
        result = component.detect_model_type(sample_completion_data)
        assert result == "completion"

    def test_detect_model_type_chat(self, component, sample_chat_data):
        """Test model type detection for chat data."""
        result = component.detect_model_type(sample_chat_data)
        assert result == "chat"

    def test_detect_model_type_mixed(self, component):
        """Test model type detection with mixed data."""
        mixed_data = [
            {"prompt": "test", "completion": "test"},  # completion indicator
            {"messages": [{"role": "user", "content": "test"}]},  # chat indicator
        ]
        result = component.detect_model_type(mixed_data)
        # Should default to completion when tied
        assert result == "completion"

    def test_determine_model_type_auto(self, component, sample_completion_data):
        """Test model type determination with auto setting."""
        component.model_type = "auto"
        result = component.determine_model_type(sample_completion_data)
        assert result == "completion"

    def test_determine_model_type_manual(self, component, sample_completion_data):
        """Test model type determination with manual setting."""
        component.model_type = "chat"
        result = component.determine_model_type(sample_completion_data)
        assert result == "chat"

    def test_extract_field_value_simple(self, component):
        """Test simple field extraction."""
        record = {"field1": "value1", "field2": "value2"}
        result = component.extract_field_value(record, "field1")
        assert result == "value1"

    def test_extract_field_value_nested(self, component):
        """Test nested field extraction."""
        record = {"level1": {"level2": {"level3": "value"}}}
        result = component.extract_field_value(record, "level1.level2.level3")
        assert result == "value"

    def test_extract_field_value_missing(self, component):
        """Test field extraction with missing field."""
        record = {"field1": "value1"}
        result = component.extract_field_value(record, "missing")
        assert result is None

    def test_extract_field_value_nested_missing(self, component):
        """Test nested field extraction with missing field."""
        record = {"level1": {"level2": "value"}}
        result = component.extract_field_value(record, "level1.level2.level3")
        assert result is None

    def test_auto_detect_field_mappings(self, component, sample_nested_data):
        """Test automatic field mapping detection."""
        mappings = component.auto_detect_field_mappings(sample_nested_data)
        assert "prompt" in mappings
        assert "completion" in mappings
        assert "ideal_response" in mappings
        assert mappings["prompt"] == "request.prompt"
        assert mappings["completion"] == "response.text"
        assert mappings["ideal_response"] == "response.expected_answer"

    def test_auto_detect_field_mappings_no_matches(self, component):
        """Test automatic field mapping with no matches."""
        data = [{"unknown_field": "value"}]
        mappings = component.auto_detect_field_mappings(data)
        assert mappings == {}

    def test_determine_field_mappings_custom(self, component):
        """Test field mapping with custom mappings."""
        component.custom_field_mappings = {"prompt": "input", "completion": "output"}
        data = [{"input": "test", "output": "result"}]
        mappings = component.determine_field_mappings(data)
        assert mappings == {"prompt": "input", "completion": "output"}

    def test_determine_field_mappings_auto(self, component, sample_nested_data):
        """Test field mapping with auto detection."""
        component.custom_field_mappings = None
        mappings = component.determine_field_mappings(sample_nested_data)
        assert "prompt" in mappings
        assert "completion" in mappings

    def test_transform_to_completion_format_training(self, component):
        """Test completion format transformation for training."""
        component.custom_field_mappings = {"prompt": "input", "completion": "output"}
        record = {"input": "What is 2+2?", "output": "4"}
        result = component.transform_to_completion_format(record, is_evaluation=False)
        assert result == {"prompt": "What is 2+2?", "completion": "4"}

    def test_transform_to_completion_format_evaluation(self, component):
        """Test completion format transformation for evaluation."""
        component.custom_field_mappings = {"prompt": "input", "ideal_response": "expected"}
        record = {"input": "What is 2+2?", "expected": "4"}
        result = component.transform_to_completion_format(record, is_evaluation=True)
        assert result == {"prompt": "What is 2+2?", "ideal_response": "4"}

    def test_transform_to_chat_format(self, component):
        """Test chat format transformation."""
        component.custom_field_mappings = {"prompt": "user_input", "completion": "assistant_response"}
        record = {"user_input": "Hello", "assistant_response": "Hi there!"}
        result = component.transform_to_chat_format(record)
        assert "messages" in result
        assert len(result["messages"]) == 2
        assert result["messages"][0]["role"] == "user"
        assert result["messages"][1]["role"] == "assistant"

    def test_transform_record_completion(self, component):
        """Test record transformation for completion model."""
        component.model_type = "completion"
        component.custom_field_mappings = {"prompt": "input", "completion": "output"}
        record = {"input": "test", "output": "result"}
        result = component.transform_record(record)
        assert "prompt" in result
        assert "completion" in result

    def test_transform_record_chat(self, component):
        """Test record transformation for chat model."""
        component.model_type = "chat"
        component.custom_field_mappings = {"prompt": "input", "completion": "output"}
        record = {"input": "test", "output": "result"}
        result = component.transform_record(record)
        assert "messages" in result

    def test_process_data_basic(self, component, sample_completion_data):
        """Test basic data processing."""
        component.model_type = "completion"
        component.evaluation_records_count = 0
        component.strict_field_extraction = False

        result = component.process_data(sample_completion_data)
        assert len(result) == 3
        assert all("prompt" in record for record in result)
        assert all("completion" in record for record in result)

    def test_process_data_with_evaluation_split(self, component, sample_completion_data):
        """Test data processing with evaluation split."""
        component.model_type = "completion"
        component.evaluation_records_count = 1
        component.strict_field_extraction = False

        result = component.process_data(sample_completion_data)
        assert len(result) == 3

        # Check that one record is marked as evaluation
        evaluation_records = [r for r in result if r.get("_is_evaluation")]
        assert len(evaluation_records) == 1

    def test_process_data_strict_field_extraction(self, component):
        """Test data processing with strict field extraction."""
        component.model_type = "completion"
        component.evaluation_records_count = 0
        component.strict_field_extraction = True

        # Data with missing required fields
        data = [{"prompt": "test"}, {"completion": "result"}]  # Missing fields

        result = component.process_data(data)
        # Should discard records with missing fields
        assert len(result) < len(data)

    def test_process_data_error_handling(self, component):
        """Test data processing error handling."""
        component.model_type = "completion"
        component.evaluation_records_count = 0
        component.strict_field_extraction = False

        # Data that will cause errors
        data = [{"prompt": "test"}, None, {"completion": "result"}]

        result = component.process_data(data)
        # Should handle errors gracefully
        assert len(result) < len(data)

    def test_prepare_data(self, component, sample_completion_data):
        """Test prepare_data method."""
        component.input_data = sample_completion_data
        component.model_type = "completion"
        component.evaluation_records_count = 0

        result = component.prepare_data()
        assert isinstance(result, list)
        assert all(isinstance(item, Data) for item in result)
        assert len(result) == 3

    def test_prepare_dataframe(self, component, sample_completion_data):
        """Test prepare_dataframe method."""
        component.input_data = sample_completion_data
        component.model_type = "completion"
        component.evaluation_records_count = 0

        result = component.prepare_dataframe()
        assert isinstance(result, pd.DataFrame)
        assert len(result) == 3
        assert "prompt" in result.columns
        assert "completion" in result.columns

    def test_prepare_data_empty_input(self, component):
        """Test prepare_data with empty input."""
        component.input_data = []
        result = component.prepare_data()
        assert result == []

    def test_prepare_dataframe_empty_input(self, component):
        """Test prepare_dataframe with empty input."""
        component.input_data = []
        result = component.prepare_dataframe()
        assert result.empty

    def test_preparation_statistics(self, component, sample_completion_data):
        """Test that preparation statistics are properly recorded."""
        component.input_data = sample_completion_data
        component.model_type = "completion"
        component.evaluation_records_count = 1

        component.prepare_data()

        stats = component.preparation_stats
        assert "total_input_records" in stats
        assert "successfully_prepared" in stats
        assert "discarded_records" in stats
        assert "evaluation_records" in stats
        assert "model_type" in stats
        assert "field_mappings" in stats
        assert stats["total_input_records"] == 3
        assert stats["evaluation_records"] == 1

    def test_logging(self, component, sample_completion_data):
        """Test that logging is called with appropriate messages."""
        component.input_data = sample_completion_data
        component.model_type = "completion"
        component.evaluation_records_count = 0

        component.prepare_data()

        # Check that log was called
        assert component.log.called
        log_calls = [call.args[0] for call in component.log.call_args_list]

        # Check for expected log messages
        assert any("Data preparation completed" in call for call in log_calls)
        assert any("Model type:" in call for call in log_calls)
        assert any("Field mappings:" in call for call in log_calls)

    def test_evaluation_records_format(self, component, sample_nested_data):
        """Test that evaluation records use the correct format."""
        component.input_data = sample_nested_data
        component.model_type = "completion"
        component.custom_field_mappings = {
            "prompt": "request.prompt",
            "completion": "response.text",
            "ideal_response": "response.expected_answer",
        }
        component.evaluation_records_count = 1

        result = component.prepare_data()

        # Find evaluation record
        evaluation_record = None
        for record in result:
            if record.data.get("_is_evaluation"):
                evaluation_record = record.data
                break

        assert evaluation_record is not None
        assert "prompt" in evaluation_record
        assert "ideal_response" in evaluation_record  # Should use ideal_response, not completion
        assert "completion" not in evaluation_record  # Should not have completion field

    def test_field_mapping_edge_cases(self, component):
        """Test field mapping with edge cases."""
        # Test with empty data
        mappings = component.auto_detect_field_mappings([])
        assert mappings == {}

        # Test with data containing None values
        data = [{"prompt": None, "completion": "test"}]
        mappings = component.auto_detect_field_mappings(data)
        assert "prompt" in mappings

        # Test with deeply nested data
        data = [{"level1": {"level2": {"level3": {"prompt": "test"}}}}]
        mappings = component.auto_detect_field_mappings(data)
        assert "prompt" in mappings

    def test_model_type_detection_edge_cases(self, component):
        """Test model type detection with edge cases."""
        # Test with empty data
        result = component.detect_model_type([])
        assert result == "completion"  # Default

        # Test with data containing None values
        data = [{"prompt": None, "completion": None}]
        result = component.detect_model_type(data)
        assert result == "completion"

        # Test with malformed data
        data = [{"messages": None}, {"prompt": "test"}]
        result = component.detect_model_type(data)
        assert result == "completion"

    @pytest.mark.parametrize(
        "evaluation_count,expected_evaluation",
        [
            (0, 0),
            (1, 1),
            (5, 3),  # More than available records
            (2, 2),
        ],
    )
    def test_evaluation_split_counts(self, component, sample_completion_data, evaluation_count, expected_evaluation):
        """Test different evaluation split counts."""
        component.input_data = sample_completion_data
        component.model_type = "completion"
        component.evaluation_records_count = evaluation_count

        result = component.prepare_data()

        evaluation_records = [r for r in result if r.data.get("_is_evaluation")]
        assert len(evaluation_records) == expected_evaluation

    def test_preserve_unmapped_fields(self, component):
        """Test preserve_unmapped_fields functionality."""
        component.input_data = [{"prompt": "test", "completion": "result", "extra_field": "extra"}]
        component.model_type = "completion"
        component.preserve_unmapped_fields = True
        component.evaluation_records_count = 0

        result = component.prepare_data()
        assert len(result) == 1
        record = result[0].data
        assert "prompt" in record
        assert "completion" in record
        assert "extra_field" in record  # Should be preserved

    def test_strict_field_extraction_behavior(self, component):
        """Test strict field extraction behavior."""
        # Data with missing required fields
        component.input_data = [
            {"prompt": "test", "completion": "result"},  # Valid
            {"prompt": "test"},  # Missing completion
            {"completion": "result"},  # Missing prompt
        ]
        component.model_type = "completion"
        component.strict_field_extraction = True
        component.evaluation_records_count = 0

        result = component.prepare_data()
        # Should only keep the valid record
        assert len(result) == 1
        assert result[0].data["prompt"] == "test"
        assert result[0].data["completion"] == "result"

from typing import Any

import pandas as pd
from loguru import logger

from langflow.custom import Component
from langflow.io import (
    BoolInput,
    DictInput,
    DropdownInput,
    HandleInput,
    IntInput,
    Output,
)
from langflow.schema import Data
from langflow.schema.dataframe import DataFrame


class NeMoDataPreparationComponent(Component):
    display_name = "NeMo Data Preparation"
    description = (
        "Transform and validate input data into NeMo-compatible format with field mapping and format transformation"
    )
    icon = "NVIDIA"
    name = "NVIDIANeMoDataPreparation"
    beta = True

    inputs = [
        # Data Input (supports both list[Data] and DataFrame)
        HandleInput(
            name="input_data",
            display_name="Input Data",
            info="Input data as list[Data] or DataFrame",
            required=True,
            input_types=["Data", "DataFrame"],
        ),
        # Model Type Configuration
        DropdownInput(
            name="model_type",
            display_name="Model Type",
            info="Target model type for data format",
            options=["auto", "chat", "completion"],
            value="auto",
            required=False,
        ),
        # Field Mapping Configuration
        DictInput(
            name="custom_field_mappings",
            display_name="Custom Field Mappings",
            info="Custom field mappings as dictionary (e.g., {'prompt': 'input', 'completion': 'response'}). Leave empty for auto-detection.",
            advanced=True,
            required=False,
        ),
        # Processing Configuration
        BoolInput(
            name="strict_field_extraction",
            display_name="Strict Field Extraction",
            info="If True, discard records with missing required fields. If False, use defaults.",
            value=True,
            required=False,
        ),
        BoolInput(
            name="preserve_unmapped_fields",
            display_name="Preserve Unmapped Fields",
            info="If True, include all original fields in output. If False, only include mapped fields.",
            value=True,
            required=False,
        ),
        # Evaluation Split Configuration
        IntInput(
            name="evaluation_records_count",
            display_name="Evaluation Records Count",
            info="Number of records to randomly select and mark as evaluation data. Set to 0 to disable evaluation split.",
            value=20,
            required=False,
        ),
    ]

    outputs = [
        Output(
            display_name="Prepared Data (Data)",
            name="prepared_data",
            method="prepare_data",
            info="Data prepared for NeMo dataset upload as list[Data]",
        ),
        Output(
            display_name="Prepared Data (DataFrame)",
            name="prepared_dataframe",
            method="prepare_dataframe",
            info="Data prepared for NeMo dataset upload as DataFrame",
        ),
    ]

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.processed_data = None
        self.preparation_stats = {}

    def convert_to_dataframe(self, input_data: list[Data] | pd.DataFrame) -> pd.DataFrame:
        """Convert input data to DataFrame for processing."""
        if isinstance(input_data, pd.DataFrame):
            return input_data.copy()

        if isinstance(input_data, list):
            # Check if all items are Data objects
            if all(isinstance(item, Data) for item in input_data):
                # Convert list[Data] to DataFrame
                records = []
                for data_item in input_data:
                    if hasattr(data_item, "data"):
                        if isinstance(data_item.data, dict):
                            records.append(data_item.data)
                        elif isinstance(data_item.data, list):
                            records.extend(data_item.data)
                        else:
                            records.append({"data": data_item.data})
                    else:
                        records.append({"data": data_item})

                return pd.DataFrame(records)

        error_msg = "Input data must be DataFrame or list[Data]"
        raise ValueError(error_msg)

    def detect_model_type(self, data: list[dict]) -> str:
        """Detect whether data is intended for chat or completion models."""
        chat_indicators = 0
        completion_indicators = 0

        for record in data:
            # Check for messages structure (chat model)
            if record.get("messages") or (
                isinstance(record.get("data"), dict) and "messages" in record.get("data", {})
            ):
                chat_indicators += 1

            # Check for prompt/completion structure (completion model)
            if ("prompt" in record and "completion" in record) or (
                isinstance(record.get("data"), dict)
                and "prompt" in record.get("data", {})
                and "completion" in record.get("data", {})
            ):
                completion_indicators += 1

        # Determine based on majority
        if chat_indicators > completion_indicators:
            return "chat"
        if completion_indicators > chat_indicators:
            return "completion"
        # Default to completion for backward compatibility
        return "completion"

    def determine_model_type(self, data: list[dict]) -> str:
        """Determine model type with fallback to configuration."""
        if self.model_type == "auto":
            return self.detect_model_type(data)
        return self.model_type

    def extract_field_value(self, record: dict, field_path: str) -> Any:
        """Extract field value using JSONPath or simple field access."""
        if "." in field_path:
            # Use JSONPath-like extraction
            return self._extract_nested_field(record, field_path)
        # Simple field access
        return record.get(field_path)

    def _extract_nested_field(self, record: dict, field_path: str) -> Any:
        """Extract nested field using dot notation."""
        keys = field_path.split(".")
        value = record

        for key in keys:
            if isinstance(value, dict) and key in value:
                value = value[key]
            else:
                return None

        return value

    def auto_detect_field_mappings(self, sample_records: list[dict]) -> dict[str, str]:
        """Automatically detect field mappings from sample records."""
        # Common field name mappings
        PROMPT_FIELDS = [
            "prompt",
            "input",
            "question",
            "text",
            "content",
            "query",
            "instruction",
            "task",
            "context",
            "source",
            "original",
        ]

        COMPLETION_FIELDS = [
            "completion",
            "response",
            "answer",
            "output",
            "result",
            "generated",
            "target",
            "label",
            "category",
        ]

        IDEAL_RESPONSE_FIELDS = [
            "ideal_response",
            "expected_answer",
            "reference",
            "ground_truth",
            "correct_answer",
            "expected_output",
            "target_response",
        ]

        # Get all unique field names from sample records
        all_fields = set()
        for record in sample_records:
            all_fields.update(self._get_all_field_paths(record))

        mappings = {}

        # Find best matches for each NeMo field
        prompt_field = self._find_best_match(all_fields, PROMPT_FIELDS)
        if prompt_field:
            mappings["prompt"] = prompt_field

        completion_field = self._find_best_match(all_fields, COMPLETION_FIELDS)
        if completion_field:
            mappings["completion"] = completion_field

        ideal_response_field = self._find_best_match(all_fields, IDEAL_RESPONSE_FIELDS)
        if ideal_response_field:
            mappings["ideal_response"] = ideal_response_field

        return mappings

    def _get_all_field_paths(self, record: dict, prefix: str = "") -> set:
        """Recursively get all field paths from a record."""
        paths = set()

        for key, value in record.items():
            current_path = f"{prefix}.{key}" if prefix else key
            paths.add(current_path)

            if isinstance(value, dict):
                paths.update(self._get_all_field_paths(value, current_path))

        return paths

    def _find_best_match(self, available_fields: set, target_fields: list[str]) -> str:
        """Find the best matching field from available fields."""
        # Exact matches first
        for target in target_fields:
            if target in available_fields:
                return target

        # Partial matches (case-insensitive)
        available_lower = {f.lower(): f for f in available_fields}
        for target in target_fields:
            if target.lower() in available_lower:
                return available_lower[target.lower()]

        # Substring matches
        for target in target_fields:
            for available in available_fields:
                if target.lower() in available.lower() or available.lower() in target.lower():
                    return available

        return None

    def determine_field_mappings(self, data: list[dict]) -> dict[str, str]:
        """Determine field mappings based on configuration."""
        # Use custom mappings if provided
        if self.custom_field_mappings:
            return self.custom_field_mappings

        # Use automatic field detection
        sample_records = data[:10]  # Sample first 10 records
        return self.auto_detect_field_mappings(sample_records)

    def transform_to_chat_format(self, record: dict, is_evaluation: bool = False) -> dict:
        """Transform record to chat model format (messages schema)."""
        # Extract conversation data using field mappings
        field_mappings = self.determine_field_mappings([record])

        user_content = self.extract_field_value(record, field_mappings.get("prompt", "prompt"))
        assistant_content = self.extract_field_value(record, field_mappings.get("completion", "completion"))

        messages = []

        # Add user message
        if user_content:
            messages.append({"role": "user", "content": str(user_content)})

        # Add assistant message (for training) or ideal response (for evaluation)
        if assistant_content:
            messages.append({"role": "assistant", "content": str(assistant_content)})

        result = {"messages": messages}

        # Preserve unmapped fields if enabled
        if self.preserve_unmapped_fields:
            # Get all mapped field names
            mapped_fields = set(field_mappings.values())
            mapped_fields.update(["prompt", "completion", "messages"])  # Add standard fields

            # Add all unmapped fields from original record
            for key, value in record.items():
                if key not in mapped_fields:
                    result[key] = value

        return result

    def transform_to_completion_format(self, record: dict, is_evaluation: bool = False) -> dict:
        """Transform record to completion model format (prompt-completion schema)."""
        field_mappings = self.determine_field_mappings([record])

        prompt = self.extract_field_value(record, field_mappings.get("prompt", "prompt"))
        completion = self.extract_field_value(record, field_mappings.get("completion", "completion"))

        if is_evaluation:
            # For evaluation, use the same completion field but output as ideal_response
            result = {
                "prompt": str(prompt) if prompt else "",
                "ideal_response": str(completion) if completion else "",
            }
        else:
            # For training, use completion field
            result = {"prompt": str(prompt) if prompt else "", "completion": str(completion) if completion else ""}

        # Preserve unmapped fields if enabled
        if self.preserve_unmapped_fields:
            # Get all mapped field names
            mapped_fields = set(field_mappings.values())
            mapped_fields.update(["prompt", "completion", "ideal_response"])  # Add standard fields

            # Add all unmapped fields from original record
            for key, value in record.items():
                if key not in mapped_fields:
                    result[key] = value

        return result

    def is_evaluation_record(self, record: dict) -> bool:
        """Check if a record is already in evaluation format."""
        # Check for evaluation indicators
        return bool(record.get("ideal_response") or record.get("expected_answer") or record.get("ground_truth"))

    def transform_record(self, record: dict, is_evaluation: bool = False) -> dict:
        """Transform record based on detected model type."""
        # If record is already in evaluation format, preserve it
        if self.is_evaluation_record(record):
            return record

        model_type = self.determine_model_type([record])

        if model_type == "chat":
            return self.transform_to_chat_format(record, is_evaluation)
        return self.transform_to_completion_format(record, is_evaluation)

    def process_data(self, input_data: list[Data] | pd.DataFrame) -> list[dict]:
        """Process input data and return transformed records."""
        # Convert to DataFrame for processing
        df = self.convert_to_dataframe(input_data)

        # Convert DataFrame to list of dictionaries
        records = df.to_dict("records")

        # Determine field mappings
        field_mappings = self.determine_field_mappings(records)

        # Determine model type
        model_type = self.determine_model_type(records)

        # Transform records
        transformed_records = []
        valid_count = 0
        discarded_count = 0
        evaluation_count = 0

        for i, record in enumerate(records):
            try:
                # Check if record is already in evaluation format
                is_already_evaluation = self.is_evaluation_record(record)

                # Only apply random evaluation split if record is not already evaluation format
                is_random_evaluation = False
                if not is_already_evaluation and self.evaluation_records_count > 0:
                    # Check if we should randomly mark this as evaluation
                    if evaluation_count < self.evaluation_records_count:
                        is_random_evaluation = True

                is_evaluation = is_already_evaluation or is_random_evaluation
                transformed_record = self.transform_record(record, is_evaluation)

                # Validate required fields
                if self.strict_field_extraction:
                    if model_type == "chat":
                        if not transformed_record.get("messages"):
                            discarded_count += 1
                            continue
                    elif not transformed_record.get("prompt"):
                        discarded_count += 1
                        continue

                # Add evaluation flag if this is an evaluation record
                if is_evaluation:
                    transformed_record["_is_evaluation"] = True

                transformed_records.append(transformed_record)
                valid_count += 1

                # Track evaluation count
                if is_evaluation:
                    evaluation_count += 1

            except Exception as e:
                logger.warning(f"Failed to transform record {i}: {e}")
                discarded_count += 1
                continue

        # Update preparation statistics
        self.preparation_stats = {
            "total_input_records": len(records),
            "successfully_prepared": valid_count,
            "discarded_records": discarded_count,
            "evaluation_records": evaluation_count,
            "model_type": model_type,
            "field_mappings": field_mappings,
            "preserve_unmapped_fields": self.preserve_unmapped_fields,
            "processing_time": 0.0,  # TODO: Add actual timing
        }

        # Log statistics
        self.log(
            f"Data preparation completed: {valid_count} valid, {discarded_count} discarded, {evaluation_count} evaluation records"
        )
        self.log(f"Model type: {model_type}")
        self.log(f"Field mappings: {field_mappings}")

        if self.preserve_unmapped_fields:
            self.log("Unmapped fields will be preserved in output")
        else:
            self.log("Only mapped fields will be included in output")

        return transformed_records

    def prepare_data(self) -> list[Data]:
        """Prepare data and return as list[Data]."""
        if self.input_data is None:
            return []

        processed_records = self.process_data(self.input_data)
        return [Data(data=record) for record in processed_records]

    def prepare_dataframe(self) -> DataFrame:
        """Prepare data and return as DataFrame."""
        if self.input_data is None:
            return DataFrame()

        processed_records = self.process_data(self.input_data)
        return DataFrame(processed_records)

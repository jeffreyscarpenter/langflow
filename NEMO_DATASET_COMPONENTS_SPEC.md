# NeMo Dataset Components Specification

## Overview

This specification defines the refactored NeMo dataset components that separate data preparation from dataset upload functionality. The design provides intelligent data classification, configurable training/validation splits, partial success handling, and detailed metadata reporting.

## Component Architecture

### 1. NeMo Data Preparation Component
**Purpose**: Transform and validate input data into NeMo-compatible format, including field mapping, data type detection, and all necessary transformations

### 2. NeMo Dataset Uploader Component
**Purpose**: Intelligently upload data to NeMo Data Store with automatic classification and configurable splits

## Component Specifications

### NeMo Data Preparation Component

#### Inputs
```python
inputs = [
    # Data Input (supports both list[Data] and DataFrame)
    HandleInput(
        name="input_data",
        display_name="Input Data",
        info="Input data as list[Data] or DataFrame",
        required=True,
        input_types=["Data"],
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
        range_spec=RangeSpec(min=0, max=10000, step=1),
        required=False,
    ),


]
```

#### Outputs
```python
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
```

#### Methods

##### `prepare_data() -> list[Data]`
- Parse input data format (JSON, JSONL, DataFrame, etc.)
- Detect or determine model type (chat vs completion)
- Extract fields using JSONPath or field names
- Apply data type detection logic
- Transform to NeMo-compatible format based on model type
- Validate required fields based on data type and model type
- Apply field transformations and defaults
- If evaluation_records_count > 0, randomly select that many records and mark them as evaluation data
- Return standardized list[Data] containing both training and evaluation records

##### `prepare_dataframe() -> DataFrame`
- Same processing logic as `prepare_data()` but returns DataFrame format
- Convert processed data to pandas DataFrame
- Maintain all metadata and field mappings in DataFrame structure
- If evaluation_records_count > 0, randomly select that many records and mark them as evaluation data
- Return DataFrame with standardized column names and data types, containing both training and evaluation records

#### Logging
Both `prepare_data()` and `prepare_dataframe()` methods will log preparation statistics using `self.log()`:
- Counts of processed, valid, and discarded records
- Field mapping information and extraction errors
- Model type detection results
- Processing time and performance metrics
- Evaluation split information (if enabled): number of records selected for evaluation

#### Model Type Detection Logic
```python
def detect_model_type(self, data: list[Data]) -> str:
    """Detect whether data is intended for chat or completion models."""
    chat_indicators = 0
    completion_indicators = 0

    for record in data:
        # Check for messages structure (chat model)
        if hasattr(record, 'messages') and record.messages:
            chat_indicators += 1
        elif hasattr(record, 'data') and 'messages' in record.data:
            chat_indicators += 1

        # Check for prompt/completion structure (completion model)
        if hasattr(record, 'prompt') and hasattr(record, 'completion'):
            completion_indicators += 1
        elif hasattr(record, 'data') and 'prompt' in record.data and 'completion' in record.data:
            completion_indicators += 1

    # Determine based on majority
    if chat_indicators > completion_indicators:
        return "chat"
    elif completion_indicators > chat_indicators:
        return "completion"
    else:
        # Default to completion for backward compatibility
        return "completion"

def determine_model_type(self, data: list[Data]) -> str:
    """Determine model type with fallback to configuration."""
    if self.model_type == "auto":
        return self.detect_model_type(data)
    else:
        return self.model_type
```

#### Format Transformation Logic

**Training Data Format:**
- **Completion Model**: `{"prompt": "...", "completion": "..."}`
- **Chat Model**: `{"messages": [{"role": "user", "content": "..."}, {"role": "assistant", "content": "..."}]}`

**Evaluation Data Format:**
- **Completion Model**: `{"prompt": "...", "ideal_response": "..."}`
- **Chat Model**: `{"messages": [{"role": "user", "content": "..."}, {"role": "assistant", "content": "..."}]}` (same as training)

```python
def transform_to_chat_format(self, record: dict, is_evaluation: bool = False) -> dict:
    """Transform record to chat model format (messages schema)."""
    # Extract conversation data
    user_content = self.extract_field_value(record, self.prompt_field_path)
    assistant_content = self.extract_field_value(record, self.completion_field_path)

    messages = []

    # Add system message if configured
    if self.system_message:
        messages.append({
            "role": "system",
            "content": self.system_message
        })

    # Add user message
    if user_content:
        messages.append({
            "role": "user",
            "content": user_content
        })

    # Add assistant message (for training) or ideal response (for evaluation)
    if assistant_content:
        messages.append({
            "role": "assistant",
            "content": assistant_content
        })

    return {"messages": messages}

def transform_to_completion_format(self, record: dict, is_evaluation: bool = False) -> dict:
    """Transform record to completion model format (prompt-completion schema)."""
    prompt = self.extract_field_value(record, self.prompt_field_path)

    if is_evaluation:
        # For evaluation, use ideal_response field
        ideal_response = self.extract_field_value(record, self.ideal_response_field_path)
        return {
            "prompt": prompt,
            "ideal_response": ideal_response
        }
    else:
        # For training, use completion field
        completion = self.extract_field_value(record, self.completion_field_path)
        return {
            "prompt": prompt,
            "completion": completion
        }

def transform_record(self, record: dict, is_evaluation: bool = False) -> dict:
    """Transform record based on detected model type."""
    model_type = self.determine_model_type([record])  # Single record for type detection

    if model_type == "chat":
        return self.transform_to_chat_format(record, is_evaluation)
    else:
        return self.transform_to_completion_format(record, is_evaluation)
```

#### Field Extraction Logic
```python
def extract_field_value(self, record: dict, field_path: str) -> Any:
    """Extract field value using JSONPath or simple field access."""
    if "." in field_path:
        # Use JSONPath-like extraction
        return self._extract_nested_field(record, field_path)
    else:
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
```

#### Data Type Detection Logic
```python
def classify_record_type(self, record: dict) -> str:
    """Classify a record as training, evaluation, or unknown."""
    if self.data_type_detection == "field_presence":
        return self._classify_by_field_presence(record)
    elif self.data_type_detection == "workload_pattern":
        return self._classify_by_workload_pattern(record)
    elif self.data_type_detection == "client_pattern":
        return self._classify_by_client_pattern(record)
    elif self.data_type_detection == "timestamp_range":
        return self._classify_by_timestamp_range(record)
    else:
        return "unknown"

def _classify_by_field_presence(self, record: dict) -> str:
    """Classify based on presence of evaluation indicator field."""
    indicator_value = self.extract_field_value(record, self.evaluation_indicator_field)

    if indicator_value is not None:
        if self.evaluation_indicator_value:
            return "evaluation" if indicator_value == self.evaluation_indicator_value else "training"
        else:
            return "evaluation"  # Any non-null value indicates evaluation
    else:
        return "training"
```

#### Automatic Field Mapping Logic
```python
class AutoFieldMapper:
    """Automatically maps common field names to NeMo standard fields."""

    # Common field name mappings
    PROMPT_FIELDS = [
        "prompt", "input", "question", "text", "content", "query",
        "instruction", "task", "context", "source", "original"
    ]

    COMPLETION_FIELDS = [
        "completion", "response", "answer", "output", "result",
        "generated", "target", "label", "category"
    ]

    IDEAL_RESPONSE_FIELDS = [
        "ideal_response", "expected_answer", "reference", "ground_truth",
        "correct_answer", "expected_output", "target_response"
    ]

    CATEGORY_FIELDS = [
        "category", "type", "class", "label", "document_type",
        "dataset_type", "task_type"
    ]

    SOURCE_FIELDS = [
        "source", "dataset_name", "workload_id", "client_id",
        "file_name", "origin", "reference"
    ]

    def detect_field_mappings(self, sample_records: list[dict]) -> dict:
        """Detect field mappings from sample records."""
        mappings = {
            "prompt": None,
            "completion": None,
            "ideal_response": None,
            "category": None,
            "source": None
        }

        # Get all unique field names from sample records
        all_fields = set()
        for record in sample_records:
            all_fields.update(self._get_all_field_paths(record))

        # Find best matches for each NeMo field
        mappings["prompt"] = self._find_best_match(all_fields, self.PROMPT_FIELDS)
        mappings["completion"] = self._find_best_match(all_fields, self.COMPLETION_FIELDS)
        mappings["ideal_response"] = self._find_best_match(all_fields, self.IDEAL_RESPONSE_FIELDS)
        mappings["category"] = self._find_best_match(all_fields, self.CATEGORY_FIELDS)
        mappings["source"] = self._find_best_match(all_fields, self.SOURCE_FIELDS)

        return {k: v for k, v in mappings.items() if v is not None}

    def _get_all_field_paths(self, record: dict, prefix: str = "") -> set:
        """Recursively get all field paths from a record."""
        paths = set()

        for key, value in record.items():
            current_path = f"{prefix}.{key}" if prefix else key
            paths.add(current_path)

            if isinstance(value, dict):
                paths.update(self._get_all_field_paths(value, current_path))

        return paths

    def _find_best_match(self, available_fields: set, target_fields: list) -> str:
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
```

#### Field Mapping Logic
```python
def determine_field_mappings(self, data: list[Data]) -> dict:
    """Determine field mappings based on configuration mode."""
    if self.field_mapping_mode == "auto" and self.enable_auto_mapping:
        # Use automatic field detection
        sample_records = [self._data_to_dict(record) for record in data[:10]]  # Sample first 10 records
        auto_mapper = AutoFieldMapper()
        auto_mappings = auto_mapper.detect_field_mappings(sample_records)

        # Override with explicit mappings if provided
        if self.prompt_field_path:
            auto_mappings["prompt"] = self.prompt_field_path
        if self.completion_field_path:
            auto_mappings["completion"] = self.completion_field_path
        if self.ideal_response_field_path:
            auto_mappings["ideal_response"] = self.ideal_response_field_path
        if self.category_field_path:
            auto_mappings["category"] = self.category_field_path
        if self.source_field_path:
            auto_mappings["source"] = self.source_field_path

        return auto_mappings
    else:
        # Use explicit mappings only
        mappings = {}
        if self.prompt_field_path:
            mappings["prompt"] = self.prompt_field_path
        if self.completion_field_path:
            mappings["completion"] = self.completion_field_path
        if self.ideal_response_field_path:
            mappings["ideal_response"] = self.ideal_response_field_path
        if self.category_field_path:
            mappings["category"] = self.category_field_path
        if self.source_field_path:
            mappings["source"] = self.source_field_path

        return mappings
```


#### Enhanced Preparation Statistics
```python
{
    "total_input_records": 1000,
    "successfully_prepared": 950,
    "discarded_records": 50,
    "filtered_records": 200,
    "training_records": 600,
    "evaluation_records": 350,
    "unknown_records": 50,
    "model_type": "completion",
    "field_mapping": {
        "mode": "auto",
        "auto_detected": {
            "prompt": "input",
            "completion": "completion",
            "category": "document_type",
            "source": "dataset_name"
        },
        "explicit_overrides": {
            "prompt": "input"  # User explicitly set this
        },
        "confidence": {
            "prompt": "high",
            "completion": "high",
            "category": "medium",
            "source": "high"
        }
    },
    "filtering": {
        "enabled": True,
        "filters_applied": [
            {
                "field": "document_type",
                "operator": "in",
                "value": "question_answer,summarization",
                "records_filtered": 150
            },
            {
                "field": "dataset_name",
                "operator": "not_equals",
                "value": "test_dataset",
                "records_filtered": 50
            }
        ],
        "filter_logic": "AND",
        "total_filtered": 200,
        "filter_effectiveness": "20% of records filtered"
    },
    "field_detection_analysis": {
        "available_fields": ["input", "completion", "dataset_name", "document_type"],
        "mapped_fields": 4,
        "unmapped_fields": 0,
        "detection_method": "automatic_with_override"
    }
}
```

### NeMo Dataset Uploader Component

#### Inputs
```python
inputs = [
    # Authentication and Configuration
    SecretStrInput(
        name="auth_token",
        display_name="Authentication Token",
        info="Bearer token for firewall authentication",
        required=True,
    ),
    StrInput(
        name="base_url",
        display_name="Base API URL",
        info="Base URL for the NeMo services",
        required=True,
        value="https://us-west-2.api-dev.ai.datastax.com/nvidia/nemo",
    ),
    StrInput(
        name="namespace",
        display_name="Namespace",
        info="Namespace for the dataset",
        required=True,
        value="default",
    ),
    StrInput(
        name="dataset_name",
        display_name="Dataset Name",
        info="Name for the dataset (will be created if it doesn't exist)",
        required=True,
    ),
    StrInput(
        name="description",
        display_name="Description",
        info="Description of the dataset",
        value="Dataset created via Langflow NeMo Dataset Uploader",
    ),

    # Data Input
    DataInput(
        name="data",
        display_name="Data",
        info="Data to upload (list[Data] or DataFrame) - will be automatically classified",
        is_list=True,
        required=True,
    ),

    # Model Type Configuration
    DropdownInput(
        name="model_type",
        display_name="Model Type",
        info="Type of model this dataset is intended for (affects upload format)",
        options=["auto", "chat", "completion"],
        value="auto",
        required=False,
    ),

    # Split Configuration
    FloatInput(
        name="training_split_ratio",
        display_name="Training Split Ratio",
        info="Ratio of training data to use for training (0.0-1.0). Validation = 1.0 - training_split_ratio",
        value=0.9,
        range_spec=RangeSpec(min=0.0, max=1.0, step=0.01),
        required=False,
    ),
    IntInput(
        name="min_validation_records",
        display_name="Minimum Validation Records",
        info="Minimum number of records to use for validation (overrides ratio if needed)",
        value=1,
        range_spec=RangeSpec(min=1, max=1000, step=1),
        required=False,
    ),
    IntInput(
        name="min_training_records",
        display_name="Minimum Training Records",
        info="Minimum number of records to use for training (overrides ratio if needed)",
        value=2,
        range_spec=RangeSpec(min=1, max=100000, step=1),
        required=False,
    ),

    # Upload Configuration
    BoolInput(
        name="partial_success",
        display_name="Allow Partial Success",
        info="If True, upload data even if some records are invalid or unclassified",
        value=True,
        required=False,
    ),
    BoolInput(
        name="create_validation_split",
        display_name="Create Validation Split",
        info="If True, create validation split from training data",
        value=True,
        required=False,
    ),
    IntInput(
        name="chunk_size",
        display_name="Upload Chunk Size",
        info="Number of records per upload chunk",
        value=100000,
        range_spec=RangeSpec(min=1000, max=500000, step=1000),
        required=False,
    ),
]
```

#### Outputs
```python
outputs = [
    Output(
        display_name="Dataset Info",
        name="dataset_info",
        method="upload_dataset",
        info="Information about the created dataset",
    ),
    Output(
        display_name="Upload Statistics",
        name="upload_stats",
        method="get_upload_stats",
        info="Detailed statistics about the upload process",
    ),
]
```

#### Methods

##### `upload_dataset() -> Data`
- Analyze input data and classify records
- Determine model type (chat vs completion)
- Apply configurable training/validation split
- Upload data to HuggingFace repository in appropriate format
- Register dataset in NeMo entity store
- Return dataset information

##### `get_upload_stats() -> Data`
- Return detailed upload statistics
- Include counts by data type and split
- Include discarded record information
- Include file paths and sizes
- Include model type information

#### Data Classification Logic
```python
def classify_data(self, data: list[Data]) -> dict:
    """Classify input data into training, evaluation, and unknown categories."""
    training_records = []
    evaluation_records = []
    unknown_records = []

    for record in data:
        record_type = self._classify_record_type(record)

        if record_type == "training":
            training_records.append(record)
        elif record_type == "evaluation":
            evaluation_records.append(record)
        else:
            unknown_records.append(record)

    return {
        "training": training_records,
        "evaluation": evaluation_records,
        "unknown": unknown_records,
        "total": len(data)
    }
```

#### Split Logic
```python
def apply_training_split(self, training_records: list[Data]) -> tuple[list[Data], list[Data]]:
    """Apply configurable training/validation split to training data."""
    total_records = len(training_records)

    if total_records < self.min_training_records + self.min_validation_records:
        raise ValueError(f"Insufficient records for split: {total_records} < {self.min_training_records + self.min_validation_records}")

    # Calculate split based on ratio and minimums
    validation_count = max(
        self.min_validation_records,
        int(total_records * (1.0 - self.training_split_ratio))
    )

    # Ensure we don't exceed available records
    validation_count = min(validation_count, total_records - self.min_training_records)

    # Split the records
    validation_records = training_records[:validation_count]
    actual_training_records = training_records[validation_count:]

    return actual_training_records, validation_records
```

## Metadata and Statistics

### Preparation Statistics
```python
{
    "total_input_records": 1000,
    "successfully_prepared": 950,
    "discarded_records": 50,
    "training_records": 600,
    "evaluation_records": 350,
    "unknown_records": 50,
    "model_type": "chat",
    "model_type_detection": {
        "method": "auto",
        "chat_indicators": 800,
        "completion_indicators": 200,
        "confidence": "high"
    },
    "field_mappings": {
        "prompt": "request.prompt",
        "completion": "response.text",
        "ideal_response": "response.expected_answer",
        "source": "workload_id"
    },
    "format_transformation": {
        "input_format": "custom_json",
        "output_format": "messages_schema",
        "system_message": "You are a helpful assistant."
    },
    "extraction_errors": [
        {"record_index": 45, "error": "Missing required field 'request.prompt'"},
        {"record_index": 123, "error": "Invalid JSON structure"}
    ],
    "data_type_detection": {
        "method": "field_presence",
        "training_indicator": "response.expected_answer is null",
        "evaluation_indicator": "response.expected_answer is not null"
    },
    "validation_errors": [
        {"record_index": 45, "error": "Missing required field 'prompt'"},
        {"record_index": 123, "error": "Empty completion field"}
    ]
}
```

### Upload Statistics
```python
{
    "dataset_info": {
        "id": "dataset-123",
        "name": "my-dataset",
        "namespace": "default",
        "files_url": "hf://datasets/default/my-dataset",
        "created_at": "2024-01-15T10:30:00Z",
        "model_type": "chat"
    },
    "upload_summary": {
        "total_input_records": 1000,
        "successfully_uploaded": 950,
        "discarded_records": 50,
        "partial_success": True,
        "model_type": "chat"
    },
    "data_distribution": {
        "training": {
            "total_records": 540,
            "uploaded_records": 540,
            "discarded_records": 0,
            "files": ["training/dataset_chunk_1.jsonl", "training/dataset_chunk_2.jsonl"]
        },
        "validation": {
            "total_records": 60,
            "uploaded_records": 60,
            "discarded_records": 0,
            "files": ["validation/dataset_validation.jsonl"]
        },
        "evaluation": {
            "total_records": 350,
            "uploaded_records": 350,
            "discarded_records": 0,
            "files": ["input.json", "output.json"]
        },
        "unknown": {
            "total_records": 50,
            "uploaded_records": 0,
            "discarded_records": 50,
            "reason": "Unable to classify records - missing required fields"
        }
    },
    "file_sizes": {
        "training/dataset_chunk_1.jsonl": "2.1MB",
        "training/dataset_chunk_2.jsonl": "1.8MB",
        "validation/dataset_validation.jsonl": "150KB",
        "input.json": "200KB",
        "output.json": "180KB"
    },
    "split_configuration": {
        "training_split_ratio": 0.9,
        "min_validation_records": 1,
        "min_training_records": 2,
        "actual_validation_ratio": 0.1
    }
}
```

## Error Handling

### Partial Success Scenarios
1. **Invalid Records**: Records with missing required fields are discarded
2. **Unclassified Records**: Records that don't match training or evaluation patterns are discarded
3. **Upload Failures**: Individual chunk upload failures don't stop the entire process
4. **Insufficient Data**: Graceful handling when minimum record requirements aren't met
5. **Model Type Conflicts**: Handle cases where data structure doesn't match intended model type

### Error Reporting
- Detailed error messages for each discarded record
- Reason codes for different types of failures
- Statistics on error types and frequencies
- Recommendations for fixing common issues
- Model type detection confidence levels

## File Structure

### Training Data
```
training/
├── dataset_chunk_1.jsonl
├── dataset_chunk_2.jsonl
└── ...
```

### Validation Data
```
validation/
└── dataset_validation.jsonl
```

### Evaluation Data
```
input.json      # Evaluation prompts and ideal responses
output.json     # Model responses and metadata
```

## Usage Examples

### Example 1: Your Data Structure Flow (Chat Model)
```python
# Flow: File Loader → NeMo Data Preparation → NeMo Dataset Uploader

# Configuration for your JSON structure:
# NeMo Data Preparation Component:
model_type = "auto"  # Will detect chat model from data structure
prompt_field_path = "request.prompt"
completion_field_path = "response.text"
ideal_response_field_path = "response.expected_answer"
source_field_path = "workload_id"
system_message = "You are a helpful assistant."
data_type_detection = "field_presence"
evaluation_indicator_field = "response.expected_answer"

# This will:
# 1. Detect model type as "chat" based on data structure
# 2. Extract prompt from request.prompt
# 3. Extract completion from response.text
# 4. Extract ideal_response from response.expected_answer
# 5. Use workload_id as source
# 6. Classify as evaluation if response.expected_answer exists
# 7. Transform to messages schema format
# 8. Validate and prepare for upload
```

### Example 2: Completion Model Configuration
```python
# For completion model training:
model_type = "completion"
prompt_field_path = "question"
completion_field_path = "answer"

# This will create prompt-completion format:
# {"prompt": "What is 2+2?", "completion": "4"}
```

### Example 3: Custom Split Configuration
```python
# 80/20 training/validation split with minimums
training_split_ratio = 0.8
min_validation_records = 10
min_training_records = 50
```

### Example 4: Partial Success Upload
```python
# Upload with invalid records discarded
partial_success = True
# Statistics will show discarded records and reasons
```

## Migration Path

### From Current Component
1. **Deprecation Period**: Keep existing component with deprecation warning
2. **Feature Parity**: Ensure new components support all existing functionality
3. **Documentation**: Provide migration guide with examples
4. **Testing**: Comprehensive testing with existing datasets

### Backward Compatibility
- Support existing input formats
- Maintain same output structure for dataset info
- Preserve authentication and configuration patterns
- Support both chat and completion model formats

## Implementation Phases

### Phase 1: Core Upload Component
- Implement intelligent data classification
- Add configurable split logic
- Implement partial success handling
- Add comprehensive statistics
- Add model type detection and format handling

### Phase 2: Data Preparation Component
- Implement field extraction with JSONPath support
- Add data type detection logic
- Support multiple input formats
- Add field mapping and validation
- Add model type detection and format transformation
- Add preparation statistics

### Phase 3: Integration and Testing
- End-to-end testing with various data sources
- Performance optimization
- Documentation and examples
- Migration tools

## Success Criteria

1. **Flexibility**: Support various input data formats and sources
2. **Intelligence**: Automatic classification of training vs evaluation data
3. **Model Type Support**: Handle both chat and completion model formats
4. **Configurability**: Adjustable split ratios and minimums
5. **Reliability**: Robust error handling and partial success support
6. **Transparency**: Detailed statistics and error reporting
7. **Performance**: Efficient processing of large datasets
8. **Usability**: Clear documentation and migration path

## Detailed Implementation Plan

### Core Requirements Implementation

#### 1. Configurable Training/Validation Split

**Implementation Details:**
```python
class SplitConfiguration:
    def __init__(self, training_ratio: float, min_training: int, min_validation: int):
        self.training_ratio = training_ratio
        self.min_training = min_training
        self.min_validation = min_validation

    def calculate_split(self, total_records: int) -> tuple[int, int]:
        """Calculate training and validation counts based on configuration."""
        # Ensure minimum requirements are met
        if total_records < self.min_training + self.min_validation:
            raise ValueError(f"Insufficient records: {total_records} < {self.min_training + self.min_validation}")

        # Calculate based on ratio
        validation_count = max(
            self.min_validation,
            int(total_records * (1.0 - self.training_ratio))
        )

        # Ensure we don't exceed available records
        validation_count = min(validation_count, total_records - self.min_training)
        training_count = total_records - validation_count

        return training_count, validation_count
```

**Usage Examples:**
```python
# Default 90/10 split
split_config = SplitConfiguration(0.9, 2, 1)

# Custom 80/20 split with minimums
split_config = SplitConfiguration(0.8, 50, 10)

# No validation split
split_config = SplitConfiguration(1.0, 2, 0)
```

#### 2. Partial Success Handling

**Implementation Details:**
```python
class PartialSuccessHandler:
    def __init__(self, allow_partial: bool = True):
        self.allow_partial = allow_partial
        self.discarded_records = []
        self.upload_failures = []
        self.classification_errors = []

    def handle_invalid_record(self, record: Data, reason: str, record_index: int):
        """Handle a record that couldn't be processed."""
        error_info = {
            "record_index": record_index,
            "reason": reason,
            "record_data": record.data if hasattr(record, 'data') else str(record)
        }

        if reason == "missing_required_fields":
            self.classification_errors.append(error_info)
        elif reason == "upload_failure":
            self.upload_failures.append(error_info)
        else:
            self.discarded_records.append(error_info)

    def should_continue(self, total_records: int, valid_records: int) -> bool:
        """Determine if upload should continue despite errors."""
        if not self.allow_partial:
            return valid_records == total_records

        # Continue if we have at least some valid records
        return valid_records > 0

    def get_error_summary(self) -> dict:
        """Get summary of all errors encountered."""
        return {
            "total_discarded": len(self.discarded_records),
            "classification_errors": len(self.classification_errors),
            "upload_failures": len(self.upload_failures),
            "error_details": {
                "discarded_records": self.discarded_records,
                "classification_errors": self.classification_errors,
                "upload_failures": self.upload_failures
            }
        }
```

#### 3. Comprehensive Metadata Statistics

**Implementation Details:**
```python
class UploadStatistics:
    def __init__(self):
        self.reset()

    def reset(self):
        """Reset all statistics."""
        self.total_input_records = 0
        self.classified_records = {
            "training": 0,
            "evaluation": 0,
            "unknown": 0
        }
        self.uploaded_records = {
            "training": 0,
            "validation": 0,
            "evaluation": 0
        }
        self.discarded_records = {
            "training": 0,
            "evaluation": 0,
            "unknown": 0
        }
        self.file_info = {}
        self.split_config = {}
        self.error_summary = {}
        self.processing_time = 0.0
        self.model_type = None

    def record_classification(self, record_type: str, count: int):
        """Record classification statistics."""
        if record_type in self.classified_records:
            self.classified_records[record_type] += count

    def record_upload(self, data_type: str, count: int, files: list[str]):
        """Record upload statistics."""
        if data_type in self.uploaded_records:
            self.uploaded_records[data_type] += count
            if data_type not in self.file_info:
                self.file_info[data_type] = []
            self.file_info[data_type].extend(files)

    def record_discard(self, record_type: str, count: int, reason: str):
        """Record discarded record statistics."""
        if record_type in self.discarded_records:
            self.discarded_records[record_type] += count

    def set_split_configuration(self, config: dict):
        """Set split configuration details."""
        self.split_config = config

    def set_error_summary(self, error_summary: dict):
        """Set error summary details."""
        self.error_summary = error_summary

    def set_model_type(self, model_type: str):
        """Set model type information."""
        self.model_type = model_type

    def get_comprehensive_stats(self) -> dict:
        """Get comprehensive upload statistics."""
        total_uploaded = sum(self.uploaded_records.values())
        total_discarded = sum(self.discarded_records.values())

        return {
            "upload_summary": {
                "total_input_records": self.total_input_records,
                "successfully_uploaded": total_uploaded,
                "discarded_records": total_discarded,
                "partial_success": total_discarded > 0 and total_uploaded > 0,
                "processing_time_seconds": self.processing_time,
                "model_type": self.model_type
            },
            "data_distribution": {
                "training": {
                    "total_records": self.classified_records["training"],
                    "uploaded_records": self.uploaded_records["training"],
                    "discarded_records": self.discarded_records["training"],
                    "files": self.file_info.get("training", [])
                },
                "validation": {
                    "total_records": self.uploaded_records["validation"],
                    "uploaded_records": self.uploaded_records["validation"],
                    "discarded_records": 0,  # Validation records come from training
                    "files": self.file_info.get("validation", [])
                },
                "evaluation": {
                    "total_records": self.classified_records["evaluation"],
                    "uploaded_records": self.uploaded_records["evaluation"],
                    "discarded_records": self.discarded_records["evaluation"],
                    "files": self.file_info.get("evaluation", [])
                },
                "unknown": {
                    "total_records": self.classified_records["unknown"],
                    "uploaded_records": 0,
                    "discarded_records": self.discarded_records["unknown"],
                    "reason": "Unable to classify records - missing required fields"
                }
            },
            "split_configuration": self.split_config,
            "error_summary": self.error_summary
        }
```

### Integration with Existing NeMo Infrastructure

#### Authentication and Client Management
```python
class NeMoClientManager:
    def __init__(self, auth_token: str, base_url: str):
        self.auth_token = auth_token
        self.base_url = base_url
        self.hf_api = None
        self.nemo_client = None

    def get_hf_api(self) -> AuthenticatedHfApi:
        """Get authenticated HuggingFace API client."""
        if not self.hf_api:
            self.hf_api = AuthenticatedHfApi(
                endpoint=f"{self.base_url}/v1/hf",
                auth_token=self.auth_token,
                token=self.auth_token,
            )
        return self.hf_api

    def get_nemo_client(self) -> AsyncNeMoMicroservices:
        """Get authenticated NeMo microservices client."""
        if not self.nemo_client:
            self.nemo_client = AsyncNeMoMicroservices(
                base_url=self.base_url,
            )
        return self.nemo_client
```

#### File Upload Strategy
```python
class ChunkedUploader:
    def __init__(self, chunk_size: int = 100000):
        self.chunk_size = chunk_size
        self.chunk_number = 1

    async def upload_training_chunks(self, records: list[Data], repo_id: str, hf_api, model_type: str) -> list[str]:
        """Upload training data in chunks."""
        uploaded_files = []

        for i in range(0, len(records), self.chunk_size):
            chunk = records[i:i + self.chunk_size]

            # Format based on model type
            if model_type == "chat":
                chunk_data = [self._format_chat_record(r) for r in chunk]
            else:
                chunk_data = [self._format_completion_record(r) for r in chunk]

            chunk_df = pd.DataFrame(chunk_data)

            filename = f"training/dataset_chunk_{self.chunk_number}.jsonl"
            await self._upload_chunk(chunk_df, filename, repo_id, hf_api)

            uploaded_files.append(filename)
            self.chunk_number += 1

        return uploaded_files

    def _format_chat_record(self, record: Data) -> dict:
        """Format record for chat model (messages schema)."""
        if hasattr(record, 'messages'):
            return {"messages": record.messages}
        else:
            # Transform from prompt/completion to messages
            return {
                "messages": [
                    {"role": "user", "content": record.prompt},
                    {"role": "assistant", "content": record.completion}
                ]
            }

    def _format_completion_record(self, record: Data) -> dict:
        """Format record for completion model (prompt-completion schema)."""
        return {
            "prompt": record.prompt,
            "completion": record.completion
        }

    async def upload_validation_data(self, records: list[Data], repo_id: str, hf_api, model_type: str) -> str:
        """Upload validation data."""
        if not records:
            return None

        # Format based on model type
        if model_type == "chat":
            validation_data = [self._format_chat_record(r) for r in records]
        else:
            validation_data = [self._format_completion_record(r) for r in records]

        validation_df = pd.DataFrame(validation_data)
        filename = "validation/dataset_validation.jsonl"
        await self._upload_chunk(validation_df, filename, repo_id, hf_api)

        return filename
```

### Error Handling and Recovery

#### Graceful Degradation
```python
class UploadErrorHandler:
    def __init__(self, partial_success: bool = True):
        self.partial_success = partial_success
        self.errors = []

    async def safe_upload_chunk(self, chunk_df: pd.DataFrame, filename: str, repo_id: str, hf_api) -> bool:
        """Safely upload a chunk with error handling."""
        try:
            await self._upload_chunk(chunk_df, filename, repo_id, hf_api)
            return True
        except Exception as e:
            error_info = {
                "filename": filename,
                "error": str(e),
                "record_count": len(chunk_df)
            }
            self.errors.append(error_info)

            if not self.partial_success:
                raise

            return False

    def get_upload_errors(self) -> list[dict]:
        """Get list of upload errors."""
        return self.errors
```

### Testing Strategy

#### Unit Tests
```python
class TestNeMoDatasetUploader:
    def test_configurable_split(self):
        """Test configurable training/validation split."""
        # Test various split configurations
        # Test minimum record requirements
        # Test edge cases

    def test_partial_success_handling(self):
        """Test partial success scenarios."""
        # Test with invalid records
        # Test with upload failures
        # Test with classification errors

    def test_comprehensive_statistics(self):
        """Test comprehensive statistics generation."""
        # Test all statistic fields
        # Test error reporting
        # Test file information

    def test_model_type_detection(self):
        """Test model type detection logic."""
        # Test chat model detection
        # Test completion model detection
        # Test auto-detection confidence
```

#### Integration Tests
```python
class TestNeMoIntegration:
    def test_end_to_end_upload(self):
        """Test complete upload workflow."""
        # Test with real NeMo services
        # Test with various data formats
        # Test error scenarios

    def test_backward_compatibility(self):
        """Test compatibility with existing datasets."""
        # Test with existing dataset formats
        # Test migration scenarios

    def test_model_type_formats(self):
        """Test both chat and completion model formats."""
        # Test chat model upload
        # Test completion model upload
        # Test format validation
```

### Performance Considerations

#### Large Dataset Handling
```python
class PerformanceOptimizer:
    def __init__(self, max_memory_usage: int = 1024 * 1024 * 1024):  # 1GB
        self.max_memory_usage = max_memory_usage

    def optimize_chunk_size(self, total_records: int, record_size: int) -> int:
        """Calculate optimal chunk size based on memory constraints."""
        max_records_per_chunk = self.max_memory_usage // record_size
        return min(100000, max_records_per_chunk)  # Cap at 100k

    def stream_process_large_dataset(self, data_iterator, chunk_size: int):
        """Stream process large datasets to minimize memory usage."""
        chunk = []
        for record in data_iterator:
            chunk.append(record)
            if len(chunk) >= chunk_size:
                yield chunk
                chunk = []

        if chunk:  # Yield remaining records
            yield chunk
```

### Processing Logic

#### Input Handling and DataFrame Conversion
```python
import pandas as pd
from typing import Union, List, Dict, Any
from langflow import Data

class DataFrameProcessor:
    """Handle input data conversion and DataFrame operations."""

    def __init__(self, config: dict):
        self.output_format = config.get("output_format", "list[Data]")
        self.strict_field_extraction = config.get("strict_field_extraction", True)

    def convert_to_dataframe(self, input_data: Union[List[Data], pd.DataFrame]) -> pd.DataFrame:
        """Convert input data to DataFrame for processing."""
        if isinstance(input_data, pd.DataFrame):
            return input_data.copy()

        if isinstance(input_data, list) and all(isinstance(item, Data) for item in input_data):
            # Convert list[Data] to DataFrame
            records = []
            for data_item in input_data:
                if hasattr(data_item, 'data'):
                    if isinstance(data_item.data, dict):
                        records.append(data_item.data)
                    elif isinstance(data_item.data, list):
                        records.extend(data_item.data)
                    else:
                        records.append({"data": data_item.data})
                else:
                    records.append({"data": data_item})

            return pd.DataFrame(records)

        raise ValueError("Input data must be DataFrame or list[Data]")

    def convert_from_dataframe(self, df: pd.DataFrame) -> Union[List[Data], pd.DataFrame]:
        """Convert DataFrame back to requested output format."""
        if self.output_format == "DataFrame":
            return df

        # Convert to list[Data]
        records = df.to_dict('records')
        return [Data(data=record) for record in records]
```

#### DataFrame-Based Filtering
```python
class DataFrameFilter:
    """Filter DataFrame using pandas operations."""

    def __init__(self, config: dict):
        self.enable_filtering = config.get("enable_filtering", False)
        self.filter_field = config.get("filter_field", "")
        self.filter_operator = config.get("filter_operator", "equals")
        self.filter_value = config.get("filter_value", "")
        self.filter_field_2 = config.get("filter_field_2", "")
        self.filter_operator_2 = config.get("filter_operator_2", "equals")
        self.filter_value_2 = config.get("filter_value_2", "")
        self.filter_logic = config.get("filter_logic", "AND")
        self.custom_filter_expression = config.get("custom_filter_expression", "")
        self.case_sensitive = config.get("case_sensitive_filtering", False)

    def apply_filters(self, df: pd.DataFrame) -> pd.DataFrame:
        """Apply filters to DataFrame using pandas operations."""
        if not self.enable_filtering:
            return df

        original_count = len(df)

        # Custom filter expression takes precedence
        if self.custom_filter_expression:
            try:
                df = df.query(self.custom_filter_expression)
                return df
            except Exception as e:
                logger.warning(f"Custom filter expression failed: {e}")
                return df

        # Build filter mask
        mask = pd.Series([True] * len(df))

        # Apply first filter
        if self.filter_field and self.filter_field in df.columns:
            mask = mask & self._build_filter_mask(df, self.filter_field, self.filter_operator, self.filter_value)

        # Apply second filter
        if self.filter_field_2 and self.filter_field_2 in df.columns:
            mask2 = self._build_filter_mask(df, self.filter_field_2, self.filter_operator_2, self.filter_value_2)

            if self.filter_logic == "AND":
                mask = mask & mask2
            else:  # OR
                mask = mask | mask2

        filtered_df = df[mask].copy()

        # Log filtering statistics
        filtered_count = len(filtered_df)
        logger.info(f"Filtering: {original_count} -> {filtered_count} records ({original_count - filtered_count} filtered out)")

        return filtered_df

    def _build_filter_mask(self, df: pd.DataFrame, field: str, operator: str, value: str) -> pd.Series:
        """Build pandas mask for a single filter condition."""
        if field not in df.columns:
            return pd.Series([True] * len(df))

        column = df[field]

        # Handle null checks
        if operator in ["is_null", "is_not_null"]:
            is_null = column.isna() | (column == "")
            return is_null if operator == "is_null" else ~is_null

        # Handle string operations
        if column.dtype == 'object':
            if not self.case_sensitive:
                column = column.str.lower()
                value = value.lower()

            if operator == "equals":
                return column == value
            elif operator == "not_equals":
                return column != value
            elif operator == "contains":
                return column.str.contains(value, na=False)
            elif operator == "not_contains":
                return ~column.str.contains(value, na=False)
            elif operator == "starts_with":
                return column.str.startswith(value)
            elif operator == "ends_with":
                return column.str.endswith(value)
            elif operator == "in":
                value_list = [v.strip() for v in value.split(",")]
                if not self.case_sensitive:
                    value_list = [v.lower() for v in value_list]
                return column.isin(value_list)
            elif operator == "not_in":
                value_list = [v.strip() for v in value.split(",")]
                if not self.case_sensitive:
                    value_list = [v.lower() for v in value_list]
                return ~column.isin(value_list)

        # Handle numeric operations
        if pd.api.types.is_numeric_dtype(column):
            try:
                numeric_value = float(value)
                if operator == "equals":
                    return column == numeric_value
                elif operator == "not_equals":
                    return column != numeric_value
                elif operator == "greater_than":
                    return column > numeric_value
                elif operator == "less_than":
                    return column < numeric_value
            except ValueError:
                return pd.Series([False] * len(df))

        return pd.Series([True] * len(df))
```

#### DataFrame-Based Field Mapping
```python
class DataFrameFieldMapper:
    """Map fields using DataFrame operations."""

    def __init__(self, config: dict):
        self.field_mapping_mode = config.get("field_mapping_mode", "auto")
        self.enable_auto_mapping = config.get("enable_auto_mapping", True)
        self.prompt_field = config.get("prompt_field", "")
        self.completion_field = config.get("completion_field", "")
        self.category_field = config.get("category_field", "")
        self.source_field = config.get("source_field", "")
        self.dataset_name_field = config.get("dataset_name_field", "")
        self.document_type_field = config.get("document_type_field", "")
        self.metadata_fields = config.get("metadata_fields", "")

    def map_fields(self, df: pd.DataFrame) -> pd.DataFrame:
        """Map input fields to NeMo standard fields using DataFrame operations."""
        if self.field_mapping_mode == "explicit":
            return self._apply_explicit_mapping(df)
        else:
            return self._apply_auto_mapping(df)

    def _apply_explicit_mapping(self, df: pd.DataFrame) -> pd.DataFrame:
        """Apply explicit field mappings."""
        mapping = {}

        if self.prompt_field and self.prompt_field in df.columns:
            mapping["prompt"] = self.prompt_field
        if self.completion_field and self.completion_field in df.columns:
            mapping["completion"] = self.completion_field
        if self.category_field and self.category_field in df.columns:
            mapping["category"] = self.category_field
        if self.source_field and self.source_field in df.columns:
            mapping["source"] = self.source_field

        return self._rename_columns(df, mapping)

    def _apply_auto_mapping(self, df: pd.DataFrame) -> pd.DataFrame:
        """Apply automatic field mapping."""
        if not self.enable_auto_mapping:
            return df

        auto_mapper = AutoFieldMapper()
        mappings = auto_mapper.detect_field_mappings(df.to_dict('records'))

        # Apply auto-detected mappings
        return self._rename_columns(df, mappings)

    def _rename_columns(self, df: pd.DataFrame, mapping: Dict[str, str]) -> pd.DataFrame:
        """Rename DataFrame columns based on mapping."""
        if not mapping:
            return df

        # Create reverse mapping for rename
        rename_dict = {v: k for k, v in mapping.items()}

        # Rename columns that exist
        existing_columns = {col: rename_dict[col] for col in rename_dict if col in df.columns}

        if existing_columns:
            df = df.rename(columns=existing_columns)

        return df
```

#### DataFrame-Based Format Transformation
```python
class DataFrameFormatTransformer:
    """Transform DataFrame to different NeMo formats."""

    def __init__(self, config: dict):
        self.model_type = config.get("model_type", "auto")
        self.system_message = config.get("system_message", "You are a helpful assistant.")
        self.user_role_field = config.get("user_role_field", "user")
        self.assistant_role_field = config.get("assistant_role_field", "assistant")
        self.system_role_field = config.get("system_role_field", "system")
        self.default_category = config.get("default_category", "Generation")

    def transform_format(self, df: pd.DataFrame) -> pd.DataFrame:
        """Transform DataFrame to target NeMo format."""
        if self.model_type == "auto":
            model_type = self._detect_model_type(df)
        else:
            model_type = self.model_type

        if model_type == "chat":
            return self._transform_to_chat_format(df)
        else:
            return self._transform_to_completion_format(df)

    def _detect_model_type(self, df: pd.DataFrame) -> str:
        """Detect model type based on DataFrame structure."""
        # Check for messages column (chat format)
        if "messages" in df.columns:
            return "chat"

        # Check for prompt/completion columns (completion format)
        if "prompt" in df.columns and "completion" in df.columns:
            return "completion"

        # Default to completion format
        return "completion"

    def _transform_to_chat_format(self, df: pd.DataFrame) -> pd.DataFrame:
        """Transform to chat format with messages column."""
        def create_messages(row):
            messages = []

            # Add system message if not present
            if self.system_role_field not in row or pd.isna(row[self.system_role_field]):
                messages.append({"role": "system", "content": self.system_message})

            # Add user message
            if "prompt" in row and not pd.isna(row["prompt"]):
                messages.append({"role": "user", "content": str(row["prompt"])})

            # Add assistant message
            if "completion" in row and not pd.isna(row["completion"]):
                messages.append({"role": "assistant", "content": str(row["completion"])})

            return messages

        df = df.copy()
        df["messages"] = df.apply(create_messages, axis=1)

        # Remove individual prompt/completion columns
        columns_to_drop = ["prompt", "completion"]
        df = df.drop(columns=[col for col in columns_to_drop if col in df.columns])

        return df

    def _transform_to_completion_format(self, df: pd.DataFrame) -> pd.DataFrame:
        """Transform to completion format with prompt/completion columns."""
        # Ensure required columns exist
        if "prompt" not in df.columns:
            df["prompt"] = ""
        if "completion" not in df.columns:
            df["completion"] = ""

        # Fill missing categories
        if "category" in df.columns:
            df["category"] = df["category"].fillna(self.default_category)
        else:
            df["category"] = self.default_category

        return df
```

#### Main Processing Method
```python
def process_data(self, input_data: Union[List[Data], pd.DataFrame]) -> Union[List[Data], pd.DataFrame]:
    """Main processing method using DataFrame operations."""
    # Convert to DataFrame
    processor = DataFrameProcessor(self.config)
    df = processor.convert_to_dataframe(input_data)

    # Apply filtering
    if self.config.get("enable_filtering", False):
        filter_processor = DataFrameFilter(self.config)
        df = filter_processor.apply_filters(df)

    # Apply field mapping
    mapper = DataFrameFieldMapper(self.config)
    df = mapper.map_fields(df)

    # Transform format
    transformer = DataFrameFormatTransformer(self.config)
    df = transformer.transform_format(df)

    # Convert back to requested format
    return processor.convert_from_dataframe(df)
```

This detailed implementation plan provides a comprehensive framework for building the refactored NeMo dataset components with all the requested features: configurable splits, partial success handling, detailed metadata statistics, and support for both chat and completion model formats as documented in the [NeMo Microservices documentation](https://docs.nvidia.com/nemo/microservices/latest/fine-tune/tutorials/format-training-dataset.html).
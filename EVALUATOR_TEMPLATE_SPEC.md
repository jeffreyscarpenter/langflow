# NeMo Evaluator Template System Specification

## Overview

This document specifies the template-based approach for NeMo evaluation configurations, designed to work within the constraints of the NeMo Microservices API while providing a flexible and user-friendly interface.

## Problem Statement

The NeMo Microservices API requires datasets to be baked into evaluation configurations at creation time, which creates several limitations:

1. **Configs become dataset-specific**: A config created for dataset A cannot be reused for dataset B
2. **No runtime dataset selection**: Users cannot choose which dataset to use when running an evaluation
3. **Poor separation of concerns**: Configs should define *how* to evaluate, not *what* to evaluate
4. **Config proliferation**: Each dataset requires its own config, leading to config explosion

## Solution: Template-Based Configuration System

### Core Concept

The template system creates a two-layer approach:
1. **Template Configs**: Define evaluation logic without specific datasets
2. **Runtime Configs**: Created on-demand by copying templates and injecting specific datasets

### Evaluation Type Classification

#### Template-Based Evaluation Types
These evaluation types require custom datasets and use the template system:

- **Custom Evaluation** (Chat/Completion Tasks)
- **Similarity Metrics**
- **LLM As A Judge**
- **BigCode Evaluation Harness**

#### Direct Config Evaluation Types
These evaluation types use built-in datasets and bypass the template system:

- **LM Evaluation Harness** (uses academic benchmarks)

## Implementation Strategy

### 1. Config Creation with Optional Dataset

When users create evaluation configs via dialog, they can choose between template and direct config creation:

```python
async def create_config(self, config_name: str, evaluation_type: str, params: dict, dataset_url: str = None) -> str:
    """Create config with optional dataset - defaults to template if no dataset provided"""

    if dataset_url:
        # Direct config creation with specific dataset
        config = {
            "name": config_name,
            "type": evaluation_type,
            "params": params,
            "tasks": {
                "main_task": {
                    "dataset": {"files_url": dataset_url}
                }
            }
        }
        return await self.nemo_client.create_evaluation_config(config)
    else:
        # Template config creation (default behavior)
        return await self.create_template_config(config_name, evaluation_type, params)

async def create_template_config(self, config_name: str, evaluation_type: str, params: dict) -> str:
    """Create template config with user-provided name"""

    # Generate template config with placeholder dataset
    template_config = {
        "name": f"{config_name}_template",
        "type": evaluation_type,
        "params": params,
        "tasks": {
            "template_task": {
                "dataset": {
                    "files_url": "hf://datasets/template/placeholder"
                }
            }
        }
    }

    # Create via NeMo API
    config_response = await self.nemo_client.create_evaluation_config(template_config)
    return config_response["id"]
```

### 2. Runtime Config Creation

When running evaluations with datasets:

```python
async def create_runtime_config(self, template_config_id: str, dataset_url: str, config_name: str) -> str:
    """Create runtime config from template with specific dataset"""

    # Fetch template config
    template_config = await self.nemo_client.get_evaluation_config(template_config_id)

    # Generate runtime config name
    dataset_id = self.extract_dataset_id(dataset_url)
    runtime_name = f"{config_name}_dataset_{dataset_id}"

    # Check if runtime config already exists
    existing_config = await self.find_runtime_config(template_config_id, dataset_id)
    if existing_config:
        return existing_config["id"]

    # Create runtime config by copying template and replacing dataset
    runtime_config = self.copy_template_config(template_config, dataset_url, runtime_name)

    # Create via NeMo API
    config_response = await self.nemo_client.create_evaluation_config(runtime_config)
    return config_response["id"]
```

### 3. Naming Conventions

#### Template Config Names
- Format: `{user_config_name}_template`
- Examples: `my_custom_eval_template`, `code_review_template`

#### Direct Config Names
- Format: `{user_config_name}_dataset_{dataset_id}`
- Examples: `my_custom_eval_dataset_code_samples_v1`, `code_review_dataset_reviews_v2`

#### Runtime Config Names
- Format: `{user_config_name}_dataset_{dataset_id}`
- Examples: `my_custom_eval_dataset_code_samples_v1`, `code_review_dataset_reviews_v2`

#### Dataset ID Extraction
```python
def extract_dataset_id(self, dataset_url: str) -> str:
    """Extract dataset ID from HF URL"""
    # hf://datasets/default/my-dataset/v1 -> my-dataset_v1
    # hf://datasets/namespace/dataset-name/version -> dataset-name_version
    parts = dataset_url.split('/')
    if len(parts) >= 4:
        dataset_name = parts[-2]
        version = parts[-1]
        return f"{dataset_name}_{version}"
    return "unknown_dataset"
```

### 4. Config Deduplication

To prevent config explosion, the system:

1. **Checks for existing runtime configs** before creating new ones
2. **Uses deterministic naming** based on template + dataset combination
3. **Optionally cleans up runtime configs** after job completion

```python
async def find_runtime_config(self, template_config_id: str, dataset_id: str) -> dict | None:
    """Find existing runtime config for template + dataset combination"""

    # Get template config to extract base name
    template_config = await self.nemo_client.get_evaluation_config(template_config_id)
    base_name = template_config["name"].replace("_template", "")

    # Search for runtime config with matching pattern
    expected_name = f"{base_name}_dataset_{dataset_id}"

    # List all configs and find matching one
    configs = await self.nemo_client.list_evaluation_configs()
    for config in configs:
        if config["name"] == expected_name:
            return config

    return None
```

### 5. Optional Cleanup Strategy

After evaluation job completion:

```python
async def cleanup_runtime_config(self, runtime_config_id: str):
    """Clean up runtime config after job completion"""

    # Only cleanup if job completed successfully
    job_status = await self.get_job_status(job_id)
    if job_status == "completed":
        await self.nemo_client.delete_evaluation_config(runtime_config_id)
```

## User Interface Design

### 1. Config Creation Dialog

Users create configs with optional dataset selection:

- **Config Name**: User-provided name for the configuration
- **Evaluation Type**: Selection from available types
- **Dataset Selection (Optional)**: Choose to create template (no dataset) or direct config (with dataset)
- **Evaluation Parameters**: Type-specific parameters

**Note**: Implementation details for the dialog interface are flexible and can be adjusted based on technical constraints and user experience considerations. The core requirement is providing users the choice between template and direct config creation.

### 2. Evaluation Job Creation

When running evaluations:
1. **Template Selection**: Choose from user's template configs
2. **Dataset Selection**: Choose dataset from NeMo Data Store or component inputs
3. **Runtime Config Creation**: System automatically creates runtime config
4. **Job Execution**: Run evaluation with runtime config

### 3. Config Management (managed separately from evaluator langflow component)

Users can:
- **View Template Configs**: See all their template configurations
- **View Direct Configs**: See all direct configs created with specific datasets
- **View Runtime Configs**: See all runtime configs created from templates
- **Delete Templates**: Remove template configs (with confirmation)
- **Delete Direct Configs**: Remove direct configs (with confirmation)
- **Cleanup Runtime Configs**: Manually cleanup runtime configs

## Technical Implementation Details

### Component Structure

```python
class NvidiaEvaluatorComponent(Component):
    """NeMo Evaluator with template system support"""

    # Inputs
    config: DropdownInput  # Template configs
    dataset: HandleInput   # Dataset from NeMo Dataset Creator
    existing_dataset: DropdownInput  # Existing datasets (reuses existing fetch_existing_datasets)

    # Methods
    async def create_config(self, config_name: str, evaluation_type: str, params: dict, dataset_url: str = None) -> str
    async def create_template_config(self, ...) -> str
    async def create_runtime_config(self, ...) -> str
    async def find_runtime_config(self, ...) -> dict | None
    async def cleanup_runtime_config(self, ...) -> None
    def extract_dataset_id(self, ...) -> str

    # Existing methods (reused)
    async def fetch_existing_datasets(self, ...) -> tuple[list[str], list[dict[str, Any]]]
```

### Dialog Structure

```python
@dataclass
class ConfigCreationInput:
    """Dialog for creating configs with optional dataset selection"""
    "01_config_name": StrInput(
        display_name="Config Name",
        info="Name for this evaluation configuration",
        required=True,
    ),
    "02_evaluation_type": DropdownInput(
        display_name="Evaluation Type",
        options=["Custom Evaluation", "Similarity Metrics", "LLM As A Judge", "BigCode Evaluation Harness"],
        required=True,
    ),
    "03_dataset_selection": DropdownInput(
        display_name="Dataset Selection",
        options=["Create template (no dataset)", "Use existing dataset", "Use input dataset"],
        info="Choose whether to create a template or direct config",
        required=False,
    ),
    "04_existing_dataset": DropdownInput(
        display_name="Existing Dataset",
        options=[],  # Populated via fetch_existing_datasets() method
        info="Choose from your existing datasets in NeMo Data Store",
        required=False,
        refresh_button=True,
    ),
    # ... type-specific fields
```

**Note**: The dataset selection feature reuses the existing `fetch_existing_datasets()` method from the current `NvidiaEvaluatorComponent` implementation. This method already provides:
- Dataset listing with pagination
- Dataset metadata (size, records, timestamps)
- Error handling for NeMo API calls
- Authentication header management

### Error Handling

1. **Template Not Found**: Graceful fallback to direct config creation
2. **Dataset Not Available**: Clear error message with dataset requirements
3. **Config Creation Failed**: Retry logic with fallback options
4. **Cleanup Failed**: Log warning but don't fail job

## Benefits

### 1. **User Experience**
- **Flexible workflow**: Choose between template and direct config creation
- **Intuitive defaults**: Template creation promoted as best practice
- **Reduced complexity**: Users don't need to understand NeMo API constraints
- **Flexible dataset selection**: Choose datasets at runtime or creation time

### 2. **System Maintainability**
- **Clean separation**: Templates define logic, runtime configs define data
- **Predictable naming**: Deterministic config naming prevents confusion
- **Efficient storage**: Deduplication prevents config explosion
- **Backward compatibility**: Supports existing workflows

### 3. **API Compatibility**
- **Works within constraints**: Leverages existing NeMo API
- **No API changes required**: Uses standard config creation/management
- **Reuses existing functionality**: Dataset listing uses existing `fetch_existing_datasets()` method
- **Future-proof**: Can adapt to API changes

## Limitations and Considerations

### 1. **API Dependencies**
- Relies on NeMo API stability
- Requires config creation permissions
- May be rate-limited by API calls

### 2. **Storage Considerations**
- Runtime configs consume storage quota
- Direct configs may lead to config proliferation
- Cleanup strategy affects storage efficiency
- Config listing may be slow with many configs

### 3. **Error Scenarios**
- Template config deletion affects runtime configs
- Direct config creation may fail if dataset unavailable
- Network failures during config creation
- Dataset URL changes invalidate runtime configs

### 4. **User Experience Considerations**
- Dialog complexity may increase with optional dataset selection
- Users may need guidance on template vs direct config benefits
- Implementation flexibility needed for dynamic dialog behavior

## Future Enhancements

### 1. **Advanced Template Features**
- **Template versioning**: Track template changes over time
- **Template sharing**: Share templates between users
- **Template validation**: Validate templates before creation

### 2. **Optimization Strategies**
- **Config caching**: Cache frequently used runtime configs
- **Batch operations**: Create multiple runtime configs efficiently
- **Background cleanup**: Automated cleanup of old runtime configs

### 3. **User Experience Improvements**
- **Template preview**: Show what template will create
- **Dataset compatibility**: Validate dataset compatibility with templates
- **Usage analytics**: Track template usage and effectiveness

## Conclusion

The hybrid template-direct approach provides a robust solution to the NeMo API constraints while maintaining excellent user experience and flexibility. By supporting both template creation (for reusability) and direct config creation (for simplicity), the system accommodates different user workflows while promoting best practices.

The implementation strategy balances simplicity with functionality, providing users with clear choices while handling the complexity of config management behind the scenes. The flexible dialog implementation allows for technical constraints while maintaining the core user experience goals.
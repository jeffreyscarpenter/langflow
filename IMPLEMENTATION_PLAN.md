# NeMo Evaluator Template System Implementation Plan

## Overview

This document outlines the implementation plan for the template-based evaluation configuration system as specified in `EVALUATOR_TEMPLATE_SPEC.md`. The implementation is divided into phases, with Phase 1 focusing on the core evaluator component functionality.

## Phase 1: Core Evaluator Component Implementation

### 1.1 Backend Component Updates

#### 1.1.1 Update NvidiaEvaluatorComponent Class
**File**: `src/backend/base/langflow/components/nvidia/nvidia_evaluator.py`

**Changes**:
- Add new `create_config()` method with optional dataset parameter
- Add `create_template_config()` method for template creation
- Add `create_runtime_config()` method for runtime config creation
- Add `find_runtime_config()` method for deduplication
- Add `extract_dataset_id()` method for naming conventions
- Update dialog structure to support optional dataset selection

**Key Methods to Implement**:
```python
async def create_config(self, config_name: str, evaluation_type: str, params: dict, dataset_url: str = None) -> str
async def create_template_config(self, config_name: str, evaluation_type: str, params: dict) -> str
async def create_runtime_config(self, template_config_id: str, dataset_url: str, config_name: str) -> str
async def find_runtime_config(self, template_config_id: str, dataset_id: str) -> dict | None
def extract_dataset_id(self, dataset_url: str) -> str
```

#### 1.1.2 Update Dialog Structure
**File**: `src/backend/base/langflow/components/nvidia/nvidia_evaluator.py`

**Changes**:
- Update `DynamicEvaluationConfigInput` to include dataset selection
- Add `existing_dataset` field with refresh functionality
- Implement conditional field visibility based on dataset selection
- Add validation for dataset selection logic

**New Dialog Fields**:
```python
"03_dataset_selection": DropdownInput(
    display_name="Dataset Selection",
    options=["Create template (no dataset)", "Use existing dataset", "Use input dataset"],
    required=False,
),
"04_existing_dataset": DropdownInput(
    display_name="Existing Dataset",
    options=[],  # Populated via existing fetch_existing_datasets()
    required=False,
    refresh_button=True,
),
```

#### 1.1.3 Implement Config Creation Logic
**File**: `src/backend/base/langflow/components/nvidia/nvidia_evaluator.py`

**Changes**:
- Modify `_create_new_evaluation_config()` to handle template vs direct config creation
- Add dataset URL extraction from existing dataset selection
- Implement naming conventions for template and direct configs
- Add error handling for dataset availability

### 1.2 Frontend Component Updates

#### 1.2.1 Update Dialog Interface
**File**: `src/frontend/src/components/nemo/` (relevant dialog components)

**Changes**:
- Add dataset selection dropdown to config creation dialog
- Implement conditional field visibility
- Add refresh functionality for existing datasets
- Update validation logic for dataset selection

#### 1.2.2 Update Config Management UI
**File**: `src/frontend/src/components/nemo/` (config management components)

**Changes**:
- Update config listing to distinguish between template and direct configs
- Add visual indicators for config types
- Implement config type filtering
- Update config creation flow

### 1.3 Testing Implementation

#### 1.3.1 Unit Tests
**File**: `src/backend/tests/unit/components/nvidia/test_nvidia_evaluator.py`

**New Test Cases**:
- Test template config creation
- Test direct config creation with dataset
- Test runtime config creation from template
- Test config deduplication logic
- Test dataset ID extraction
- Test error handling for missing datasets

**Test Structure**:
```python
@pytest.mark.asyncio
async def test_create_template_config():
    """Test template config creation without dataset"""

@pytest.mark.asyncio
async def test_create_direct_config():
    """Test direct config creation with dataset"""

@pytest.mark.asyncio
async def test_create_runtime_config():
    """Test runtime config creation from template"""

@pytest.mark.asyncio
async def test_find_runtime_config():
    """Test runtime config deduplication"""

@pytest.mark.asyncio
async def test_extract_dataset_id():
    """Test dataset ID extraction from URLs"""
```

#### 1.3.2 Integration Tests
**File**: `src/backend/tests/integration/test_nemo_evaluator.py`

**New Test Cases**:
- Test end-to-end template creation workflow
- Test end-to-end direct config creation workflow
- Test config reuse with different datasets
- Test error scenarios and recovery

### 1.4 Documentation Updates

#### 1.4.1 Component Documentation
**File**: `src/backend/base/langflow/components/nvidia/nvidia_evaluator.py`

**Changes**:
- Update docstrings for new methods
- Add examples for template vs direct config usage
- Document naming conventions
- Add error handling documentation

#### 1.4.2 User Documentation
**File**: `docs/docs/Components/bundles-nvidia.mdx`

**Changes**:
- Add template system explanation
- Update configuration creation instructions
- Add examples for different use cases
- Document best practices

## Phase 2: Advanced Features (Future Implementation)

### 2.1 Config Lifecycle Management
**Note**: This will be implemented separately by a colleague and is not part of the evaluator component.

**Features**:
- Template config management UI
- Runtime config cleanup strategies
- Config versioning and history
- Config sharing between users
- Config validation and testing

### 2.2 Template System Enhancements
**Features**:
- Template versioning
- Template sharing and collaboration
- Advanced template validation
- Template preview functionality
- Template usage analytics

### 2.3 Performance Optimizations
**Features**:
- Config caching strategies
- Batch config operations
- Background cleanup processes
- Optimized dataset listing

## Implementation Timeline

### Week 1: Backend Core Implementation
- [ ] Implement `create_config()` method with optional dataset
- [ ] Implement `create_template_config()` method
- [ ] Implement `create_runtime_config()` method
- [ ] Implement `find_runtime_config()` method
- [ ] Implement `extract_dataset_id()` method
- [ ] Update dialog structure with dataset selection

### Week 2: Frontend Integration
- [ ] Update dialog interface for dataset selection
- [ ] Implement conditional field visibility
- [ ] Add refresh functionality for existing datasets
- [ ] Update config management UI
- [ ] Add visual indicators for config types

### Week 3: Testing and Documentation
- [ ] Write unit tests for all new methods
- [ ] Write integration tests for workflows
- [ ] Update component documentation
- [ ] Update user documentation
- [ ] Perform end-to-end testing

### Week 4: Refinement and Deployment
- [ ] Address feedback and bug fixes
- [ ] Performance testing and optimization
- [ ] Security review
- [ ] Deployment preparation
- [ ] User acceptance testing

## Technical Considerations

### 1. Backward Compatibility
- Ensure existing configs continue to work
- Maintain compatibility with current API usage
- Provide migration path for existing users

### 2. Error Handling
- Handle dataset unavailability gracefully
- Provide clear error messages for users
- Implement retry logic for API failures
- Log errors for debugging

### 3. Performance
- Optimize dataset listing for large datasets
- Implement caching for frequently accessed configs
- Minimize API calls during config creation
- Handle pagination for dataset lists

### 4. Security
- Validate dataset access permissions
- Sanitize config names and parameters
- Implement proper authentication for config operations
- Audit config creation and modification

## Success Criteria

### Phase 1 Success Criteria
- [ ] Users can create template configs without datasets
- [ ] Users can create direct configs with specific datasets
- [ ] Template configs can be reused with different datasets
- [ ] Config naming follows specified conventions
- [ ] Existing functionality remains unchanged
- [ ] All tests pass
- [ ] Documentation is complete and accurate

### Quality Gates
- [ ] Code review completed
- [ ] All unit tests passing
- [ ] Integration tests passing
- [ ] Performance benchmarks met
- [ ] Security review completed
- [ ] User acceptance testing completed

## Risk Mitigation

### 1. Technical Risks
- **Risk**: NeMo API changes affecting implementation
- **Mitigation**: Use existing API patterns and implement defensive programming

- **Risk**: Dialog complexity causing user confusion
- **Mitigation**: Implement progressive disclosure and clear messaging

- **Risk**: Performance issues with large dataset lists
- **Mitigation**: Implement pagination and caching strategies

### 2. User Experience Risks
- **Risk**: Users not understanding template vs direct configs
- **Mitigation**: Provide clear documentation and examples

- **Risk**: Config proliferation due to direct config usage
- **Mitigation**: Promote template usage with clear benefits messaging

### 3. Integration Risks
- **Risk**: Conflicts with existing config management
- **Mitigation**: Maintain backward compatibility and gradual rollout

## Dependencies

### Internal Dependencies
- Existing `fetch_existing_datasets()` method
- Current NeMo API client implementation
- Existing dialog system infrastructure
- Current testing framework

### External Dependencies
- NeMo Microservices API stability
- Frontend component library compatibility
- Documentation system updates

## Conclusion

This implementation plan provides a structured approach to implementing the template system while maintaining focus on the core evaluator component. Phase 1 delivers the essential functionality for template and direct config creation, while Phase 2 outlines future enhancements that can be implemented separately.

The plan emphasizes backward compatibility, comprehensive testing, and user experience considerations to ensure a smooth transition to the new template system.
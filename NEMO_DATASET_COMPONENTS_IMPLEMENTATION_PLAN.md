# NeMo Dataset Components Implementation Plan

## Overview

This plan outlines the implementation of the refactored NeMo dataset components as specified in `NEMO_DATASET_COMPONENTS_SPEC.md`. The implementation will create two new components:

1. **NeMo Data Preparation Component** - Handles data transformation, filtering, and field mapping
2. **NeMo Dataset Uploader Component** - Manages dataset creation and upload with configurable splits

## Phase 1: Foundation and Infrastructure (Week 1)

### 1.1 Project Setup and Dependencies
- [ ] **Task 1.1.1**: Create new component directory structure
  ```
  src/backend/base/langflow/components/nvidia/
  ├── nvidia_data_preparation.py
  ├── nvidia_dataset_uploader.py
  └── __init__.py
  ```
- [ ] **Task 1.1.2**: Create shared utilities module
  ```
  src/backend/base/langflow/components/nvidia/utils/
  ├── __init__.py
  ├── dataframe_processor.py
  ├── field_mapper.py
  ├── filter_processor.py
  └── format_transformer.py
  ```

### 1.3 Base Classes and Interfaces
- [ ] **Task 1.3.1**: Create `DataFrameProcessor` class
  - Input/output format conversion
  - DataFrame validation
  - Error handling for malformed data
- [ ] **Task 1.3.2**: Create `AutoFieldMapper` class
  - Field detection logic
  - Confidence scoring
  - Common field name mappings
- [ ] **Task 1.3.3**: Create `RecordFilter` class
  - Filter condition evaluation
  - Compound filter logic
  - Custom expression support

### 1.4 Testing Infrastructure
- [ ] **Task 1.4.1**: Create test directory structure
  ```
  src/backend/tests/unit/components/nvidia/
  ├── test_data_preparation.py
  ├── test_dataset_uploader.py
  └── test_utils/
      ├── test_dataframe_processor.py
      ├── test_field_mapper.py
      ├── test_filter_processor.py
      └── test_format_transformer.py
  ```
- [ ] **Task 1.4.2**: Create test data fixtures
  - Sample list[Data] inputs
  - Sample DataFrame inputs
  - Various field mapping scenarios
  - Filter test cases

## Phase 2: Data Preparation Component Implementation (Week 2)

### 2.1 Core Component Structure
- [ ] **Task 2.1.1**: Create `NvidiaDataPreparationComponent` class
  ```python
  class NvidiaDataPreparationComponent(Component):
      display_name = "NeMo Data Preparation"
      description = "Transform and prepare data for NeMo dataset creation"
      icon = "NVIDIA"
      name = "NVIDIANeMoDataPreparation"
      beta = True
  ```

### 2.2 Input Configuration
- [ ] **Task 2.1.2**: Implement all input fields from specification
  - Data input (HandleInput with input_types=["Data"])
  - Model type configuration
  - Field mapping configuration
  - Filtering configuration
  - Metadata configuration

### 2.3 Processing Logic Implementation
- [ ] **Task 2.1.3**: Implement `DataFrameProcessor`
  - Convert list[Data] to DataFrame
  - Convert DataFrame to list[Data]
  - Handle nested data structures
  - Error handling for malformed inputs
- [ ] **Task 2.1.4**: Implement `DataFrameFilter`
  - Pandas-based filtering operations
  - Custom query expression support
  - Compound filter logic (AND/OR)
  - Case-sensitive/insensitive filtering
- [ ] **Task 2.1.5**: Implement `DataFrameFieldMapper`
  - Automatic field detection
  - Explicit field mapping
  - Column renaming operations
  - Confidence scoring
- [ ] **Task 2.1.6**: Implement `DataFrameFormatTransformer`
  - Chat format transformation
  - Completion format transformation
  - Model type auto-detection
  - Messages column creation

### 2.4 Output Methods
- [ ] **Task 2.1.7**: Implement output methods
  - `prepare_data()` method returning list[Data]
  - `prepare_dataframe()` method returning DataFrame
  - `get_preparation_stats()` method returning statistics
  - Consistent processing logic across all methods

### 2.5 Testing
- [ ] **Task 2.1.8**: Unit tests for Data Preparation Component
  - Input format conversion tests
  - Filtering operation tests
  - Field mapping tests
  - Format transformation tests
  - Error handling tests

## Phase 3: Dataset Uploader Component Implementation (Week 3)

### 3.1 Core Component Structure
- [ ] **Task 3.1.1**: Create `NvidiaDatasetUploaderComponent` class
  ```python
  class NvidiaDatasetUploaderComponent(Component):
      display_name = "NeMo Dataset Uploader"
      description = "Upload prepared data to NeMo Data Store"
      icon = "NVIDIA"
      name = "NVIDIANeMoDatasetUploader"
      beta = True
  ```

### 3.2 Input Configuration
- [ ] **Task 3.1.2**: Implement all input fields from specification
  - Authentication and connection settings
  - Dataset configuration
  - Split configuration
  - Upload settings
  - Partial success handling

### 3.3 Upload Logic Implementation
- [ ] **Task 3.1.3**: Implement `SplitConfiguration`
  - Training/validation split logic
  - Minimum record requirements
  - Split ratio validation
  - Random seed handling
- [ ] **Task 3.1.4**: Implement `PartialSuccessHandler`
  - Record validation
  - Error collection and reporting
  - Discard logic for invalid records
  - Success/failure tracking
- [ ] **Task 3.1.5**: Implement `UploadStatistics`
  - Record counting
  - Split statistics
  - Error reporting
  - Performance metrics

### 3.4 NeMo Integration
- [ ] **Task 3.1.6**: Implement dataset creation
  - NeMo client integration
  - HuggingFace repo creation
  - Dataset registration
  - File upload handling
- [ ] **Task 3.1.7**: Implement file processing
  - Training file creation
  - Validation file creation
  - Evaluation file creation
  - Metadata preservation

### 3.5 Testing
- [ ] **Task 3.1.8**: Unit tests for Dataset Uploader Component
  - Split configuration tests
  - Upload logic tests
  - Error handling tests
  - Statistics collection tests

## Phase 4: Integration and Advanced Features (Week 4)

### 4.1 Component Integration
- [ ] **Task 4.1.1**: Test component chaining
  - Data Preparation → Dataset Uploader flow
  - Multiple data sources → Preparation → Uploader
  - Error propagation between components
- [ ] **Task 4.1.2**: Implement shared utilities
  - Common NeMo client handling
  - Authentication management
  - Error handling patterns

### 4.2 Advanced Features
- [ ] **Task 4.1.3**: Implement custom filter expressions
  - Safe evaluation environment
  - Pandas query syntax support
  - Error handling for malformed expressions
- [ ] **Task 4.1.4**: Implement metadata preservation
  - Custom field mapping
  - Dataset name and document type handling
  - Additional metadata fields

### 4.3 Performance Optimization
- [ ] **Task 4.1.5**: Optimize DataFrame operations
  - Vectorized operations for filtering
  - Efficient field mapping
  - Memory usage optimization
- [ ] **Task 4.1.6**: Implement chunked processing
  - Large dataset handling
  - Memory-efficient processing
  - Progress tracking

### 4.4 Documentation and Examples
- [ ] **Task 4.1.7**: Create component documentation
  - Usage examples
  - Configuration guides
  - Troubleshooting tips
- [ ] **Task 4.1.8**: Create flow examples
  - Basic data preparation flow
  - Advanced filtering flow
  - Multi-source data flow

## Phase 5: Testing and Validation (Week 5)

### 5.1 Comprehensive Testing
- [ ] **Task 5.1.1**: Integration tests
  - End-to-end flow testing
  - Component interaction testing
  - Error scenario testing
- [ ] **Task 5.1.2**: Performance testing
  - Large dataset processing
  - Memory usage testing
  - Processing speed validation
- [ ] **Task 5.1.3**: Edge case testing
  - Empty datasets
  - Malformed data
  - Network failures
  - Authentication errors

### 5.2 Real-world Validation
- [ ] **Task 5.1.4**: Test with actual data sources
  - Elasticsearch data
  - AstraDB data
  - File loader data
  - Custom JSON data
- [ ] **Task 5.1.5**: Validate NeMo integration
  - Dataset creation in NeMo
  - File upload to HuggingFace
  - Dataset registration
  - Compatibility with existing NeMo components

### 5.3 Code Quality
- [ ] **Task 5.1.6**: Code review and refactoring
  - Code style compliance
  - Performance optimization
  - Error handling improvement
- [ ] **Task 5.1.7**: Documentation review
  - API documentation
  - User guides
  - Developer documentation

## Phase 6: Deployment and Integration (Week 6)

### 6.1 Frontend Integration
- [ ] **Task 6.1.1**: Add component icons
  - Data Preparation component icon
  - Dataset Uploader component icon
  - Icon consistency with existing NeMo components
- [ ] **Task 6.1.2**: Update frontend component registry
  - Register new components
  - Update component metadata
  - Test component loading

### 6.2 Backend Integration
- [ ] **Task 6.1.3**: Update component imports
  - Add to `__init__.py`
  - Update component discovery
  - Test component registration
- [ ] **Task 6.1.4**: Update API endpoints
  - Add component endpoints if needed
  - Update component metadata endpoints
  - Test API integration

### 6.3 Documentation and Release
- [ ] **Task 6.1.5**: Update project documentation
  - README updates
  - Component documentation
  - Example flows
- [ ] **Task 6.1.6**: Create release notes
  - Feature descriptions
  - Breaking changes
  - Migration guide

## Implementation Dependencies

### Technical Dependencies
- **pandas**: For DataFrame operations and filtering
- **nemo_microservices**: For NeMo API integration
- **huggingface_hub**: For HuggingFace repository management
- **langflow**: For component framework

### Component Dependencies
- **Data Preparation Component**: Independent, can be used standalone
- **Dataset Uploader Component**: Depends on Data Preparation for input format
- **Existing NeMo Components**: Integration with evaluator and customizer

### Testing Dependencies
- **pytest**: For unit and integration testing
- **pytest-asyncio**: For async test support
- **pytest-mock**: For mocking external dependencies

## Risk Assessment and Mitigation

### High-Risk Items
1. **NeMo API Changes**: Monitor NeMo microservices updates
2. **Performance with Large Datasets**: Implement chunked processing
3. **Data Type Compatibility**: Comprehensive input validation
4. **Authentication Issues**: Robust error handling for auth failures

### Mitigation Strategies
1. **API Versioning**: Support multiple NeMo API versions
2. **Memory Management**: Implement streaming for large datasets
3. **Input Validation**: Comprehensive data validation and error reporting
4. **Authentication**: Multiple auth methods and clear error messages

## Success Criteria

### Functional Requirements
- [ ] Both components can be used independently
- [ ] Components can be chained together effectively
- [ ] Support for all specified input formats (list[Data], DataFrame)
- [ ] Comprehensive filtering capabilities
- [ ] Automatic and explicit field mapping
- [ ] Configurable training/validation splits
- [ ] Partial success handling with detailed statistics
- [ ] Integration with existing NeMo components

### Performance Requirements
- [ ] Handle datasets up to 100,000 records efficiently
- [ ] Memory usage stays under 2GB for typical datasets
- [ ] Processing time under 5 minutes for 10,000 records
- [ ] Successful upload to NeMo Data Store

### Quality Requirements
- [ ] 90%+ test coverage
- [ ] All edge cases handled gracefully
- [ ] Comprehensive error messages
- [ ] Clear documentation and examples

## Timeline Summary

| Phase | Duration | Key Deliverables |
|-------|----------|------------------|
| Phase 1 | Week 1 | Foundation, dependencies, base classes |
| Phase 2 | Week 2 | Data Preparation Component |
| Phase 3 | Week 3 | Dataset Uploader Component |
| Phase 4 | Week 4 | Integration, advanced features |
| Phase 5 | Week 5 | Testing, validation |
| Phase 6 | Week 6 | Deployment, documentation |

**Total Duration**: 6 weeks
**Estimated Effort**: 240 hours (40 hours/week)

## Next Steps

1. **Review and Approve Plan**: Stakeholder review of implementation plan
2. **Set Up Development Environment**: Configure development tools and dependencies
3. **Begin Phase 1**: Start with foundation and infrastructure setup
4. **Regular Check-ins**: Weekly progress reviews and milestone tracking

This implementation plan provides a structured approach to building the refactored NeMo dataset components while ensuring quality, performance, and maintainability.
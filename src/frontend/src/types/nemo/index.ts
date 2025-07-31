// NeMo Microservices Type Definitions
// Includes types for Data Store, Customizer, and Evaluator

// =============================================================================
// Dataset Types (Data Store)
// =============================================================================

export interface NeMoDataset {
  id: string;
  name: string;
  namespace: string;
  description: string | null;
  format: string;
  created_at: string;
  updated_at: string;
  files_url: string;
  hf_endpoint: string | null;
  split: string | null;
  limit: string | null;
  project: string;
  custom_fields: Record<string, any>;
  // Legacy fields for backward compatibility
  type?: string;
  metadata?: {
    file_count: number;
    total_size: string;
    format: string;
    tags: string[];
  };
  files?: NeMoFile[];
  file_count?: number;
}

export interface NeMoFile {
  filename: string;
  size: number;
  content_type: string;
  uploaded_at: string;
}

export interface CreateDatasetRequest {
  name: string;
  namespace: string;
  description?: string;
  dataset_type?: string;
}

export interface CreateDatasetResponse {
  id: string;
  name: string;
  description: string | null;
  type: string;
  created_at: string;
  updated_at: string;
  metadata: {
    file_count: number;
    total_size: string;
    format: string;
    tags: string[];
  };
}

export interface UploadFilesRequest {
  datasetId: string;
  files: File[];
}

export interface UploadFilesResponse {
  message: string;
  files: NeMoFile[];
}

export interface DeleteDatasetResponse {
  message: string;
}

// =============================================================================
// Customization Target Types
// =============================================================================

export interface NeMoCustomizationTarget {
  id: string;
  name: string;
  namespace: string;
  description: string;
  enabled: boolean;
  base_model: string;
  model_path: string;
  model_uri: string;
  tokenizer: Record<string, any>;
  num_parameters: number;
  precision: string;
  status: string;
  ownership: Record<string, any>;
  custom_fields: Record<string, any>;
  created_at: string;
  updated_at: string;
}

export interface NeMoCustomizationTargetList {
  data: NeMoCustomizationTarget[];
  page: number;
  page_size: number;
  total: number;
  total_pages: number;
  has_next: boolean;
  has_prev: boolean;
  error?: string;
}

// =============================================================================
// Customization Config Types
// =============================================================================

export interface NeMoCustomizationConfig {
  id: string;
  name: string;
  namespace: string;
  target: {
    id: string;
    name: string;
  };
  params: {
    training_type: string;
    finetuning_type: string;
    max_seq_length: number;
    prompt_template: string;
    training_precision: string;
    lora?: {
      adapter_dim: number;
      alpha: number;
      target_modules: string[];
    };
  };
  created_at: string;
  updated_at: string;
}

export interface NeMoCustomizationConfigList {
  data: NeMoCustomizationConfig[];
  page: number;
  page_size: number;
  total: number;
  total_pages: number;
  has_next: boolean;
  has_prev: boolean;
  error?: string;
}

// =============================================================================
// Evaluation Target Types
// =============================================================================

export interface NeMoEvaluationTarget {
  id: string;
  name: string;
  namespace: string;
  description: string;
  type: string; // "data_source" | "llm_model" | "retriever_pipeline" | "rag_pipeline"
  params: {
    // For LLM Model targets
    endpoint?: string;
    model?: string;
    api_key?: string;

    // For Data Source targets
    dataset?: string;
    rows?: any[];

    // For Retriever Pipeline targets
    embedding_model?: string;
    re_ranker?: string;

    // For RAG Pipeline targets
    retriever?: any;
    generator?: any;
  };
  created_at: string;
  updated_at: string;
}

export interface NeMoEvaluationTargetList {
  data: NeMoEvaluationTarget[];
  page: number;
  page_size: number;
  total: number;
  total_pages: number;
  has_next: boolean;
  has_prev: boolean;
  error?: string;
}

// =============================================================================
// Evaluation Config Types
// =============================================================================

export interface NeMoEvaluationConfig {
  id: string;
  name: string;
  namespace: string;
  target?: {
    id: string;
    name: string;
  };
  params: {
    evaluation_type: string; // "agentic" | "bfcl" | "bigcode" | "custom" | "lm_harness" | "rag" | "retriever"
    metrics?: string[];
    task?: string;
    dataset?: string;

    // For custom evaluations
    custom_metrics?: any[];

    // For LM Harness evaluations
    tasks?: string[];

    // For RAG evaluations
    retrieval_metrics?: string[];
    generation_metrics?: string[];
  };
  created_at: string;
  updated_at: string;
}

export interface NeMoEvaluationConfigList {
  data: NeMoEvaluationConfig[];
  page: number;
  page_size: number;
  total: number;
  total_pages: number;
  has_next: boolean;
  has_prev: boolean;
  error?: string;
}

// =============================================================================
// Job Types (Customizer) - Real NeMo API Structure
// =============================================================================

export interface NeMoJobConfig {
  schema_version: string;
  id: string;
  namespace: string;
  created_at: string;
  updated_at: string;
  custom_fields: Record<string, any>;
  name: string;
  base_model: string;
  model_path: string;
  training_types: string[];
  finetuning_types: string[];
  precision: string;
  num_gpus: number;
  num_nodes: number;
  micro_batch_size: number;
  tensor_parallel_size: number;
  max_seq_length: number;
}

export interface NeMoJobHyperparameters {
  finetuning_type: string;
  training_type: string;
  batch_size: number;
  epochs: number;
  learning_rate: number;
  lora?: {
    adapter_dim: number;
    adapter_dropout?: number;
  };
}

export interface NeMoTrainingLossEntry {
  step: number;
  value: number;
  timestamp: string;
}

export interface NeMoValidationLossEntry {
  epoch: number;
  value: number;
  timestamp: string;
}

export interface NeMoJobStatusLog {
  updated_at: string;
  message: string;
  detail?: string;
}

export interface NeMoJobStatusDetails {
  created_at: string;
  updated_at: string;
  steps_completed: number;
  epochs_completed: number;
  percentage_done: number;
  status_logs: NeMoJobStatusLog[];
  training_loss: NeMoTrainingLossEntry[];
  validation_loss: NeMoValidationLossEntry[];
}

export type NeMoJobStatus =
  | "created"
  | "running"
  | "completed"
  | "failed"
  | "cancelled";

export interface NeMoCustomizerJob {
  id: string;
  created_at: string;
  updated_at: string;
  namespace: string;
  config: NeMoJobConfig;
  dataset: string;
  hyperparameters: NeMoJobHyperparameters;
  output_model: string;
  status: NeMoJobStatus;
  status_details: NeMoJobStatusDetails;
  custom_fields: Record<string, any>;
}

export interface NeMoJobStatusResponse {
  id: string;
  status: NeMoJobStatus;
  status_details: NeMoJobStatusDetails;
  created_at: string;
  updated_at: string;
}

// =============================================================================
// Job Tracking for Langflow Dashboard
// =============================================================================

export interface TrackedJob {
  job_id: string;
  status: NeMoJobStatus;
  created_at: string;
  updated_at: string;
  config: string;
  dataset: string;
  progress: number;
  output_model?: string;
  hyperparameters?: NeMoJobHyperparameters;
  custom_fields?: Record<string, any>;
}

export interface TrackJobRequest {
  job_id: string;
  metadata?: Record<string, any>;
}

export interface TrackJobResponse {
  job_id: string;
  tracked_at: string;
  metadata: Record<string, any>;
  message: string;
}

export interface StopTrackingResponse {
  message: string;
}

// =============================================================================
// Legacy Support (for backward compatibility)
// =============================================================================

export interface DatasetType {
  id: string;
  name: string;
  description?: string;
  type: string;
  created_at: string;
  updated_at: string;
  metadata?: {
    file_count?: number;
    total_size?: string;
    format?: string;
    tags?: string[];
  };
  disabled?: boolean;
}

// Legacy job types for backward compatibility
export interface LegacyNeMoCustomizerJob {
  id: string;
  created_at: string;
  updated_at: string;
  namespace: string;
  config: {
    name: string;
    base_model: string;
    training_types: string[];
    finetuning_types: string[];
    precision: string;
    num_gpus: number;
    num_nodes?: number;
    micro_batch_size?: number;
    tensor_parallel_size?: number;
    max_seq_length?: number;
  };
  dataset: string;
  hyperparameters: NeMoJobHyperparameters;
  output_model: string;
  status: NeMoJobStatus;
  progress?: {
    current_epoch: number;
    total_epochs: number;
    percentage: number;
    training_loss: Array<{
      step: number;
      loss: number;
      timestamp: string;
    }>;
    validation_loss: Array<{
      epoch: number;
      loss: number;
      timestamp: string;
    }>;
  };
  custom_fields?: Record<string, any>;
  component_id?: string;
  created_by_component?: string;
  stored_at?: string;
}

export interface StoreJobRequest {
  job_info: Partial<LegacyNeMoCustomizerJob>;
  component_id?: string;
  created_by_component?: string;
}

export interface StoreJobResponse {
  job_info: LegacyNeMoCustomizerJob;
  component_id?: string;
  created_by_component?: string;
  stored_at: string;
}

// =============================================================================
// Evaluator Job Types (Evaluator) - Real NeMo API Structure
// =============================================================================

export type NeMoEvaluatorJobStatus =
  | "created"
  | "running"
  | "completed"
  | "failed"
  | "cancelled";

export interface NeMoEvaluatorJobStatusDetails {
  created_at: string;
  updated_at: string;
  message?: string;
  percentage_done?: number;
}

export interface NeMoEvaluatorJob {
  id: string;
  created_at: string;
  updated_at: string;
  namespace: string;
  target: any; // API returns object, not string
  config: any; // API returns object, not string
  tags?: string[]; // Optional since it might not be in response
  status: NeMoEvaluatorJobStatus;
  status_details: NeMoEvaluatorJobStatusDetails;
}

export interface NeMoEvaluatorJobList {
  data: NeMoEvaluatorJob[];
}

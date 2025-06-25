// NeMo Microservices Type Definitions
// Includes types for Data Store, Customizer, and Evaluator

// =============================================================================
// Dataset Types (Data Store)
// =============================================================================

export interface NeMoDataset {
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
// Job Types (Customizer)
// =============================================================================

export interface NeMoCustomizerJob {
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
  hyperparameters: {
    training_type: string;
    finetuning_type: string;
    epochs: number;
    batch_size: number;
    learning_rate: number;
    lora?: {
      adapter_dim: number;
      adapter_dropout?: number;
    };
  };
  output_model: string;
  status: "created" | "running" | "completed" | "failed" | "cancelled";
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
  job_info: Partial<NeMoCustomizerJob>;
  component_id?: string;
  created_by_component?: string;
}

export interface StoreJobResponse {
  job_info: NeMoCustomizerJob;
  component_id?: string;
  created_by_component?: string;
  stored_at: string;
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

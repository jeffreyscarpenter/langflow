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

export interface UploadFilesResponse {
  message: string;
  files: NeMoFile[];
}

export interface DeleteDatasetResponse {
  message: string;
}

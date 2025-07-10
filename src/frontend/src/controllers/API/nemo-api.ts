import axios, { AxiosInstance } from "axios";
import { BASE_URL_API_V2 } from "@/constants/constants";

interface NeMoConfig {
  baseUrl: string;
  authToken: string;
  namespace: string;
}

class NeMoApiClient {
  private config: NeMoConfig | null = null;
  private client: AxiosInstance;

  constructor() {
    this.client = axios.create({
      baseURL: BASE_URL_API_V2,
    });
    this.loadConfig();
  }

  private loadConfig(): void {
    try {
      const savedConfig = localStorage.getItem('nemo-config');
      if (savedConfig) {
        this.config = JSON.parse(savedConfig);
      }
    } catch (error) {
      console.error('Error loading NeMo config:', error);
      this.config = null;
    }
  }

  private getHeaders(): Record<string, string> {
    if (!this.config?.authToken) {
      throw new Error('NeMo configuration not found. Please configure your connection.');
    }

    return {
      'X-NeMo-Base-URL': this.config.baseUrl,
      'X-NeMo-Auth-Token': this.config.authToken,
      'X-NeMo-Namespace': this.config.namespace,
      'Content-Type': 'application/json',
      'Accept': 'application/json'
    };
  }

  getNamespace(): string {
    if (!this.config?.namespace) {
      throw new Error('NeMo configuration not found. Please configure your connection.');
    }
    return this.config.namespace;
  }

  // Dataset operations (use backend proxy)
  async getDatasets(page: number = 1, pageSize: number = 10, datasetName?: string) {
    this.loadConfig(); // Refresh config
    const params: any = {
      page,
      page_size: pageSize
    };

    if (datasetName) {
      params.dataset_name = datasetName;
    }

    const response = await this.client.get('/nemo/datasets', {
      headers: this.getHeaders(),
      params
    });
    return response.data;
  }

  async createDataset(data: any) {
    this.loadConfig(); // Refresh config

    // Prepare form data for the API call
    const formData = new URLSearchParams();
    formData.append('name', data.name);
    formData.append('namespace', data.namespace);
    if (data.description) {
      formData.append('description', data.description);
    }
    formData.append('dataset_type', data.dataset_type || 'fileset');

    const response = await this.client.post('/nemo/datasets', formData, {
      headers: {
        ...this.getHeaders(),
        'Content-Type': 'application/x-www-form-urlencoded'
      }
    });
    return response.data;
  }

  async deleteDataset(datasetName: string, namespace?: string) {
    this.loadConfig(); // Refresh config
    const params: any = {};

    if (namespace) {
      params.namespace = namespace;
    }

    const response = await this.client.delete(`/nemo/datasets/${datasetName}`, {
      headers: this.getHeaders(),
      params
    });
    return response.data;
  }

  async getDataset(datasetName: string, namespace?: string) {
    this.loadConfig(); // Refresh config
    const params: any = {};

    if (namespace) {
      params.namespace = namespace;
    }

    const response = await this.client.get(`/nemo/datasets/${datasetName}`, {
      headers: this.getHeaders(),
      params
    });
    return response.data;
  }

  async getDatasetFiles(datasetName: string) {
    this.loadConfig(); // Refresh config
    const response = await this.client.get(`/nemo/datasets/${datasetName}/files`, {
      headers: this.getHeaders()
    });
    return response.data;
  }

  async uploadFiles(params: {
    datasetId: string;
    files: File[];
  }) {
    this.loadConfig(); // Refresh config

    const formData = new FormData();

    // Add each file to the form data
    params.files.forEach((file, index) => {
      formData.append(`files`, file);
    });

    const headers = {
      'X-NeMo-Base-URL': this.config!.baseUrl,
      'X-NeMo-Auth-Token': this.config!.authToken,
      'X-NeMo-Namespace': this.config!.namespace,
      'Accept': 'application/json'
      // Don't set Content-Type - let browser set it with boundary for multipart/form-data
    };

    const response = await this.client.post(
      `/nemo/datasets/${params.datasetId}/files`,
      formData,
      { headers }
    );
    return response.data;
  }

  async getDatasetDetails(datasetName: string, namespace?: string) {
    this.loadConfig(); // Refresh config
    const params: any = {};

    if (namespace) {
      params.namespace = namespace;
    }

    const response = await this.client.get(`/nemo/datasets/${datasetName}/details`, {
      headers: this.getHeaders(),
      params
    });
    return response.data;
  }

  async uploadDatasetFiles(params: {
    datasetId: string;
    datasetName: string;
    namespace: string;
    path: string;
    files: File[];
  }) {
    this.loadConfig(); // Refresh config

    const formData = new FormData();
    formData.append('path', params.path);
    formData.append('namespace', params.namespace);

    // Add each file to the form data
    params.files.forEach((file, index) => {
      formData.append(`files`, file);
    });

    const headers = {
      'X-NeMo-Base-URL': this.config!.baseUrl,
      'X-NeMo-Auth-Token': this.config!.authToken,
      'X-NeMo-Namespace': this.config!.namespace,
      'Accept': 'application/json'
      // Don't set Content-Type - let browser set it with boundary for multipart/form-data
    };

    const response = await this.client.post(
      `/nemo/datasets/${params.datasetName}/upload`,
      formData,
      { headers }
    );
    return response.data;
  }

  // Customizer job operations
  async getCustomizerJobs() {
    this.loadConfig(); // Refresh config
    const response = await this.client.get('/nemo/v1/customization/jobs', {
      headers: this.getHeaders()
    });
    return response.data;
  }

  async getTrackedJobs() {
    this.loadConfig(); // Refresh config
    const response = await this.client.get('/nemo/jobs/tracked', {
      headers: this.getHeaders()
    });
    return response.data;
  }

  async getCustomizerJob(jobId: string) {
    this.loadConfig(); // Refresh config
    const response = await this.client.get(`/nemo/v1/customization/jobs/${jobId}`, {
      headers: this.getHeaders()
    });
    return response.data;
  }

  async getCustomizerJobStatus(jobId: string) {
    this.loadConfig(); // Refresh config
    const response = await this.client.get(`/nemo/v1/customization/jobs/${jobId}/status`, {
      headers: this.getHeaders()
    });
    return response.data;
  }

  async cancelCustomizerJob(jobId: string) {
    this.loadConfig(); // Refresh config
    const response = await this.client.post(`/nemo/v1/customization/jobs/${jobId}/cancel`, {}, {
      headers: this.getHeaders()
    });
    return response.data;
  }

  // Evaluator job operations
  async getEvaluatorJobs() {
    this.loadConfig(); // Refresh config
    const response = await this.client.get('/nemo/v1/evaluation/jobs', {
      headers: this.getHeaders()
    });
    return response.data;
  }

  async getEvaluatorJob(jobId: string) {
    this.loadConfig(); // Refresh config
    const response = await this.client.get(`/nemo/v1/evaluation/jobs/${jobId}`, {
      headers: this.getHeaders()
    });
    return response.data;
  }

  async getEvaluatorJobStatus(jobId: string) {
    this.loadConfig(); // Refresh config
    const response = await this.client.get(`/nemo/v1/evaluation/jobs/${jobId}/status`, {
      headers: this.getHeaders()
    });
    return response.data;
  }

  // Check if configuration is valid
  isConfigured(): boolean {
    this.loadConfig();
    return !!(this.config?.baseUrl && this.config?.authToken && this.config?.namespace);
  }
}

// Export a singleton instance
export const nemoApi = new NeMoApiClient();

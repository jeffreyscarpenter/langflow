import { useMutation } from "@tanstack/react-query";
import { nemoApi } from "../../nemo-api";

interface UploadDatasetFilesParams {
  datasetId: string;
  datasetName: string;
  namespace: string;
  path: string;
  files: File[];
}

async function uploadDatasetFiles(params: UploadDatasetFilesParams): Promise<void> {
  return await nemoApi.uploadDatasetFiles(params);
}

export function useUploadDatasetFiles() {
  return useMutation({
    mutationFn: uploadDatasetFiles,
  });
}

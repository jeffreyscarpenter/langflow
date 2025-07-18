import { useQueryFunctionType } from "@/types/api";
import { NeMoDataset } from "@/types/nemo";
import { nemoApi } from "../../nemo-api";
import { UseRequestProcessor } from "../../services/request-processor";

interface UseGetDatasetsParams {
  page?: number;
  pageSize?: number;
  datasetName?: string;
}

interface PaginatedDatasetsResponse {
  data: NeMoDataset[];
  page: number;
  page_size: number;
  total: number;
  has_next: boolean;
  has_prev: boolean;
  error?: string;
}

export const useGetDatasets = (params: UseGetDatasetsParams = {}, options?: any) => {
  const { query } = UseRequestProcessor();
  const { page = 1, pageSize = 10, datasetName } = params;

  const getDatasetsFn = async (): Promise<PaginatedDatasetsResponse> => {
    return await nemoApi.getDatasets(page, pageSize, datasetName);
  };

  const queryResult = query(
    ["useGetDatasets", page, pageSize, datasetName],
    getDatasetsFn,
    {
      refetchOnWindowFocus: false,
      ...options,
    }
  );

  return queryResult;
};

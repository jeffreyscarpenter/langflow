import { useQueryFunctionType } from "@/types/api";
import { nemoApi } from "../../nemo-api";
import { UseRequestProcessor } from "../../services/request-processor";

interface DatasetDetailsResponse {
  id: string;
  author: string;
  sha: string;
  name: string;
  created_at: string;
  last_modified: string;
  siblings: Array<{
    rfilename: string;
    size?: number;
  }>;
}

interface UseGetDatasetDetailsParams {
  datasetName: string;
  namespace?: string;
}

export const useGetDatasetDetails = (params: UseGetDatasetDetailsParams, options?: any) => {
  const { query } = UseRequestProcessor();
  const { datasetName, namespace } = params;

  const getDatasetDetailsFn = async (): Promise<DatasetDetailsResponse> => {
    return await nemoApi.getDatasetDetails(datasetName, namespace);
  };

  const queryResult = query(
    ["useGetDatasetDetails", datasetName, namespace],
    getDatasetDetailsFn,
    {
      refetchOnWindowFocus: false,
      enabled: !!datasetName, // Only run if datasetName is provided
      ...options,
    }
  );

  return queryResult;
};

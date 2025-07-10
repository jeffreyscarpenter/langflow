import { useQueryFunctionType } from "@/types/api";
import { NeMoDataset } from "@/types/nemo";
import { nemoApi } from "../../nemo-api";
import { UseRequestProcessor } from "../../services/request-processor";

interface GetDatasetParams {
  datasetName: string;
  namespace?: string;
}

export const useGetDataset: useQueryFunctionType<
  GetDatasetParams,
  NeMoDataset
> = (params, options) => {
  const { query } = UseRequestProcessor();

  const getDatasetFn = async () => {
    return await nemoApi.getDataset(params.datasetName, params.namespace);
  };

  const queryResult = query(
    ["useGetDataset", params.datasetName, params.namespace],
    getDatasetFn,
    {
      refetchOnWindowFocus: false,
      ...options,
    }
  );

  return queryResult;
};

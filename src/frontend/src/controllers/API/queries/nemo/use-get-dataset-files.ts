import { useQueryFunctionType } from "@/types/api";
import { NeMoFile } from "@/types/nemo";
import { nemoApi } from "../../nemo-api";
import { UseRequestProcessor } from "../../services/request-processor";

interface GetDatasetFilesParams {
  datasetName: string;
}

export const useGetDatasetFiles: useQueryFunctionType<
  GetDatasetFilesParams,
  NeMoFile[]
> = (params, options) => {
  const { query } = UseRequestProcessor();

  const getDatasetFilesFn = async () => {
    return await nemoApi.getDatasetFiles(params.datasetName);
  };

  const queryResult = query(
    ["useGetDatasetFiles", params.datasetName],
    getDatasetFilesFn,
    {
      refetchOnWindowFocus: false,
      ...options,
    }
  );

  return queryResult;
};

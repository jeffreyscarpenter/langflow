import { useQueryFunctionType } from "@/types/api";
import { NeMoFile } from "@/types/nemo";
import { api } from "../../api";
import { getURL } from "../../helpers/constants";
import { UseRequestProcessor } from "../../services/request-processor";

interface GetDatasetFilesParams {
  datasetId: string;
}

export const useGetDatasetFiles: useQueryFunctionType<
  GetDatasetFilesParams,
  NeMoFile[]
> = (params, options) => {
  const { query } = UseRequestProcessor();

  const getDatasetFilesFn = async () => {
    const response = await api.get<NeMoFile[]>(
      `${getURL("NEMO", undefined, true)}/datasets/${params.datasetId}/files`
    );
    return response.data;
  };

  const queryResult = query(
    ["useGetDatasetFiles", params.datasetId],
    getDatasetFilesFn,
    {
      refetchOnWindowFocus: false,
      ...options,
    }
  );

  return queryResult;
};

import { useQueryFunctionType } from "@/types/api";
import { NeMoDataset } from "@/types/nemo-datastore";
import { api } from "../../api";
import { getURL } from "../../helpers/constants";
import { UseRequestProcessor } from "../../services/request-processor";

interface GetDatasetParams {
  datasetId: string;
}

export const useGetDataset: useQueryFunctionType<
  GetDatasetParams,
  NeMoDataset
> = (params, options) => {
  const { query } = UseRequestProcessor();

  const getDatasetFn = async () => {
    const response = await api.get<NeMoDataset>(
      `${getURL("NEMO_DATASTORE", undefined, true)}/datasets/${params.datasetId}`
    );
    return response.data;
  };

  const queryResult = query(
    ["useGetDataset", params.datasetId],
    getDatasetFn,
    {
      refetchOnWindowFocus: false,
      ...options,
    }
  );

  return queryResult;
};

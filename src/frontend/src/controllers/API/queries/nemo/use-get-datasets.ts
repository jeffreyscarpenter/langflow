import { useQueryFunctionType } from "@/types/api";
import { NeMoDataset } from "@/types/nemo";
import { api } from "../../api";
import { getURL } from "../../helpers/constants";
import { UseRequestProcessor } from "../../services/request-processor";

export const useGetDatasets: useQueryFunctionType<
  undefined,
  NeMoDataset[]
> = (options) => {
  const { query } = UseRequestProcessor();

  const getDatasetsFn = async () => {
    const response = await api.get<NeMoDataset[]>(
      `${getURL("NEMO", undefined, true)}/datasets`
    );
    return response.data;
  };

  const queryResult = query(
    ["useGetDatasets"],
    getDatasetsFn,
    {
      refetchOnWindowFocus: false,
      ...options,
    }
  );

  return queryResult;
};

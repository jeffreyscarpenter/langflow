import { useMutationFunctionType } from "@/types/api";
import { CreateDatasetRequest, CreateDatasetResponse } from "@/types/nemo-datastore";
import { UseMutationResult } from "@tanstack/react-query";
import { api } from "../../api";
import { getURL } from "../../helpers/constants";
import { UseRequestProcessor } from "../../services/request-processor";

export const useCreateDataset: useMutationFunctionType<
  undefined,
  CreateDatasetRequest,
  CreateDatasetResponse
> = (options?) => {
  const { mutate, queryClient } = UseRequestProcessor();

  async function createDatasetFn(data: CreateDatasetRequest): Promise<CreateDatasetResponse> {
    const params = new URLSearchParams();
    params.append("name", data.name);
    if (data.description) {
      params.append("description", data.description);
    }
    if (data.dataset_type) {
      params.append("dataset_type", data.dataset_type);
    }

    const response = await api.post<CreateDatasetResponse>(
      `${getURL("NEMO_DATASTORE", undefined, true)}/datasets?${params.toString()}`
    );
    return response.data;
  }

  const mutation: UseMutationResult<CreateDatasetResponse, any, CreateDatasetRequest> = mutate(
    ["useCreateDataset"],
    createDatasetFn,
    {
      ...options,
      onSuccess: () => {
        // Invalidate and refetch datasets list
        queryClient.invalidateQueries({ queryKey: ["useGetDatasets"] });
      },
    }
  );

  return mutation;
};

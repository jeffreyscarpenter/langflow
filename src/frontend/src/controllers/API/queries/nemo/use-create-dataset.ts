import { UseMutationResult } from "@tanstack/react-query";
import { useMutationFunctionType } from "@/types/api";
import { CreateDatasetRequest, CreateDatasetResponse } from "@/types/nemo";
import { nemoApi } from "../../nemo-api";
import { UseRequestProcessor } from "../../services/request-processor";

export const useCreateDataset: useMutationFunctionType<
  undefined,
  CreateDatasetRequest,
  CreateDatasetResponse
> = (options?) => {
  const { mutate, queryClient } = UseRequestProcessor();

  async function createDatasetFn(
    data: CreateDatasetRequest,
  ): Promise<CreateDatasetResponse> {
    // Add namespace to the request data
    const dataWithNamespace = {
      ...data,
      namespace: data.namespace || nemoApi.getNamespace(),
    };

    const response = await nemoApi.createDataset(dataWithNamespace);
    return response;
  }

  const mutation: UseMutationResult<
    CreateDatasetResponse,
    any,
    CreateDatasetRequest
  > = mutate(["useCreateDataset"], createDatasetFn, {
    ...options,
    onSuccess: () => {
      // Invalidate and refetch datasets list
      queryClient.invalidateQueries({ queryKey: ["useGetDatasets"] });
    },
  });

  return mutation;
};

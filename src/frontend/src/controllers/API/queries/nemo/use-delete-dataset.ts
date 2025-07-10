import { useMutationFunctionType } from "@/types/api";
import { DeleteDatasetResponse } from "@/types/nemo";
import { UseMutationResult } from "@tanstack/react-query";
import { nemoApi } from "../../nemo-api";
import { UseRequestProcessor } from "../../services/request-processor";

interface DeleteDatasetParams {
  datasetName: string;
  namespace?: string;
}

export const useDeleteDataset: useMutationFunctionType<
  undefined,
  DeleteDatasetParams,
  DeleteDatasetResponse
> = (options?) => {
  const { mutate, queryClient } = UseRequestProcessor();

  async function deleteDatasetFn(data: DeleteDatasetParams): Promise<DeleteDatasetResponse> {
    return await nemoApi.deleteDataset(data.datasetName, data.namespace);
  }

  const mutation: UseMutationResult<DeleteDatasetResponse, any, DeleteDatasetParams> = mutate(
    ["useDeleteDataset"],
    deleteDatasetFn,
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

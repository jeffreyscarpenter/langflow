import { useMutationFunctionType } from "@/types/api";
import { DeleteDatasetResponse } from "@/types/nemo-datastore";
import { UseMutationResult } from "@tanstack/react-query";
import { api } from "../../api";
import { getURL } from "../../helpers/constants";
import { UseRequestProcessor } from "../../services/request-processor";

interface DeleteDatasetParams {
  datasetId: string;
}

export const useDeleteDataset: useMutationFunctionType<
  undefined,
  DeleteDatasetParams,
  DeleteDatasetResponse
> = (options?) => {
  const { mutate, queryClient } = UseRequestProcessor();

  async function deleteDatasetFn(data: DeleteDatasetParams): Promise<DeleteDatasetResponse> {
    const response = await api.delete<DeleteDatasetResponse>(
      `${getURL("NEMO_DATASTORE", undefined, true)}/datasets/${data.datasetId}`
    );
    return response.data;
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

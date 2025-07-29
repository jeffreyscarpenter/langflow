import { UseMutationResult } from "@tanstack/react-query";
import { useMutationFunctionType } from "@/types/api";
import { UploadFilesResponse } from "@/types/nemo";
import { nemoApi } from "../../nemo-api";
import { UseRequestProcessor } from "../../services/request-processor";

interface UploadFilesParams {
  datasetId: string;
  files: File[];
}

export const useUploadFiles: useMutationFunctionType<
  undefined,
  UploadFilesParams,
  UploadFilesResponse
> = (options?) => {
  const { mutate, queryClient } = UseRequestProcessor();

  async function uploadFilesFn(
    data: UploadFilesParams,
  ): Promise<UploadFilesResponse> {
    return await nemoApi.uploadFiles({
      datasetId: data.datasetId,
      files: data.files,
    });
  }

  const mutation: UseMutationResult<
    UploadFilesResponse,
    any,
    UploadFilesParams
  > = mutate(["useUploadFiles"], uploadFilesFn, {
    ...options,
    onSuccess: () => {
      // Invalidate and refetch dataset files and dataset details
      queryClient.invalidateQueries({ queryKey: ["useGetDatasetFiles"] });
      queryClient.invalidateQueries({ queryKey: ["useGetDataset"] });
      queryClient.invalidateQueries({ queryKey: ["useGetDatasets"] });
    },
  });

  return mutation;
};

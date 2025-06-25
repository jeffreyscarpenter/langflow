import { useMutationFunctionType } from "@/types/api";
import { UploadFilesResponse } from "@/types/nemo";
import { UseMutationResult } from "@tanstack/react-query";
import { api } from "../../api";
import { getURL } from "../../helpers/constants";
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

  async function uploadFilesFn(data: UploadFilesParams): Promise<UploadFilesResponse> {
    const formData = new FormData();
    data.files.forEach((file) => {
      formData.append("files", file);
    });

    const response = await api.post<UploadFilesResponse>(
      `${getURL("NEMO", undefined, true)}/datasets/${data.datasetId}/files`,
      formData,
      {
        headers: {
          "Content-Type": "multipart/form-data",
        },
      }
    );
    return response.data;
  }

  const mutation: UseMutationResult<UploadFilesResponse, any, UploadFilesParams> = mutate(
    ["useUploadFiles"],
    uploadFilesFn,
    {
      ...options,
      onSuccess: () => {
        // Invalidate and refetch dataset files and dataset details
        queryClient.invalidateQueries({ queryKey: ["useGetDatasetFiles"] });
        queryClient.invalidateQueries({ queryKey: ["useGetDataset"] });
        queryClient.invalidateQueries({ queryKey: ["useGetDatasets"] });
      },
    }
  );

  return mutation;
};

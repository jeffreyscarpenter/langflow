import { useMutation, useQueryClient } from "@tanstack/react-query";
import { nemoApi } from "../../nemo-api";

export function useCancelCustomizerJob() {
  const queryClient = useQueryClient();

  return useMutation({
    mutationFn: async (jobId: string) => {
      return await nemoApi.cancelCustomizerJob(jobId);
    },
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ["nemo", "customizer-jobs"] });
      queryClient.invalidateQueries({ queryKey: ["nemo", "tracked-jobs"] });
    },
  });
}

export function useGetCustomizerJobLogs() {
  return useMutation({
    mutationFn: async (jobId: string) => {
      return await nemoApi.getCustomizerJobContainerLogs(jobId);
    },
  });
}

import { useMutation, useQueryClient } from "@tanstack/react-query";
import { nemoApi } from "../../nemo-api";

async function cancelJob(jobId: string) {
  return await nemoApi.cancelCustomizerJob(jobId);
}

export function useCancelJob() {
  const queryClient = useQueryClient();

  return useMutation({
    mutationFn: cancelJob,
    onSuccess: (data, jobId) => {
      // Invalidate job status queries to refresh the data
      queryClient.invalidateQueries({
        queryKey: ["nemo", "jobs", "status", jobId],
      });
      queryClient.invalidateQueries({ queryKey: ["nemo", "jobs", "tracked"] });
    },
  });
}

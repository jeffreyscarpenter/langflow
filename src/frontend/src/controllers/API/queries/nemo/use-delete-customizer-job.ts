import { useMutation, useQueryClient } from "@tanstack/react-query";
import { nemoApi } from "../../nemo-api";

async function deleteCustomizerJob(jobId: string) {
  return await nemoApi.deleteCustomizerJob(jobId);
}

export function useDeleteCustomizerJob() {
  const queryClient = useQueryClient();

  return useMutation({
    mutationFn: deleteCustomizerJob,
    onSuccess: (data, jobId) => {
      // Invalidate job-related queries to refresh the data
      queryClient.invalidateQueries({ queryKey: ["nemo", "jobs", "status", jobId] });
      queryClient.invalidateQueries({ queryKey: ["nemo", "jobs", "tracked"] });
      queryClient.invalidateQueries({ queryKey: ["nemo", "customizer", "jobs"] });
    },
  });
}

import { useMutation, useQueryClient } from "@tanstack/react-query";
import { nemoApi } from "../../nemo-api";

async function deleteEvaluatorJob(jobId: string) {
  return await nemoApi.deleteEvaluatorJob(jobId);
}

export function useDeleteEvaluatorJob() {
  const queryClient = useQueryClient();

  return useMutation({
    mutationFn: deleteEvaluatorJob,
    onSuccess: (data, jobId) => {
      // Invalidate evaluator job queries to refresh the data
      queryClient.invalidateQueries({
        queryKey: ["nemo", "evaluator", "jobs"],
      });
      queryClient.invalidateQueries({
        queryKey: ["nemo", "evaluation", "jobs"],
      });
    },
  });
}

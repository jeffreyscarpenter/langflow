import { useMutation, useQueryClient } from "@tanstack/react-query";
import { nemoApi } from "../../nemo-api";

export function useDeleteEvaluatorJob() {
  const queryClient = useQueryClient();

  return useMutation({
    mutationFn: async (jobId: string) => {
      return await nemoApi.deleteEvaluatorJob(jobId);
    },
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ["nemo", "evaluator-jobs"] });
    },
  });
}

export function useGetEvaluatorJobLogs() {
  return useMutation({
    mutationFn: async (jobId: string) => {
      return await nemoApi.getEvaluatorJobLogs(jobId);
    },
  });
}

export function useGetEvaluatorJobResults() {
  return useMutation({
    mutationFn: async (jobId: string) => {
      return await nemoApi.getEvaluatorJobResults(jobId);
    },
  });
}

export function useDownloadEvaluatorJobResults() {
  return useMutation({
    mutationFn: async (jobId: string) => {
      return await nemoApi.downloadEvaluatorJobResults(jobId);
    },
  });
}

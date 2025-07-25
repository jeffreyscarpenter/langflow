import { useQuery } from "@tanstack/react-query";
import { nemoApi } from "../../nemo-api";
import { NeMoEvaluatorJobList } from "@/types/nemo";

async function getEvaluatorJobs(page: number = 1, pageSize: number = 10): Promise<NeMoEvaluatorJobList> {
  return await nemoApi.getEvaluatorJobs(page, pageSize);
}

export function useGetEvaluatorJobs(page: number = 1, pageSize: number = 10) {
  return useQuery({
    queryKey: ["nemo", "evaluator-jobs", page, pageSize],
    queryFn: () => getEvaluatorJobs(page, pageSize),
    refetchInterval: 30000, // 30 seconds
  });
}

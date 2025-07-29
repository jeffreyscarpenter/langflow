import { useQuery } from "@tanstack/react-query";
import { NeMoEvaluatorJobList } from "@/types/nemo";
import { nemoApi } from "../../nemo-api";

async function getEvaluatorJobs(
  page: number = 1,
  pageSize: number = 10,
): Promise<NeMoEvaluatorJobList> {
  return await nemoApi.getEvaluatorJobs(page, pageSize);
}

export function useGetEvaluatorJobs(page: number = 1, pageSize: number = 10) {
  return useQuery({
    queryKey: ["nemo", "evaluator-jobs", page, pageSize],
    queryFn: () => getEvaluatorJobs(page, pageSize),
    refetchInterval: 30000, // 30 seconds
  });
}

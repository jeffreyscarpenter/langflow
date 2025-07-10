import { useQuery } from "@tanstack/react-query";
import { nemoApi } from "../../nemo-api";
import { NeMoEvaluatorJobList } from "@/types/nemo";

async function getEvaluatorJobs(): Promise<NeMoEvaluatorJobList> {
  return await nemoApi.getEvaluatorJobs();
}

export function useGetEvaluatorJobs() {
  return useQuery({
    queryKey: ["nemo", "evaluator-jobs"],
    queryFn: getEvaluatorJobs,
    refetchInterval: 30000, // 30 seconds
  });
}

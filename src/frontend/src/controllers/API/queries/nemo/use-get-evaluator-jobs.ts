import { useQuery } from "@tanstack/react-query";
import { api } from "../../api";
import { getURL } from "../../helpers/constants";
import { NeMoEvaluatorJobList } from "@/types/nemo";

async function getEvaluatorJobs(): Promise<NeMoEvaluatorJobList> {
  const response = await api.get(
    `${getURL("NEMO", undefined, true)}/v1/evaluation/jobs`
  );
  return response.data;
}

export function useGetEvaluatorJobs() {
  return useQuery({
    queryKey: ["nemo", "evaluator-jobs"],
    queryFn: getEvaluatorJobs,
    refetchInterval: 30000, // 30 seconds
  });
}

import { useQuery } from "@tanstack/react-query";
import { nemoApi } from "../../nemo-api";

async function getCustomizerJobs(page: number = 1, pageSize: number = 10) {
  return await nemoApi.getCustomizerJobs(page, pageSize);
}

export function useGetCustomizerJobs(page: number = 1, pageSize: number = 10) {
  return useQuery({
    queryKey: ["nemo", "customizer-jobs", page, pageSize],
    queryFn: () => getCustomizerJobs(page, pageSize),
    refetchInterval: 30000, // 30 seconds
  });
}

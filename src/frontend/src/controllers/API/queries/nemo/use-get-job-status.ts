import { useQuery } from "@tanstack/react-query";
import { nemoApi } from "../../nemo-api";

async function getJobStatus(jobId: string) {
  return await nemoApi.getCustomizerJobStatus(jobId);
}

export function useGetJobStatus(jobId: string | null) {
  return useQuery({
    queryKey: ["nemo", "jobs", "status", jobId],
    queryFn: () => getJobStatus(jobId!),
    enabled: !!jobId,
    refetchInterval: 10000, // Refetch every 10 seconds for live updates
  });
}

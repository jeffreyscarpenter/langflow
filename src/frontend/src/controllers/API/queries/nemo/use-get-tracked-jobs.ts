import { useQuery } from "@tanstack/react-query";
import { TrackedJob } from "@/types/nemo";
import { nemoApi } from "../../nemo-api";

async function getTrackedJobs(): Promise<TrackedJob[]> {
  return await nemoApi.getTrackedJobs();
}

export function useGetTrackedJobs() {
  return useQuery({
    queryKey: ["nemo", "jobs", "tracked"],
    queryFn: getTrackedJobs,
    refetchInterval: 30000, // Refetch every 30 seconds for real-time updates
  });
}

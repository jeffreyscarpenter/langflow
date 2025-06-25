import { useQuery } from "@tanstack/react-query";
import { api } from "../../api";
import { getURL } from "../../helpers/constants";
import { TrackedJob } from "@/types/nemo";

async function getTrackedJobs(): Promise<TrackedJob[]> {
  const response = await api.get(`${getURL("NEMO", undefined, true)}/jobs/tracked`);
  return response.data;
}

export function useGetTrackedJobs() {
  return useQuery({
    queryKey: ["nemo", "jobs", "tracked"],
    queryFn: getTrackedJobs,
    refetchInterval: 30000, // Refetch every 30 seconds for real-time updates
  });
}

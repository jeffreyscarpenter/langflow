import { useQuery } from "@tanstack/react-query";
import { api } from "../../api";
import { getURL } from "../../helpers/constants";
import { NeMoJobStatusResponse } from "@/types/nemo";

async function getJobStatus(jobId: string): Promise<NeMoJobStatusResponse> {
  const response = await api.get(`${getURL("NEMO", undefined, true)}/v1/customization/jobs/${jobId}/status`);
  return response.data;
}

export function useGetJobStatus(jobId: string | null) {
  return useQuery({
    queryKey: ["nemo", "jobs", "status", jobId],
    queryFn: () => getJobStatus(jobId!),
    enabled: !!jobId,
    refetchInterval: 10000, // Refetch every 10 seconds for live updates
  });
}

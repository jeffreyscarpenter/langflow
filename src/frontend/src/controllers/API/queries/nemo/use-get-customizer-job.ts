import { useQuery } from "@tanstack/react-query";
import { TrackedJob } from "@/types/nemo";
import { nemoApi } from "../../nemo-api";

interface UseGetCustomizerJobParams {
  jobId: string;
  enabled?: boolean;
}

export const useGetCustomizerJob = ({
  jobId,
  enabled = true,
}: UseGetCustomizerJobParams) => {
  return useQuery({
    queryKey: ["nemo", "customizer-job", jobId],
    queryFn: async (): Promise<TrackedJob> => {
      const result = await nemoApi.getCustomizerJob(jobId);
      // Map the direct API response to TrackedJob format
      return {
        job_id: result.id,
        status: result.status,
        created_at: result.created_at,
        updated_at: result.updated_at,
        config: result.config,
        dataset: result.dataset,
        progress: result.status_details?.percentage_done || 0,
        output_model: result.output_model,
        hyperparameters: result.hyperparameters,
        custom_fields: {
          description: result.description,
          namespace: result.namespace,
          ...result.status_details,
        },
      };
    },
    enabled: enabled && !!jobId,
    retry: false, // Don't retry on 404 (job not found)
    refetchOnWindowFocus: false,
  });
};

import { useQuery } from "@tanstack/react-query";
import { NeMoDataset } from "@/types/nemo";
import { nemoApi } from "../../nemo-api";

interface UseGetDatasetByNameParams {
  namespace: string;
  datasetName: string;
  enabled?: boolean;
}

export const useGetDatasetByName = ({
  namespace,
  datasetName,
  enabled = true,
}: UseGetDatasetByNameParams) => {
  return useQuery({
    queryKey: ["nemo", "dataset", namespace, datasetName],
    queryFn: async (): Promise<NeMoDataset> => {
      return await nemoApi.getDatasetByName(namespace, datasetName);
    },
    enabled: enabled && !!namespace && !!datasetName,
    retry: false, // Don't retry on 404 (dataset not found)
    refetchOnWindowFocus: false,
  });
};

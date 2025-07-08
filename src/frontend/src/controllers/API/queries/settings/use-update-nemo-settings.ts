import { useMutationFunctionType } from "../../../../types/api";
import { api } from "../../api";
import { getURL } from "../../helpers/constants";
import { UseRequestProcessor } from "../../services/request-processor";

export interface NeMoSettingsUpdate {
  nemo_use_mock?: boolean;
  nemo_api_key?: string;
  nemo_base_url?: string;
}

export interface NeMoSettingsResponse {
  nemo_use_mock: boolean;
  nemo_base_url: string;
  nemo_api_key: string | null;
}

export const useUpdateNeMoSettings: useMutationFunctionType<
  NeMoSettingsUpdate,
  NeMoSettingsResponse
> = (options) => {
  const { mutation } = UseRequestProcessor();

  const updateNeMoSettingsFn = async (settings: NeMoSettingsUpdate) => {
    const response = await api.patch<NeMoSettingsResponse>(
      `${getURL("SETTINGS")}/nemo`,
      settings,
    );
    return response.data;
  };

  const mutationResult = mutation(updateNeMoSettingsFn, {
    ...options,
  });

  return mutationResult;
};

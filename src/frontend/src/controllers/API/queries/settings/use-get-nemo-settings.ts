import { useQueryFunctionType } from "../../../../types/api";
import { api } from "../../api";
import { getURL } from "../../helpers/constants";
import { UseRequestProcessor } from "../../services/request-processor";

export interface NeMoSettingsResponse {
  nemo_use_mock: boolean;
  nemo_base_url: string;
  nemo_api_key: string | null;
}

export const useGetNeMoSettings: useQueryFunctionType<undefined, NeMoSettingsResponse> = (
  options,
) => {
  const { query } = UseRequestProcessor();

  const getNeMoSettingsFn = async () => {
    const response = await api.get<NeMoSettingsResponse>(`${getURL("SETTINGS")}/nemo`);
    return response.data;
  };

  const queryResult = query(["useGetNeMoSettings"], getNeMoSettingsFn, {
    refetchOnWindowFocus: false,
    ...options,
  });

  return queryResult;
};

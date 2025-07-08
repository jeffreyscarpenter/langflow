import {
  EDIT_PASSWORD_ALERT_LIST,
  EDIT_PASSWORD_ERROR_ALERT,
  SAVE_ERROR_ALERT,
  SAVE_SUCCESS_ALERT,
} from "@/constants/alerts_constants";
import { usePostAddApiKey } from "@/controllers/API/queries/api-keys";
import {
  useResetPassword,
  useUpdateUser,
} from "@/controllers/API/queries/auth";
import { useGetProfilePicturesQuery } from "@/controllers/API/queries/files";
import { useGetNeMoSettings } from "@/controllers/API/queries/settings/use-get-nemo-settings";
import { useUpdateNeMoSettings } from "@/controllers/API/queries/settings/use-update-nemo-settings";
import { ENABLE_PROFILE_ICONS } from "@/customization/feature-flags";
import useAuthStore from "@/stores/authStore";
import { cloneDeep } from "lodash";
import { useContext, useEffect, useState } from "react";
import { useParams } from "react-router-dom";
import { CONTROL_PATCH_USER_STATE } from "../../../../constants/constants";
import { AuthContext } from "../../../../contexts/authContext";
import useAlertStore from "../../../../stores/alertStore";
import { useStoreStore } from "../../../../stores/storeStore";
import {
  inputHandlerEventType,
  patchUserInputStateType,
} from "../../../../types/components";
import useScrollToElement from "../hooks/use-scroll-to-element";
import GeneralPageHeaderComponent from "./components/GeneralPageHeader";
import NeMoSettingsForm from "./components/NeMoSettingsForm";
import PasswordFormComponent from "./components/PasswordForm";
import ProfilePictureFormComponent from "./components/ProfilePictureForm";

export const GeneralPage = () => {
  const { scrollId } = useParams();

  const [inputState, setInputState] = useState<patchUserInputStateType>({
    ...CONTROL_PATCH_USER_STATE,
    nemoUseMock: true,
    nemoApiKey: "",
    nemoBaseUrl: "https://us-west-2.api-dev.ai.datastax.com/nvidia/nemo",
  });

  // Get NeMo settings from API
  const { data: nemoSettings, isLoading: isLoadingNeMoSettings } = useGetNeMoSettings();

  // Update NeMo settings mutation
  const { mutate: updateNeMoSettings, isPending: isUpdatingNeMoSettings } = useUpdateNeMoSettings({
    onSuccess: () => {
      setSuccessData({ title: "NeMo settings saved successfully" });
    },
    onError: (error) => {
      setErrorData({
        title: "Failed to save NeMo settings",
        list: [(error as any)?.response?.data?.detail || "Unknown error"],
      });
    },
  });

  const setSuccessData = useAlertStore((state) => state.setSuccessData);
  const setErrorData = useAlertStore((state) => state.setErrorData);
  const { userData, setUserData } = useContext(AuthContext);
  const { password, cnfPassword, profilePicture, nemoUseMock, nemoApiKey, nemoBaseUrl } = inputState;
  const autoLogin = useAuthStore((state) => state.autoLogin);

  const { storeApiKey } = useContext(AuthContext);
  const setHasApiKey = useStoreStore((state) => state.updateHasApiKey);
  const setValidApiKey = useStoreStore((state) => state.updateValidApiKey);
  const setLoadingApiKey = useStoreStore((state) => state.updateLoadingApiKey);

  const { mutate: mutateResetPassword } = useResetPassword();
  const { mutate: mutatePatchUser } = useUpdateUser();

  const handlePatchPassword = () => {
    if (password !== cnfPassword) {
      setErrorData({
        title: EDIT_PASSWORD_ERROR_ALERT,
        list: [EDIT_PASSWORD_ALERT_LIST],
      });
      return;
    }

    if (password !== "") {
      mutateResetPassword(
        { user_id: userData!.id, password: { password } },
        {
          onSuccess: () => {
            handleInput({ target: { name: "password", value: "" } });
            handleInput({ target: { name: "cnfPassword", value: "" } });
            setSuccessData({ title: SAVE_SUCCESS_ALERT });
          },
          onError: (error) => {
            setErrorData({
              title: SAVE_ERROR_ALERT,
              list: [(error as any)?.response?.data?.detail],
            });
          },
        },
      );
    }
  };

  const handleGetProfilePictures = useGetProfilePicturesQuery();

  const handlePatchProfilePicture = (profile_picture) => {
    if (profile_picture !== "") {
      mutatePatchUser(
        { user_id: userData!.id, user: { profile_image: profile_picture } },
        {
          onSuccess: () => {
            let newUserData = cloneDeep(userData);
            newUserData!.profile_image = profile_picture;
            setUserData(newUserData);
            setSuccessData({ title: SAVE_SUCCESS_ALERT });
          },
          onError: (error) => {
            setErrorData({
              title: SAVE_ERROR_ALERT,
              list: [(error as any)?.response?.data?.detail],
            });
          },
        },
      );
    }
  };

  useScrollToElement(scrollId);

  const { mutate } = usePostAddApiKey({
    onSuccess: () => {
      setSuccessData({ title: "API key saved successfully" });
      setHasApiKey(true);
      setValidApiKey(true);
      setLoadingApiKey(false);
      handleInput({ target: { name: "apikey", value: "" } });
    },
    onError: (error) => {
      setErrorData({
        title: "API key save error",
        list: [(error as any)?.response?.data?.detail],
      });
      setHasApiKey(false);
      setValidApiKey(false);
      setLoadingApiKey(false);
    },
  });

  const handleSaveKey = (apikey: string) => {
    if (apikey) {
      mutate({ key: apikey });
      storeApiKey(apikey);
    }
  };

  function handleInput({
    target: { name, value },
  }: inputHandlerEventType): void {
    setInputState((prev) => ({ ...prev, [name]: value }));
  }

  // Update local state when API data loads
  useEffect(() => {
    if (nemoSettings) {
      setInputState(prev => ({
        ...prev,
        nemoUseMock: nemoSettings.nemo_use_mock,
        nemoApiKey: nemoSettings.nemo_api_key || "",
        nemoBaseUrl: nemoSettings.nemo_base_url,
      }));
    }
  }, [nemoSettings]);

  const handleSaveNeMoSettings = () => {
    updateNeMoSettings({
      nemo_use_mock: nemoUseMock,
      nemo_api_key: nemoApiKey,
      nemo_base_url: nemoBaseUrl,
    });
  };

  return (
    <div className="flex h-full w-full flex-col gap-6 overflow-x-hidden">
      <GeneralPageHeaderComponent />

      <div className="flex w-full flex-col gap-6">
        {ENABLE_PROFILE_ICONS && (
          <ProfilePictureFormComponent
            profilePicture={profilePicture}
            handleInput={handleInput}
            handlePatchProfilePicture={handlePatchProfilePicture}
            handleGetProfilePictures={handleGetProfilePictures}
            userData={userData}
          />
        )}

        {!autoLogin && (
          <PasswordFormComponent
            password={password}
            cnfPassword={cnfPassword}
            handleInput={handleInput}
            handlePatchPassword={handlePatchPassword}
          />
        )}

        <NeMoSettingsForm
          nemoUseMock={nemoUseMock}
          nemoApiKey={nemoApiKey}
          nemoBaseUrl={nemoBaseUrl}
          handleInput={handleInput}
          onSave={handleSaveNeMoSettings}
          isSaving={isUpdatingNeMoSettings}
          isLoading={isLoadingNeMoSettings}
        />
      </div>
    </div>
  );
};

export default GeneralPage;

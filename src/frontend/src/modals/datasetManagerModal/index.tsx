import useAlertStore from "@/stores/alertStore";
import { NeMoDataset } from "@/types/nemo-datastore";
import { useQueryClient } from "@tanstack/react-query";
import { ReactNode, useEffect, useState } from "react";
import { ForwardedIconComponent } from "../../components/common/genericIconComponent";
import BaseModal from "../baseModal";
import CreateDatasetComponent from "./components/createDatasetComponent";
import RecentDatasetsComponent from "./components/recentDatasetsComponent";

export default function DatasetManagerModal({
  children,
  handleSubmit,
  selectedDatasets,
  disabled,
  datasets,
  types,
  isList,
}: {
  children?: ReactNode;
  selectedDatasets?: string[];
  open?: boolean;
  handleSubmit: (datasets: string[]) => void;
  setOpen?: (open: boolean) => void;
  disabled?: boolean;
  datasets: NeMoDataset[];
  types: string[];
  isList?: boolean;
}): JSX.Element {
  const [internalOpen, internalSetOpen] = useState(false);

  const setErrorData = useAlertStore((state) => state.setErrorData);

  const queryClient = useQueryClient();

  useEffect(() => {
    queryClient.refetchQueries({
      queryKey: ["useGetDatasets"],
    });
  }, [internalOpen]);

  const [internalSelectedDatasets, setInternalSelectedDatasets] = useState<string[]>(
    selectedDatasets || [],
  );

  useEffect(() => {
    setInternalSelectedDatasets(selectedDatasets || []);
  }, [internalOpen]);

  const handleCreate = (datasetIds: string[]) => {
    setInternalSelectedDatasets(
      isList ? [...internalSelectedDatasets, ...datasetIds] : [datasetIds[0]],
    );
  };

  return (
    <>
      <BaseModal
        size="smaller-h-full"
        open={!disabled && internalOpen}
        setOpen={internalSetOpen}
        onSubmit={() => {
          if (internalSelectedDatasets.length === 0) {
            setErrorData({
              title: "Please select at least one dataset",
            });
            return;
          }
          handleSubmit(internalSelectedDatasets);
          internalSetOpen(false);
        }}
      >
        <BaseModal.Trigger asChild>
          {children ? children : <></>}
        </BaseModal.Trigger>
        <BaseModal.Header description={null}>
          <span className="flex items-center gap-2 font-medium">
            <div className="rounded-md bg-muted p-1.5">
              <ForwardedIconComponent name="Database" className="h-5 w-5" />
            </div>
            NeMo Datasets
          </span>
        </BaseModal.Header>
        <BaseModal.Content overflowHidden>
          <div className="flex flex-col gap-4 overflow-hidden">
            <div className="flex shrink-0 flex-col">
              <CreateDatasetComponent
                onCreate={handleCreate}
                types={types}
                isList={isList ?? false}
              />
            </div>
            <div className="flex flex-1 flex-col overflow-hidden">
              <RecentDatasetsComponent
                datasets={datasets}
                selectedDatasets={internalSelectedDatasets}
                setSelectedDatasets={setInternalSelectedDatasets}
                types={types}
                isList={isList ?? false}
              />
            </div>
          </div>
        </BaseModal.Content>

        <BaseModal.Footer
          submit={{
            label: `Select datasets`,
            dataTestId: "select-datasets-modal-button",
          }}
        ></BaseModal.Footer>
      </BaseModal>
    </>
  );
}
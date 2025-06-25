import { ICON_STROKE_WIDTH } from "@/constants/constants";
import { useGetDatasets } from "@/controllers/API/queries/nemo-datastore";
import { ENABLE_NEMO_DATASTORE } from "@/customization/feature-flags";
import DatasetManagerModal from "@/modals/datasetManagerModal";
import DatasetsRendererComponent from "@/modals/datasetManagerModal/components/datasetsRendererComponent";
import { cn } from "@/utils/utils";
import { useEffect } from "react";
import useAlertStore from "../../../../../stores/alertStore";
import useFlowsManagerStore from "../../../../../stores/flowsManagerStore";
import IconComponent, {
  ForwardedIconComponent,
} from "../../../../common/genericIconComponent";
import { Button } from "../../../../ui/button";
import { DatasetComponentType, InputProps } from "../../types";

export default function InputDatasetComponent({
  value,
  dataset_path,
  handleOnNewValue,
  disabled,
  datasetTypes,
  isList,
  editNode = false,
  id,
}: InputProps<string, DatasetComponentType>): JSX.Element {
  const currentFlowId = useFlowsManagerStore((state) => state.currentFlowId);
  const setErrorData = useAlertStore((state) => state.setErrorData);

  // Clear component state
  useEffect(() => {
    if (disabled && value !== "") {
      handleOnNewValue({ value: "", dataset_path: "" }, { skipSnapshot: true });
    }
  }, [disabled, handleOnNewValue]);

  const { data: datasets, isLoading } = useGetDatasets();

  const selectedDatasets = Array.isArray(dataset_path)
    ? dataset_path
    : dataset_path
    ? [dataset_path]
    : [];

  const isDisabled = disabled || isLoading;

  return (
    <div className="w-full">
      <div className="flex flex-col gap-2.5">
        <div className="flex items-center gap-2.5">
          {ENABLE_NEMO_DATASTORE ? (
            datasets && (
              <div className="relative flex w-full flex-col gap-2">
                <div className="nopan nowheel flex max-h-44 flex-col overflow-y-auto">
                  <DatasetsRendererComponent
                    datasets={datasets.filter((dataset) =>
                      selectedDatasets.includes(dataset.id),
                    )}
                    handleRemove={(id) => {
                      const newSelectedDatasets = selectedDatasets.filter(
                        (datasetId) => datasetId !== id,
                      );
                      handleOnNewValue({
                        value: isList
                          ? newSelectedDatasets.map(
                              (id) =>
                                datasets.find((d) => d.id === id)?.name,
                            )
                          : (datasets.find((d) => d.id === newSelectedDatasets[0])?.name ?? ""),
                        dataset_path: isList
                          ? newSelectedDatasets
                          : (newSelectedDatasets[0] ?? ""),
                      });
                    }}
                  />
                </div>
                <DatasetManagerModal
                  datasets={datasets}
                  selectedDatasets={selectedDatasets}
                  handleSubmit={(selectedDatasets) => {
                    handleOnNewValue({
                      value: isList
                        ? selectedDatasets.map(
                            (id) => datasets.find((d) => d.id === id)?.name,
                          )
                        : (datasets.find((d) => d.id === selectedDatasets[0])?.name ?? ""),
                      dataset_path: isList
                        ? selectedDatasets
                        : (selectedDatasets[0] ?? ""),
                    });
                  }}
                  disabled={isDisabled}
                  types={datasetTypes}
                  isList={isList}
                >
                  {
                    <div data-testid="input-dataset-component" className="w-full">
                      <Button
                        disabled={isDisabled}
                        variant={
                          selectedDatasets.length !== 0 ? "ghost" : "default"
                        }
                        size={selectedDatasets.length !== 0 ? "iconMd" : "default"}
                        className={cn(
                          selectedDatasets.length !== 0
                            ? "hit-area-icon absolute -top-8 right-0"
                            : "w-full",
                          "font-semibold",
                        )}
                        data-testid="button_open_dataset_management"
                      >
                        {selectedDatasets.length !== 0 ? (
                          <ForwardedIconComponent
                            name="Plus"
                            className="icon-size"
                            strokeWidth={ICON_STROKE_WIDTH}
                          />
                        ) : (
                          <div>Select dataset{isList ? "s" : ""}</div>
                        )}
                      </Button>
                    </div>
                  }
                </DatasetManagerModal>
              </div>
            )
          ) : (
            <div className="relative flex w-full">
              <div className="w-full">
                <input
                  data-testid="input-dataset-component"
                  type="text"
                  className={cn(
                    "primary-input h-9 w-full cursor-pointer rounded-r-none text-sm focus:border-border focus:outline-none focus:ring-0",
                    !value && "text-placeholder-foreground",
                    editNode && "h-6",
                  )}
                  value={value || "Select a dataset..."}
                  readOnly
                  disabled={isDisabled}
                  onClick={() => {
                    setErrorData({
                      title: "NeMo Data Store not enabled",
                      list: ["Please enable NeMo Data Store integration to use this feature."],
                    });
                  }}
                />
              </div>
              <div>
                <Button
                  className={cn(
                    "h-9 w-9 rounded-l-none",
                    value &&
                      "bg-accent-emerald-foreground ring-accent-emerald-foreground hover:bg-accent-emerald-foreground",
                    isDisabled &&
                      "relative top-[1px] h-9 ring-1 ring-border ring-offset-0 hover:ring-border",
                    editNode && "h-6",
                  )}
                  onClick={() => {
                    setErrorData({
                      title: "NeMo Data Store not enabled",
                      list: ["Please enable NeMo Data Store integration to use this feature."],
                    });
                  }}
                  disabled={isDisabled}
                  size="icon"
                  data-testid="button_select_dataset"
                >
                  <IconComponent
                    name={value ? "CircleCheckBig" : "Database"}
                    className={cn(
                      value && "text-background",
                      isDisabled && "text-muted-foreground",
                      "h-4 w-4",
                    )}
                    strokeWidth={2}
                  />
                </Button>
              </div>
            </div>
          )}
        </div>
      </div>
    </div>
  );
}
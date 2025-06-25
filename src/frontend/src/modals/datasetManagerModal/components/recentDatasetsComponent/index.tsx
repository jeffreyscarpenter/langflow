import { NeMoDataset } from "@/types/nemo";

interface RecentDatasetsComponentProps {
  datasets: NeMoDataset[];
  selectedDatasets: string[];
  setSelectedDatasets: (datasets: string[]) => void;
  types: string[];
  isList: boolean;
}

export default function RecentDatasetsComponent({
  datasets,
  selectedDatasets,
  setSelectedDatasets,
  types,
  isList,
}: RecentDatasetsComponentProps) {
  return (
    <div className="space-y-2">
      <h3 className="text-sm font-semibold">Available Datasets</h3>
      <div className="space-y-1">
        {datasets.map((dataset) => (
          <div
            key={dataset.id}
            className={`p-2 border rounded cursor-pointer ${
              selectedDatasets.includes(dataset.id)
                ? "bg-accent border-accent-foreground"
                : "hover:bg-muted"
            }`}
            onClick={() => {
              if (isList) {
                if (selectedDatasets.includes(dataset.id)) {
                  setSelectedDatasets(selectedDatasets.filter(id => id !== dataset.id));
                } else {
                  setSelectedDatasets([...selectedDatasets, dataset.id]);
                }
              } else {
                setSelectedDatasets([dataset.id]);
              }
            }}
          >
            <div className="flex items-center justify-between">
              <div>
                <p className="text-sm font-medium">{dataset.name}</p>
                <p className="text-xs text-muted-foreground">{dataset.description}</p>
              </div>
              <div className="text-xs text-muted-foreground">
                {dataset.metadata?.file_count || 0} files
              </div>
            </div>
          </div>
        ))}
      </div>
    </div>
  );
}
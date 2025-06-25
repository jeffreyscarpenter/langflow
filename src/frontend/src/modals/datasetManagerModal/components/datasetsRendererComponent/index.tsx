import { NeMoDataset } from "@/types/nemo-datastore";
import { Button } from "@/components/ui/button";
import { X } from "lucide-react";

interface DatasetsRendererComponentProps {
  datasets: NeMoDataset[];
  handleRemove: (id: string) => void;
}

export default function DatasetsRendererComponent({
  datasets,
  handleRemove,
}: DatasetsRendererComponentProps) {
  if (datasets.length === 0) {
    return null;
  }

  return (
    <div className="space-y-1">
      {datasets.map((dataset) => (
        <div
          key={dataset.id}
          className="flex items-center justify-between p-2 bg-muted rounded"
        >
          <div className="flex items-center space-x-2">
            <span className="text-sm font-medium">{dataset.name}</span>
            <span className="text-xs text-muted-foreground">
              {dataset.metadata?.file_count || 0} files
            </span>
          </div>
          <Button
            size="sm"
            variant="ghost"
            onClick={() => handleRemove(dataset.id)}
          >
            <X className="h-4 w-4" />
          </Button>
        </div>
      ))}
    </div>
  );
}
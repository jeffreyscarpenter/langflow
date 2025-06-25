import { Button } from "@/components/ui/button";
import { useState } from "react";

interface CreateDatasetComponentProps {
  onCreate: (datasetIds: string[]) => void;
  types: string[];
  isList: boolean;
}

export default function CreateDatasetComponent({
  onCreate,
  types,
  isList,
}: CreateDatasetComponentProps) {
  return (
    <div className="flex flex-col items-center justify-center p-4 border-2 border-dashed border-muted-foreground/25 rounded-lg">
      <h3 className="text-sm font-semibold mb-2">Create New Dataset</h3>
      <p className="text-xs text-muted-foreground mb-4 text-center">
        Create a new dataset for your NeMo training and evaluation workflows.
      </p>
      <Button
        variant="outline"
        size="sm"
        onClick={() => {
          // TODO: Open create dataset dialog
          console.log("Create dataset clicked");
        }}
      >
        Create Dataset
      </Button>
    </div>
  );
}
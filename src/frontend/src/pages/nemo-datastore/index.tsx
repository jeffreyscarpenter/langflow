import React, { useState } from "react";
import { NeMoDataset } from "@/types/nemo-datastore";
import DatasetList from "@/components/nemo-datastore/DatasetList";
import DatasetFiles from "@/components/nemo-datastore/DatasetFiles";
import { Button } from "@/components/ui/button";
import { ArrowLeft, Database } from "lucide-react";

const NeMoDataStorePage: React.FC = () => {
  const [selectedDataset, setSelectedDataset] = useState<NeMoDataset | null>(null);

  const handleDatasetSelect = (dataset: NeMoDataset) => {
    setSelectedDataset(dataset);
  };

  const handleBackToList = () => {
    setSelectedDataset(null);
  };

  return (
    <div className="container mx-auto p-6 max-w-7xl">
      <div className="mb-6">
        <div className="flex items-center space-x-2 mb-2">
          <Database className="h-6 w-6 text-blue-500" />
          <h1 className="text-3xl font-bold">NeMo Data Store</h1>
        </div>
        <p className="text-muted-foreground">
          Manage your datasets and files for NeMo training and inference.
        </p>
      </div>

      {selectedDataset ? (
        <div className="space-y-4">
          <div className="flex items-center space-x-4">
            <Button
              variant="outline"
              onClick={handleBackToList}
              className="flex items-center space-x-2"
            >
              <ArrowLeft className="h-4 w-4" />
              <span>Back to Datasets</span>
            </Button>
            <div className="h-6 w-px bg-border" />
            <h2 className="text-xl font-semibold">{selectedDataset.name}</h2>
          </div>
          <DatasetFiles
            datasetId={selectedDataset.id}
            datasetName={selectedDataset.name}
          />
        </div>
      ) : (
        <DatasetList onDatasetSelect={handleDatasetSelect} />
      )}
    </div>
  );
};

export default NeMoDataStorePage;
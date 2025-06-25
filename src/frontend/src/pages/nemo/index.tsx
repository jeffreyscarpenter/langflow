import React, { useState } from "react";
import { NeMoDataset } from "@/types/nemo";
import DatasetList from "@/components/nemo/DatasetList";
import DatasetFiles from "@/components/nemo/DatasetFiles";
import { Button } from "@/components/ui/button";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import { ArrowLeft, Database, Wrench, Activity } from "lucide-react";

const NeMoMicroservicesPage: React.FC = () => {
  const [selectedDataset, setSelectedDataset] = useState<NeMoDataset | null>(null);
  const [activeTab, setActiveTab] = useState("datasets");

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
          <Activity className="h-6 w-6 text-blue-500" />
          <h1 className="text-3xl font-bold">NeMo Microservices</h1>
        </div>
        <p className="text-muted-foreground">
          Manage datasets, monitor customization jobs, and track evaluations across the NeMo platform.
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
        <Tabs value={activeTab} onValueChange={setActiveTab}>
          <TabsList className="grid w-full grid-cols-2 mb-6">
            <TabsTrigger value="datasets" className="flex items-center space-x-2">
              <Database className="h-4 w-4" />
              <span>Datasets</span>
            </TabsTrigger>
            <TabsTrigger value="jobs" className="flex items-center space-x-2">
              <Wrench className="h-4 w-4" />
              <span>Customizer Jobs</span>
            </TabsTrigger>
          </TabsList>

          <TabsContent value="datasets">
            <DatasetList onDatasetSelect={handleDatasetSelect} />
          </TabsContent>

          <TabsContent value="jobs">
            <div className="text-center py-12">
              <Wrench className="h-12 w-12 text-muted-foreground mx-auto mb-4" />
              <h3 className="text-lg font-semibold mb-2">Job Tracking Coming Soon</h3>
              <p className="text-muted-foreground">
                Monitor your NeMo Customizer jobs with real-time progress and metrics.
              </p>
            </div>
          </TabsContent>
        </Tabs>
      )}
    </div>
  );
};

export default NeMoMicroservicesPage;
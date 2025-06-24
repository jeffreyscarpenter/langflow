import React, { useState } from "react";
import { useGetDatasets, useDeleteDataset } from "@/controllers/API/queries/nemo-datastore";
import { NeMoDataset } from "@/types/nemo-datastore";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { Trash2, Plus, Folder, FileText, Calendar, HardDrive } from "lucide-react";
import { formatDistanceToNow } from "date-fns";
import { formatBytes } from "@/utils/utils";
import useAlertStore from "@/stores/alertStore";
import CreateDatasetDialog from "./CreateDatasetDialog";

interface DatasetListProps {
  onDatasetSelect?: (dataset: NeMoDataset) => void;
  selectedDatasetId?: string;
}

const DatasetList: React.FC<DatasetListProps> = ({ onDatasetSelect, selectedDatasetId }) => {
  const [isCreateDialogOpen, setIsCreateDialogOpen] = useState(false);
  const setSuccessData = useAlertStore((state) => state.setSuccessData);
  const setErrorData = useAlertStore((state) => state.setErrorData);

  const { data: datasets, isLoading, error, refetch } = useGetDatasets();
  const deleteDatasetMutation = useDeleteDataset();

  const handleDeleteDataset = (datasetId: string, datasetName: string) => {
    if (confirm(`Are you sure you want to delete the dataset "${datasetName}"?`)) {
      deleteDatasetMutation.mutate({ datasetId }, {
        onSuccess: () => {
          setSuccessData({
            title: "Dataset deleted",
          });
        },
        onError: (error) => {
          setErrorData({
            title: "Error",
            list: [error?.message || "Failed to delete dataset"],
          });
        },
      });
    }
  };

  const handleDatasetClick = (dataset: NeMoDataset) => {
    onDatasetSelect?.(dataset);
  };

  if (isLoading) {
    return (
      <div className="flex items-center justify-center p-8">
        <div className="animate-spin rounded-full h-8 w-8 border-b-2 border-primary"></div>
      </div>
    );
  }

  if (error) {
    return (
      <div className="flex flex-col items-center justify-center p-8 text-center">
        <p className="text-muted-foreground mb-4">Failed to load datasets</p>
        <Button onClick={() => refetch()} variant="outline">
          Try Again
        </Button>
      </div>
    );
  }

  return (
    <div className="space-y-4">
      <div className="flex items-center justify-between">
        <h2 className="text-2xl font-bold">Datasets</h2>
        <Button onClick={() => setIsCreateDialogOpen(true)}>
          <Plus className="h-4 w-4 mr-2" />
          New Dataset
        </Button>
      </div>

      {datasets && datasets.length > 0 ? (
        <div className="grid gap-4 md:grid-cols-2 lg:grid-cols-3">
          {datasets.map((dataset) => (
            <Card
              key={dataset.id}
              className={`cursor-pointer transition-all hover:shadow-md ${
                selectedDatasetId === dataset.id ? "ring-2 ring-primary" : ""
              }`}
              onClick={() => handleDatasetClick(dataset)}
            >
              <CardHeader className="pb-3">
                <div className="flex items-start justify-between">
                  <div className="flex items-center space-x-2">
                    <Folder className="h-5 w-5 text-blue-500" />
                    <CardTitle className="text-lg">{dataset.name}</CardTitle>
                  </div>
                  <Button
                    variant="ghost"
                    size="sm"
                    onClick={(e) => {
                      e.stopPropagation();
                      handleDeleteDataset(dataset.id, dataset.name);
                    }}
                    disabled={deleteDatasetMutation.isPending}
                  >
                    <Trash2 className="h-4 w-4 text-red-500" />
                  </Button>
                </div>
                {dataset.description && (
                  <CardDescription className="text-sm">
                    {dataset.description}
                  </CardDescription>
                )}
              </CardHeader>
              <CardContent className="pt-0">
                <div className="space-y-2">
                  <div className="flex items-center justify-between text-sm">
                    <div className="flex items-center space-x-1">
                      <FileText className="h-4 w-4 text-muted-foreground" />
                      <span>{dataset.metadata.file_count} files</span>
                    </div>
                    <Badge variant="secondary">{dataset.type}</Badge>
                  </div>
                  <div className="flex items-center justify-between text-sm text-muted-foreground">
                    <div className="flex items-center space-x-1">
                      <HardDrive className="h-4 w-4" />
                      <span>{dataset.metadata.total_size}</span>
                    </div>
                    <div className="flex items-center space-x-1">
                      <Calendar className="h-4 w-4" />
                      <span>
                        {formatDistanceToNow(new Date(dataset.created_at), { addSuffix: true })}
                      </span>
                    </div>
                  </div>
                </div>
              </CardContent>
            </Card>
          ))}
        </div>
      ) : (
        <Card>
          <CardContent className="flex flex-col items-center justify-center py-12">
            <Folder className="h-12 w-12 text-muted-foreground mb-4" />
            <h3 className="text-lg font-semibold mb-2">No datasets found</h3>
            <p className="text-muted-foreground text-center mb-4">
              Create your first dataset to get started with NeMo Data Store.
            </p>
            <Button onClick={() => setIsCreateDialogOpen(true)}>
              <Plus className="h-4 w-4 mr-2" />
              Create Dataset
            </Button>
          </CardContent>
        </Card>
      )}

      <CreateDatasetDialog
        open={isCreateDialogOpen}
        onOpenChange={setIsCreateDialogOpen}
      />
    </div>
  );
};

export default DatasetList;
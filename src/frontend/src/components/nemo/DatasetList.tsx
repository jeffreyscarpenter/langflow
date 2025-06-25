import React, { useState } from "react";
import { useQuery } from "@tanstack/react-query";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { Alert, AlertDescription } from "@/components/ui/alert";
import { Loader2, Database, Plus, Eye, Trash2, Calendar, FileText } from "lucide-react";
import { useGetDatasets } from "@/controllers/API/queries/nemo/use-get-datasets";
import { useDeleteDataset } from "@/controllers/API/queries/nemo/use-delete-dataset";
import { NeMoDataset } from "@/types/nemo";
import CreateDatasetDialog from "./CreateDatasetDialog";
import DatasetPreview from "./DatasetPreview";

interface DatasetListProps {
  onDatasetSelect: (dataset: NeMoDataset) => void;
}

const DatasetList: React.FC<DatasetListProps> = ({ onDatasetSelect }) => {
  const [isCreateDialogOpen, setIsCreateDialogOpen] = useState(false);
  const [previewDataset, setPreviewDataset] = useState<NeMoDataset | null>(null);

  const { data: datasets, isLoading, error, refetch } = useGetDatasets();

  const deleteDatasetMutation = useDeleteDataset();

  const handleDeleteDataset = async (datasetId: string) => {
    if (confirm("Are you sure you want to delete this dataset? This action cannot be undone.")) {
      try {
        await deleteDatasetMutation.mutateAsync({ datasetId });
        refetch();
      } catch (error) {
        console.error("Failed to delete dataset:", error);
        alert("Failed to delete dataset");
      }
    }
  };

  const formatDate = (dateString: string) => {
    return new Date(dateString).toLocaleDateString("en-US", {
      year: "numeric",
      month: "short",
      day: "numeric",
      hour: "2-digit",
      minute: "2-digit",
    });
  };

  if (isLoading) {
    return (
      <div className="flex items-center justify-center p-8">
        <Loader2 className="h-8 w-8 animate-spin" />
        <span className="ml-2">Loading datasets...</span>
      </div>
    );
  }

  if (error) {
    return (
      <Alert variant="destructive">
        <AlertDescription>
          Failed to load datasets: {error.message}
        </AlertDescription>
      </Alert>
    );
  }

  return (
    <div className="space-y-6">
      <div className="flex items-center justify-between">
        <div>
          <h2 className="text-2xl font-bold">Datasets</h2>
          <p className="text-muted-foreground">
            Manage your datasets for NeMo training and evaluation
          </p>
        </div>
        <Button onClick={() => setIsCreateDialogOpen(true)} className="flex items-center space-x-2">
          <Plus className="h-4 w-4" />
          <span>Create Dataset</span>
        </Button>
      </div>

      {datasets && datasets.length > 0 ? (
        <div className="grid gap-4 md:grid-cols-2 lg:grid-cols-3">
          {datasets.map((dataset) => (
            <Card key={dataset.id} className="hover:shadow-md transition-shadow">
              <CardHeader>
                <div className="flex items-start justify-between">
                  <div className="flex items-center space-x-2">
                    <Database className="h-5 w-5 text-blue-500" />
                    <CardTitle className="text-lg">{dataset.name}</CardTitle>
                  </div>
                  <Badge variant="outline">{dataset.namespace}</Badge>
                </div>
                <CardDescription className="line-clamp-2">
                  {dataset.description || "No description provided"}
                </CardDescription>
              </CardHeader>
              <CardContent>
                <div className="space-y-3">
                  <div className="flex items-center justify-between text-sm text-muted-foreground">
                    <div className="flex items-center space-x-1">
                      <Calendar className="h-4 w-4" />
                      <span>Created</span>
                    </div>
                    <span>{formatDate(dataset.created_at)}</span>
                  </div>

                  <div className="flex items-center justify-between text-sm text-muted-foreground">
                    <div className="flex items-center space-x-1">
                      <FileText className="h-4 w-4" />
                      <span>Files</span>
                    </div>
                    <span>{dataset.file_count || 0}</span>
                  </div>

                  <div className="flex items-center space-x-2 pt-2">
                    <Button
                      variant="outline"
                      size="sm"
                      onClick={() => onDatasetSelect(dataset)}
                      className="flex-1"
                    >
                      <FileText className="h-4 w-4 mr-1" />
                      Manage Files
                    </Button>
                    <Button
                      variant="outline"
                      size="sm"
                      onClick={() => setPreviewDataset(dataset)}
                    >
                      <Eye className="h-4 w-4" />
                    </Button>
                    <Button
                      variant="outline"
                      size="sm"
                      onClick={() => handleDeleteDataset(dataset.id)}
                      className="text-red-600 hover:text-red-700"
                    >
                      <Trash2 className="h-4 w-4" />
                    </Button>
                  </div>
                </div>
              </CardContent>
            </Card>
          ))}
        </div>
      ) : (
        <Card className="p-8 text-center">
          <Database className="h-12 w-12 text-muted-foreground mx-auto mb-4" />
          <h3 className="text-lg font-semibold mb-2">No datasets yet</h3>
          <p className="text-muted-foreground mb-4">
            Create your first dataset to get started with NeMo training and evaluation
          </p>
          <Button onClick={() => setIsCreateDialogOpen(true)}>
            <Plus className="h-4 w-4 mr-2" />
            Create Dataset
          </Button>
        </Card>
      )}

      <CreateDatasetDialog
        open={isCreateDialogOpen}
        onOpenChange={setIsCreateDialogOpen}
        onSuccess={() => {
          setIsCreateDialogOpen(false);
          refetch();
        }}
      />

      {previewDataset && (
        <div className="fixed inset-0 bg-black bg-opacity-50 flex items-center justify-center z-50 p-4">
          <div className="bg-background rounded-lg max-w-4xl w-full max-h-[90vh] overflow-y-auto">
            <DatasetPreview
              dataset={previewDataset}
              onClose={() => setPreviewDataset(null)}
            />
          </div>
        </div>
      )}
    </div>
  );
};

export default DatasetList;
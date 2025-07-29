import { useQuery } from "@tanstack/react-query";
import {
  Calendar,
  ChevronLeft,
  ChevronRight,
  Database,
  Eye,
  FileText,
  Loader2,
  Plus,
  Search,
  Trash2,
} from "lucide-react";
import React, { useState } from "react";
import { Alert, AlertDescription } from "@/components/ui/alert";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import {
  Card,
  CardContent,
  CardDescription,
  CardHeader,
  CardTitle,
} from "@/components/ui/card";
import { Input } from "@/components/ui/input";
import { useDeleteDataset } from "@/controllers/API/queries/nemo/use-delete-dataset";
import { useGetDatasetByName } from "@/controllers/API/queries/nemo/use-get-dataset-by-name";
import { useGetDatasets } from "@/controllers/API/queries/nemo/use-get-datasets";
import { NeMoDataset } from "@/types/nemo";
import AddFilesDialog from "./AddFilesDialog";
import CreateDatasetDialog from "./CreateDatasetDialog";
import DatasetPreview from "./DatasetPreview";

interface DatasetListProps {
  onDatasetSelect: (dataset: NeMoDataset) => void;
}

const DatasetList: React.FC<DatasetListProps> = ({ onDatasetSelect }) => {
  const [isCreateDialogOpen, setIsCreateDialogOpen] = useState(false);
  const [previewDataset, setPreviewDataset] = useState<NeMoDataset | null>(
    null,
  );
  const [addFilesDataset, setAddFilesDataset] = useState<NeMoDataset | null>(
    null,
  );
  const [currentPage, setCurrentPage] = useState(1);
  const [searchQuery, setSearchQuery] = useState("");
  const [activeSearchQuery, setActiveSearchQuery] = useState("");

  const handleSearch = () => {
    setActiveSearchQuery(searchQuery.trim());
    setCurrentPage(1); // Reset to first page when searching
  };

  const handleClearSearch = () => {
    setSearchQuery("");
    setActiveSearchQuery("");
    setCurrentPage(1);
  };

  const handleKeyDown = (e: React.KeyboardEvent) => {
    if (e.key === "Enter") {
      handleSearch();
    }
  };

  const {
    data: response,
    isLoading,
    error,
    refetch,
  } = useGetDatasets({
    page: currentPage,
    pageSize: 10,
    datasetName: undefined, // Remove datasetName filtering from paginated call
  });

  // Search for specific dataset by name using direct API call (only when actively searching)
  const {
    data: searchedDataset,
    isLoading: isSearchLoading,
    error: searchError,
  } = useGetDatasetByName({
    namespace: "default",
    datasetName: activeSearchQuery,
    enabled: !!activeSearchQuery, // Only run when there's an active search query
  });

  // Determine which datasets to show: search result or paginated list
  const datasets = activeSearchQuery
    ? searchedDataset
      ? [searchedDataset]
      : []
    : response?.data || [];

  const pagination = {
    page: activeSearchQuery ? 1 : response?.page || 1,
    pageSize: activeSearchQuery ? 1 : response?.page_size || 10,
    total: activeSearchQuery ? (searchedDataset ? 1 : 0) : response?.total || 0,
    hasNext: activeSearchQuery ? false : response?.has_next || false,
    hasPrev: activeSearchQuery ? false : response?.has_prev || false,
  };
  const authError = response?.error;
  const currentLoading = activeSearchQuery ? isSearchLoading : isLoading;
  const currentError = activeSearchQuery ? searchError : error;

  const deleteDatasetMutation = useDeleteDataset();

  const handleDeleteDataset = async (dataset: NeMoDataset) => {
    if (
      confirm(
        "Are you sure you want to delete this dataset? This action cannot be undone.",
      )
    ) {
      try {
        await deleteDatasetMutation.mutateAsync({
          datasetName: dataset.name,
          namespace: dataset.namespace,
        });
        refetch();
      } catch (error) {
        console.error("Failed to delete dataset:", error);
        alert("Failed to delete dataset");
      }
    }
  };

  const handleNextPage = () => {
    if (pagination.hasNext) {
      setCurrentPage((prev) => prev + 1);
    }
  };

  const handlePrevPage = () => {
    if (pagination.hasPrev) {
      setCurrentPage((prev) => prev - 1);
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

  if (currentLoading) {
    return (
      <div className="flex items-center justify-center p-8">
        <Loader2 className="h-8 w-8 animate-spin" />
        <span className="ml-2">Loading datasets...</span>
      </div>
    );
  }

  if (currentError) {
    return (
      <Alert variant="destructive">
        <AlertDescription>
          Failed to load datasets: {currentError.message}
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
        <Button
          onClick={() => setIsCreateDialogOpen(true)}
          className="flex items-center space-x-2"
        >
          <Plus className="h-4 w-4" />
          <span>Create Dataset</span>
        </Button>
      </div>

      {/* Authentication Error Alert */}
      {authError && (
        <Alert variant="destructive">
          <AlertDescription>
            {authError} Please configure your NeMo credentials in the settings
            to access datasets.
          </AlertDescription>
        </Alert>
      )}

      {/* Find Dataset Controls */}
      <div className="flex items-center space-x-4">
        <div className="flex items-center space-x-2 flex-1 max-w-md">
          <div className="relative flex-1">
            <Search className="absolute left-3 top-1/2 transform -translate-y-1/2 text-muted-foreground h-4 w-4" />
            <Input
              placeholder="Enter dataset name to find..."
              value={searchQuery}
              onChange={(e) => setSearchQuery(e.target.value)}
              onKeyDown={handleKeyDown}
              className="pl-10"
            />
          </div>
          <Button
            variant="default"
            size="sm"
            onClick={handleSearch}
            disabled={!searchQuery.trim() || currentLoading}
          >
            Find
          </Button>
          {activeSearchQuery && (
            <Button variant="outline" size="sm" onClick={handleClearSearch}>
              Clear
            </Button>
          )}
        </div>
        <div className="text-sm text-muted-foreground">
          {pagination.total > 0 && (
            <span>
              Showing {(pagination.page - 1) * pagination.pageSize + 1} to{" "}
              {Math.min(
                pagination.page * pagination.pageSize,
                pagination.total,
              )}{" "}
              of {pagination.total} datasets
            </span>
          )}
        </div>
      </div>

      {datasets && datasets.length > 0 ? (
        <div className="grid gap-4 md:grid-cols-2 lg:grid-cols-3">
          {datasets.map((dataset) => (
            <Card
              key={dataset.id}
              className="hover:shadow-md transition-shadow"
            >
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

                  <div className="flex items-center space-x-2 pt-2">
                    <Button
                      variant="outline"
                      size="sm"
                      onClick={() => setAddFilesDataset(dataset)}
                      className="flex-1"
                    >
                      <FileText className="h-4 w-4 mr-1" />
                      Add Files
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
                      onClick={() => handleDeleteDataset(dataset)}
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
      ) : debouncedSearchQuery ? (
        <Card className="p-8 text-center">
          <Search className="h-12 w-12 text-muted-foreground mx-auto mb-4" />
          <h3 className="text-lg font-semibold mb-2">Dataset not found</h3>
          <p className="text-muted-foreground mb-4">
            No dataset named "{activeSearchQuery}" found in the "default"
            namespace.
          </p>
          <Button variant="outline" onClick={handleClearSearch}>
            Clear search
          </Button>
        </Card>
      ) : (
        <Card className="p-8 text-center">
          <Database className="h-12 w-12 text-muted-foreground mx-auto mb-4" />
          <h3 className="text-lg font-semibold mb-2">No datasets yet</h3>
          <p className="text-muted-foreground mb-4">
            Create your first dataset to get started with NeMo training and
            evaluation
          </p>
          <Button onClick={() => setIsCreateDialogOpen(true)}>
            <Plus className="h-4 w-4 mr-2" />
            Create Dataset
          </Button>
        </Card>
      )}

      {/* Pagination Controls */}
      {pagination.total > pagination.pageSize && (
        <div className="flex items-center justify-between">
          <div className="text-sm text-muted-foreground">
            Page {pagination.page} of{" "}
            {Math.ceil(pagination.total / pagination.pageSize)}
            {pagination.total > 0 && (
              <span className="ml-2">({pagination.total} total datasets)</span>
            )}
          </div>
          <div className="flex items-center space-x-2">
            <Button
              variant="outline"
              size="sm"
              onClick={handlePrevPage}
              disabled={!pagination.hasPrev || currentLoading}
              className="flex items-center space-x-1"
              type="button"
            >
              <ChevronLeft className="h-4 w-4" />
              <span>Previous</span>
            </Button>
            <Button
              variant="outline"
              size="sm"
              onClick={handleNextPage}
              disabled={!pagination.hasNext || currentLoading}
              className="flex items-center space-x-1"
              type="button"
            >
              <span>Next</span>
              <ChevronRight className="h-4 w-4" />
            </Button>
          </div>
        </div>
      )}

      <CreateDatasetDialog
        open={isCreateDialogOpen}
        onOpenChange={setIsCreateDialogOpen}
        onSuccess={() => {
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

      {addFilesDataset && (
        <AddFilesDialog
          isOpen={!!addFilesDataset}
          onClose={() => setAddFilesDataset(null)}
          datasetId={addFilesDataset.id}
          datasetName={addFilesDataset.name}
          namespace={addFilesDataset.namespace}
          onSuccess={() => {
            setAddFilesDataset(null);
            refetch();
          }}
        />
      )}
    </div>
  );
};

export default DatasetList;

import {
  BarChart3,
  CheckSquare,
  Database,
  Eye,
  Filter,
  Loader2,
  Settings,
  Tag,
  Trash2,
} from "lucide-react";
import React, { useCallback, useEffect, useState } from "react";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import {
  Card,
  CardContent,
  CardDescription,
  CardHeader,
  CardTitle,
} from "@/components/ui/card";
import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogFooter,
  DialogHeader,
  DialogTitle,
} from "@/components/ui/dialog";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import {
  Table,
  TableBody,
  TableCell,
  TableHead,
  TableHeader,
  TableRow,
} from "@/components/ui/table";
import { nemoApi } from "@/controllers/API/nemo-api";
import type {
  NeMoEvaluationConfig,
  NeMoEvaluationConfigList,
} from "@/types/nemo";

const EvaluationConfigList: React.FC = () => {
  const [configs, setConfigs] = useState<NeMoEvaluationConfig[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [currentPage, setCurrentPage] = useState(1);
  const [totalPages, setTotalPages] = useState(1);
  const [totalConfigs, setTotalConfigs] = useState(0);
  const [pageSize] = useState(10);
  const [selectedConfig, setSelectedConfig] =
    useState<NeMoEvaluationConfig | null>(null);
  const [showDetails, setShowDetails] = useState(false);
  const [showDeleteConfirm, setShowDeleteConfirm] = useState(false);
  const [configToDelete, setConfigToDelete] =
    useState<NeMoEvaluationConfig | null>(null);
  const [deleting, setDeleting] = useState(false);

  // Filters
  const [targetIdFilter, setTargetIdFilter] = useState("");
  const [searchTerm, setSearchTerm] = useState("");

  const loadConfigs = useCallback(
    async (page: number = 1) => {
      try {
        setLoading(true);
        setError(null);
        console.log("EvaluationConfigList: Loading configs page", page);

        const filter: any = {};
        if (targetIdFilter) filter.target_id = targetIdFilter;

        console.log("EvaluationConfigList: Calling API with filter:", filter);
        const response: NeMoEvaluationConfigList =
          await nemoApi.getEvaluationConfigs(page, pageSize, filter);
        console.log("EvaluationConfigList: Raw API response:", response);
        console.log(
          "EvaluationConfigList: Configs in response:",
          response.data?.map((c) => ({ id: c.id, name: c.name })),
        );

        let filteredConfigs = (response.data || []).filter(
          (config) => config && config.id && config.name,
        );

        // Apply client-side search filter
        if (searchTerm) {
          filteredConfigs = filteredConfigs.filter(
            (config) =>
              config.name?.toLowerCase().includes(searchTerm.toLowerCase()) ||
              config.target?.name
                ?.toLowerCase()
                .includes(searchTerm.toLowerCase()) ||
              config.params?.evaluation_type
                ?.toLowerCase()
                .includes(searchTerm.toLowerCase()),
          );
        }

        setConfigs(filteredConfigs);
        setCurrentPage(response.page || page);
        setTotalPages(response.total_pages || 1);
        setTotalConfigs(response.total || 0);

        if (response.error) {
          setError(response.error);
        }
      } catch (err: any) {
        console.error("Error loading evaluation configs:", err);
        setError(
          `Failed to load evaluation configs: ${err.message || "Unknown error"}`,
        );
        setConfigs([]);
      } finally {
        setLoading(false);
      }
    },
    [targetIdFilter, searchTerm, pageSize],
  );

  useEffect(() => {
    loadConfigs(1);
  }, [loadConfigs]);

  useEffect(() => {
    if (currentPage !== 1) {
      loadConfigs(currentPage);
    }
  }, [currentPage, loadConfigs]);

  const handlePageChange = (page: number) => {
    setCurrentPage(page);
  };

  const handleFilterChange = () => {
    setCurrentPage(1);
    loadConfigs(1);
  };

  const handleViewDetails = async (config: NeMoEvaluationConfig) => {
    try {
      const detailedConfig = await nemoApi.getEvaluationConfig(config.id);
      setSelectedConfig(detailedConfig);
      setShowDetails(true);
    } catch (err) {
      console.error("Error loading config details:", err);
      setSelectedConfig(config);
      setShowDetails(true);
    }
  };

  const handleDeleteClick = (config: NeMoEvaluationConfig) => {
    setConfigToDelete(config);
    setShowDeleteConfirm(true);
  };

  const handleDeleteConfirm = async () => {
    if (!configToDelete) return;

    try {
      setDeleting(true);
      console.log(
        "EvaluationConfigList: Starting delete for config:",
        configToDelete.id,
      );

      const deleteResult = await nemoApi.deleteEvaluationConfig(
        configToDelete.id,
      );

      // Only remove from UI if delete actually succeeded (optimistic update)
      const deletedConfigId = configToDelete.id;
      setConfigs((currentConfigs) =>
        currentConfigs.filter((config) => config.id !== deletedConfigId),
      );
      setTotalConfigs((prev) => Math.max(0, prev - 1));

      // Show success message briefly
      setError(`✅ Config "${configToDelete.name}" deleted successfully.`);

      setShowDeleteConfirm(false);
      setConfigToDelete(null);

      // Clear success message and refresh after short delay
      setTimeout(async () => {
        try {
          await loadConfigs(currentPage);
          setError(null); // Clear success message
        } catch (refreshErr) {
          console.error("Error during background refresh:", refreshErr);
        } finally {
          setDeleting(false);
        }
      }, 2000);
    } catch (err: any) {
      console.error("Error deleting config:", err);

      let errorMessage = "Unknown error";
      if (err.response?.status === 404) {
        errorMessage = "Config not found - may have been deleted already";
      } else if (err.response?.status === 500) {
        errorMessage = "Server error - the config may not support deletion";
      } else if (err.message) {
        errorMessage = err.message;
      }

      setError(`Failed to delete config: ${errorMessage}`);

      // Refresh list to get current state
      setTimeout(() => loadConfigs(currentPage), 1000);

      setDeleting(false);
    }
  };

  const handleDeleteCancel = () => {
    setShowDeleteConfirm(false);
    setConfigToDelete(null);
  };

  const getEvaluationTypeColor = (type: string) => {
    switch (type?.toLowerCase()) {
      case "agentic":
        return "bg-blue-500 hover:bg-blue-600";
      case "bfcl":
        return "bg-purple-500 hover:bg-purple-600";
      case "bigcode":
        return "bg-green-500 hover:bg-green-600";
      case "custom":
        return "bg-orange-500 hover:bg-orange-600";
      case "lm_harness":
        return "bg-cyan-500 hover:bg-cyan-600";
      case "rag":
        return "bg-red-500 hover:bg-red-600";
      case "retriever":
        return "bg-yellow-500 hover:bg-yellow-600";
      default:
        return "bg-gray-500 hover:bg-gray-600";
    }
  };

  const formatEvaluationType = (type: string) => {
    switch (type?.toLowerCase()) {
      case "agentic":
        return "Agentic";
      case "bfcl":
        return "BFCL";
      case "bigcode":
        return "BigCode";
      case "custom":
        return "Custom";
      case "lm_harness":
        return "LM Harness";
      case "rag":
        return "RAG";
      case "retriever":
        return "Retriever";
      default:
        return type || "Unknown";
    }
  };

  return (
    <div className="space-y-6">
      <Card>
        <CardHeader>
          <CardTitle className="flex items-center gap-2">
            <BarChart3 className="h-5 w-5" />
            Evaluation Configs
          </CardTitle>
          <CardDescription>
            Manage and view evaluation configurations for different evaluation
            types
          </CardDescription>
        </CardHeader>
        <CardContent>
          {/* Filters */}
          <div className="mb-6 grid grid-cols-1 gap-4 md:grid-cols-3">
            <div>
              <Label htmlFor="search">Search</Label>
              <Input
                id="search"
                placeholder="Search configs..."
                value={searchTerm}
                onChange={(e) => setSearchTerm(e.target.value)}
                onKeyDown={(e) => e.key === "Enter" && handleFilterChange()}
              />
            </div>
            <div>
              <Label htmlFor="targetId">Target ID</Label>
              <Input
                id="targetId"
                placeholder="Filter by target ID..."
                value={targetIdFilter}
                onChange={(e) => setTargetIdFilter(e.target.value)}
                onKeyDown={(e) => e.key === "Enter" && handleFilterChange()}
              />
            </div>
            <div className="flex items-end">
              <Button onClick={handleFilterChange} className="w-full">
                <Filter className="mr-2 h-4 w-4" />
                Apply Filters
              </Button>
            </div>
          </div>

          {loading ? (
            <div className="flex justify-center py-8">
              <Loader2 className="h-8 w-8 animate-spin" />
            </div>
          ) : error ? (
            <div className="rounded-md bg-red-50 p-4">
              <p className="text-sm text-red-700">{error}</p>
            </div>
          ) : configs.length === 0 ? (
            <div className="text-center py-8">
              <BarChart3 className="mx-auto h-12 w-12 text-gray-400" />
              <h3 className="mt-2 text-sm font-medium text-gray-900">
                No configs found
              </h3>
              <p className="mt-1 text-sm text-gray-500">
                No evaluation configs match your current filters.
              </p>
            </div>
          ) : (
            <>
              {/* Configs Table */}
              <div className="rounded-md border">
                <Table>
                  <TableHeader>
                    <TableRow>
                      <TableHead>Name</TableHead>
                      <TableHead>Target</TableHead>
                      <TableHead>Evaluation Type</TableHead>
                      <TableHead>Metrics</TableHead>
                      <TableHead>Updated</TableHead>
                      <TableHead>Actions</TableHead>
                    </TableRow>
                  </TableHeader>
                  <TableBody>
                    {configs.map((config) => (
                      <TableRow key={config.id}>
                        <TableCell>
                          <div>
                            <p className="font-medium">
                              {config.name || "N/A"}
                            </p>
                            <p className="text-sm text-gray-500">
                              {config.namespace || "N/A"}
                            </p>
                          </div>
                        </TableCell>
                        <TableCell>
                          <div>
                            <p className="text-sm font-medium">
                              {config.target?.name || "N/A"}
                            </p>
                            <code className="text-xs text-gray-500">
                              {config.target?.id || "N/A"}
                            </code>
                          </div>
                        </TableCell>
                        <TableCell>
                          <Badge
                            className={getEvaluationTypeColor(
                              config.params?.evaluation_type || "unknown",
                            )}
                          >
                            {formatEvaluationType(
                              config.params?.evaluation_type || "unknown",
                            )}
                          </Badge>
                        </TableCell>
                        <TableCell>
                          <div className="flex flex-wrap gap-1">
                            {config.params?.metrics
                              ?.slice(0, 2)
                              .map((metric, index) => (
                                <Badge
                                  key={index}
                                  variant="outline"
                                  className="text-xs"
                                >
                                  {metric}
                                </Badge>
                              )) || (
                              <span className="text-sm text-gray-500">N/A</span>
                            )}
                            {config.params?.metrics &&
                              config.params.metrics.length > 2 && (
                                <Badge variant="outline" className="text-xs">
                                  +{config.params.metrics.length - 2} more
                                </Badge>
                              )}
                          </div>
                        </TableCell>
                        <TableCell>
                          <span className="text-sm text-gray-500">
                            {config.updated_at
                              ? new Date(config.updated_at).toLocaleDateString()
                              : "N/A"}
                          </span>
                        </TableCell>
                        <TableCell>
                          <div className="flex items-center space-x-2">
                            <Button
                              variant="outline"
                              size="sm"
                              onClick={() => handleViewDetails(config)}
                            >
                              <Eye className="h-4 w-4" />
                            </Button>
                            <Button
                              variant="outline"
                              size="sm"
                              onClick={() => handleDeleteClick(config)}
                              className="text-red-600 hover:text-red-700 hover:bg-red-50"
                            >
                              <Trash2 className="h-4 w-4" />
                            </Button>
                          </div>
                        </TableCell>
                      </TableRow>
                    ))}
                  </TableBody>
                </Table>
              </div>

              {/* Pagination */}
              <div className="flex items-center justify-between">
                <p className="text-sm text-gray-700">
                  Showing {configs.length} of {totalConfigs} configs
                </p>
                <div className="flex items-center space-x-2">
                  <Button
                    variant="outline"
                    size="sm"
                    onClick={() => handlePageChange(currentPage - 1)}
                    disabled={currentPage <= 1}
                  >
                    Previous
                  </Button>
                  <span className="text-sm">
                    Page {currentPage} of {totalPages}
                  </span>
                  <Button
                    variant="outline"
                    size="sm"
                    onClick={() => handlePageChange(currentPage + 1)}
                    disabled={currentPage >= totalPages}
                  >
                    Next
                  </Button>
                </div>
              </div>
            </>
          )}
        </CardContent>
      </Card>

      {/* Config Details Modal */}
      <Dialog open={showDetails} onOpenChange={setShowDetails}>
        <DialogContent className="max-w-4xl">
          <DialogHeader>
            <DialogTitle>Evaluation Config Details</DialogTitle>
          </DialogHeader>
          {selectedConfig && (
            <div className="space-y-6">
              {/* Basic Info */}
              <div className="grid grid-cols-2 gap-4">
                <div>
                  <Label className="font-semibold">Name</Label>
                  <p className="text-sm">{selectedConfig.name || "N/A"}</p>
                </div>
                <div>
                  <Label className="font-semibold">Namespace</Label>
                  <p className="text-sm">{selectedConfig.namespace || "N/A"}</p>
                </div>
                <div>
                  <Label className="font-semibold">Target Name</Label>
                  <p className="text-sm">
                    {selectedConfig.target?.name || "N/A"}
                  </p>
                </div>
                <div>
                  <Label className="font-semibold">Target ID</Label>
                  <p className="text-sm font-mono">
                    {selectedConfig.target?.id || "N/A"}
                  </p>
                </div>
              </div>

              {/* Evaluation Parameters */}
              <div>
                <Label className="font-semibold text-base">
                  Evaluation Parameters
                </Label>
                <div className="mt-2 grid grid-cols-2 gap-4">
                  <div>
                    <Label className="font-semibold">Evaluation Type</Label>
                    <Badge
                      className={getEvaluationTypeColor(
                        selectedConfig.params?.evaluation_type || "unknown",
                      )}
                    >
                      {formatEvaluationType(
                        selectedConfig.params?.evaluation_type || "unknown",
                      )}
                    </Badge>
                  </div>
                  {selectedConfig.params?.task && (
                    <div>
                      <Label className="font-semibold">Task</Label>
                      <p className="text-sm">{selectedConfig.params.task}</p>
                    </div>
                  )}
                  {selectedConfig.params?.dataset && (
                    <div>
                      <Label className="font-semibold">Dataset</Label>
                      <p className="text-sm">{selectedConfig.params.dataset}</p>
                    </div>
                  )}
                </div>
              </div>

              {/* Metrics */}
              {selectedConfig.params?.metrics &&
                selectedConfig.params.metrics.length > 0 && (
                  <div>
                    <Label className="font-semibold text-base flex items-center gap-2">
                      <CheckSquare className="h-4 w-4" />
                      Metrics
                    </Label>
                    <div className="mt-2 flex flex-wrap gap-2">
                      {selectedConfig.params.metrics.map((metric, index) => (
                        <Badge key={index} variant="outline">
                          {metric}
                        </Badge>
                      ))}
                    </div>
                  </div>
                )}

              {/* Tasks (for LM Harness) */}
              {selectedConfig.params?.tasks &&
                selectedConfig.params.tasks.length > 0 && (
                  <div>
                    <Label className="font-semibold text-base flex items-center gap-2">
                      <Tag className="h-4 w-4" />
                      Tasks (LM Harness)
                    </Label>
                    <div className="mt-2 flex flex-wrap gap-2">
                      {selectedConfig.params.tasks.map((task, index) => (
                        <Badge key={index} variant="outline">
                          {task}
                        </Badge>
                      ))}
                    </div>
                  </div>
                )}

              {/* RAG Metrics */}
              {(selectedConfig.params?.retrieval_metrics ||
                selectedConfig.params?.generation_metrics) && (
                <div>
                  <Label className="font-semibold text-base">RAG Metrics</Label>
                  <div className="mt-2 grid grid-cols-2 gap-4">
                    {selectedConfig.params.retrieval_metrics && (
                      <div>
                        <Label className="font-semibold">
                          Retrieval Metrics
                        </Label>
                        <div className="flex flex-wrap gap-1 mt-1">
                          {selectedConfig.params.retrieval_metrics.map(
                            (metric, index) => (
                              <Badge
                                key={index}
                                variant="outline"
                                className="text-xs"
                              >
                                {metric}
                              </Badge>
                            ),
                          )}
                        </div>
                      </div>
                    )}
                    {selectedConfig.params.generation_metrics && (
                      <div>
                        <Label className="font-semibold">
                          Generation Metrics
                        </Label>
                        <div className="flex flex-wrap gap-1 mt-1">
                          {selectedConfig.params.generation_metrics.map(
                            (metric, index) => (
                              <Badge
                                key={index}
                                variant="outline"
                                className="text-xs"
                              >
                                {metric}
                              </Badge>
                            ),
                          )}
                        </div>
                      </div>
                    )}
                  </div>
                </div>
              )}

              {/* Custom Metrics */}
              {selectedConfig.params?.custom_metrics &&
                selectedConfig.params.custom_metrics.length > 0 && (
                  <div>
                    <Label className="font-semibold text-base">
                      Custom Metrics
                    </Label>
                    <div className="mt-2 rounded bg-gray-100 p-3">
                      <pre className="text-sm whitespace-pre-wrap">
                        {JSON.stringify(
                          selectedConfig.params.custom_metrics,
                          null,
                          2,
                        )}
                      </pre>
                    </div>
                  </div>
                )}

              {/* Full Parameters (if not covered above) */}
              <div>
                <Label className="font-semibold text-base">
                  Full Parameters
                </Label>
                <div className="mt-2 rounded bg-gray-100 p-3 max-h-60 overflow-y-auto">
                  <pre className="text-sm whitespace-pre-wrap">
                    {JSON.stringify(selectedConfig.params, null, 2)}
                  </pre>
                </div>
              </div>

              {/* Timestamps */}
              <div className="grid grid-cols-2 gap-4">
                <div>
                  <Label className="font-semibold">Created</Label>
                  <p className="text-sm">
                    {selectedConfig.created_at
                      ? new Date(selectedConfig.created_at).toLocaleString()
                      : "N/A"}
                  </p>
                </div>
                <div>
                  <Label className="font-semibold">Updated</Label>
                  <p className="text-sm">
                    {selectedConfig.updated_at
                      ? new Date(selectedConfig.updated_at).toLocaleString()
                      : "N/A"}
                  </p>
                </div>
              </div>
            </div>
          )}
        </DialogContent>
      </Dialog>

      {/* Delete Confirmation Dialog */}
      <Dialog open={showDeleteConfirm} onOpenChange={setShowDeleteConfirm}>
        <DialogContent>
          <DialogHeader>
            <DialogTitle>Delete Config</DialogTitle>
            <DialogDescription>
              Are you sure you want to delete the config "{configToDelete?.name}
              "? This action cannot be undone.
            </DialogDescription>
          </DialogHeader>
          <DialogFooter>
            <Button
              variant="outline"
              onClick={handleDeleteCancel}
              disabled={deleting}
            >
              Cancel
            </Button>
            <Button
              variant="destructive"
              onClick={handleDeleteConfirm}
              disabled={deleting}
            >
              {deleting ? (
                <>
                  <Loader2 className="mr-2 h-4 w-4 animate-spin" />
                  Deleting...
                </>
              ) : (
                <>
                  <Trash2 className="mr-2 h-4 w-4" />
                  Delete
                </>
              )}
            </Button>
          </DialogFooter>
        </DialogContent>
      </Dialog>
    </div>
  );
};

export default EvaluationConfigList;

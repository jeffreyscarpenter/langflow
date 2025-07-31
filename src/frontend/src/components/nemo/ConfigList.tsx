import {
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
  NeMoCustomizationConfig,
  NeMoCustomizationConfigList,
} from "@/types/nemo";

const ConfigList: React.FC = () => {
  const [configs, setConfigs] = useState<NeMoCustomizationConfig[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [currentPage, setCurrentPage] = useState(1);
  const [totalPages, setTotalPages] = useState(1);
  const [totalConfigs, setTotalConfigs] = useState(0);
  const [pageSize] = useState(10);
  const [selectedConfig, setSelectedConfig] =
    useState<NeMoCustomizationConfig | null>(null);
  const [showDetails, setShowDetails] = useState(false);
  const [showDeleteConfirm, setShowDeleteConfirm] = useState(false);
  const [configToDelete, setConfigToDelete] =
    useState<NeMoCustomizationConfig | null>(null);
  const [deleting, setDeleting] = useState(false);

  // Filters
  const [targetIdFilter, setTargetIdFilter] = useState("");
  const [searchTerm, setSearchTerm] = useState("");

  const loadConfigs = useCallback(
    async (page: number = 1) => {
      try {
        setLoading(true);
        setError(null);
        const filter: any = {};
        if (targetIdFilter) filter.target_id = targetIdFilter;

        const response: NeMoCustomizationConfigList =
          await nemoApi.getCustomizationConfigs(page, pageSize, filter);

        // Fix: Don't require 'id' field since configs don't have it, just require 'name'
        let filteredConfigs = (response.data || []).filter(
          (config) => config && config.name,
        );

        // Apply client-side search filter
        if (searchTerm) {
          filteredConfigs = filteredConfigs.filter(
            (config) =>
              config.name?.toLowerCase().includes(searchTerm.toLowerCase()) ||
              config.target?.name
                ?.toLowerCase()
                .includes(searchTerm.toLowerCase()) ||
              config.params?.training_type
                ?.toLowerCase()
                .includes(searchTerm.toLowerCase()) ||
              config.params?.finetuning_type
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
        console.error("Error loading configs:", err);
        setError(`Failed to load configs: ${err.message || "Unknown error"}`);
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

  const handleViewDetails = async (config: NeMoCustomizationConfig) => {
    try {
      const detailedConfig = await nemoApi.getCustomizationConfig(config.id);
      setSelectedConfig(detailedConfig);
      setShowDetails(true);
    } catch (err) {
      console.error("Error loading config details:", err);
      setSelectedConfig(config);
      setShowDetails(true);
    }
  };

  const handleDeleteClick = (config: NeMoCustomizationConfig) => {
    setConfigToDelete(config);
    setShowDeleteConfirm(true);
  };

  const handleDeleteConfirm = async () => {
    if (!configToDelete) return;

    try {
      setDeleting(true);
      console.log("ConfigList: Starting delete for config:", configToDelete.id);

      const deleteResult = await nemoApi.deleteCustomizationConfig(
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

  const getTrainingTypeColor = (type: string) => {
    switch (type.toLowerCase()) {
      case "sft":
        return "bg-blue-500 hover:bg-blue-600";
      case "dpo":
        return "bg-purple-500 hover:bg-purple-600";
      case "ppo":
        return "bg-green-500 hover:bg-green-600";
      default:
        return "bg-gray-500 hover:bg-gray-600";
    }
  };

  const getFinetuningTypeColor = (type: string) => {
    switch (type.toLowerCase()) {
      case "lora":
        return "bg-orange-500 hover:bg-orange-600";
      case "full":
        return "bg-red-500 hover:bg-red-600";
      case "ptuning":
        return "bg-cyan-500 hover:bg-cyan-600";
      default:
        return "bg-gray-500 hover:bg-gray-600";
    }
  };

  return (
    <div className="space-y-6">
      <Card>
        <CardHeader>
          <CardTitle className="flex items-center gap-2">
            <Settings className="h-5 w-5" />
            Customization Configs
          </CardTitle>
          <CardDescription>
            Manage and view training configurations for model customization
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
              <Database className="mx-auto h-12 w-12 text-gray-400" />
              <h3 className="mt-2 text-sm font-medium text-gray-900">
                No configs found
              </h3>
              <p className="mt-1 text-sm text-gray-500">
                No configs match your current filters.
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
                      <TableHead>Training Type</TableHead>
                      <TableHead>Finetuning Type</TableHead>
                      <TableHead>Max Seq Length</TableHead>
                      <TableHead>Updated</TableHead>
                      <TableHead>Actions</TableHead>
                    </TableRow>
                  </TableHeader>
                  <TableBody>
                    {configs.map((config) => (
                      <TableRow key={config.id}>
                        <TableCell>
                          <div>
                            <p className="font-medium">{config.name}</p>
                            <p className="text-sm text-gray-500">
                              {config.namespace}
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
                            className={getTrainingTypeColor(
                              config.params?.training_type || "unknown",
                            )}
                          >
                            {(
                              config.params?.training_type || "unknown"
                            ).toUpperCase()}
                          </Badge>
                        </TableCell>
                        <TableCell>
                          <Badge
                            className={getFinetuningTypeColor(
                              config.params?.finetuning_type || "unknown",
                            )}
                          >
                            {(
                              config.params?.finetuning_type || "unknown"
                            ).toUpperCase()}
                          </Badge>
                        </TableCell>
                        <TableCell>
                          <span className="text-sm">
                            {config.params?.max_seq_length?.toLocaleString() ||
                              "N/A"}
                          </span>
                        </TableCell>
                        <TableCell>
                          <span className="text-sm text-gray-500">
                            {new Date(config.updated_at).toLocaleDateString()}
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
        <DialogContent className="max-w-3xl">
          <DialogHeader>
            <DialogTitle>Config Details</DialogTitle>
          </DialogHeader>
          {selectedConfig && (
            <div className="space-y-6">
              {/* Basic Info */}
              <div className="grid grid-cols-2 gap-4">
                <div>
                  <Label className="font-semibold">Name</Label>
                  <p className="text-sm">{selectedConfig.name}</p>
                </div>
                <div>
                  <Label className="font-semibold">Namespace</Label>
                  <p className="text-sm">{selectedConfig.namespace}</p>
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

              {/* Training Parameters */}
              <div>
                <Label className="font-semibold text-base">
                  Training Parameters
                </Label>
                <div className="mt-2 grid grid-cols-2 gap-4">
                  <div>
                    <Label className="font-semibold">Training Type</Label>
                    <Badge
                      className={getTrainingTypeColor(
                        selectedConfig.params?.training_type || "unknown",
                      )}
                    >
                      {(
                        selectedConfig.params?.training_type || "unknown"
                      ).toUpperCase()}
                    </Badge>
                  </div>
                  <div>
                    <Label className="font-semibold">Finetuning Type</Label>
                    <Badge
                      className={getFinetuningTypeColor(
                        selectedConfig.params?.finetuning_type || "unknown",
                      )}
                    >
                      {(
                        selectedConfig.params?.finetuning_type || "unknown"
                      ).toUpperCase()}
                    </Badge>
                  </div>
                  <div>
                    <Label className="font-semibold">Max Sequence Length</Label>
                    <p className="text-sm">
                      {selectedConfig.params?.max_seq_length?.toLocaleString() ||
                        "N/A"}
                    </p>
                  </div>
                  <div>
                    <Label className="font-semibold">Training Precision</Label>
                    <p className="text-sm">
                      {selectedConfig.params?.training_precision || "N/A"}
                    </p>
                  </div>
                </div>
                <div className="mt-4">
                  <Label className="font-semibold">Prompt Template</Label>
                  <div className="mt-1 rounded bg-gray-100 p-3">
                    <code className="text-sm">
                      {selectedConfig.params?.prompt_template || "N/A"}
                    </code>
                  </div>
                </div>
              </div>

              {/* LoRA Parameters (if applicable) */}
              {selectedConfig.params?.lora && (
                <div>
                  <Label className="font-semibold text-base flex items-center gap-2">
                    <Tag className="h-4 w-4" />
                    LoRA Parameters
                  </Label>
                  <div className="mt-2 grid grid-cols-2 gap-4">
                    <div>
                      <Label className="font-semibold">Adapter Dimension</Label>
                      <p className="text-sm">
                        {selectedConfig.params.lora?.adapter_dim || "N/A"}
                      </p>
                    </div>
                    <div>
                      <Label className="font-semibold">Alpha</Label>
                      <p className="text-sm">
                        {selectedConfig.params.lora?.alpha || "N/A"}
                      </p>
                    </div>
                    <div className="col-span-2">
                      <Label className="font-semibold">Target Modules</Label>
                      <div className="mt-1 flex flex-wrap gap-2">
                        {selectedConfig.params.lora?.target_modules?.map(
                          (module, index) => (
                            <Badge key={index} variant="outline">
                              {module}
                            </Badge>
                          ),
                        ) || <span className="text-sm text-gray-500">N/A</span>}
                      </div>
                    </div>
                  </div>
                </div>
              )}

              {/* Timestamps */}
              <div className="grid grid-cols-2 gap-4">
                <div>
                  <Label className="font-semibold">Created</Label>
                  <p className="text-sm">
                    {new Date(selectedConfig.created_at).toLocaleString()}
                  </p>
                </div>
                <div>
                  <Label className="font-semibold">Updated</Label>
                  <p className="text-sm">
                    {new Date(selectedConfig.updated_at).toLocaleString()}
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

export default ConfigList;

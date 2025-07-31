import { Cpu, Database, Eye, Filter, Loader2, Trash2 } from "lucide-react";
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
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select";
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
  NeMoCustomizationTarget,
  NeMoCustomizationTargetList,
} from "@/types/nemo";

const TargetList: React.FC = () => {
  const [targets, setTargets] = useState<NeMoCustomizationTarget[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [currentPage, setCurrentPage] = useState(1);
  const [totalPages, setTotalPages] = useState(1);
  const [totalTargets, setTotalTargets] = useState(0);
  const [pageSize] = useState(10);
  const [selectedTarget, setSelectedTarget] =
    useState<NeMoCustomizationTarget | null>(null);
  const [showDetails, setShowDetails] = useState(false);
  const [showDeleteConfirm, setShowDeleteConfirm] = useState(false);
  const [targetToDelete, setTargetToDelete] =
    useState<NeMoCustomizationTarget | null>(null);
  const [deleting, setDeleting] = useState(false);

  // Filters
  const [baseModelFilter, setBaseModelFilter] = useState("");
  const [statusFilter, setStatusFilter] = useState("all");
  const [searchTerm, setSearchTerm] = useState("");

  const loadTargets = useCallback(
    async (page: number = 1) => {
      try {
        setLoading(true);
        setError(null);

        const filter: any = {};
        if (baseModelFilter) filter.base_model = baseModelFilter;
        if (statusFilter && statusFilter !== "all")
          filter.status = statusFilter;

        const response: NeMoCustomizationTargetList =
          await nemoApi.getCustomizationTargets(page, pageSize, filter);

        let filteredTargets = (response.data || []).filter(
          (target) => target && target.id && target.name,
        );

        // Apply client-side search filter
        if (searchTerm) {
          filteredTargets = filteredTargets.filter(
            (target) =>
              target.name?.toLowerCase().includes(searchTerm.toLowerCase()) ||
              target.description
                ?.toLowerCase()
                .includes(searchTerm.toLowerCase()) ||
              target.base_model
                ?.toLowerCase()
                .includes(searchTerm.toLowerCase()),
          );
        }

        setTargets(filteredTargets);
        setCurrentPage(response.page || page);
        setTotalPages(response.total_pages || 1);
        setTotalTargets(response.total || 0);

        if (response.error) {
          setError(response.error);
        }
      } catch (err) {
        console.error("Error loading targets:", err);
        setError(
          "Failed to load targets. Please check your NeMo configuration.",
        );
        setTargets([]);
      } finally {
        setLoading(false);
      }
    },
    [baseModelFilter, statusFilter, searchTerm, pageSize],
  );

  useEffect(() => {
    loadTargets(1);
  }, [loadTargets]);

  useEffect(() => {
    if (currentPage !== 1) {
      loadTargets(currentPage);
    }
  }, [currentPage, loadTargets]);

  const handlePageChange = (page: number) => {
    setCurrentPage(page);
  };

  const handleFilterChange = () => {
    setCurrentPage(1);
    loadTargets(1);
  };

  const handleViewDetails = async (target: NeMoCustomizationTarget) => {
    try {
      const detailedTarget = await nemoApi.getCustomizationTarget(target.id);
      setSelectedTarget(detailedTarget);
      setShowDetails(true);
    } catch (err) {
      console.error("Error loading target details:", err);
      setSelectedTarget(target);
      setShowDetails(true);
    }
  };

  const handleDeleteClick = (target: NeMoCustomizationTarget) => {
    setTargetToDelete(target);
    setShowDeleteConfirm(true);
  };

  const handleDeleteConfirm = async () => {
    if (!targetToDelete) return;

    try {
      setDeleting(true);
      await nemoApi.deleteCustomizationTarget(targetToDelete.id);
      setShowDeleteConfirm(false);
      setTargetToDelete(null);
      // Refresh the current page
      loadTargets(currentPage);
    } catch (err: any) {
      console.error("Error deleting target:", err);

      let errorMessage = "Unknown error";
      if (err.response?.status === 404) {
        errorMessage = "Target was already deleted or does not exist";
      } else if (err.response?.status === 500) {
        errorMessage = "Server error - the target may not support deletion";
      } else if (err.message) {
        errorMessage = err.message;
      }

      setError(`Failed to delete target: ${errorMessage}`);
    } finally {
      setDeleting(false);
    }
  };

  const handleDeleteCancel = () => {
    setShowDeleteConfirm(false);
    setTargetToDelete(null);
  };

  const formatNumber = (num: number) => {
    if (num >= 1e9) return `${(num / 1e9).toFixed(1)}B`;
    if (num >= 1e6) return `${(num / 1e6).toFixed(1)}M`;
    if (num >= 1e3) return `${(num / 1e3).toFixed(1)}K`;
    return num.toString();
  };

  const getStatusColor = (status: string) => {
    switch (status.toLowerCase()) {
      case "ready":
        return "bg-green-500 hover:bg-green-600";
      case "downloading":
        return "bg-blue-500 hover:bg-blue-600";
      case "error":
        return "bg-red-500 hover:bg-red-600";
      case "pending":
        return "bg-yellow-500 hover:bg-yellow-600";
      default:
        return "bg-gray-500 hover:bg-gray-600";
    }
  };

  return (
    <div className="space-y-6">
      <Card>
        <CardHeader>
          <CardTitle className="flex items-center gap-2">
            <Cpu className="h-5 w-5" />
            Customization Targets
          </CardTitle>
          <CardDescription>
            Manage and view available targets for model customization
          </CardDescription>
        </CardHeader>
        <CardContent>
          {/* Filters */}
          <div className="mb-6 grid grid-cols-1 gap-4 md:grid-cols-4">
            <div>
              <Label htmlFor="search">Search</Label>
              <Input
                id="search"
                placeholder="Search targets..."
                value={searchTerm}
                onChange={(e) => setSearchTerm(e.target.value)}
                onKeyDown={(e) => e.key === "Enter" && handleFilterChange()}
              />
            </div>
            <div>
              <Label htmlFor="baseModel">Base Model</Label>
              <Input
                id="baseModel"
                placeholder="Filter by base model..."
                value={baseModelFilter}
                onChange={(e) => setBaseModelFilter(e.target.value)}
                onKeyDown={(e) => e.key === "Enter" && handleFilterChange()}
              />
            </div>
            <div>
              <Label htmlFor="status">Status</Label>
              <Select value={statusFilter} onValueChange={setStatusFilter}>
                <SelectTrigger>
                  <SelectValue placeholder="All statuses" />
                </SelectTrigger>
                <SelectContent>
                  <SelectItem value="all">All statuses</SelectItem>
                  <SelectItem value="ready">Ready</SelectItem>
                  <SelectItem value="downloading">Downloading</SelectItem>
                  <SelectItem value="error">Error</SelectItem>
                  <SelectItem value="pending">Pending</SelectItem>
                </SelectContent>
              </Select>
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
          ) : targets.length === 0 ? (
            <div className="text-center py-8">
              <Database className="mx-auto h-12 w-12 text-gray-400" />
              <h3 className="mt-2 text-sm font-medium text-gray-900">
                No targets found
              </h3>
              <p className="mt-1 text-sm text-gray-500">
                No targets match your current filters.
              </p>
            </div>
          ) : (
            <>
              {/* Targets Table */}
              <div className="rounded-md border">
                <Table>
                  <TableHeader>
                    <TableRow>
                      <TableHead>Name</TableHead>
                      <TableHead>Base Model</TableHead>
                      <TableHead>Status</TableHead>
                      <TableHead>Parameters</TableHead>
                      <TableHead>Precision</TableHead>
                      <TableHead>Updated</TableHead>
                      <TableHead>Actions</TableHead>
                    </TableRow>
                  </TableHeader>
                  <TableBody>
                    {targets.map((target) => (
                      <TableRow key={target.id}>
                        <TableCell>
                          <div>
                            <p className="font-medium">
                              {target.name || "N/A"}
                            </p>
                            <p className="text-sm text-gray-500 truncate max-w-[200px]">
                              {target.description || "No description"}
                            </p>
                          </div>
                        </TableCell>
                        <TableCell>
                          <code className="text-sm bg-gray-100 px-2 py-1 rounded">
                            {target.base_model || "N/A"}
                          </code>
                        </TableCell>
                        <TableCell>
                          <Badge
                            className={getStatusColor(
                              target.status || "unknown",
                            )}
                          >
                            {target.status || "unknown"}
                          </Badge>
                        </TableCell>
                        <TableCell>
                          {formatNumber(target.num_parameters || 0)}
                        </TableCell>
                        <TableCell>
                          <span className="text-sm">
                            {target.precision || "N/A"}
                          </span>
                        </TableCell>
                        <TableCell>
                          <span className="text-sm text-gray-500">
                            {target.updated_at
                              ? new Date(target.updated_at).toLocaleDateString()
                              : "N/A"}
                          </span>
                        </TableCell>
                        <TableCell>
                          <div className="flex items-center space-x-2">
                            <Button
                              variant="outline"
                              size="sm"
                              onClick={() => handleViewDetails(target)}
                            >
                              <Eye className="h-4 w-4" />
                            </Button>
                            <Button
                              variant="outline"
                              size="sm"
                              onClick={() => handleDeleteClick(target)}
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
                  Showing {targets.length} of {totalTargets} targets
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

      {/* Target Details Modal */}
      <Dialog open={showDetails} onOpenChange={setShowDetails}>
        <DialogContent className="max-w-2xl">
          <DialogHeader>
            <DialogTitle>Target Details</DialogTitle>
          </DialogHeader>
          {selectedTarget && (
            <div className="space-y-4">
              <div className="grid grid-cols-2 gap-4">
                <div>
                  <Label className="font-semibold">Name</Label>
                  <p className="text-sm">{selectedTarget.name || "N/A"}</p>
                </div>
                <div>
                  <Label className="font-semibold">Status</Label>
                  <Badge
                    className={getStatusColor(
                      selectedTarget.status || "unknown",
                    )}
                  >
                    {selectedTarget.status || "unknown"}
                  </Badge>
                </div>
                <div>
                  <Label className="font-semibold">Base Model</Label>
                  <p className="text-sm font-mono">
                    {selectedTarget.base_model || "N/A"}
                  </p>
                </div>
                <div>
                  <Label className="font-semibold">Namespace</Label>
                  <p className="text-sm">{selectedTarget.namespace || "N/A"}</p>
                </div>
                <div>
                  <Label className="font-semibold">Parameters</Label>
                  <p className="text-sm">
                    {formatNumber(selectedTarget.num_parameters || 0)}
                  </p>
                </div>
                <div>
                  <Label className="font-semibold">Precision</Label>
                  <p className="text-sm">{selectedTarget.precision || "N/A"}</p>
                </div>
                <div>
                  <Label className="font-semibold">Model URI</Label>
                  <p className="text-sm font-mono break-all">
                    {selectedTarget.model_uri || "N/A"}
                  </p>
                </div>
                <div>
                  <Label className="font-semibold">Model Path</Label>
                  <p className="text-sm font-mono">
                    {selectedTarget.model_path || "N/A"}
                  </p>
                </div>
              </div>
              <div>
                <Label className="font-semibold">Description</Label>
                <p className="text-sm">
                  {selectedTarget.description || "No description"}
                </p>
              </div>
              <div className="grid grid-cols-2 gap-4">
                <div>
                  <Label className="font-semibold">Created</Label>
                  <p className="text-sm">
                    {selectedTarget.created_at
                      ? new Date(selectedTarget.created_at).toLocaleString()
                      : "N/A"}
                  </p>
                </div>
                <div>
                  <Label className="font-semibold">Updated</Label>
                  <p className="text-sm">
                    {selectedTarget.updated_at
                      ? new Date(selectedTarget.updated_at).toLocaleString()
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
            <DialogTitle>Delete Target</DialogTitle>
            <DialogDescription>
              Are you sure you want to delete the target "{targetToDelete?.name}
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

export default TargetList;

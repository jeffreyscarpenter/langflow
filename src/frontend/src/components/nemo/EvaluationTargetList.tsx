import {
  Database,
  Eye,
  Filter,
  Globe,
  Loader2,
  Search,
  Server,
  Target,
  Trash2,
  Users,
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
  NeMoEvaluationTarget,
  NeMoEvaluationTargetList,
} from "@/types/nemo";

const EvaluationTargetList: React.FC = () => {
  const [targets, setTargets] = useState<NeMoEvaluationTarget[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [currentPage, setCurrentPage] = useState(1);
  const [totalPages, setTotalPages] = useState(1);
  const [totalTargets, setTotalTargets] = useState(0);
  const [pageSize] = useState(10);
  const [selectedTarget, setSelectedTarget] =
    useState<NeMoEvaluationTarget | null>(null);
  const [showDetails, setShowDetails] = useState(false);
  const [showDeleteConfirm, setShowDeleteConfirm] = useState(false);
  const [targetToDelete, setTargetToDelete] =
    useState<NeMoEvaluationTarget | null>(null);
  const [deleting, setDeleting] = useState(false);

  // Filters
  const [targetTypeFilter, setTargetTypeFilter] = useState("all");
  const [searchTerm, setSearchTerm] = useState("");

  const loadTargets = useCallback(
    async (page: number = 1) => {
      try {
        setLoading(true);
        setError(null);

        const filter: any = {};
        if (targetTypeFilter && targetTypeFilter !== "all")
          filter.target_type = targetTypeFilter;

        const response: NeMoEvaluationTargetList =
          await nemoApi.getEvaluationTargets(page, pageSize, filter);

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
              target.type?.toLowerCase().includes(searchTerm.toLowerCase()),
          );
        }

        setTargets(filteredTargets);
        setCurrentPage(response.page || page);
        setTotalPages(response.total_pages || 1);
        setTotalTargets(response.total || 0);

        if (response.error) {
          setError(response.error);
        }
      } catch (err: any) {
        console.error("Error loading evaluation targets:", err);
        setError(
          `Failed to load evaluation targets: ${err.message || "Unknown error"}`,
        );
        setTargets([]);
      } finally {
        setLoading(false);
      }
    },
    [targetTypeFilter, searchTerm, pageSize],
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

  const handleViewDetails = async (target: NeMoEvaluationTarget) => {
    try {
      const detailedTarget = await nemoApi.getEvaluationTarget(target.id);
      setSelectedTarget(detailedTarget);
      setShowDetails(true);
    } catch (err) {
      console.error("Error loading target details:", err);
      setSelectedTarget(target);
      setShowDetails(true);
    }
  };

  const handleDeleteClick = (target: NeMoEvaluationTarget) => {
    setTargetToDelete(target);
    setShowDeleteConfirm(true);
  };

  const handleDeleteConfirm = async () => {
    if (!targetToDelete) return;

    try {
      setDeleting(true);
      await nemoApi.deleteEvaluationTarget(targetToDelete.id);
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

  const getTargetTypeIcon = (type: string) => {
    switch (type?.toLowerCase()) {
      case "data_source":
        return <Database className="h-4 w-4" />;
      case "llm_model":
        return <Globe className="h-4 w-4" />;
      case "retriever_pipeline":
        return <Search className="h-4 w-4" />;
      case "rag_pipeline":
        return <Server className="h-4 w-4" />;
      default:
        return <Target className="h-4 w-4" />;
    }
  };

  const getTargetTypeColor = (type: string) => {
    switch (type?.toLowerCase()) {
      case "data_source":
        return "bg-blue-500 hover:bg-blue-600";
      case "llm_model":
        return "bg-green-500 hover:bg-green-600";
      case "retriever_pipeline":
        return "bg-purple-500 hover:bg-purple-600";
      case "rag_pipeline":
        return "bg-orange-500 hover:bg-orange-600";
      default:
        return "bg-gray-500 hover:bg-gray-600";
    }
  };

  const formatTargetType = (type: string) => {
    switch (type?.toLowerCase()) {
      case "data_source":
        return "Data Source";
      case "llm_model":
        return "LLM Model";
      case "retriever_pipeline":
        return "Retriever Pipeline";
      case "rag_pipeline":
        return "RAG Pipeline";
      default:
        return type || "Unknown";
    }
  };

  return (
    <div className="space-y-6">
      <Card>
        <CardHeader>
          <CardTitle className="flex items-center gap-2">
            <Target className="h-5 w-5" />
            Evaluation Targets
          </CardTitle>
          <CardDescription>
            Manage and view targets for model evaluation across different target
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
                placeholder="Search targets..."
                value={searchTerm}
                onChange={(e) => setSearchTerm(e.target.value)}
                onKeyDown={(e) => e.key === "Enter" && handleFilterChange()}
              />
            </div>
            <div>
              <Label htmlFor="targetType">Target Type</Label>
              <Select
                value={targetTypeFilter}
                onValueChange={setTargetTypeFilter}
              >
                <SelectTrigger>
                  <SelectValue placeholder="All types" />
                </SelectTrigger>
                <SelectContent>
                  <SelectItem value="all">All types</SelectItem>
                  <SelectItem value="data_source">Data Source</SelectItem>
                  <SelectItem value="llm_model">LLM Model</SelectItem>
                  <SelectItem value="retriever_pipeline">
                    Retriever Pipeline
                  </SelectItem>
                  <SelectItem value="rag_pipeline">RAG Pipeline</SelectItem>
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
              <Target className="mx-auto h-12 w-12 text-gray-400" />
              <h3 className="mt-2 text-sm font-medium text-gray-900">
                No targets found
              </h3>
              <p className="mt-1 text-sm text-gray-500">
                No evaluation targets match your current filters.
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
                      <TableHead>Type</TableHead>
                      <TableHead>Description</TableHead>
                      <TableHead>Namespace</TableHead>
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
                            <code className="text-xs text-gray-500">
                              {target.id}
                            </code>
                          </div>
                        </TableCell>
                        <TableCell>
                          <Badge
                            className={`${getTargetTypeColor(target.type || "unknown")} flex items-center gap-1 w-fit`}
                          >
                            {getTargetTypeIcon(target.type || "unknown")}
                            {formatTargetType(target.type || "unknown")}
                          </Badge>
                        </TableCell>
                        <TableCell>
                          <p className="text-sm text-gray-600 truncate max-w-[200px]">
                            {target.description || "No description"}
                          </p>
                        </TableCell>
                        <TableCell>
                          <span className="text-sm">
                            {target.namespace || "N/A"}
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
        <DialogContent className="max-w-3xl">
          <DialogHeader>
            <DialogTitle>Evaluation Target Details</DialogTitle>
          </DialogHeader>
          {selectedTarget && (
            <div className="space-y-6">
              {/* Basic Info */}
              <div className="grid grid-cols-2 gap-4">
                <div>
                  <Label className="font-semibold">Name</Label>
                  <p className="text-sm">{selectedTarget.name || "N/A"}</p>
                </div>
                <div>
                  <Label className="font-semibold">Type</Label>
                  <Badge
                    className={`${getTargetTypeColor(selectedTarget.type || "unknown")} flex items-center gap-1 w-fit`}
                  >
                    {getTargetTypeIcon(selectedTarget.type || "unknown")}
                    {formatTargetType(selectedTarget.type || "unknown")}
                  </Badge>
                </div>
                <div>
                  <Label className="font-semibold">Namespace</Label>
                  <p className="text-sm">{selectedTarget.namespace || "N/A"}</p>
                </div>
                <div>
                  <Label className="font-semibold">ID</Label>
                  <p className="text-sm font-mono">{selectedTarget.id}</p>
                </div>
              </div>

              <div>
                <Label className="font-semibold">Description</Label>
                <p className="text-sm">
                  {selectedTarget.description || "No description"}
                </p>
              </div>

              {/* Parameters */}
              {selectedTarget.params &&
                Object.keys(selectedTarget.params).length > 0 && (
                  <div>
                    <Label className="font-semibold text-base">
                      Parameters
                    </Label>
                    <div className="mt-2 rounded bg-gray-100 p-3">
                      <pre className="text-sm whitespace-pre-wrap">
                        {JSON.stringify(selectedTarget.params, null, 2)}
                      </pre>
                    </div>
                  </div>
                )}

              {/* Timestamps */}
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

export default EvaluationTargetList;

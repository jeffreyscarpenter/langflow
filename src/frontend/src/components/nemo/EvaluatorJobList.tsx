import React, { useState } from "react";
import { useGetEvaluatorJobs } from "@/controllers/API/queries/nemo/use-get-evaluator-jobs";
import { useDeleteEvaluatorJob, useGetEvaluatorJobLogs, useGetEvaluatorJobResults, useDownloadEvaluatorJobResults } from "@/controllers/API/queries/nemo/use-evaluator-job-actions";
import { Alert, AlertDescription } from "@/components/ui/alert";
import { Skeleton } from "@/components/ui/skeleton";
import { RefreshCw, AlertCircle, ActivitySquare, Search } from "lucide-react";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { NeMoEvaluatorJob } from "@/types/nemo";
import { Progress } from "@/components/ui/progress";
import useAlertStore from "@/stores/alertStore";
import {
  Dialog,
  DialogContent,
  DialogHeader,
  DialogTitle,
} from "@/components/ui/dialog";
import { Badge } from "@/components/ui/badge";
import { ScrollArea } from "@/components/ui/scroll-area";
import { Separator } from "@/components/ui/separator";
import { Table, TableBody, TableCell, TableHead, TableHeader, TableRow } from "@/components/ui/table";
import {
  Clock,
  CheckCircle,
  XCircle,
  PlayCircle,
  Pause,
  BarChart3,
  Calendar,
  Database,
  Settings,
  AlertTriangle,
  Tag,
  Target,
  FileText,
  Trash2,
  Download,
  Eye,
  Terminal
} from "lucide-react";
import { formatDistanceToNow } from "date-fns";

// Helper function to parse evaluation results for display
const parseEvaluationResults = (data: any) => {
  if (!data || !data.tasks) return [];

  const metrics: Array<{
    metric: string;
    value: number;
    count: number;
    mean: number | null;
    min: number | null;
    max: number | null;
    stddev: number | null;
  }> = [];

  // Iterate through tasks (typically "default_task")
  Object.values(data.tasks).forEach((task: any) => {
    if (task.metrics) {
      // Iterate through each metric type (accuracy, bleu, rouge, etc.)
      Object.entries(task.metrics).forEach(([metricName, metricData]: [string, any]) => {
        if (metricData.scores) {
          // Get the first scoring method (typically "string-check")
          const firstScore = Object.values(metricData.scores)[0] as any;
          if (firstScore) {
            metrics.push({
              metric: metricName.toUpperCase(),
              value: firstScore.value || 0,
              count: firstScore.stats?.count || 0,
              mean: firstScore.stats?.mean,
              min: firstScore.stats?.min,
              max: firstScore.stats?.max,
              stddev: firstScore.stats?.stddev,
            });
          }
        }
      });
    }
  });

  return metrics;
};

const statusColorMap: Record<string, string> = {
  running: "bg-blue-100 text-blue-800 dark:bg-blue-900 dark:text-blue-200",
  completed: "bg-green-100 text-green-800 dark:bg-green-900 dark:text-green-200",
  failed: "bg-red-100 text-red-800 dark:bg-red-900 dark:text-red-200",
  created: "bg-gray-100 text-gray-800 dark:bg-gray-900 dark:text-gray-200",
  cancelled: "bg-gray-100 text-gray-800 dark:bg-gray-900 dark:text-gray-200",
};

const getStatusIcon = (status: string) => {
  switch (status) {
    case "running":
      return <PlayCircle className="h-5 w-5 text-blue-500" />;
    case "completed":
      return <CheckCircle className="h-5 w-5 text-green-500" />;
    case "failed":
      return <XCircle className="h-5 w-5 text-red-500" />;
    case "cancelled":
      return <Pause className="h-5 w-5 text-gray-500" />;
    default:
      return <Clock className="h-5 w-5 text-yellow-500" />;
  }
};

const getStatusColor = (status: string): string => {
  switch (status) {
    case "running":
      return "bg-blue-500 hover:bg-blue-600";
    case "completed":
      return "bg-green-500 hover:bg-green-600";
    case "failed":
      return "bg-red-500 hover:bg-red-600";
    case "cancelled":
      return "bg-gray-500 hover:bg-gray-600";
    default:
      return "bg-yellow-500 hover:bg-yellow-600";
  }
};

const EvaluatorJobListSkeleton: React.FC = () => (
  <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
    {[...Array(6)].map((_, i) => (
      <div key={i} className="space-y-4 p-6 border rounded-lg">
        <div className="flex items-center justify-between">
          <Skeleton className="h-6 w-32" />
          <Skeleton className="h-6 w-20" />
        </div>
        <Skeleton className="h-2 w-full" />
        <div className="space-y-2">
          <Skeleton className="h-4 w-full" />
          <Skeleton className="h-4 w-3/4" />
        </div>
        <Skeleton className="h-8 w-full" />
      </div>
    ))}
  </div>
);

const EmptyState: React.FC = () => (
  <div className="text-center py-12">
    <ActivitySquare className="h-16 w-16 text-muted-foreground mx-auto mb-4" />
    <h3 className="text-xl font-semibold mb-2">No Evaluator Jobs Found</h3>
    <p className="text-muted-foreground mb-4">
      No evaluator jobs are currently available. Jobs will appear here when you create them using the NeMo Evaluator component.
    </p>
    <p className="text-sm text-muted-foreground">
      Jobs are automatically refreshed every 30 seconds to show the latest status.
    </p>
  </div>
);

const EvaluatorJobCard: React.FC<{
  job: NeMoEvaluatorJob;
  jobType: 'customizer' | 'evaluator';
  onViewDetails: (jobId: string) => void;
  onDelete?: (jobId: string) => void;
  onViewLogs?: (jobId: string) => void;
  onViewResults?: (jobId: string) => void;
  onDownloadResults?: (jobId: string) => void;
}> = ({ job, jobType, onViewDetails, onDelete, onViewLogs, onViewResults, onDownloadResults }) => {
  const statusColor = statusColorMap[job.status] || "bg-gray-100 text-gray-800";

  // Safe string conversion for complex objects
  const safeString = (value: any): string => {
    if (value === null || value === undefined) return "";
    if (typeof value === "string") return value;
    if (typeof value === "object") return JSON.stringify(value);
    return String(value);
  };

  // Safe array handling - tags might be undefined
  const safeTags = Array.isArray(job.tags) ? job.tags : [];

  return (
    <div className="space-y-2 p-6 border rounded-lg hover:shadow-md transition">
      <div className="flex items-center justify-between">
        <div className="font-semibold truncate max-w-[60%]">
          {safeString(job.id)}
        </div>
        <div className="flex items-center gap-2">
          <span className={`text-xs px-2 py-1 rounded ${statusColor} capitalize`}>
            {safeString(job.status)}
          </span>
          <div className="flex gap-1 flex-wrap">
            {/* View Details button for evaluator jobs */}
            <Button
              variant="outline"
              size="sm"
              className="text-gray-600 hover:text-gray-700 hover:bg-gray-50 p-1 h-auto"
              onClick={(e) => {
                e.preventDefault();
                e.stopPropagation();
                onViewDetails(job.id);
              }}
            >
              <Eye className="h-3 w-3" />
            </Button>

            {/* Delete button for evaluator jobs */}
            {jobType === 'evaluator' && onDelete && (
              <Button
                variant="outline"
                size="sm"
                className="text-red-600 hover:text-red-700 hover:bg-red-50 p-1 h-auto"
                onClick={(e) => {
                  e.preventDefault();
                  e.stopPropagation();
                  if (confirm(`Are you sure you want to delete evaluator job ${safeString(job.id).slice(-8)}?`)) {
                    onDelete(job.id);
                  }
                }}
              >
                <Trash2 className="h-3 w-3" />
              </Button>
            )}

            {/* Logs button */}
            {onViewLogs && (
              <Button
                variant="outline"
                size="sm"
                className="text-blue-600 hover:text-blue-700 hover:bg-blue-50 p-1 h-auto"
                onClick={(e) => {
                  e.preventDefault();
                  e.stopPropagation();
                  onViewLogs(job.id);
                }}
              >
                <FileText className="h-3 w-3" />
              </Button>
            )}

            {/* Results and Download buttons for completed evaluator jobs */}
            {jobType === 'evaluator' && job.status === 'completed' && (
              <>
                {onViewResults && (
                  <Button
                    variant="outline"
                    size="sm"
                    className="text-green-600 hover:text-green-700 hover:bg-green-50 p-1 h-auto"
                    onClick={(e) => {
                      e.preventDefault();
                      e.stopPropagation();
                      onViewResults(job.id);
                    }}
                  >
                    <BarChart3 className="h-3 w-3" />
                  </Button>
                )}

                {onDownloadResults && (
                  <Button
                    variant="outline"
                    size="sm"
                    className="text-purple-600 hover:text-purple-700 hover:bg-purple-50 p-1 h-auto"
                    onClick={(e) => {
                      e.preventDefault();
                      e.stopPropagation();
                      onDownloadResults(job.id);
                    }}
                  >
                    <Download className="h-3 w-3" />
                  </Button>
                )}
              </>
            )}
          </div>
        </div>
      </div>
      <div className="flex flex-wrap gap-1 mb-1">
        {safeTags.map((tag, index) => (
          <span key={`tag-${index}`} className="text-xs bg-gray-200 dark:bg-gray-700 rounded px-2 py-0.5">
            {safeString(tag)}
          </span>
        ))}
      </div>
      <div className="text-sm text-muted-foreground truncate">Namespace: {safeString(job.namespace)}</div>
      <div className="text-sm text-muted-foreground truncate">Target: {safeString(job.target)}</div>
      <div className="text-sm text-muted-foreground truncate">Config: {safeString(job.config)}</div>
      <div className="text-xs text-gray-400">Created: {safeString(job.created_at) ? new Date(safeString(job.created_at)).toLocaleString() : "-"}</div>
      <div className="text-xs text-gray-400">Updated: {safeString(job.updated_at) ? new Date(safeString(job.updated_at)).toLocaleString() : "-"}</div>
      <div className="text-xs text-gray-600 font-medium mt-2 mb-1">
        Status Message: {job.status_details?.message ? safeString(job.status_details.message) : "-"}
      </div>
      {typeof job.status_details?.percentage_done === "number" && (
        <Progress value={job.status_details.percentage_done} className="h-2" />
      )}
    </div>
  );
};

const EvaluatorJobDetailsModal: React.FC<{
  job: NeMoEvaluatorJob | null;
  isOpen: boolean;
  onClose: () => void;
}> = ({ job, isOpen, onClose }) => {
  if (!job) return null;

  // Safe string conversion for complex objects
  const safeString = (value: any): string => {
    if (value === null || value === undefined) return "";
    if (typeof value === "string") return value;
    if (typeof value === "object") return JSON.stringify(value);
    return String(value);
  };

  // Safe array handling - tags might be undefined
  const safeTags = Array.isArray(job.tags) ? job.tags : [];

  return (
    <Dialog open={isOpen} onOpenChange={onClose}>
      <DialogContent className="max-w-4xl max-h-[90vh]">
        <DialogHeader>
          <DialogTitle className="flex items-center space-x-2">
            <span>Evaluator Job Details: {safeString(job.id).slice(-8)}</span>
            {getStatusIcon(safeString(job.status))}
            <Badge className={`${getStatusColor(safeString(job.status))} text-white`}>
              {safeString(job.status).toUpperCase()}
            </Badge>
          </DialogTitle>
        </DialogHeader>

        <ScrollArea className="max-h-[70vh]">
          <div className="space-y-6">
            {/* Progress Section */}
            <div className="space-y-4">
              <h3 className="text-lg font-semibold flex items-center space-x-2">
                <BarChart3 className="h-5 w-5" />
                <span>Progress</span>
              </h3>

              <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
                <div className="space-y-2">
                  <div className="flex items-center justify-between text-sm">
                    <span className="text-muted-foreground">Overall Progress</span>
                    <span className="font-medium">{job.status_details?.percentage_done || 0}%</span>
                  </div>
                  <Progress value={job.status_details?.percentage_done || 0} className="h-2" />
                </div>

                <div className="text-center">
                  <div className="text-2xl font-bold text-blue-600">
                    {job.status === "completed" ? "✓" : job.status === "failed" ? "✗" : "○"}
                  </div>
                  <div className="text-sm text-muted-foreground">Status</div>
                </div>

                <div className="text-center">
                  <div className="text-2xl font-bold text-green-600">
                    {job.tags?.length || 0}
                  </div>
                  <div className="text-sm text-muted-foreground">Tags</div>
                </div>
              </div>
            </div>

            <Separator />

            {/* Job Configuration */}
            <div className="space-y-4">
              <h3 className="text-lg font-semibold flex items-center space-x-2">
                <Settings className="h-5 w-5" />
                <span>Job Configuration</span>
              </h3>

              <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
                <div className="space-y-3">
                  <h4 className="font-medium flex items-center space-x-2">
                    <Database className="h-4 w-4" />
                    <span>Basic Info</span>
                  </h4>
                  <div className="space-y-2">
                    <div className="flex justify-between text-sm p-2 bg-gray-50 dark:bg-gray-800 rounded">
                      <span>Namespace</span>
                      <span className="font-mono">{safeString(job.namespace)}</span>
                    </div>
                    <div className="flex justify-between text-sm p-2 bg-gray-50 dark:bg-gray-800 rounded">
                      <span>Job ID</span>
                      <span className="font-mono">{safeString(job.id)}</span>
                    </div>
                  </div>
                </div>

                <div className="space-y-3">
                  <h4 className="font-medium flex items-center space-x-2">
                    <Target className="h-4 w-4" />
                    <span>Target & Config</span>
                  </h4>
                  <div className="space-y-2">
                    <div className="flex justify-between text-sm p-2 bg-gray-50 dark:bg-gray-800 rounded">
                      <span>Target</span>
                      <span className="font-mono">{safeString(job.target)}</span>
                    </div>
                    <div className="flex justify-between text-sm p-2 bg-gray-50 dark:bg-gray-800 rounded">
                      <span>Config</span>
                      <span className="font-mono">{safeString(job.config)}</span>
                    </div>
                  </div>
                </div>
              </div>
            </div>

            <Separator />

            {/* Tags */}
            {safeTags && safeTags.length > 0 && (
              <>
                <div className="space-y-4">
                  <h3 className="text-lg font-semibold flex items-center space-x-2">
                    <Tag className="h-5 w-5" />
                    <span>Tags</span>
                  </h3>
                  <div className="flex flex-wrap gap-2">
                    {safeTags.map((tag, index) => (
                      <Badge key={index} variant="secondary">
                        {safeString(tag)}
                      </Badge>
                    ))}
                  </div>
                </div>
                <Separator />
              </>
            )}

            {/* Status Information */}
            <div className="space-y-4">
              <h3 className="text-lg font-semibold flex items-center space-x-2">
                <FileText className="h-5 w-5" />
                <span>Status Information</span>
              </h3>

              <div className="space-y-2">
                <div className="p-3 border rounded-lg space-y-1">
                  <div className="flex items-center justify-between">
                    <span className="font-medium">Status Message</span>
                  </div>
                  <div className="text-sm text-muted-foreground">
                    {job.status_details?.message ? safeString(job.status_details.message) : "No status message available"}
                  </div>
                </div>
              </div>
            </div>

            <Separator />

            {/* Timestamps */}
            <div className="space-y-4">
              <h3 className="text-lg font-semibold flex items-center space-x-2">
                <Clock className="h-5 w-5" />
                <span>Timestamps</span>
              </h3>

              <div className="grid grid-cols-1 md:grid-cols-2 gap-4 text-sm">
                <div className="space-y-2">
                  <div className="flex items-center space-x-2">
                    <Calendar className="h-4 w-4 text-muted-foreground" />
                    <span className="text-muted-foreground">Created:</span>
                    <span>{safeString(job.created_at) ? formatDistanceToNow(new Date(safeString(job.created_at)), { addSuffix: true }) : "-"}</span>
                  </div>
                </div>
                <div className="space-y-2">
                  <div className="flex items-center space-x-2">
                    <Clock className="h-4 w-4 text-muted-foreground" />
                    <span className="text-muted-foreground">Last Updated:</span>
                    <span>{safeString(job.updated_at) ? formatDistanceToNow(new Date(safeString(job.updated_at)), { addSuffix: true }) : "-"}</span>
                  </div>
                </div>
              </div>
            </div>
          </div>
        </ScrollArea>
      </DialogContent>
    </Dialog>
  );
};

const EvaluatorJobList: React.FC = () => {
  const [currentPage, setCurrentPage] = useState(1);
  const [pageSize] = useState(10);
  const [searchQuery, setSearchQuery] = useState("");
  const [activeSearchQuery, setActiveSearchQuery] = useState("");

  const { data: jobs, isLoading, error, refetch, isFetching } = useGetEvaluatorJobs(currentPage, pageSize);
  const deleteEvaluatorJob = useDeleteEvaluatorJob();
  const getEvaluatorJobLogs = useGetEvaluatorJobLogs();
  const getEvaluatorJobResults = useGetEvaluatorJobResults();
  const downloadEvaluatorJobResults = useDownloadEvaluatorJobResults();
  const setSuccessData = useAlertStore((state) => state.setSuccessData);
  const setErrorData = useAlertStore((state) => state.setErrorData);

  // Safe jobs handling - ensure we have a valid array
  const safeJobs: NeMoEvaluatorJob[] = React.useMemo(() => {
    if (jobs && typeof jobs === 'object' && Array.isArray(jobs.data)) {
      return jobs.data;
    }
    // Fallback: if jobs is directly an array (for backward compatibility)
    if (Array.isArray(jobs)) {
      return jobs;
    }
    // Otherwise return empty array
    return [];
  }, [jobs]);

  // Extract pagination metadata
  const paginationInfo = React.useMemo(() => {
    if (jobs && typeof jobs === 'object' && !Array.isArray(jobs)) {
      return {
        page: jobs.page || currentPage,
        pageSize: jobs.page_size || pageSize,
        total: jobs.total || 0,
        totalPages: jobs.total_pages || 0,
        hasNext: jobs.has_next || false,
        hasPrev: jobs.has_prev || false,
      };
    }
    return {
      page: currentPage,
      pageSize,
      total: safeJobs.length,
      totalPages: 1,
      hasNext: false,
      hasPrev: false,
    };
  }, [jobs, currentPage, pageSize, safeJobs.length]);

  // Filter jobs based on active search query (by job ID)
  const filteredJobs = React.useMemo(() => {
    if (!activeSearchQuery.trim()) return safeJobs;
    return safeJobs.filter(job =>
      job.id.toLowerCase().includes(activeSearchQuery.toLowerCase())
    );
  }, [safeJobs, activeSearchQuery]);

  const handleSearch = () => {
    setActiveSearchQuery(searchQuery.trim());
  };

  const handleClearSearch = () => {
    setSearchQuery("");
    setActiveSearchQuery("");
  };

  const handleKeyDown = (e: React.KeyboardEvent) => {
    if (e.key === 'Enter') {
      handleSearch();
    }
  };

  // Check if backend returned an error (auth issues, etc.)
  const backendError = React.useMemo(() => {
    if (jobs && typeof jobs === 'object' && !Array.isArray(jobs) && 'error' in jobs) {
      return (jobs as any).error;
    }
    return null;
  }, [jobs]);

  const [selectedJob, setSelectedJob] = useState<NeMoEvaluatorJob | null>(null);
  const [resultsData, setResultsData] = useState<any>(null);
  const [showResultsDialog, setShowResultsDialog] = useState(false);
  const [logsData, setLogsData] = useState<any>(null);
  const [showLogsDialog, setShowLogsDialog] = useState(false);

  const handleRefresh = () => {
    refetch();
  };

  const handleViewDetails = (jobId: string) => {
    const job = safeJobs.find((j) => j.id === jobId) || null;
    setSelectedJob(job);
  };

  const handleCloseDetails = () => {
    setSelectedJob(null);
  };

  const handleDeleteJob = async (jobId: string) => {
    try {
      await deleteEvaluatorJob.mutateAsync(jobId);
      setSuccessData({
        title: "Evaluator job deleted successfully",
      });
    } catch (error) {
      console.error("Error deleting evaluator job:", error);
      setErrorData({
        title: "Failed to delete evaluator job",
        list: ["Please try again."],
      });
    }
  };

  const handleViewLogs = async (jobId: string) => {
    try {
      const logs = await getEvaluatorJobLogs.mutateAsync(jobId);
      console.log('Job logs:', logs);

      // Check if the response contains actual logs or just a message
      if (logs && typeof logs === 'object' && 'message' in logs) {
        // Service returned a message (e.g., "Job not found")
        setErrorData({
          title: "No logs available",
          list: [logs.message],
        });
      } else if (logs && (logs.logs || logs.content || typeof logs === 'string' || Array.isArray(logs))) {
        // Service returned actual logs
        setLogsData(logs);
        setShowLogsDialog(true);
      } else {
        // Unexpected response format
        setErrorData({
          title: "No logs available",
          list: ["No logs found for this job."],
        });
      }
    } catch (error) {
      console.error("Error getting logs:", error);
      setErrorData({
        title: "Failed to get job logs",
        list: ["Please try again."],
      });
    }
  };

  const handleViewResults = async (jobId: string) => {
    try {
      const results = await getEvaluatorJobResults.mutateAsync(jobId);
      console.log('Job results:', results);
      setResultsData(results);
      setShowResultsDialog(true);
      setSuccessData({
        title: "Results retrieved successfully",
      });
    } catch (error) {
      console.error("Error getting results:", error);
      setErrorData({
        title: "Failed to get job results",
        list: ["Please try again."],
      });
    }
  };

  const handleDownloadResults = async (jobId: string) => {
    try {
      const downloadData = await downloadEvaluatorJobResults.mutateAsync(jobId);
      console.log('Download data:', downloadData);

      // Handle file download based on the response data
      if (downloadData) {
        let fileContent: string;
        let fileName: string;
        let mimeType: string;

        // Check if the response contains structured content
        if (downloadData.content && downloadData.content_type) {
          // Handle structured response with content and metadata
          fileContent = downloadData.content;
          mimeType = downloadData.content_type || 'application/octet-stream';

          // Use filename from backend if provided, otherwise determine from content type
          if (downloadData.filename) {
            fileName = downloadData.filename;
          } else {
            // Determine file extension based on content type
            let extension = 'txt';
            if (mimeType.includes('zip')) {
              extension = 'zip';
            } else if (mimeType.includes('json')) {
              extension = 'json';
              mimeType = 'application/json';
            } else if (mimeType.includes('csv')) {
              extension = 'csv';
            } else if (mimeType.includes('xml')) {
              extension = 'xml';
            }

            fileName = `evaluation-results-${jobId}.${extension}`;
          }

          // Handle base64 encoded content
          if (downloadData.encoding === 'base64') {
            try {
              const binaryContent = atob(fileContent);
              const bytes = new Uint8Array(binaryContent.length);
              for (let i = 0; i < binaryContent.length; i++) {
                bytes[i] = binaryContent.charCodeAt(i);
              }
              const blob = new Blob([bytes], { type: mimeType });
              const url = window.URL.createObjectURL(blob);

              const link = document.createElement('a');
              link.href = url;
              link.download = fileName;
              document.body.appendChild(link);
              link.click();

              document.body.removeChild(link);
              window.URL.revokeObjectURL(url);
            } catch (error) {
              console.error('Error decoding base64 content:', error);
              setErrorData({
                title: "Failed to decode download content",
                list: ["The downloaded content could not be processed."],
              });
              return;
            }
          } else {
            // Handle text content
            const blob = new Blob([fileContent], { type: mimeType });
            const url = window.URL.createObjectURL(blob);

            const link = document.createElement('a');
            link.href = url;
            link.download = fileName;
            document.body.appendChild(link);
            link.click();

            document.body.removeChild(link);
            window.URL.revokeObjectURL(url);
          }
        } else if (downloadData.error) {
          // Handle error response
          setErrorData({
            title: "Download failed",
            list: [downloadData.message || downloadData.error],
          });
          return;
        } else {
          // Fallback: treat entire response as JSON data
          fileContent = JSON.stringify(downloadData, null, 2);
          fileName = `evaluation-results-${jobId}.json`;
          mimeType = 'application/json';

          const blob = new Blob([fileContent], { type: mimeType });
          const url = window.URL.createObjectURL(blob);

          const link = document.createElement('a');
          link.href = url;
          link.download = fileName;
          document.body.appendChild(link);
          link.click();

          document.body.removeChild(link);
          window.URL.revokeObjectURL(url);
        }

        setSuccessData({
          title: "Results downloaded successfully",
        });
      } else {
        setErrorData({
          title: "No download data available",
          list: ["The response did not contain downloadable data."],
        });
      }
    } catch (error) {
      console.error("Error downloading results:", error);
      setErrorData({
        title: "Failed to download job results",
        list: ["Please try again."],
      });
    }
  };

  if (error) {
    return (
      <Alert variant="destructive">
        <AlertCircle className="h-4 w-4" />
        <AlertDescription>
          Failed to load evaluator jobs. Please check your connection and try again.
        </AlertDescription>
      </Alert>
    );
  }

  // Show backend error if exists (e.g., authentication issues)
  if (backendError && safeJobs.length === 0) {
    return (
      <Alert variant="destructive">
        <AlertCircle className="h-4 w-4" />
        <AlertDescription>
          <div className="space-y-2">
            <div className="font-medium">Authentication Error</div>
            <div className="text-sm">
              Unable to authenticate with NeMo services. Please check your NeMo configuration (API token and base URL) and try again.
            </div>
            <div className="text-xs text-muted-foreground mt-2">
              Error details: {backendError}
            </div>
          </div>
        </AlertDescription>
      </Alert>
    );
  }

  return (
    <div className="space-y-6">
      {/* Header with search and refresh */}
      <div className="flex items-center justify-between">
        <div>
          <h2 className="text-2xl font-bold">Evaluator Jobs</h2>
          <p className="text-muted-foreground">
            Monitor your NeMo Evaluator jobs and view their status
          </p>
        </div>
        <div className="flex items-center space-x-4">
          {/* Find by Job ID */}
          <div className="flex items-center space-x-2">
            <div className="relative">
              <Search className="absolute left-3 top-1/2 transform -translate-y-1/2 h-4 w-4 text-muted-foreground" />
              <Input
                placeholder="Enter Job ID to find..."
                value={searchQuery}
                onChange={(e) => setSearchQuery(e.target.value)}
                onKeyDown={handleKeyDown}
                className="pl-10 w-64"
              />
            </div>
            <Button
              variant="default"
              size="sm"
              onClick={handleSearch}
              disabled={!searchQuery.trim() || isFetching}
            >
              Find
            </Button>
            {activeSearchQuery && (
              <Button
                variant="outline"
                size="sm"
                onClick={handleClearSearch}
              >
                Clear
              </Button>
            )}
          </div>
          <Button
            variant="outline"
            onClick={handleRefresh}
            disabled={isFetching}
            className="flex items-center space-x-2"
          >
            <RefreshCw className={`h-4 w-4 ${isFetching ? 'animate-spin' : ''}`} />
            <span>Refresh</span>
          </Button>
        </div>
      </div>

      {/* Job Stats */}
      {safeJobs.length > 0 && (
        <div className="grid grid-cols-1 md:grid-cols-4 gap-4">
          <div className="bg-blue-50 dark:bg-blue-950 p-4 rounded-lg">
            <div className="text-2xl font-bold text-blue-600 dark:text-blue-400">
              {safeJobs.filter(job => job.status === 'running').length}
            </div>
            <div className="text-sm text-blue-600 dark:text-blue-400">Running</div>
          </div>
          <div className="bg-green-50 dark:bg-green-950 p-4 rounded-lg">
            <div className="text-2xl font-bold text-green-600 dark:text-green-400">
              {safeJobs.filter(job => job.status === 'completed').length}
            </div>
            <div className="text-sm text-green-600 dark:text-green-400">Completed</div>
          </div>
          <div className="bg-red-50 dark:bg-red-950 p-4 rounded-lg">
            <div className="text-2xl font-bold text-red-600 dark:text-red-400">
              {safeJobs.filter(job => job.status === 'failed').length}
            </div>
            <div className="text-sm text-red-600 dark:text-red-400">Failed</div>
          </div>
          <div className="bg-gray-50 dark:bg-gray-950 p-4 rounded-lg">
            <div className="text-2xl font-bold text-gray-600 dark:text-gray-400">
              {safeJobs.length}
            </div>
            <div className="text-sm text-gray-600 dark:text-gray-400">Total</div>
          </div>
        </div>
      )}

      {/* Jobs Grid */}
      {isLoading ? (
        <EvaluatorJobListSkeleton />
      ) : filteredJobs.length > 0 ? (
        <>
          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
            {filteredJobs.map((job) => (
              <EvaluatorJobCard
                key={job.id}
                job={job}
                jobType="evaluator"
                onViewDetails={handleViewDetails}
                onDelete={handleDeleteJob}
                onViewLogs={handleViewLogs}
                onViewResults={handleViewResults}
                onDownloadResults={handleDownloadResults}
              />
            ))}
          </div>

          {/* Pagination Controls */}
          {paginationInfo.totalPages > 1 && (
            <div className="flex items-center justify-between">
              <div className="text-sm text-muted-foreground">
                Showing page {paginationInfo.page} of {paginationInfo.totalPages}
                ({paginationInfo.total} total jobs)
              </div>
              <div className="flex items-center space-x-2">
                <Button
                  variant="outline"
                  size="sm"
                  onClick={() => setCurrentPage(paginationInfo.page - 1)}
                  disabled={!paginationInfo.hasPrev || isFetching}
                >
                  Previous
                </Button>
                <Button
                  variant="outline"
                  size="sm"
                  onClick={() => setCurrentPage(paginationInfo.page + 1)}
                  disabled={!paginationInfo.hasNext || isFetching}
                >
                  Next
                </Button>
              </div>
            </div>
          )}
        </>
      ) : activeSearchQuery ? (
        <div className="text-center py-12">
          <Search className="h-16 w-16 text-muted-foreground mx-auto mb-4" />
          <h3 className="text-xl font-semibold mb-2">No jobs found</h3>
          <p className="text-muted-foreground mb-4">
            No evaluator jobs match your search for "{activeSearchQuery}".
          </p>
          <Button variant="outline" onClick={handleClearSearch}>
            Clear search
          </Button>
        </div>
      ) : (
        <EmptyState />
      )}

      {/* Job Details Modal */}
      <EvaluatorJobDetailsModal
        job={selectedJob}
        isOpen={!!selectedJob}
        onClose={handleCloseDetails}
      />

      {/* Results Dialog */}
      <Dialog open={showResultsDialog} onOpenChange={setShowResultsDialog}>
        <DialogContent className="max-w-4xl max-h-[80vh]">
          <DialogHeader>
            <DialogTitle className="flex items-center space-x-2">
              <BarChart3 className="h-5 w-5" />
              <span>Evaluation Results</span>
            </DialogTitle>
          </DialogHeader>
          <ScrollArea className="max-h-[60vh]">
            <div className="space-y-6">
              {resultsData ? (
                <>
                  {/* Evaluation Metrics Table */}
                  <div className="space-y-4">
                    <h3 className="font-medium text-lg">Evaluation Metrics</h3>
                    {(() => {
                      const parsedMetrics = parseEvaluationResults(resultsData);
                      return parsedMetrics.length > 0 ? (
                        <div className="border rounded-lg overflow-hidden">
                          <Table>
                            <TableHeader>
                              <TableRow>
                                <TableHead className="font-semibold">Metric</TableHead>
                                <TableHead className="font-semibold text-right">Score</TableHead>
                                <TableHead className="font-semibold text-right">Count</TableHead>
                                <TableHead className="font-semibold text-right">Mean</TableHead>
                                <TableHead className="font-semibold text-right">Min</TableHead>
                                <TableHead className="font-semibold text-right">Max</TableHead>
                                <TableHead className="font-semibold text-right">Std Dev</TableHead>
                              </TableRow>
                            </TableHeader>
                            <TableBody>
                              {parsedMetrics.map((metric, index) => (
                                <TableRow key={index}>
                                  <TableCell className="font-medium">
                                    <Badge variant="outline" className="font-mono">
                                      {metric.metric}
                                    </Badge>
                                  </TableCell>
                                  <TableCell className="text-right font-mono">
                                    {metric.value.toFixed(4)}
                                  </TableCell>
                                  <TableCell className="text-right font-mono">
                                    {metric.count}
                                  </TableCell>
                                  <TableCell className="text-right font-mono">
                                    {metric.mean !== null ? metric.mean.toFixed(4) : '-'}
                                  </TableCell>
                                  <TableCell className="text-right font-mono">
                                    {metric.min !== null ? metric.min.toFixed(4) : '-'}
                                  </TableCell>
                                  <TableCell className="text-right font-mono">
                                    {metric.max !== null ? metric.max.toFixed(4) : '-'}
                                  </TableCell>
                                  <TableCell className="text-right font-mono">
                                    {metric.stddev !== null ? metric.stddev.toFixed(4) : '-'}
                                  </TableCell>
                                </TableRow>
                              ))}
                            </TableBody>
                          </Table>
                        </div>
                      ) : (
                        <div className="text-center py-6 text-muted-foreground">
                          No metrics found in results data
                        </div>
                      );
                    })()}
                  </div>

                  {/* Job Information */}
                  <div className="space-y-4">
                    <h3 className="font-medium text-lg">Job Information</h3>
                    <div className="grid grid-cols-2 gap-4">
                      <div className="space-y-2">
                        <div className="text-sm font-medium text-muted-foreground">Job ID</div>
                        <div className="text-sm font-mono bg-gray-100 dark:bg-gray-800 p-2 rounded">
                          {resultsData.job || 'N/A'}
                        </div>
                      </div>
                      <div className="space-y-2">
                        <div className="text-sm font-medium text-muted-foreground">Result ID</div>
                        <div className="text-sm font-mono bg-gray-100 dark:bg-gray-800 p-2 rounded">
                          {resultsData.id || 'N/A'}
                        </div>
                      </div>
                      <div className="space-y-2">
                        <div className="text-sm font-medium text-muted-foreground">Namespace</div>
                        <div className="text-sm font-mono bg-gray-100 dark:bg-gray-800 p-2 rounded">
                          {resultsData.namespace || 'N/A'}
                        </div>
                      </div>
                      <div className="space-y-2">
                        <div className="text-sm font-medium text-muted-foreground">Created At</div>
                        <div className="text-sm font-mono bg-gray-100 dark:bg-gray-800 p-2 rounded">
                          {resultsData.created_at ? new Date(resultsData.created_at).toLocaleString() : 'N/A'}
                        </div>
                      </div>
                    </div>
                  </div>

                  {/* Raw Data (Collapsible) */}
                  <details className="space-y-2">
                    <summary className="font-medium text-lg cursor-pointer hover:text-primary">
                      Raw Data (Click to expand)
                    </summary>
                    <div className="bg-gray-50 dark:bg-gray-900 p-4 rounded-lg">
                      <pre className="text-xs bg-white dark:bg-black p-3 rounded border overflow-auto max-h-64">
                        {JSON.stringify(resultsData, null, 2)}
                      </pre>
                    </div>
                  </details>
                </>
              ) : (
                <div className="text-center py-8">
                  <div className="text-muted-foreground">No results data available</div>
                </div>
              )}
            </div>
          </ScrollArea>
        </DialogContent>
      </Dialog>

      {/* Logs Dialog */}
      <Dialog open={showLogsDialog} onOpenChange={setShowLogsDialog}>
        <DialogContent className="max-w-4xl max-h-[80vh]">
          <DialogHeader>
            <DialogTitle className="flex items-center space-x-2">
              <Terminal className="h-5 w-5" />
              <span>Job Logs</span>
            </DialogTitle>
          </DialogHeader>
          <ScrollArea className="max-h-[60vh]">
            <div className="space-y-4">
              {logsData ? (
                <div className="bg-gray-900 text-green-400 p-4 rounded-lg font-mono text-sm overflow-auto">
                  {(() => {
                    // Handle different log formats
                    if (typeof logsData === 'string') {
                      return <pre className="whitespace-pre-wrap">{logsData}</pre>;
                    } else if (Array.isArray(logsData)) {
                      return (
                        <pre className="whitespace-pre-wrap">
                          {logsData.map((line, index) => `${index + 1}: ${line}`).join('\n')}
                        </pre>
                      );
                    } else if (logsData.logs) {
                      return (
                        <pre className="whitespace-pre-wrap">
                          {typeof logsData.logs === 'string'
                            ? logsData.logs
                            : JSON.stringify(logsData.logs, null, 2)}
                        </pre>
                      );
                    } else if (logsData.content) {
                      return <pre className="whitespace-pre-wrap">{logsData.content}</pre>;
                    } else {
                      return (
                        <pre className="whitespace-pre-wrap">
                          {JSON.stringify(logsData, null, 2)}
                        </pre>
                      );
                    }
                  })()}
                </div>
              ) : (
                <div className="text-center py-8 text-gray-500">
                  No logs available
                </div>
              )}
            </div>
          </ScrollArea>
        </DialogContent>
      </Dialog>
    </div>
  );
};

export default EvaluatorJobList;
import React, { useState } from "react";
import { useGetEvaluatorJobs } from "@/controllers/API/queries/nemo/use-get-evaluator-jobs";
import { Alert, AlertDescription } from "@/components/ui/alert";
import { Skeleton } from "@/components/ui/skeleton";
import { RefreshCw, AlertCircle, ActivitySquare } from "lucide-react";
import { Button } from "@/components/ui/button";
import { NeMoEvaluatorJob } from "@/types/nemo";
import { Progress } from "@/components/ui/progress";
import {
  Dialog,
  DialogContent,
  DialogHeader,
  DialogTitle,
} from "@/components/ui/dialog";
import { Badge } from "@/components/ui/badge";
import { ScrollArea } from "@/components/ui/scroll-area";
import { Separator } from "@/components/ui/separator";
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
  FileText
} from "lucide-react";
import { formatDistanceToNow } from "date-fns";

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
  onViewDetails: (jobId: string) => void;
}> = ({ job, onViewDetails }) => {
  const statusColor = statusColorMap[job.status] || "bg-gray-100 text-gray-800";
  return (
    <div className="space-y-2 p-6 border rounded-lg cursor-pointer hover:shadow-md transition" onClick={() => onViewDetails(job.id)}>
      <div className="flex items-center justify-between">
        <div className="font-semibold truncate max-w-[60%]">{job.id}</div>
        <span className={`text-xs px-2 py-1 rounded ${statusColor} capitalize`}>
          {job.status}
        </span>
      </div>
      <div className="flex flex-wrap gap-1 mb-1">
        {job.tags?.map((tag) => (
          <span key={tag} className="text-xs bg-gray-200 dark:bg-gray-700 rounded px-2 py-0.5">
            {tag}
          </span>
        ))}
      </div>
      <div className="text-sm text-muted-foreground truncate">Namespace: {job.namespace}</div>
      <div className="text-sm text-muted-foreground truncate">Target: {job.target}</div>
      <div className="text-sm text-muted-foreground truncate">Config: {job.config}</div>
      <div className="text-xs text-gray-400">Created: {new Date(job.created_at).toLocaleString()}</div>
      <div className="text-xs text-gray-400">Updated: {new Date(job.updated_at).toLocaleString()}</div>
      <div className="text-xs text-gray-600 font-medium mt-2 mb-1">
        Status Message: {job.status_details?.message || "-"}
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

  return (
    <Dialog open={isOpen} onOpenChange={onClose}>
      <DialogContent className="max-w-4xl max-h-[90vh]">
        <DialogHeader>
          <DialogTitle className="flex items-center space-x-2">
            <span>Evaluator Job Details: {job.id.slice(-8)}</span>
            {getStatusIcon(job.status)}
            <Badge className={`${getStatusColor(job.status)} text-white`}>
              {job.status.toUpperCase()}
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
                      <span className="font-mono">{job.namespace}</span>
                    </div>
                    <div className="flex justify-between text-sm p-2 bg-gray-50 dark:bg-gray-800 rounded">
                      <span>Job ID</span>
                      <span className="font-mono">{job.id}</span>
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
                      <span className="font-mono">{job.target}</span>
                    </div>
                    <div className="flex justify-between text-sm p-2 bg-gray-50 dark:bg-gray-800 rounded">
                      <span>Config</span>
                      <span className="font-mono">{job.config}</span>
                    </div>
                  </div>
                </div>
              </div>
            </div>

            <Separator />

            {/* Tags */}
            {job.tags && job.tags.length > 0 && (
              <>
                <div className="space-y-4">
                  <h3 className="text-lg font-semibold flex items-center space-x-2">
                    <Tag className="h-5 w-5" />
                    <span>Tags</span>
                  </h3>
                  <div className="flex flex-wrap gap-2">
                    {job.tags.map((tag, index) => (
                      <Badge key={index} variant="secondary">
                        {tag}
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
                    {job.status_details?.message || "No status message available"}
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
                    <span>{formatDistanceToNow(new Date(job.created_at), { addSuffix: true })}</span>
                  </div>
                </div>
                <div className="space-y-2">
                  <div className="flex items-center space-x-2">
                    <Clock className="h-4 w-4 text-muted-foreground" />
                    <span className="text-muted-foreground">Last Updated:</span>
                    <span>{formatDistanceToNow(new Date(job.updated_at), { addSuffix: true })}</span>
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
  const { data: jobs, isLoading, error, refetch, isFetching } = useGetEvaluatorJobs();
  const safeJobs: NeMoEvaluatorJob[] = Array.isArray(jobs) ? jobs : [];
  const [selectedJob, setSelectedJob] = useState<NeMoEvaluatorJob | null>(null);

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

  return (
    <div className="space-y-6">
      {/* Header with refresh */}
      <div className="flex items-center justify-between">
        <div>
          <h2 className="text-2xl font-bold">Evaluator Jobs</h2>
          <p className="text-muted-foreground">
            Monitor your NeMo Evaluator jobs and view their status
          </p>
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
      ) : safeJobs.length > 0 ? (
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
          {safeJobs.map((job) => (
            <EvaluatorJobCard
              key={job.id}
              job={job}
              onViewDetails={handleViewDetails}
            />
          ))}
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
    </div>
  );
};

export default EvaluatorJobList;
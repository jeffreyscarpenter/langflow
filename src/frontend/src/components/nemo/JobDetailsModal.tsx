import { formatDistanceToNow } from "date-fns";
import {
  AlertTriangle,
  BarChart3,
  Calendar,
  CheckCircle,
  Clock,
  Cpu,
  Database,
  Pause,
  PlayCircle,
  Settings,
  StopCircle,
  TrendingDown,
  XCircle,
} from "lucide-react";
import React from "react";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogHeader,
  DialogTitle,
} from "@/components/ui/dialog";
import { Progress } from "@/components/ui/progress";
import { ScrollArea } from "@/components/ui/scroll-area";
import { Separator } from "@/components/ui/separator";
import { Skeleton } from "@/components/ui/skeleton";
import { useCancelJob, useGetJobStatus } from "@/controllers/API/queries/nemo";
import useAlertStore from "@/stores/alertStore";

interface JobDetailsModalProps {
  jobId: string;
  isOpen: boolean;
  onClose: () => void;
}

const getStatusIcon = (status: string) => {
  switch (status?.toLowerCase()) {
    case "running":
      return <PlayCircle className="h-5 w-5 text-blue-500" />;
    case "completed":
      return <CheckCircle className="h-5 w-5 text-green-500" />;
    case "failed":
      return <XCircle className="h-5 w-5 text-red-500" />;
    case "cancelled":
      return <Pause className="h-5 w-5 text-gray-500" />;
    case "created":
    case "pending":
      return <Clock className="h-5 w-5 text-yellow-500" />;
    default:
      return <Clock className="h-5 w-5 text-yellow-500" />;
  }
};

const getStatusColor = (status: string): string => {
  switch (status?.toLowerCase()) {
    case "running":
      return "bg-blue-500 hover:bg-blue-600";
    case "completed":
      return "bg-green-500 hover:bg-green-600";
    case "failed":
      return "bg-red-500 hover:bg-red-600";
    case "cancelled":
      return "bg-gray-500 hover:bg-gray-600";
    case "created":
    case "pending":
      return "bg-yellow-500 hover:bg-yellow-600";
    default:
      return "bg-yellow-500 hover:bg-yellow-600";
  }
};

const JobDetailsModal: React.FC<JobDetailsModalProps> = ({
  jobId,
  isOpen,
  onClose,
}) => {
  const { data: jobStatus, isLoading, error } = useGetJobStatus(jobId);
  const cancelJobMutation = useCancelJob();
  const setSuccessData = useAlertStore((state) => state.setSuccessData);
  const setErrorData = useAlertStore((state) => state.setErrorData);

  const handleCancelJob = async () => {
    try {
      await cancelJobMutation.mutateAsync(jobId);
      setSuccessData({
        title: "Job cancellation requested successfully",
      });
    } catch (error) {
      console.error("Failed to cancel job:", error);
      setErrorData({
        title: "Failed to cancel job. Please try again.",
      });
    }
  };

  const canCancelJob =
    jobStatus?.status?.toLowerCase() === "created" ||
    jobStatus?.status?.toLowerCase() === "pending";

  if (!isOpen) return null;

  return (
    <Dialog open={isOpen} onOpenChange={onClose}>
      <DialogContent className="max-w-4xl max-h-[90vh]">
        <DialogHeader>
          <DialogTitle className="flex items-center justify-between">
            <div className="flex items-center space-x-2">
              <span>Job Details: {jobId.slice(-8)}</span>
              {jobStatus && (
                <>
                  {getStatusIcon(jobStatus.status)}
                  <Badge
                    className={`${getStatusColor(jobStatus.status)} text-white`}
                  >
                    {jobStatus.status.toUpperCase()}
                  </Badge>
                </>
              )}
            </div>
            {canCancelJob && (
              <Button
                variant="destructive"
                size="sm"
                onClick={handleCancelJob}
                disabled={cancelJobMutation.isPending}
                className="flex items-center space-x-2"
              >
                <StopCircle className="h-4 w-4" />
                <span>
                  {cancelJobMutation.isPending ? "Cancelling..." : "Cancel Job"}
                </span>
              </Button>
            )}
          </DialogTitle>
          <DialogDescription>
            Detailed information about the customization job including progress,
            metrics, and logs.
          </DialogDescription>
        </DialogHeader>

        <ScrollArea className="max-h-[70vh]">
          {isLoading ? (
            <div className="space-y-6 p-4">
              <Skeleton className="h-4 w-full" />
              <Skeleton className="h-32 w-full" />
              <Skeleton className="h-64 w-full" />
            </div>
          ) : error ? (
            <div className="text-center py-8">
              <AlertTriangle className="h-12 w-12 text-red-500 mx-auto mb-4" />
              <p className="text-red-600">Failed to load job details</p>
            </div>
          ) : jobStatus ? (
            <div className="space-y-6">
              {/* Progress Section */}
              <div className="space-y-4">
                <h3 className="text-lg font-semibold flex items-center space-x-2">
                  <BarChart3 className="h-5 w-5" />
                  <span>Progress</span>
                </h3>

                <div className="grid grid-cols-1 md:grid-cols-4 gap-4">
                  <div className="space-y-2">
                    <div className="flex items-center justify-between text-sm">
                      <span className="text-muted-foreground">
                        Overall Progress
                      </span>
                      <span className="font-medium">
                        {typeof jobStatus.percentage_done === "number"
                          ? jobStatus.percentage_done
                          : 0}
                        %
                      </span>
                    </div>
                    <Progress
                      value={
                        typeof jobStatus.percentage_done === "number"
                          ? jobStatus.percentage_done
                          : 0
                      }
                      className="h-2"
                    />
                  </div>

                  <div className="text-center">
                    <div className="text-2xl font-bold text-blue-600">
                      {typeof jobStatus.epochs_completed === "number"
                        ? jobStatus.epochs_completed
                        : 0}
                    </div>
                    <div className="text-sm text-muted-foreground">
                      Epochs Completed
                    </div>
                  </div>

                  <div className="text-center">
                    <div className="text-2xl font-bold text-green-600">
                      {typeof jobStatus.steps_completed === "number"
                        ? jobStatus.steps_completed
                        : 0}
                    </div>
                    <div className="text-sm text-muted-foreground">
                      Steps Completed
                    </div>
                  </div>

                  <div className="text-center">
                    <div className="text-2xl font-bold text-purple-600">
                      {typeof jobStatus.elapsed_time === "number"
                        ? jobStatus.elapsed_time
                        : 0}
                      s
                    </div>
                    <div className="text-sm text-muted-foreground">
                      Elapsed Time
                    </div>
                  </div>
                </div>
              </div>

              <Separator />

              {/* Training Metrics */}
              <div className="space-y-4">
                <h3 className="text-lg font-semibold flex items-center space-x-2">
                  <TrendingDown className="h-5 w-5" />
                  <span>Training Metrics</span>
                </h3>

                <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
                  {/* Current Loss Values */}
                  <div className="space-y-3">
                    <h4 className="font-medium">Current Loss Values</h4>
                    <div className="space-y-2">
                      <div className="flex justify-between text-sm p-3 bg-blue-50 dark:bg-blue-950 rounded">
                        <span className="text-blue-600 dark:text-blue-400">
                          Training Loss
                        </span>
                        <span className="font-mono font-bold text-blue-600 dark:text-blue-400">
                          {jobStatus.train_loss
                            ? typeof jobStatus.train_loss === "number"
                              ? jobStatus.train_loss.toFixed(4)
                              : String(jobStatus.train_loss)
                            : "N/A"}
                        </span>
                      </div>
                      <div className="flex justify-between text-sm p-3 bg-green-50 dark:bg-green-950 rounded">
                        <span className="text-green-600 dark:text-green-400">
                          Validation Loss
                        </span>
                        <span className="font-mono font-bold text-green-600 dark:text-green-400">
                          {jobStatus.val_loss
                            ? typeof jobStatus.val_loss === "number"
                              ? jobStatus.val_loss.toFixed(4)
                              : String(jobStatus.val_loss)
                            : "N/A"}
                        </span>
                      </div>
                    </div>
                  </div>

                  {/* Training Progress */}
                  <div className="space-y-3">
                    <h4 className="font-medium">Training Progress</h4>
                    <div className="space-y-2">
                      <div className="flex justify-between text-sm p-3 bg-gray-50 dark:bg-gray-800 rounded">
                        <span>Steps per Epoch</span>
                        <span className="font-mono">
                          {jobStatus.steps_per_epoch || 0}
                        </span>
                      </div>
                      <div className="flex justify-between text-sm p-3 bg-gray-50 dark:bg-gray-800 rounded">
                        <span>Best Epoch</span>
                        <span className="font-mono">
                          {jobStatus.best_epoch || 0}
                        </span>
                      </div>
                    </div>
                  </div>

                  {/* Historical Metrics */}
                  {jobStatus.metrics && (
                    <div className="col-span-full space-y-3">
                      <h4 className="font-medium">Historical Metrics</h4>
                      <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                        {/* Training Loss History */}
                        <div className="space-y-2">
                          <h5 className="text-sm font-medium text-muted-foreground">
                            Training Loss History
                          </h5>
                          <div className="space-y-1 max-h-32 overflow-y-auto">
                            {jobStatus.metrics.metrics?.train_loss?.map(
                              (metric, index) => (
                                <div
                                  key={index}
                                  className="flex justify-between text-xs p-2 bg-gray-50 dark:bg-gray-800 rounded"
                                >
                                  <span>
                                    Step{" "}
                                    {typeof metric === "object" && metric.step
                                      ? metric.step
                                      : index + 1}
                                  </span>
                                  <span className="font-mono">
                                    {typeof metric === "object" &&
                                    metric.value !== undefined
                                      ? typeof metric.value === "number"
                                        ? metric.value.toFixed(4)
                                        : String(metric.value)
                                      : typeof metric === "number"
                                        ? metric.toFixed(4)
                                        : String(metric)}
                                  </span>
                                </div>
                              ),
                            ) || (
                              <div className="text-xs text-muted-foreground p-2">
                                No training loss history available
                              </div>
                            )}
                          </div>
                        </div>

                        {/* Validation Loss History */}
                        <div className="space-y-2">
                          <h5 className="text-sm font-medium text-muted-foreground">
                            Validation Loss History
                          </h5>
                          <div className="space-y-1 max-h-32 overflow-y-auto">
                            {jobStatus.metrics.metrics?.val_loss?.map(
                              (metric, index) => (
                                <div
                                  key={index}
                                  className="flex justify-between text-xs p-2 bg-gray-50 dark:bg-gray-800 rounded"
                                >
                                  <span>
                                    Epoch{" "}
                                    {typeof metric === "object" && metric.epoch
                                      ? metric.epoch
                                      : index + 1}
                                  </span>
                                  <span className="font-mono">
                                    {typeof metric === "object" &&
                                    metric.value !== undefined
                                      ? typeof metric.value === "number"
                                        ? metric.value.toFixed(4)
                                        : String(metric.value)
                                      : typeof metric === "number"
                                        ? metric.toFixed(4)
                                        : String(metric)}
                                  </span>
                                </div>
                              ),
                            ) || (
                              <div className="text-xs text-muted-foreground p-2">
                                No validation loss history available
                              </div>
                            )}
                          </div>
                        </div>
                      </div>
                    </div>
                  )}
                </div>
              </div>

              <Separator />

              {/* Status Logs */}
              <div className="space-y-4">
                <h3 className="text-lg font-semibold flex items-center space-x-2">
                  <Calendar className="h-5 w-5" />
                  <span>Status Logs</span>
                </h3>

                <div className="space-y-2 max-h-64 overflow-y-auto">
                  {jobStatus.status_logs?.map((log, index) => (
                    <div
                      key={index}
                      className="p-3 border rounded-lg space-y-1"
                    >
                      <div className="flex items-center justify-between">
                        <span className="font-medium">
                          {typeof log.message === "string"
                            ? log.message
                            : JSON.stringify(log.message)}
                        </span>
                        <span className="text-xs text-muted-foreground">
                          {formatDistanceToNow(new Date(log.updated_at), {
                            addSuffix: true,
                          })}
                        </span>
                      </div>
                      {log.detail && (
                        <div className="text-xs text-muted-foreground font-mono bg-gray-50 dark:bg-gray-800 p-2 rounded mt-2">
                          {typeof log.detail === "string"
                            ? log.detail
                            : JSON.stringify(log.detail, null, 2)}
                        </div>
                      )}
                    </div>
                  )) || (
                    <div className="text-sm text-muted-foreground p-2">
                      No status logs available
                    </div>
                  )}
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
                      <span>
                        {formatDistanceToNow(new Date(jobStatus.created_at), {
                          addSuffix: true,
                        })}
                      </span>
                    </div>
                  </div>
                  <div className="space-y-2">
                    <div className="flex items-center space-x-2">
                      <Clock className="h-4 w-4 text-muted-foreground" />
                      <span className="text-muted-foreground">
                        Last Updated:
                      </span>
                      <span>
                        {formatDistanceToNow(new Date(jobStatus.updated_at), {
                          addSuffix: true,
                        })}
                      </span>
                    </div>
                  </div>
                </div>
              </div>

              {/* Job Info Section */}
              <div className="space-y-4">
                <h3 className="text-lg font-semibold flex items-center space-x-2">
                  <Settings className="h-5 w-5" />
                  <span>Job Information</span>
                </h3>
                <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
                  <div className="space-y-2">
                    <div className="flex items-center space-x-2 text-sm">
                      <Clock className="h-4 w-4 text-muted-foreground" />
                      <span className="text-muted-foreground">Job ID:</span>
                      <span className="font-medium font-mono">{jobId}</span>
                    </div>
                    <div className="flex items-center space-x-2 text-sm">
                      <Database className="h-4 w-4 text-muted-foreground" />
                      <span className="text-muted-foreground">Status:</span>
                      <span className="font-medium">{jobStatus.status}</span>
                    </div>
                  </div>
                  <div className="space-y-2">
                    <div className="flex items-center space-x-2 text-sm">
                      <BarChart3 className="h-4 w-4 text-muted-foreground" />
                      <span className="text-muted-foreground">Progress:</span>
                      <span className="font-medium">
                        {typeof jobStatus.percentage_done === "number"
                          ? jobStatus.percentage_done
                          : 0}
                        %
                      </span>
                    </div>
                    <div className="flex items-center space-x-2 text-sm">
                      <Clock className="h-4 w-4 text-muted-foreground" />
                      <span className="text-muted-foreground">Elapsed:</span>
                      <span className="font-medium">
                        {jobStatus.elapsed_time || 0} seconds
                      </span>
                    </div>
                  </div>
                </div>
              </div>
            </div>
          ) : null}
        </ScrollArea>
      </DialogContent>
    </Dialog>
  );
};

export default JobDetailsModal;

import React from "react";
import { useGetJobStatus } from "@/controllers/API/queries/nemo";
import {
  Dialog,
  DialogContent,
  DialogHeader,
  DialogTitle,
} from "@/components/ui/dialog";
import { Badge } from "@/components/ui/badge";
import { Progress } from "@/components/ui/progress";
import { Skeleton } from "@/components/ui/skeleton";
import { ScrollArea } from "@/components/ui/scroll-area";
import { Separator } from "@/components/ui/separator";
import {
  Clock,
  CheckCircle,
  XCircle,
  PlayCircle,
  Pause,
  TrendingDown,
  BarChart3,
  Calendar,
  Database,
  Cpu,
  Settings,
  AlertTriangle
} from "lucide-react";
import { formatDistanceToNow } from "date-fns";
import { NeMoJobStatus } from "@/types/nemo";

interface JobDetailsModalProps {
  jobId: string;
  isOpen: boolean;
  onClose: () => void;
}

const getStatusIcon = (status: NeMoJobStatus) => {
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

const getStatusColor = (status: NeMoJobStatus): string => {
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

const JobDetailsModal: React.FC<JobDetailsModalProps> = ({ jobId, isOpen, onClose }) => {
  const { data: jobStatus, isLoading, error } = useGetJobStatus(jobId);

  if (!isOpen) return null;

  return (
    <Dialog open={isOpen} onOpenChange={onClose}>
      <DialogContent className="max-w-4xl max-h-[90vh]">
        <DialogHeader>
          <DialogTitle className="flex items-center space-x-2">
            <span>Job Details: {jobId.slice(-8)}</span>
            {jobStatus && (
              <>
                {getStatusIcon(jobStatus.status)}
                <Badge className={`${getStatusColor(jobStatus.status)} text-white`}>
                  {jobStatus.status.toUpperCase()}
                </Badge>
              </>
            )}
          </DialogTitle>
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

                <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
                  <div className="space-y-2">
                    <div className="flex items-center justify-between text-sm">
                      <span className="text-muted-foreground">Overall Progress</span>
                      <span className="font-medium">{jobStatus.status_details.percentage_done}%</span>
                    </div>
                    <Progress value={jobStatus.status_details.percentage_done} className="h-2" />
                  </div>

                  <div className="text-center">
                    <div className="text-2xl font-bold text-blue-600">
                      {jobStatus.status_details.epochs_completed}
                    </div>
                    <div className="text-sm text-muted-foreground">Epochs Completed</div>
                  </div>

                  <div className="text-center">
                    <div className="text-2xl font-bold text-green-600">
                      {jobStatus.status_details.steps_completed}
                    </div>
                    <div className="text-sm text-muted-foreground">Steps Completed</div>
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
                  {/* Training Loss */}
                  <div className="space-y-3">
                    <h4 className="font-medium">Training Loss</h4>
                    <div className="space-y-2 max-h-32 overflow-y-auto">
                      {jobStatus.status_details.training_loss.map((entry, index) => (
                        <div key={index} className="flex justify-between text-sm p-2 bg-gray-50 dark:bg-gray-800 rounded">
                          <span>Step {entry.step}</span>
                          <span className="font-mono">{entry.value.toFixed(4)}</span>
                        </div>
                      ))}
                    </div>
                  </div>

                  {/* Validation Loss */}
                  <div className="space-y-3">
                    <h4 className="font-medium">Validation Loss</h4>
                    <div className="space-y-2 max-h-32 overflow-y-auto">
                      {jobStatus.status_details.validation_loss.map((entry, index) => (
                        <div key={index} className="flex justify-between text-sm p-2 bg-gray-50 dark:bg-gray-800 rounded">
                          <span>Epoch {entry.epoch}</span>
                          <span className="font-mono">{entry.value.toFixed(4)}</span>
                        </div>
                      ))}
                    </div>
                  </div>
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
                  {jobStatus.status_details.status_logs.map((log, index) => (
                    <div key={index} className="p-3 border rounded-lg space-y-1">
                      <div className="flex items-center justify-between">
                        <span className="font-medium">{log.message}</span>
                        <span className="text-xs text-muted-foreground">
                          {formatDistanceToNow(new Date(log.updated_at), { addSuffix: true })}
                        </span>
                      </div>
                      {log.detail && (
                        <div className="text-xs text-muted-foreground font-mono bg-gray-50 dark:bg-gray-800 p-2 rounded mt-2">
                          {log.detail}
                        </div>
                      )}
                    </div>
                  ))}
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
                      <span>{formatDistanceToNow(new Date(jobStatus.created_at), { addSuffix: true })}</span>
                    </div>
                  </div>
                  <div className="space-y-2">
                    <div className="flex items-center space-x-2">
                      <Clock className="h-4 w-4 text-muted-foreground" />
                      <span className="text-muted-foreground">Last Updated:</span>
                      <span>{formatDistanceToNow(new Date(jobStatus.updated_at), { addSuffix: true })}</span>
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
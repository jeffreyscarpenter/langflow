import { formatDistanceToNow } from "date-fns";
import {
  BarChart3,
  Calendar,
  CheckCircle,
  Clock,
  Cpu,
  Database,
  Download,
  Eye,
  FileText,
  Pause,
  PlayCircle,
  Settings,
  Square,
  Trash2,
  XCircle,
} from "lucide-react";
import React from "react";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Progress } from "@/components/ui/progress";
import { NeMoJobStatus, TrackedJob } from "@/types/nemo";

interface JobCardProps {
  job: TrackedJob;
  jobType: "customizer" | "evaluator"; // New prop to determine job type
  onViewDetails?: (jobId: string) => void;
  onDelete?: (jobId: string) => void;
  onCancel?: (jobId: string) => void;
  onViewLogs?: (jobId: string) => void;
  onViewResults?: (jobId: string) => void;
  onDownloadResults?: (jobId: string) => void;
}

const getStatusIcon = (status: NeMoJobStatus) => {
  switch (status) {
    case "running":
      return <PlayCircle className="h-4 w-4" />;
    case "completed":
      return <CheckCircle className="h-4 w-4" />;
    case "failed":
      return <XCircle className="h-4 w-4" />;
    case "cancelled":
      return <Pause className="h-4 w-4" />;
    default:
      return <Clock className="h-4 w-4" />;
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

const getProgressColor = (status: NeMoJobStatus): string => {
  switch (status) {
    case "running":
      return "bg-blue-500";
    case "completed":
      return "bg-green-500";
    case "failed":
      return "bg-red-500";
    default:
      return "bg-gray-500";
  }
};

const JobCard: React.FC<JobCardProps> = ({
  job,
  jobType,
  onViewDetails,
  onDelete,
  onCancel,
  onViewLogs,
  onViewResults,
  onDownloadResults,
}) => {
  const formattedCreatedAt = formatDistanceToNow(new Date(job.created_at), {
    addSuffix: true,
  });
  const formattedUpdatedAt = formatDistanceToNow(new Date(job.updated_at), {
    addSuffix: true,
  });

  // Try to get job name from custom_fields or config
  const getJobName = () => {
    if (job.custom_fields && job.custom_fields.job_name) {
      return typeof job.custom_fields.job_name === "string"
        ? job.custom_fields.job_name
        : String(job.custom_fields.job_name);
    }
    if (job.config) {
      return typeof job.config === "string" ? job.config : String(job.config);
    }
    return `Job ${job.job_id.slice(-8)}`;
  };
  const jobName = getJobName();

  return (
    <Card className="hover:shadow-md transition-shadow">
      <CardHeader className="pb-3">
        <div className="flex items-center justify-between">
          <CardTitle className="text-lg font-semibold truncate flex items-center space-x-2">
            {getStatusIcon(job.status)}
            <span>{jobName}</span>
          </CardTitle>
          <Badge className={`${getStatusColor(job.status)} text-white`}>
            {job.status.toUpperCase()}
          </Badge>
        </div>
      </CardHeader>

      <CardContent className="space-y-4">
        {/* Progress Bar */}
        <div className="space-y-2">
          <div className="flex items-center justify-between text-sm">
            <span className="text-muted-foreground">Progress</span>
            <span className="font-medium">{job.progress}%</span>
          </div>
          <Progress
            value={job.progress}
            className="h-2"
            style={
              {
                "--progress-foreground": getProgressColor(job.status),
              } as React.CSSProperties
            }
          />
        </div>

        {/* Job Details */}
        <div className="space-y-2">
          <div className="flex items-center space-x-2 text-sm">
            <Clock className="h-4 w-4 text-muted-foreground" />
            <span className="text-muted-foreground">Job ID:</span>
            <span className="font-mono text-xs truncate" title={job.job_id}>
              {job.job_id}
            </span>
          </div>

          <div className="flex items-center space-x-2 text-sm">
            <Cpu className="h-4 w-4 text-muted-foreground" />
            <span className="text-muted-foreground">Model:</span>
            <span className="font-medium truncate">{job.config}</span>
          </div>

          <div className="flex items-center space-x-2 text-sm">
            <Database className="h-4 w-4 text-muted-foreground" />
            <span className="text-muted-foreground">Dataset:</span>
            <span className="font-medium truncate">{job.dataset}</span>
          </div>

          {/* Output Model */}
          {job.output_model && (
            <div className="flex items-center space-x-2 text-sm">
              <Cpu className="h-4 w-4 text-muted-foreground flex-shrink-0" />
              <span className="text-muted-foreground flex-shrink-0">
                Output Model:
              </span>
              <span className="font-medium break-all" title={job.output_model}>
                {job.output_model}
              </span>
            </div>
          )}

          {/* Hyperparameters */}
          {job.hyperparameters && (
            <div className="flex items-center space-x-2 text-sm">
              <Settings className="h-4 w-4 text-muted-foreground" />
              <span className="text-muted-foreground">Epochs:</span>
              <span className="font-medium">{job.hyperparameters.epochs}</span>
              <span className="text-muted-foreground">Batch Size:</span>
              <span className="font-medium">
                {job.hyperparameters.batch_size}
              </span>
            </div>
          )}
        </div>

        {/* Timestamps */}
        <div className="space-y-1">
          <div className="flex items-center space-x-2 text-xs text-muted-foreground">
            <Calendar className="h-3 w-3" />
            <span>Created {formattedCreatedAt}</span>
          </div>
          <div className="flex items-center space-x-2 text-xs text-muted-foreground">
            <Clock className="h-3 w-3" />
            <span>Updated {formattedUpdatedAt}</span>
          </div>
        </div>

        {/* Actions */}
        <div className="pt-2 flex gap-2 flex-wrap">
          <Button
            variant="outline"
            size="sm"
            className="flex-1"
            onClick={(e) => {
              e.preventDefault();
              e.stopPropagation();
              if (onViewDetails) {
                onViewDetails(job.job_id);
              }
            }}
          >
            <Eye className="h-4 w-4 mr-2" />
            View Details
          </Button>

          {/* Customizer job actions */}
          {jobType === "customizer" && (
            <>
              {/* Cancel/Stop button for non-completed jobs */}
              {job.status !== "completed" && onCancel && (
                <Button
                  variant="outline"
                  size="sm"
                  className="text-orange-600 hover:text-orange-700 hover:bg-orange-50"
                  onClick={(e) => {
                    e.preventDefault();
                    e.stopPropagation();
                    if (
                      confirm(
                        `Are you sure you want to cancel job ${job.job_id.slice(-8)}?`,
                      )
                    ) {
                      onCancel(job.job_id);
                    }
                  }}
                >
                  <Square className="h-4 w-4 mr-1" />
                  Stop
                </Button>
              )}

              {/* Logs button for all customizer jobs */}
              {onViewLogs && (
                <Button
                  variant="outline"
                  size="sm"
                  className="text-blue-600 hover:text-blue-700 hover:bg-blue-50"
                  onClick={(e) => {
                    e.preventDefault();
                    e.stopPropagation();
                    onViewLogs(job.job_id);
                  }}
                >
                  <FileText className="h-4 w-4 mr-1" />
                  Logs
                </Button>
              )}
            </>
          )}

          {/* Evaluator job actions */}
          {jobType === "evaluator" && (
            <>
              {/* Delete button for evaluator jobs */}
              {onDelete && (
                <Button
                  variant="outline"
                  size="sm"
                  className="text-red-600 hover:text-red-700 hover:bg-red-50"
                  onClick={(e) => {
                    e.preventDefault();
                    e.stopPropagation();
                    if (
                      confirm(
                        `Are you sure you want to delete job ${job.job_id.slice(-8)}?`,
                      )
                    ) {
                      onDelete(job.job_id);
                    }
                  }}
                >
                  <Trash2 className="h-4 w-4 mr-1" />
                  Delete
                </Button>
              )}

              {/* Logs button for all evaluator jobs */}
              {onViewLogs && (
                <Button
                  variant="outline"
                  size="sm"
                  className="text-blue-600 hover:text-blue-700 hover:bg-blue-50"
                  onClick={(e) => {
                    e.preventDefault();
                    e.stopPropagation();
                    onViewLogs(job.job_id);
                  }}
                >
                  <FileText className="h-4 w-4 mr-1" />
                  Logs
                </Button>
              )}

              {/* Results and Download buttons for completed evaluator jobs */}
              {job.status === "completed" && (
                <>
                  {onViewResults && (
                    <Button
                      variant="outline"
                      size="sm"
                      className="text-green-600 hover:text-green-700 hover:bg-green-50"
                      onClick={(e) => {
                        e.preventDefault();
                        e.stopPropagation();
                        onViewResults(job.job_id);
                      }}
                    >
                      <BarChart3 className="h-4 w-4 mr-1" />
                      Results
                    </Button>
                  )}

                  {onDownloadResults && (
                    <Button
                      variant="outline"
                      size="sm"
                      className="text-purple-600 hover:text-purple-700 hover:bg-purple-50"
                      onClick={(e) => {
                        e.preventDefault();
                        e.stopPropagation();
                        onDownloadResults(job.job_id);
                      }}
                    >
                      <Download className="h-4 w-4 mr-1" />
                      Download
                    </Button>
                  )}
                </>
              )}
            </>
          )}
        </div>
      </CardContent>
    </Card>
  );
};

export default JobCard;

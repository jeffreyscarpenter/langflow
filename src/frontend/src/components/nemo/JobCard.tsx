import React from "react";
import { TrackedJob, NeMoJobStatus } from "@/types/nemo";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { Progress } from "@/components/ui/progress";
import { Button } from "@/components/ui/button";
import {
  Clock,
  CheckCircle,
  XCircle,
  PlayCircle,
  Pause,
  Eye,
  Calendar,
  Database,
  Cpu,
  Settings
} from "lucide-react";
import { formatDistanceToNow } from "date-fns";

interface JobCardProps {
  job: TrackedJob;
  onViewDetails?: (jobId: string) => void;
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

const JobCard: React.FC<JobCardProps> = ({ job, onViewDetails }) => {
  const formattedCreatedAt = formatDistanceToNow(new Date(job.created_at), { addSuffix: true });
  const formattedUpdatedAt = formatDistanceToNow(new Date(job.updated_at), { addSuffix: true });

  // Try to get job name from custom_fields or config
  const jobName = (job.custom_fields && job.custom_fields.job_name) || job.config || `Job ${job.job_id.slice(-8)}`;


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
            style={{
              '--progress-foreground': getProgressColor(job.status),
            } as React.CSSProperties}
          />
        </div>

        {/* Job Details */}
        <div className="space-y-2">
          <div className="flex items-center space-x-2 text-sm">
            <Clock className="h-4 w-4 text-muted-foreground" />
            <span className="text-muted-foreground">Job ID:</span>
            <span className="font-mono text-xs truncate" title={job.job_id}>{job.job_id}</span>
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
              <span className="text-muted-foreground flex-shrink-0">Output Model:</span>
              <span className="font-medium break-all" title={job.output_model}>{job.output_model}</span>
            </div>
          )}

          {/* Hyperparameters */}
          {job.hyperparameters && (
            <div className="flex items-center space-x-2 text-sm">
              <Settings className="h-4 w-4 text-muted-foreground" />
              <span className="text-muted-foreground">Epochs:</span>
              <span className="font-medium">{job.hyperparameters.epochs}</span>
              <span className="text-muted-foreground">Batch Size:</span>
              <span className="font-medium">{job.hyperparameters.batch_size}</span>
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
        <div className="pt-2">
          <Button
            variant="outline"
            size="sm"
            className="w-full"
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
        </div>
      </CardContent>
    </Card>
  );
};

export default JobCard;
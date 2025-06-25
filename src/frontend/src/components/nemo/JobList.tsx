import React, { useState } from "react";
import { useGetTrackedJobs } from "@/controllers/API/queries/nemo";
import JobCard from "./JobCard";
import JobDetailsModal from "./JobDetailsModal";
import { Alert, AlertDescription } from "@/components/ui/alert";
import { Skeleton } from "@/components/ui/skeleton";
import { RefreshCw, AlertCircle, Wrench } from "lucide-react";
import { Button } from "@/components/ui/button";

const JobListSkeleton: React.FC = () => (
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
    <Wrench className="h-16 w-16 text-muted-foreground mx-auto mb-4" />
    <h3 className="text-xl font-semibold mb-2">No Jobs Found</h3>
    <p className="text-muted-foreground mb-4">
      No customizer jobs are currently being tracked. Jobs will appear here when you create them using the NeMo Customizer component.
    </p>
    <p className="text-sm text-muted-foreground">
      Jobs are automatically refreshed every 30 seconds to show the latest status.
    </p>
  </div>
);

const JobList: React.FC = () => {
  const [selectedJobId, setSelectedJobId] = useState<string | null>(null);
  const { data: jobs, isLoading, error, refetch, isFetching } = useGetTrackedJobs();

  // Ensure jobs is always an array to prevent filter errors
  const safeJobs = Array.isArray(jobs) ? jobs : [];

  const handleViewDetails = (jobId: string) => {
    setSelectedJobId(jobId);
  };

  const handleCloseDetails = () => {
    setSelectedJobId(null);
  };

  const handleRefresh = () => {
    refetch();
  };

  if (error) {
    return (
      <Alert variant="destructive">
        <AlertCircle className="h-4 w-4" />
        <AlertDescription>
          Failed to load customizer jobs. Please check your connection and try again.
        </AlertDescription>
      </Alert>
    );
  }

  return (
    <div className="space-y-6">
      {/* Header with refresh */}
      <div className="flex items-center justify-between">
        <div>
          <h2 className="text-2xl font-bold">Customizer Jobs</h2>
          <p className="text-muted-foreground">
            Monitor your NeMo Customizer jobs with real-time progress and metrics
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
        <JobListSkeleton />
      ) : safeJobs.length > 0 ? (
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
          {safeJobs.map((job) => (
            <JobCard
              key={job.job_id}
              job={job}
              onViewDetails={handleViewDetails}
            />
          ))}
        </div>
      ) : (
        <EmptyState />
      )}

      {/* Job Details Modal */}
      {selectedJobId && (
        <JobDetailsModal
          jobId={selectedJobId}
          isOpen={!!selectedJobId}
          onClose={handleCloseDetails}
        />
      )}
    </div>
  );
};

export default JobList;
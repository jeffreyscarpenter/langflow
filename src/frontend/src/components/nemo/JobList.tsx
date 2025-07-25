import React, { useState } from "react";
import { useGetCustomizerJobs } from "@/controllers/API/queries/nemo/use-get-customizer-jobs";
import { useGetCustomizerJob } from "@/controllers/API/queries/nemo/use-get-customizer-job";
import { useCancelCustomizerJob, useGetCustomizerJobLogs } from "@/controllers/API/queries/nemo/use-customizer-job-actions";
import JobCard from "./JobCard";
import JobDetailsModal from "./JobDetailsModal";
import { Alert, AlertDescription } from "@/components/ui/alert";
import { Skeleton } from "@/components/ui/skeleton";
import { RefreshCw, AlertCircle, Wrench, Terminal, Search } from "lucide-react";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import useAlertStore from "@/stores/alertStore";
import {
  Dialog,
  DialogContent,
  DialogHeader,
  DialogTitle,
} from "@/components/ui/dialog";
import { ScrollArea } from "@/components/ui/scroll-area";

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
  const [logsData, setLogsData] = useState<any>(null);
  const [showLogsDialog, setShowLogsDialog] = useState(false);
  const [currentPage, setCurrentPage] = useState(1);
  const [pageSize] = useState(10);
  const [searchQuery, setSearchQuery] = useState("");
  const [activeSearchQuery, setActiveSearchQuery] = useState("");
  const { data: jobs, isLoading, error, refetch, isFetching } = useGetCustomizerJobs(currentPage, pageSize);

  // Search for specific job by ID using direct API call
  const {
    data: searchedJob,
    isLoading: isSearchLoading,
    error: searchError
  } = useGetCustomizerJob({
    jobId: activeSearchQuery,
    enabled: !!activeSearchQuery
  });
  const cancelCustomizerJob = useCancelCustomizerJob();
  const getCustomizerJobLogs = useGetCustomizerJobLogs();
  const setSuccessData = useAlertStore((state) => state.setSuccessData);
  const setErrorData = useAlertStore((state) => state.setErrorData);

  // Ensure jobs is always an array and map to TrackedJob format
  const safeJobs = React.useMemo(() => {
    if (jobs && typeof jobs === 'object' && !Array.isArray(jobs) && jobs.data) {
      const jobsArray = Array.isArray(jobs.data) ? jobs.data : [];
      // Map customizer job format to TrackedJob format
      return jobsArray.map((job: any) => ({
        job_id: job.id,
        status: job.status,
        created_at: job.created_at,
        updated_at: job.updated_at,
        config: job.config,
        dataset: job.dataset,
        progress: job.status_details?.percentage_done || 0,
        output_model: job.output_model,
        hyperparameters: job.hyperparameters,
        custom_fields: {
          description: job.description,
          namespace: job.namespace,
          ...job.status_details
        }
      }));
    }
    return Array.isArray(jobs) ? jobs : [];
  }, [jobs]);

  // Extract pagination metadata
  const paginationInfo = React.useMemo(() => {
    if (jobs && typeof jobs === 'object' && !Array.isArray(jobs) && jobs.pagination) {
      const pagination = jobs.pagination;
      return {
        page: pagination.page || currentPage,
        pageSize: pagination.page_size || pageSize,
        total: pagination.total_results || 0,
        totalPages: pagination.total_pages || 0,
        hasNext: (pagination.page || currentPage) < (pagination.total_pages || 1),
        hasPrev: (pagination.page || currentPage) > 1,
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

  // Determine which jobs to show: search result or paginated list
  const displayJobs = React.useMemo(() => {
    if (activeSearchQuery) {
      return searchedJob ? [searchedJob] : [];
    }
    return safeJobs;
  }, [activeSearchQuery, searchedJob, safeJobs]);

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

  const handleViewDetails = (jobId: string) => {
    setSelectedJobId(jobId);
  };

  const handleCloseDetails = () => {
    setSelectedJobId(null);
  };

  const handleRefresh = () => {
    refetch();
  };

  const handleCancelJob = async (jobId: string) => {
    try {
      await cancelCustomizerJob.mutateAsync(jobId);
      setSuccessData({
        title: "Job cancelled successfully",
      });
    } catch (error) {
      console.error("Error cancelling job:", error);
      setErrorData({
        title: "Failed to cancel job",
        list: ["Please try again."],
      });
    }
  };

  const handleViewLogs = async (jobId: string) => {
    try {
      const logs = await getCustomizerJobLogs.mutateAsync(jobId);
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
      {/* Header with search and refresh */}
      <div className="flex items-center justify-between">
        <div>
          <h2 className="text-2xl font-bold">Customizer Jobs</h2>
          <p className="text-muted-foreground">
            Monitor your NeMo Customizer jobs with real-time progress and metrics
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
              disabled={!searchQuery.trim() || isFetching || isSearchLoading}
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
      {(isLoading || isSearchLoading) ? (
        <JobListSkeleton />
      ) : displayJobs.length > 0 ? (
        <>
          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
            {displayJobs.map((job) => (
              <JobCard
                key={job.job_id}
                job={job}
                jobType="customizer"
                onViewDetails={handleViewDetails}
                onCancel={handleCancelJob}
                onViewLogs={handleViewLogs}
              />
            ))}
          </div>

          {/* Pagination Controls - Disabled for customizer jobs due to backend pagination issues */}
          {false && paginationInfo.totalPages > 1 && (
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
            No customizer jobs match your search for "{activeSearchQuery}".
          </p>
          <Button variant="outline" onClick={handleClearSearch}>
            Clear search
          </Button>
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

export default JobList;
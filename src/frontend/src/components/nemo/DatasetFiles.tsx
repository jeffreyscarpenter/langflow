import React, { useState, useRef } from "react";
import { useGetDatasetFiles, useUploadFiles } from "@/controllers/API/queries/nemo";
import { NeMoFile } from "@/types/nemo";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { Upload, File, Calendar, HardDrive, Download, Trash2 } from "lucide-react";
import { formatDistanceToNow } from "date-fns";
import { formatBytes } from "@/utils/utils";
import useAlertStore from "@/stores/alertStore";

interface DatasetFilesProps {
  datasetId: string;
  datasetName: string;
}

const DatasetFiles: React.FC<DatasetFilesProps> = ({ datasetId, datasetName }) => {
  const [selectedFiles, setSelectedFiles] = useState<File[]>([]);
  const fileInputRef = useRef<HTMLInputElement>(null);
  const setSuccessData = useAlertStore((state) => state.setSuccessData);
  const setErrorData = useAlertStore((state) => state.setErrorData);

  const { data: files, isLoading, error, refetch } = useGetDatasetFiles({ datasetId });
  const uploadFilesMutation = useUploadFiles();

  const handleFileSelect = (event: React.ChangeEvent<HTMLInputElement>) => {
    const files = Array.from(event.target.files || []);
    setSelectedFiles(files);
  };

  const handleUpload = () => {
    if (selectedFiles.length === 0) {
      setErrorData({
        title: "Error",
        list: ["Please select files to upload"],
      });
      return;
    }

    uploadFilesMutation.mutate({
      datasetId,
      files: selectedFiles,
    }, {
      onSuccess: () => {
        setSuccessData({
          title: "Files uploaded",
        });
        setSelectedFiles([]);
        if (fileInputRef.current) {
          fileInputRef.current.value = "";
        }
      },
      onError: (error) => {
        setErrorData({
          title: "Error",
          list: [error?.message || "Failed to upload files"],
        });
      },
    });
  };

  const handleDownload = (file: NeMoFile) => {
    // This would typically trigger a download from the backend
    // For now, we'll just show a notification
    setSuccessData({
      title: "Download",
    });
  };

  const handleDeleteFile = (file: NeMoFile) => {
    if (confirm(`Are you sure you want to delete "${file.filename}"?`)) {
      // This would typically call a delete API
      setSuccessData({
        title: "Delete",
      });
    }
  };

  const getFileIcon = (contentType: string) => {
    if (contentType.startsWith("text/")) return "📄";
    if (contentType.startsWith("image/")) return "🖼️";
    if (contentType.startsWith("video/")) return "🎥";
    if (contentType.startsWith("audio/")) return "🎵";
    if (contentType.includes("pdf")) return "📕";
    if (contentType.includes("zip") || contentType.includes("tar")) return "📦";
    return "📄";
  };

  if (isLoading) {
    return (
      <div className="flex items-center justify-center p-8">
        <div className="animate-spin rounded-full h-8 w-8 border-b-2 border-primary"></div>
      </div>
    );
  }

  if (error) {
    return (
      <div className="flex flex-col items-center justify-center p-8 text-center">
        <p className="text-muted-foreground mb-4">Failed to load files</p>
        <Button onClick={() => refetch()} variant="outline">
          Try Again
        </Button>
      </div>
    );
  }

  return (
    <div className="space-y-4">
      <div className="flex items-center justify-between">
        <div>
          <h2 className="text-2xl font-bold">Files in {datasetName}</h2>
          <p className="text-muted-foreground">
            {files?.length || 0} files • {files?.reduce((total, file) => total + file.size, 0) || 0} bytes
          </p>
        </div>
        <div className="flex items-center space-x-2">
          <input
            ref={fileInputRef}
            type="file"
            multiple
            onChange={handleFileSelect}
            className="hidden"
            accept="*/*"
          />
          <Button
            variant="outline"
            onClick={() => fileInputRef.current?.click()}
            disabled={uploadFilesMutation.isPending}
          >
            <File className="h-4 w-4 mr-2" />
            Select Files
          </Button>
          {selectedFiles.length > 0 && (
            <Button
              onClick={handleUpload}
              disabled={uploadFilesMutation.isPending}
            >
              <Upload className="h-4 w-4 mr-2" />
              {uploadFilesMutation.isPending ? "Uploading..." : `Upload ${selectedFiles.length} files`}
            </Button>
          )}
        </div>
      </div>

      {selectedFiles.length > 0 && (
        <Card>
          <CardHeader>
            <CardTitle className="text-lg">Selected Files</CardTitle>
            <CardDescription>
              {selectedFiles.length} file(s) ready to upload
            </CardDescription>
          </CardHeader>
          <CardContent>
            <div className="space-y-2">
              {selectedFiles.map((file, index) => (
                <div key={index} className="flex items-center justify-between p-2 bg-muted rounded">
                  <div className="flex items-center space-x-2">
                    <File className="h-4 w-4" />
                    <span className="text-sm font-medium">{file.name}</span>
                    <Badge variant="secondary">{formatBytes(file.size)}</Badge>
                  </div>
                </div>
              ))}
            </div>
          </CardContent>
        </Card>
      )}

      {files && files.length > 0 ? (
        <div className="grid gap-4">
          {files.map((file, index) => (
            <Card key={index} className="hover:shadow-md transition-shadow">
              <CardContent className="p-4">
                <div className="flex items-center justify-between">
                  <div className="flex items-center space-x-3">
                    <span className="text-2xl">{getFileIcon(file.content_type)}</span>
                    <div>
                      <h3 className="font-medium">{file.filename}</h3>
                      <div className="flex items-center space-x-4 text-sm text-muted-foreground">
                        <div className="flex items-center space-x-1">
                          <HardDrive className="h-4 w-4" />
                          <span>{formatBytes(file.size)}</span>
                        </div>
                        <div className="flex items-center space-x-1">
                          <Calendar className="h-4 w-4" />
                          <span>
                            {formatDistanceToNow(new Date(file.uploaded_at), { addSuffix: true })}
                          </span>
                        </div>
                        <Badge variant="outline">{file.content_type}</Badge>
                      </div>
                    </div>
                  </div>
                  <div className="flex items-center space-x-2">
                    <Button
                      variant="ghost"
                      size="sm"
                      onClick={() => handleDownload(file)}
                    >
                      <Download className="h-4 w-4" />
                    </Button>
                    <Button
                      variant="ghost"
                      size="sm"
                      onClick={() => handleDeleteFile(file)}
                    >
                      <Trash2 className="h-4 w-4 text-red-500" />
                    </Button>
                  </div>
                </div>
              </CardContent>
            </Card>
          ))}
        </div>
      ) : (
        <Card>
          <CardContent className="flex flex-col items-center justify-center py-12">
            <File className="h-12 w-12 text-muted-foreground mb-4" />
            <h3 className="text-lg font-semibold mb-2">No files found</h3>
            <p className="text-muted-foreground text-center mb-4">
              Upload files to this dataset to get started.
            </p>
            <Button
              variant="outline"
              onClick={() => fileInputRef.current?.click()}
            >
              <Upload className="h-4 w-4 mr-2" />
              Upload Files
            </Button>
          </CardContent>
        </Card>
      )}
    </div>
  );
};

export default DatasetFiles;
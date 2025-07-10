import React, { useState, useRef } from "react";
import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogHeader,
  DialogTitle,
} from "@/components/ui/dialog";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { Progress } from "@/components/ui/progress";
import { Badge } from "@/components/ui/badge";
import {
  Upload,
  X,
  File,
  FolderOpen,
  AlertCircle,
  CheckCircle2,
  Loader2
} from "lucide-react";
import { Alert, AlertDescription } from "@/components/ui/alert";
import { formatBytes } from "@/utils/utils";
import { useUploadDatasetFiles } from "@/controllers/API/queries/nemo";
import useAlertStore from "@/stores/alertStore";

interface AddFilesDialogProps {
  isOpen: boolean;
  onClose: () => void;
  datasetId: string;
  datasetName: string;
  namespace: string;
  onSuccess?: () => void;
}


const AddFilesDialog: React.FC<AddFilesDialogProps> = ({
  isOpen,
  onClose,
  datasetId,
  datasetName,
  namespace,
  onSuccess,
}) => {
  const [path, setPath] = useState("");
  const [acceptedFiles, setAcceptedFiles] = useState<File[]>([]);
  const [uploadProgress, setUploadProgress] = useState(0);
  const [isUploading, setIsUploading] = useState(false);
  const [errors, setErrors] = useState<string[]>([]);
  const fileInputRef = useRef<HTMLInputElement>(null);

  const setSuccessData = useAlertStore((state) => state.setSuccessData);
  const setErrorData = useAlertStore((state) => state.setErrorData);

  const uploadMutation = useUploadDatasetFiles();

  const handleFileSelect = (event: React.ChangeEvent<HTMLInputElement>) => {
    const files = Array.from(event.target.files || []);
    setAcceptedFiles(prev => [...prev, ...files]);
    setErrors([]);
  };

  const removeFile = (index: number) => {
    setAcceptedFiles(prev => prev.filter((_, i) => i !== index));
  };

  const clearAll = () => {
    setAcceptedFiles([]);
    setPath("");
    setErrors([]);
    setUploadProgress(0);
  };

  const validateInputs = (): string[] => {
    const newErrors: string[] = [];

    if (!path.trim()) {
      newErrors.push("Path is required (e.g., 'training' or 'validation')");
    }

    if (acceptedFiles.length === 0) {
      newErrors.push("At least one file must be selected");
    }

    // Validate path format
    if (path.trim() && !/^[a-zA-Z0-9_\-\/]+$/.test(path.trim())) {
      newErrors.push("Path can only contain letters, numbers, underscores, hyphens, and forward slashes");
    }

    return newErrors;
  };

  const handleUpload = async () => {
    const validationErrors = validateInputs();
    if (validationErrors.length > 0) {
      setErrors(validationErrors);
      return;
    }

    setIsUploading(true);
    setUploadProgress(0);
    setErrors([]);

    try {
      // Simulate progress for UX
      const progressInterval = setInterval(() => {
        setUploadProgress(prev => {
          if (prev >= 90) {
            clearInterval(progressInterval);
            return prev;
          }
          return prev + 10;
        });
      }, 200);

      await uploadMutation.mutateAsync({
        datasetId,
        datasetName,
        namespace,
        path: path.trim(),
        files: acceptedFiles,
      });

      clearInterval(progressInterval);
      setUploadProgress(100);

      setSuccessData({
        title: "Files uploaded successfully",
        text: `${acceptedFiles.length} file(s) uploaded to ${path}/ in dataset ${datasetName}`,
      });

      // Reset form
      clearAll();
      onSuccess?.();
      onClose();
    } catch (error: any) {
      setErrors([error?.message || "Failed to upload files"]);
      setErrorData({
        title: "Upload failed",
        text: error?.message || "Failed to upload files",
      });
    } finally {
      setIsUploading(false);
      setUploadProgress(0);
    }
  };

  const handleClose = () => {
    if (!isUploading) {
      clearAll();
      onClose();
    }
  };

  return (
    <Dialog open={isOpen} onOpenChange={handleClose}>
      <DialogContent className="max-w-2xl max-h-[90vh] overflow-y-auto">
        <DialogHeader>
          <DialogTitle className="flex items-center space-x-2">
            <FolderOpen className="h-5 w-5" />
            <span>Add Files to {datasetName}</span>
          </DialogTitle>
          <DialogDescription>
            Upload files to your dataset with a specified path. Files will be organized as path/filename.
          </DialogDescription>
        </DialogHeader>

        <div className="space-y-6">
          {/* Path Input */}
          <div className="space-y-2">
            <Label htmlFor="path">
              Path <span className="text-red-500">*</span>
            </Label>
            <Input
              id="path"
              value={path}
              onChange={(e) => setPath(e.target.value)}
              placeholder="training, validation, test, etc."
              disabled={isUploading}
            />
            <p className="text-sm text-muted-foreground">
              Example: If you enter "validation" and upload "data.jsonl", it will be saved as "validation/data.jsonl"
            </p>
          </div>

          {/* File Selection */}
          <div className="space-y-2">
            <Label>
              Files <span className="text-red-500">*</span>
            </Label>
            <div className="border-2 border-dashed rounded-lg p-8 text-center transition-colors">
              <input
                ref={fileInputRef}
                type="file"
                multiple
                onChange={handleFileSelect}
                accept=".json,.jsonl,.txt,.csv,.parquet"
                className="hidden"
                disabled={isUploading}
              />
              <Upload className="h-12 w-12 mx-auto mb-4 text-muted-foreground" />
              <div>
                <Button
                  type="button"
                  variant="outline"
                  onClick={() => fileInputRef.current?.click()}
                  disabled={isUploading}
                  className="mb-2"
                >
                  <File className="h-4 w-4 mr-2" />
                  Select Files
                </Button>
                <p className="text-sm text-muted-foreground">
                  Supports .json, .jsonl, .txt, .csv, .parquet files
                </p>
              </div>
            </div>
          </div>

          {/* Selected Files */}
          {acceptedFiles.length > 0 && (
            <div className="space-y-3">
              <div className="flex items-center justify-between">
                <Label>Selected Files ({acceptedFiles.length})</Label>
                <Button
                  variant="ghost"
                  size="sm"
                  onClick={clearAll}
                  disabled={isUploading}
                >
                  Clear All
                </Button>
              </div>
              <div className="space-y-2 max-h-40 overflow-y-auto">
                {acceptedFiles.map((file, index) => (
                  <div
                    key={index}
                    className="flex items-center justify-between p-3 bg-muted rounded-lg"
                  >
                    <div className="flex items-center space-x-3">
                      <File className="h-4 w-4 text-muted-foreground" />
                      <div>
                        <p className="text-sm font-medium">{file.name}</p>
                        <p className="text-xs text-muted-foreground">
                          {formatBytes(file.size)}
                        </p>
                      </div>
                    </div>
                    <div className="flex items-center space-x-2">
                      <Badge variant="secondary">{file.type || "unknown"}</Badge>
                      <Button
                        variant="ghost"
                        size="sm"
                        onClick={() => removeFile(index)}
                        disabled={isUploading}
                      >
                        <X className="h-4 w-4" />
                      </Button>
                    </div>
                  </div>
                ))}
              </div>
            </div>
          )}

          {/* Upload Progress */}
          {isUploading && (
            <div className="space-y-2">
              <div className="flex items-center justify-between">
                <Label>Uploading files...</Label>
                <span className="text-sm text-muted-foreground">{uploadProgress}%</span>
              </div>
              <Progress value={uploadProgress} className="h-2" />
            </div>
          )}

          {/* Errors */}
          {errors.length > 0 && (
            <Alert variant="destructive">
              <AlertCircle className="h-4 w-4" />
              <AlertDescription>
                <ul className="list-disc list-inside space-y-1">
                  {errors.map((error, index) => (
                    <li key={index}>{error}</li>
                  ))}
                </ul>
              </AlertDescription>
            </Alert>
          )}

          {/* Action Buttons */}
          <div className="flex items-center justify-end space-x-3 pt-4 border-t">
            <Button
              variant="outline"
              onClick={handleClose}
              disabled={isUploading}
            >
              Cancel
            </Button>
            <Button
              onClick={handleUpload}
              disabled={isUploading || acceptedFiles.length === 0 || !path.trim()}
              className="flex items-center space-x-2"
            >
              {isUploading ? (
                <>
                  <Loader2 className="h-4 w-4 animate-spin" />
                  <span>Uploading...</span>
                </>
              ) : (
                <>
                  <Upload className="h-4 w-4" />
                  <span>Upload {acceptedFiles.length} file(s)</span>
                </>
              )}
            </Button>
          </div>
        </div>
      </DialogContent>
    </Dialog>
  );
};

export default AddFilesDialog;
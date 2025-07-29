import { useQuery } from "@tanstack/react-query";
import {
  Calendar,
  Database,
  Download,
  Eye,
  FileText,
  Hash,
  Loader2,
} from "lucide-react";
import React, { useState } from "react";
import { Alert, AlertDescription } from "@/components/ui/alert";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import {
  Card,
  CardContent,
  CardDescription,
  CardHeader,
  CardTitle,
} from "@/components/ui/card";
import { ScrollArea } from "@/components/ui/scroll-area";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import { useGetDatasetDetails } from "@/controllers/API/queries/nemo/use-get-dataset-details";
import { NeMoDataset } from "@/types/nemo";

interface DatasetPreviewProps {
  dataset: NeMoDataset;
  onClose: () => void;
}

interface DatasetFile {
  rfilename: string;
  size?: number;
  type: "training" | "validation" | "other";
}

const DatasetPreview: React.FC<DatasetPreviewProps> = ({
  dataset,
  onClose,
}) => {
  const [selectedFile, setSelectedFile] = useState<string | null>(null);

  const {
    data: datasetDetails,
    isLoading,
    error,
  } = useGetDatasetDetails({
    datasetName: dataset.name,
    namespace: dataset.namespace,
  });

  const getFileType = (fileName: string): DatasetFile["type"] => {
    if (fileName.includes("training") || fileName.includes("train"))
      return "training";
    if (fileName.includes("validation") || fileName.includes("eval"))
      return "validation";
    return "other";
  };

  const formatFileSize = (bytes: number): string => {
    if (bytes === 0) return "0 Bytes";
    const k = 1024;
    const sizes = ["Bytes", "KB", "MB", "GB"];
    const i = Math.floor(Math.log(bytes) / Math.log(k));
    return parseFloat((bytes / Math.pow(k, i)).toFixed(2)) + " " + sizes[i];
  };

  const formatDate = (dateString: string) => {
    return new Date(dateString).toLocaleDateString("en-US", {
      year: "numeric",
      month: "short",
      day: "numeric",
      hour: "2-digit",
      minute: "2-digit",
    });
  };

  if (isLoading) {
    return (
      <div className="flex items-center justify-center p-8">
        <Loader2 className="h-8 w-8 animate-spin" />
        <span className="ml-2">Loading dataset preview...</span>
      </div>
    );
  }

  if (error) {
    return (
      <Alert variant="destructive">
        <AlertDescription>
          Failed to load dataset preview: {error.message}
        </AlertDescription>
      </Alert>
    );
  }

  const datasetFiles =
    datasetDetails?.siblings
      ?.filter((file) => !file.rfilename.includes(".gitattributes"))
      ?.map((file) => ({
        ...file,
        type: getFileType(file.rfilename),
      })) || [];

  const trainingFiles = datasetFiles.filter((f) => f.type === "training");
  const validationFiles = datasetFiles.filter((f) => f.type === "validation");
  const otherFiles = datasetFiles.filter((f) => f.type === "other");

  return (
    <div className="space-y-4">
      <div className="flex items-center justify-between">
        <div className="flex items-center space-x-2">
          <Database className="h-5 w-5 text-blue-500" />
          <h2 className="text-xl font-semibold">{dataset.name}</h2>
        </div>
        <Button variant="outline" onClick={onClose}>
          Close
        </Button>
      </div>

      {/* Dataset Info Card */}
      {datasetDetails && (
        <Card>
          <CardHeader>
            <CardTitle className="flex items-center space-x-2">
              <Database className="h-5 w-5" />
              <span>Dataset Information</span>
            </CardTitle>
          </CardHeader>
          <CardContent className="space-y-4">
            <div className="grid grid-cols-2 gap-4">
              <div className="space-y-2">
                <div className="flex items-center space-x-2">
                  <Calendar className="h-4 w-4 text-muted-foreground" />
                  <span className="text-sm font-medium">Created:</span>
                  <span className="text-sm">
                    {formatDate(datasetDetails.created_at)}
                  </span>
                </div>
                <div className="flex items-center space-x-2">
                  <Calendar className="h-4 w-4 text-muted-foreground" />
                  <span className="text-sm font-medium">Modified:</span>
                  <span className="text-sm">
                    {formatDate(datasetDetails.last_modified)}
                  </span>
                </div>
              </div>
              <div className="space-y-2">
                <div className="flex items-center space-x-2">
                  <Hash className="h-4 w-4 text-muted-foreground" />
                  <span className="text-sm font-medium">SHA:</span>
                  <code className="text-xs bg-muted px-2 py-1 rounded">
                    {datasetDetails.sha.substring(0, 12)}...
                  </code>
                </div>
                <div className="flex items-center space-x-2">
                  <FileText className="h-4 w-4 text-muted-foreground" />
                  <span className="text-sm font-medium">Files:</span>
                  <Badge variant="secondary">{datasetFiles.length}</Badge>
                </div>
              </div>
            </div>
          </CardContent>
        </Card>
      )}

      <Card>
        <CardHeader>
          <CardTitle className="flex items-center space-x-2">
            <Eye className="h-5 w-5" />
            <span>Dataset Files</span>
          </CardTitle>
          <CardDescription>Browse the files in this dataset</CardDescription>
        </CardHeader>
        <CardContent>
          <Tabs defaultValue="training" className="w-full">
            <TabsList className="grid w-full grid-cols-3">
              <TabsTrigger
                value="training"
                disabled={trainingFiles.length === 0}
              >
                Training Data ({trainingFiles.length})
              </TabsTrigger>
              <TabsTrigger
                value="validation"
                disabled={validationFiles.length === 0}
              >
                Validation Data ({validationFiles.length})
              </TabsTrigger>
              <TabsTrigger value="other" disabled={otherFiles.length === 0}>
                Other Files ({otherFiles.length})
              </TabsTrigger>
            </TabsList>

            <TabsContent value="training" className="space-y-4">
              {trainingFiles.length === 0 ? (
                <div className="text-center text-muted-foreground py-8">
                  No training data files found
                </div>
              ) : (
                trainingFiles.map((file) => (
                  <Card key={file.rfilename} className="p-4">
                    <div className="flex items-center justify-between">
                      <div className="flex items-center space-x-2">
                        <FileText className="h-4 w-4" />
                        <span className="font-medium">{file.rfilename}</span>
                        {file.size && (
                          <Badge variant="outline">
                            {formatFileSize(file.size)}
                          </Badge>
                        )}
                      </div>
                    </div>
                  </Card>
                ))
              )}
            </TabsContent>

            <TabsContent value="validation" className="space-y-4">
              {validationFiles.length === 0 ? (
                <div className="text-center text-muted-foreground py-8">
                  No validation data files found
                </div>
              ) : (
                validationFiles.map((file) => (
                  <Card key={file.rfilename} className="p-4">
                    <div className="flex items-center justify-between">
                      <div className="flex items-center space-x-2">
                        <FileText className="h-4 w-4" />
                        <span className="font-medium">{file.rfilename}</span>
                        {file.size && (
                          <Badge variant="outline">
                            {formatFileSize(file.size)}
                          </Badge>
                        )}
                      </div>
                    </div>
                  </Card>
                ))
              )}
            </TabsContent>

            <TabsContent value="other" className="space-y-4">
              {otherFiles.length === 0 ? (
                <div className="text-center text-muted-foreground py-8">
                  No other files found
                </div>
              ) : (
                otherFiles.map((file) => (
                  <Card key={file.rfilename} className="p-4">
                    <div className="flex items-center justify-between">
                      <div className="flex items-center space-x-2">
                        <FileText className="h-4 w-4" />
                        <span className="font-medium">{file.rfilename}</span>
                        {file.size && (
                          <Badge variant="outline">
                            {formatFileSize(file.size)}
                          </Badge>
                        )}
                      </div>
                    </div>
                  </Card>
                ))
              )}
            </TabsContent>
          </Tabs>
        </CardContent>
      </Card>
    </div>
  );
};

export default DatasetPreview;

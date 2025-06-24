import React, { useState } from "react";
import { useQuery } from "@tanstack/react-query";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import { ScrollArea } from "@/components/ui/scroll-area";
import { Alert, AlertDescription } from "@/components/ui/alert";
import { Loader2, FileText, Database, Eye, Download } from "lucide-react";
import { useGetDatasetFiles } from "@/controllers/API/queries/nemo-datastore/use-get-dataset-files";
import { NeMoDataset } from "@/types/nemo-datastore";

interface DatasetPreviewProps {
  dataset: NeMoDataset;
  onClose: () => void;
}

interface DatasetFile {
  name: string;
  size: number;
  content?: string;
  type: "training" | "evaluation" | "other";
}

const DatasetPreview: React.FC<DatasetPreviewProps> = ({ dataset, onClose }) => {
  const [selectedFile, setSelectedFile] = useState<string | null>(null);

  const { data: files, isLoading, error } = useGetDatasetFiles({ datasetId: dataset.id });

  const getFileType = (fileName: string): DatasetFile["type"] => {
    if (fileName.includes("training") || fileName.includes("train")) return "training";
    if (fileName.includes("input") || fileName.includes("eval")) return "evaluation";
    return "other";
  };

  const formatFileSize = (bytes: number): string => {
    if (bytes === 0) return "0 Bytes";
    const k = 1024;
    const sizes = ["Bytes", "KB", "MB", "GB"];
    const i = Math.floor(Math.log(bytes) / Math.log(k));
    return parseFloat((bytes / Math.pow(k, i)).toFixed(2)) + " " + sizes[i];
  };

  const parseJsonContent = (content: string) => {
    try {
      const parsed = JSON.parse(content);
      return Array.isArray(parsed) ? parsed : [parsed];
    } catch {
      return [];
    }
  };

  const renderDataPreview = (data: any[], type: string) => {
    if (!data || data.length === 0) {
      return (
        <div className="text-center text-muted-foreground py-8">
          No data available
        </div>
      );
    }

    const sampleData = data.slice(0, 5); // Show first 5 records

    return (
      <div className="space-y-4">
        <div className="flex items-center justify-between">
          <Badge variant="secondary">
            {data.length} record{data.length !== 1 ? "s" : ""}
          </Badge>
          {data.length > 5 && (
            <span className="text-sm text-muted-foreground">
              Showing first 5 of {data.length} records
            </span>
          )}
        </div>

        <ScrollArea className="h-96">
          <div className="space-y-3">
            {sampleData.map((record, index) => (
              <Card key={index} className="p-4">
                <div className="space-y-2">
                  {Object.entries(record).map(([key, value]) => (
                    <div key={key} className="grid grid-cols-3 gap-2 text-sm">
                      <span className="font-medium text-muted-foreground capitalize">
                        {key.replace(/_/g, " ")}:
                      </span>
                      <span className="col-span-2 break-words">
                        {typeof value === "string" && value.length > 100
                          ? `${value.substring(0, 100)}...`
                          : String(value)}
                      </span>
                    </div>
                  ))}
                </div>
              </Card>
            ))}
          </div>
        </ScrollArea>
      </div>
    );
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

  const datasetFiles = files?.map(file => ({
    ...file,
    type: getFileType(file.name),
  })) || [];

  const trainingFiles = datasetFiles.filter(f => f.type === "training");
  const evaluationFiles = datasetFiles.filter(f => f.type === "evaluation");
  const otherFiles = datasetFiles.filter(f => f.type === "other");

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

      <Card>
        <CardHeader>
          <CardTitle className="flex items-center space-x-2">
            <Eye className="h-5 w-5" />
            <span>Dataset Preview</span>
          </CardTitle>
          <CardDescription>
            Preview the contents of your dataset before using it in NeMo components
          </CardDescription>
        </CardHeader>
        <CardContent>
          <Tabs defaultValue="training" className="w-full">
            <TabsList className="grid w-full grid-cols-3">
              <TabsTrigger value="training" disabled={trainingFiles.length === 0}>
                Training Data ({trainingFiles.length})
              </TabsTrigger>
              <TabsTrigger value="evaluation" disabled={evaluationFiles.length === 0}>
                Evaluation Data ({evaluationFiles.length})
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
                  <Card key={file.name} className="p-4">
                    <div className="flex items-center justify-between mb-4">
                      <div className="flex items-center space-x-2">
                        <FileText className="h-4 w-4" />
                        <span className="font-medium">{file.name}</span>
                        <Badge variant="outline">{formatFileSize(file.size)}</Badge>
                      </div>
                      <Button
                        variant="ghost"
                        size="sm"
                        onClick={() => setSelectedFile(file.name)}
                      >
                        <Download className="h-4 w-4" />
                      </Button>
                    </div>
                    {file.content && (
                      <div className="mt-4">
                        {renderDataPreview(parseJsonContent(file.content), "training")}
                      </div>
                    )}
                  </Card>
                ))
              )}
            </TabsContent>

            <TabsContent value="evaluation" className="space-y-4">
              {evaluationFiles.length === 0 ? (
                <div className="text-center text-muted-foreground py-8">
                  No evaluation data files found
                </div>
              ) : (
                evaluationFiles.map((file) => (
                  <Card key={file.name} className="p-4">
                    <div className="flex items-center justify-between mb-4">
                      <div className="flex items-center space-x-2">
                        <FileText className="h-4 w-4" />
                        <span className="font-medium">{file.name}</span>
                        <Badge variant="outline">{formatFileSize(file.size)}</Badge>
                      </div>
                      <Button
                        variant="ghost"
                        size="sm"
                        onClick={() => setSelectedFile(file.name)}
                      >
                        <Download className="h-4 w-4" />
                      </Button>
                    </div>
                    {file.content && (
                      <div className="mt-4">
                        {renderDataPreview(parseJsonContent(file.content), "evaluation")}
                      </div>
                    )}
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
                  <Card key={file.name} className="p-4">
                    <div className="flex items-center justify-between mb-4">
                      <div className="flex items-center space-x-2">
                        <FileText className="h-4 w-4" />
                        <span className="font-medium">{file.name}</span>
                        <Badge variant="outline">{formatFileSize(file.size)}</Badge>
                      </div>
                      <Button
                        variant="ghost"
                        size="sm"
                        onClick={() => setSelectedFile(file.name)}
                      >
                        <Download className="h-4 w-4" />
                      </Button>
                    </div>
                    {file.content && (
                      <div className="mt-4">
                        <ScrollArea className="h-64">
                          <pre className="text-sm bg-muted p-4 rounded-md overflow-auto">
                            {file.content}
                          </pre>
                        </ScrollArea>
                      </div>
                    )}
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
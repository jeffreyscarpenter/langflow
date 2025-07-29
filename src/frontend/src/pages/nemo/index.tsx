import {
  Activity,
  AlertCircle,
  ArrowLeft,
  Database,
  Save,
  Settings,
  Wrench,
} from "lucide-react";
import React, { useEffect, useState } from "react";
import DatasetFiles from "@/components/nemo/DatasetFiles";
import DatasetList from "@/components/nemo/DatasetList";
import EvaluatorJobList from "@/components/nemo/EvaluatorJobList";
import JobList from "@/components/nemo/JobList";
import { Alert, AlertDescription } from "@/components/ui/alert";
import { Button } from "@/components/ui/button";
import {
  Card,
  CardContent,
  CardDescription,
  CardHeader,
  CardTitle,
} from "@/components/ui/card";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import { NeMoDataset } from "@/types/nemo";

interface NeMoConfig {
  baseUrl: string;
  authToken: string;
  namespace: string;
}

const NeMoMicroservicesPage: React.FC = () => {
  const [selectedDataset, setSelectedDataset] = useState<NeMoDataset | null>(
    null,
  );
  const [activeTab, setActiveTab] = useState("datasets");
  const [showConfig, setShowConfig] = useState(false);
  const [config, setConfig] = useState<NeMoConfig>({
    baseUrl: "https://us-west-2.api-dev.ai.datastax.com/nvidia/nemo",
    authToken: "",
    namespace: "default",
  });
  const [isConfigValid, setIsConfigValid] = useState(false);

  // Load config from localStorage on component mount
  useEffect(() => {
    const savedConfig = localStorage.getItem("nemo-config");
    if (savedConfig) {
      try {
        const parsed = JSON.parse(savedConfig);
        setConfig(parsed);
        setIsConfigValid(
          parsed.baseUrl && parsed.authToken && parsed.namespace,
        );
      } catch (error) {
        console.error("Error parsing saved config:", error);
      }
    }
  }, []);

  // Check if config is valid whenever it changes
  useEffect(() => {
    setIsConfigValid(config.baseUrl && config.authToken && config.namespace);
  }, [config]);

  const handleDatasetSelect = (dataset: NeMoDataset) => {
    setSelectedDataset(dataset);
  };

  const handleBackToList = () => {
    setSelectedDataset(null);
  };

  const handleConfigChange = (field: keyof NeMoConfig, value: string) => {
    setConfig((prev) => ({ ...prev, [field]: value }));
  };

  const handleSaveConfig = () => {
    if (config.baseUrl && config.authToken && config.namespace) {
      localStorage.setItem("nemo-config", JSON.stringify(config));
      setShowConfig(false);
      setIsConfigValid(true);
      // Force refresh of components by updating a key or triggering re-render
      window.location.reload();
    }
  };

  const handleToggleConfig = () => {
    setShowConfig(!showConfig);
  };

  return (
    <div className="h-full w-full overflow-auto">
      <div className="container mx-auto p-6 max-w-7xl">
        <div className="mb-6">
          <div className="flex items-center justify-between mb-2">
            <div className="flex items-center space-x-2">
              <Activity className="h-6 w-6 text-blue-500" />
              <h1 className="text-3xl font-bold">NeMo Microservices</h1>
            </div>
            <Button
              variant="outline"
              onClick={handleToggleConfig}
              className="flex items-center space-x-2"
            >
              <Settings className="h-4 w-4" />
              <span>Configure</span>
            </Button>
          </div>
          <p className="text-muted-foreground">
            Manage datasets, monitor customization jobs, and track evaluations
            across the NeMo platform.
          </p>

          {!isConfigValid && (
            <Alert variant="destructive" className="mt-4">
              <AlertCircle className="h-4 w-4" />
              <AlertDescription>
                Configuration required. Please click "Configure" to set up your
                NeMo microservices connection.
              </AlertDescription>
            </Alert>
          )}
        </div>

        {showConfig ? (
          <Card className="mb-6">
            <CardHeader>
              <CardTitle>NeMo Microservices Configuration</CardTitle>
              <CardDescription>
                Configure your connection to NeMo microservices. These settings
                will be used for all API calls.
              </CardDescription>
            </CardHeader>
            <CardContent className="space-y-4">
              <div className="space-y-2">
                <Label htmlFor="baseUrl">Base URL</Label>
                <Input
                  id="baseUrl"
                  value={config.baseUrl}
                  onChange={(e) =>
                    handleConfigChange("baseUrl", e.target.value)
                  }
                  placeholder="https://us-west-2.api-dev.ai.datastax.com/nvidia/nemo"
                />
              </div>
              <div className="space-y-2">
                <Label htmlFor="authToken">Authentication Token</Label>
                <Input
                  id="authToken"
                  type="password"
                  value={config.authToken}
                  onChange={(e) =>
                    handleConfigChange("authToken", e.target.value)
                  }
                  placeholder="Enter your bearer token"
                />
              </div>
              <div className="space-y-2">
                <Label htmlFor="namespace">Namespace</Label>
                <Input
                  id="namespace"
                  value={config.namespace}
                  onChange={(e) =>
                    handleConfigChange("namespace", e.target.value)
                  }
                  placeholder="default"
                />
              </div>
              <div className="flex space-x-2">
                <Button
                  onClick={handleSaveConfig}
                  disabled={
                    !config.baseUrl || !config.authToken || !config.namespace
                  }
                  className="flex items-center space-x-2"
                >
                  <Save className="h-4 w-4" />
                  <span>Save Configuration</span>
                </Button>
                <Button variant="outline" onClick={() => setShowConfig(false)}>
                  Cancel
                </Button>
              </div>
            </CardContent>
          </Card>
        ) : selectedDataset ? (
          <div className="space-y-4">
            <div className="flex items-center space-x-4">
              <Button
                variant="outline"
                onClick={handleBackToList}
                className="flex items-center space-x-2"
              >
                <ArrowLeft className="h-4 w-4" />
                <span>Back to Datasets</span>
              </Button>
              <div className="h-6 w-px bg-border" />
              <h2 className="text-xl font-semibold">{selectedDataset.name}</h2>
            </div>
            <DatasetFiles
              datasetId={selectedDataset.id}
              datasetName={selectedDataset.name}
            />
          </div>
        ) : (
          <Tabs value={activeTab} onValueChange={setActiveTab}>
            <TabsList className="grid w-full grid-cols-3 mb-6">
              <TabsTrigger
                value="datasets"
                className="flex items-center space-x-2"
              >
                <Database className="h-4 w-4" />
                <span>Datasets</span>
              </TabsTrigger>
              <TabsTrigger value="jobs" className="flex items-center space-x-2">
                <Wrench className="h-4 w-4" />
                <span>Customizer Jobs</span>
              </TabsTrigger>
              <TabsTrigger
                value="evaluator"
                className="flex items-center space-x-2"
              >
                <Activity className="h-4 w-4" />
                <span>Evaluator Jobs</span>
              </TabsTrigger>
            </TabsList>

            <TabsContent value="datasets">
              <DatasetList onDatasetSelect={handleDatasetSelect} />
            </TabsContent>

            <TabsContent value="jobs">
              <JobList />
            </TabsContent>

            <TabsContent value="evaluator">
              <EvaluatorJobList />
            </TabsContent>
          </Tabs>
        )}
      </div>
    </div>
  );
};

export default NeMoMicroservicesPage;

import { Button } from "@/components/ui/button";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { Switch } from "@/components/ui/switch";
import { Textarea } from "@/components/ui/textarea";
import { useState } from "react";
import { inputHandlerEventType } from "../../../../../../../types/components";

interface NeMoSettingsFormProps {
  nemoUseMock: boolean;
  nemoApiKey: string;
  nemoBaseUrl: string;
  handleInput: (e: inputHandlerEventType) => void;
  onSave: () => void;
  isSaving?: boolean;
  isLoading?: boolean;
}

const NeMoSettingsForm = ({
  nemoUseMock,
  nemoApiKey,
  nemoBaseUrl,
  handleInput,
  onSave,
  isSaving = false,
  isLoading = false,
}: NeMoSettingsFormProps) => {
  const [showApiKey, setShowApiKey] = useState(false);

  return (
    <Card>
      <CardHeader>
        <CardTitle className="flex items-center gap-2">
          <span>NeMo Microservices</span>
          <div className="flex items-center gap-2 text-sm font-normal text-muted-foreground">
            <div className="h-2 w-2 rounded-full bg-green-500"></div>
            {nemoUseMock ? "Mock Mode" : "Real API Mode"}
          </div>
        </CardTitle>
        <CardDescription>
          Configure NeMo Microservices integration for model training and customization.
          Switch between mock services for development and real NeMo APIs for production.
        </CardDescription>
      </CardHeader>
      <CardContent className="space-y-6">
        {/* Service Mode Toggle */}
        <div className="flex items-center justify-between">
          <div className="space-y-0.5">
            <Label htmlFor="nemo-use-mock">Use Mock Services</Label>
            <div className="text-sm text-muted-foreground">
              {nemoUseMock
                ? "Using mock NeMo services for development and testing"
                : "Using real NeMo APIs for production"}
            </div>
          </div>
                      <Switch
              id="nemo-use-mock"
              checked={nemoUseMock}
              onCheckedChange={(checked) =>
                handleInput({ target: { name: "nemoUseMock", value: checked.toString() } })
              }
              disabled={isLoading}
            />
        </div>

        {/* API Configuration (only shown when not using mock) */}
        {!nemoUseMock && (
          <>
            <div className="space-y-2">
              <Label htmlFor="nemo-base-url">NeMo Base URL</Label>
              <Input
                id="nemo-base-url"
                name="nemoBaseUrl"
                value={nemoBaseUrl}
                onChange={handleInput}
                placeholder="https://us-west-2.api-dev.ai.datastax.com/nvidia/nemo"
                className="font-mono text-sm"
                disabled={isLoading}
              />
              <div className="text-sm text-muted-foreground">
                Base URL for NeMo Microservices APIs
              </div>
            </div>

            <div className="space-y-2">
              <Label htmlFor="nemo-api-key">API Key</Label>
              <div className="relative">
                <Input
                  id="nemo-api-key"
                  name="nemoApiKey"
                  type={showApiKey ? "text" : "password"}
                  value={nemoApiKey}
                  onChange={handleInput}
                  placeholder="Enter your NeMo API key"
                  className="font-mono text-sm pr-10"
                  disabled={isLoading}
                />
                <Button
                  type="button"
                  variant="ghost"
                  size="sm"
                  className="absolute right-0 top-0 h-full px-3 hover:bg-transparent"
                  onClick={() => setShowApiKey(!showApiKey)}
                >
                  {showApiKey ? "Hide" : "Show"}
                </Button>
              </div>
              <div className="text-sm text-muted-foreground">
                API key for authenticating with NeMo services
              </div>
            </div>
          </>
        )}

        {/* Mock Mode Info */}
        {nemoUseMock && (
          <div className="rounded-lg bg-muted p-4">
            <div className="text-sm">
              <strong>Mock Mode Active:</strong> All NeMo operations will use simulated data and responses.
              This is useful for development, testing, and demonstrations without requiring real NeMo API access.
            </div>
          </div>
        )}

        {/* Save Button */}
        <div className="flex justify-end">
          <Button onClick={onSave} disabled={isSaving || isLoading}>
            {isSaving ? "Saving..." : isLoading ? "Loading..." : "Save NeMo Settings"}
          </Button>
        </div>
      </CardContent>
    </Card>
  );
};

export default NeMoSettingsForm;
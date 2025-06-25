import React, { useState } from "react";
import { useCreateDataset } from "@/controllers/API/queries/nemo";
import { CreateDatasetRequest } from "@/types/nemo";
import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogFooter,
  DialogHeader,
  DialogTitle,
} from "@/components/ui/dialog";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { Textarea } from "@/components/ui/textarea";
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select";
import useAlertStore from "@/stores/alertStore";

interface CreateDatasetDialogProps {
  open: boolean;
  onOpenChange: (open: boolean) => void;
}

const CreateDatasetDialog: React.FC<CreateDatasetDialogProps> = ({ open, onOpenChange }) => {
  const [formData, setFormData] = useState<CreateDatasetRequest>({
    name: "",
    description: "",
    dataset_type: "general",
  });
  const setSuccessData = useAlertStore((state) => state.setSuccessData);
  const setErrorData = useAlertStore((state) => state.setErrorData);

  const createDatasetMutation = useCreateDataset();

  const handleClose = () => {
    setFormData({
      name: "",
      description: "",
      dataset_type: "general",
    });
    onOpenChange(false);
  };

  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault();
    if (!formData.name.trim()) {
      setErrorData({
        title: "Error",
        list: ["Dataset name is required"],
      });
      return;
    }
    createDatasetMutation.mutate(formData, {
      onSuccess: () => {
        setSuccessData({
          title: "Dataset created",
        });
        handleClose();
      },
      onError: (error) => {
        setErrorData({
          title: "Error",
          list: [error?.message || "Failed to create dataset"],
        });
      },
    });
  };

  const handleInputChange = (field: keyof CreateDatasetRequest, value: string) => {
    setFormData((prev) => ({
      ...prev,
      [field]: value,
    }));
  };

  return (
    <Dialog open={open} onOpenChange={onOpenChange}>
      <DialogContent className="sm:max-w-[425px]">
        <DialogHeader>
          <DialogTitle>Create New Dataset</DialogTitle>
          <DialogDescription>
            Create a new dataset to organize your files in NeMo Data Store.
          </DialogDescription>
        </DialogHeader>
        <form onSubmit={handleSubmit}>
          <div className="grid gap-4 py-4">
            <div className="grid gap-2">
              <Label htmlFor="name">Dataset Name *</Label>
              <Input
                id="name"
                value={formData.name}
                onChange={(e) => handleInputChange("name", e.target.value)}
                placeholder="Enter dataset name"
                disabled={createDatasetMutation.isPending}
              />
            </div>
            <div className="grid gap-2">
              <Label htmlFor="description">Description</Label>
              <Textarea
                id="description"
                value={formData.description}
                onChange={(e) => handleInputChange("description", e.target.value)}
                placeholder="Enter dataset description (optional)"
                rows={3}
                disabled={createDatasetMutation.isPending}
              />
            </div>
            <div className="grid gap-2">
              <Label htmlFor="type">Dataset Type</Label>
              <Select
                value={formData.dataset_type}
                onValueChange={(value) => handleInputChange("dataset_type", value)}
                disabled={createDatasetMutation.isPending}
              >
                <SelectTrigger>
                  <SelectValue placeholder="Select dataset type" />
                </SelectTrigger>
                <SelectContent>
                  <SelectItem value="general">General</SelectItem>
                  <SelectItem value="text">Text</SelectItem>
                  <SelectItem value="code">Code</SelectItem>
                  <SelectItem value="documentation">Documentation</SelectItem>
                  <SelectItem value="training">Training Data</SelectItem>
                </SelectContent>
              </Select>
            </div>
          </div>
          <DialogFooter>
            <Button
              type="button"
              variant="outline"
              onClick={handleClose}
              disabled={createDatasetMutation.isPending}
            >
              Cancel
            </Button>
            <Button
              type="submit"
              disabled={createDatasetMutation.isPending || !formData.name.trim()}
            >
              {createDatasetMutation.isPending ? "Creating..." : "Create Dataset"}
            </Button>
          </DialogFooter>
        </form>
      </DialogContent>
    </Dialog>
  );
};

export default CreateDatasetDialog;
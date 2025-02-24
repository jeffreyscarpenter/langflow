from pathlib import Path

import yaml

from langflow.base.data.utils import read_text_file
from langflow.custom import Component
from langflow.io import FileInput, MultilineInput, Output
from langflow.schema import Data


class NVIDIANemoGuardrailsComponent(Component):
    display_name = "NVIDIA NeMo Guardrails"
    description = "Apply guardrails to LLM interactions. Load guardrail definitions from a YAML file," \
                  " or provide directly as multiline text"
    icon = "NVIDIA"
    name = "NVIDIANemoGuardrails"
    beta = True

    file_types = ["yaml"]

    inputs = [
        MultilineInput(
            name="yaml_content",
            display_name="YAML Content (takes precedence)",
            info="Enter YAML content here"
        ),
        FileInput(
            name="path",
            display_name="YAML File Path",
            file_types=file_types,
            info="yaml files"
        ),
    ]

    outputs = [
        Output(display_name="Data", name="data", method="load_file"),
    ]

    def load_file(self) -> Data:
        # Prioritize MultilineInput if provided
        yaml_content = self.yaml_content
        if yaml_content:
            try:
                data_dict = yaml.safe_load(yaml_content)
                return Data(data={"text": yaml_content, "parsed_data": data_dict})
            except yaml.YAMLError as e:
                err_msg=f"Invalid YAML syntax"
                raise ValueError(err_msg) from e

        # Fall back to FileInput
        if not self.path:
            err_msg="Please, upload a file or provide YAML content."
            raise ValueError(err_msg)

        resolved_path = self.resolve_path(self.path)
        extension = Path(resolved_path).suffix[1:].lower()

        if extension not in self.file_types:
            err_msg=f"Unsupported file type: {extension}"
            raise ValueError(err_msg)

        text = read_text_file(resolved_path)
        try:
            data_dict = yaml.safe_load(text)
            return Data(data={"file_path": resolved_path, "text": text, "parsed_data": data_dict})
        except yaml.YAMLError as e:
            error_msg = f"Invalid YAML syntax in file"
            raise ValueError(error_msg) from e

from langflow.custom import Component
from langflow.io import MessageTextInput, Output
from langflow.schema import Data

from nv_ingest_client.client import NvIngestClient
from nv_ingest_client.primitives import JobSpec
from nv_ingest_client.primitives.tasks import ExtractTask
from nv_ingest_client.primitives.tasks import SplitTask
from nv_ingest_client.util.file_processing.extract import extract_file_content, EXTENSION_TO_DOCUMENT_TYPE
import logging, time

class NVIDIAIngestComponent(Component):
    display_name = "NVIDIA Ingest Component"
    description = "Ingest documents"
    documentation: str = "https://github.com/NVIDIA/nv-ingest/tree/main/docs"
    icon = "NVIDIA"
    name = "NVIDIAIngest"

    file_types = list(EXTENSION_TO_DOCUMENT_TYPE.keys())
    supported_file_types_info = f"Supported file types: {', '.join(file_types)}"

    inputs = [
        FileInput(
            name="path",
            display_name="Path",
            file_types=file_types,
            info=supported_file_types_info
        ),
        BoolInput(
            name="extract_text",
            display_name="Extract text?",
            info="Extract text or not"
        ),
        BoolInput(
            name="extract_images",
            display_name="Extract images?",
            info="Extract images or not"
        ),
        BoolInput(
            name="extract_tables",
            display_name="Extract tables?",
            info="Extract tables or not"
        ),
        BoolInput(
            name="split_text",
            display_name="Split text?",
            info="Split text into smaller chunks?"
        ),
        IntInput(
            name="chunk_overlap",
            display_name="Chunk Overlap",
            info="Number of characters to overlap between chunks.",
            value=200,
            advanced=True,
        ),
        IntInput(
            name="chunk_size",
            display_name="Chunk Size",
            info="The maximum number of characters in each chunk.",
            value=1000,
            advanced=True,
        ),
        MessageTextInput(
            name="separator",
            display_name="Separator",
            info="The character to split on. Defaults to newline.",
            value="\n",
            advanced=True,
        ),
    ]

    outputs = [
        Output(display_name="Data", name="data", method="load_file"),
    ]

    def load_file(self) -> Data:
        if not self.path:
            raise ValueError("Please, upload a file to use this component.")
        resolved_path = self.resolve_path(self.path)

        extension = Path(resolved_path).suffix[1:].lower()

        if extension not in self.file_types:
            raise ValueError(f"Unsupported file type: {extension}")

        file_content, file_type = extract_file_content(resolved_path)

        job_spec = JobSpec(
            document_type=file_type,
            payload=file_content,
            source_id=self.path,
            source_name=self.path,
            extended_options={"tracing_options": {"trace": True, "ts_send": time.time_ns()}},
        )

        extract_task = ExtractTask(
            document_type=file_type,
            extract_text=self.extract_text,
            extract_images=self.extract_images,
            extract_tables=self.extract_tables,
        )

        job_spec.add_task(extract_task)

        if self.split_text:
            split_task = SplitTask(
                split_by="word",
                split_length=self.chunk_size,
                split_overlap=self.chunk_overlap,
                max_character_length=self.chunk_size,
                sentence_window_size=0,
            )
            job_spec.add_task(split_task)

        client = NvIngestClient() # message_client_hostname="localhost", message_client_port=7670

        job_id = client.add_job(job_spec)

        client.submit_job(job_id, "morpheus_task_queue")

        result = client.fetch_job_result(job_id, timeout=60)

        data = []

        for element in result[0][0]:
            if element['document_type'] == 'text':
                data.append(Data(text=element['metadata']['content'],file_path=element['metadata']['source_metadata']['source_name'],document_type=element['document_type'],description=element['metadata']['content_metadata']['description']))
            elif element['document_type'] == 'structured':
                data.append(Data(text=element['metadata']['table_metadata']['table_content'],file_path=element['metadata']['source_metadata']['source_name'],document_type=element['document_type'],description=element['metadata']['content_metadata']['description']))
            #TODO handle image

        print(data)

        self.status = data if data else "No data"
        return data or Data()

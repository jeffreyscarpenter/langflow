# Standard library imports
import json

# Third-party imports
import httpx
import pandas as pd

# Local application imports
from arize.experimental.datasets import ArizeDatasetsClient
from loguru import logger

from langflow.custom import Component
from langflow.io import DictInput, DropdownInput, MessageTextInput, Output, SecretStrInput
from langflow.schema import Data, DataFrame


class ArizeAIDatastoreComponent(Component):
    display_name = "Arize AI Datastore"
    description = "Fetch available datasets and display details"
    icon = "Arize"
    name = "ArizeAIDatastoreComponent"
    beta = True

    # Inputs: A dropdown for dataset selection and a dictionary to store dataset metadata
    inputs = [
        SecretStrInput(
            name="api_key",
            display_name="ArizeAI API Key",
            info="The ArizeAI API Key to use.",
            advanced=False,
        ),
        MessageTextInput(
            name="space_id",
            display_name="ArizeAI Space Id",
            info="The ArizeAI Space Id to use.",
            advanced=False,
        ),
        DropdownInput(
            name="dataset_name",
            display_name="Dataset Name",
            info="Select a dataset from the available list",
            options=[],  # Dynamically populated
            refresh_button=True,  # Allow the dropdown to be refreshed
        ),
        DictInput(
            name="dataset_metadata",
            display_name="Dataset Metadata",
            info="Dictionary storing metadata for the datasets",
            advanced=True,  # This is the advanced field we populate dynamically
        ),
    ]

    # Outputs: A list of Data objects and a DataFrame
    outputs = [
        Output(name="data_list", display_name="Data List", method="get_dataset_data_list"),
        Output(name="dataframe", display_name="DataFrame", method="get_dataset_dataframe"),
    ]

    def process_messages(self, input_messages, output_messages):
        """Extracts 'user' and 'assistant' messages from the input and output messages.

        Returns the extracted user input and assistant output.
        """
        # Extract 'user' message from input messages
        user_input = next((msg["message.content"] for msg in input_messages if msg["message.role"] == "user"), None)

        # Extract "assistant" message from output messages
        assistant_output = next(
            (msg["message.content"] for msg in output_messages if msg["message.role"] == "assistant"), None
        )

        # If no messages found, return fallback from 'attributes.input.value' and 'attributes.output.value'
        if not user_input:
            user_input = None
        if not assistant_output:
            assistant_output = None

        return user_input, assistant_output

    def get_dataset_data_list(self) -> list[Data]:
        """Return the list of Data objects created from the dataset."""
        try:
            client = self.get_client()
            raw_dataset = self.fetch_raw_dataset(client)

            if raw_dataset.empty:
                logger.warning("No data available for Data List output")
                return []

            data_objects = self.process_dataset_to_data_objects(raw_dataset)
            logger.info(f"Data List output: returning {len(data_objects)} Data objects")
            return data_objects  # noqa: TRY300
        except (httpx.RequestError, ValueError, AttributeError) as e:
            logger.exception(f"Error in get_dataset_data_list: {e}")
            return []

    def get_dataset_dataframe(self) -> DataFrame:
        """Return the dataset as a DataFrame."""
        try:
            client = self.get_client()
            raw_dataset = self.fetch_raw_dataset(client)

            if raw_dataset.empty:
                logger.warning("No data available for DataFrame output")
                return DataFrame(pd.DataFrame())

            logger.info(
                f"DataFrame output: returning DataFrame with {len(raw_dataset)} rows "
                f"and {len(raw_dataset.columns)} columns"
            )
            return DataFrame(raw_dataset)
        except (httpx.RequestError, ValueError, AttributeError) as e:
            logger.exception(f"Error in get_dataset_dataframe: {e}")
            return DataFrame(pd.DataFrame({"error": [f"Error fetching dataset: {e!s}"]}))

    def process_dataset_to_data_objects(self, dataset: pd.DataFrame) -> list[Data]:
        """Process the raw dataset into a list of Data objects."""
        selected_dataset_name = getattr(self, "dataset_name", None)

        # Process the dataset row by row
        new_data = []
        for _, row in dataset.iterrows():
            # Check if the input and output messages exist in the row
            input_messages = row.get("attributes.llm.input_messages", None)
            output_messages = row.get("attributes.llm.output_messages", None)

            if input_messages and output_messages:
                # Parse the messages if they exist and are valid JSON
                try:
                    input_messages = json.loads(input_messages) if isinstance(input_messages, str) else input_messages
                    output_messages = (
                        json.loads(output_messages) if isinstance(output_messages, str) else output_messages
                    )
                except (json.JSONDecodeError, TypeError):
                    logger.exception("Error parsing JSON for row %s", row.name)
                    input_messages = []
                    output_messages = []

                # Process the messages
                input_message, output_message = self.process_messages(input_messages, output_messages)
            else:
                # Fallback to attributes.input.value and attributes.output.value if messages don't exist
                input_message = row.get("attributes.input.value", None)
                output_message = row.get("attributes.output.value", None)

            new_data.append({"input": input_message, "completion": output_message})

        # Create new DataFrame with the mapped values
        new_df = pd.DataFrame(new_data)

        # Create a list of Data objects, one for each row
        data_objects = [
            Data(
                data={"input": row["input"], "completion": row["completion"]},
                dataset_name=selected_dataset_name,
                document_type="Arize dataset",
                description="",
            )
            for _, row in new_df.iterrows()
        ]

        logger.info(f"Created {len(data_objects)} Data objects for dataset: {selected_dataset_name}")
        return data_objects

    def fetch_raw_dataset(self, client) -> pd.DataFrame:
        """Fetch the raw dataset data."""
        space_id = getattr(self, "space_id", None)
        selected_dataset_name = getattr(self, "dataset_name", None)
        dataset_metadata = getattr(self, "dataset_metadata", None)

        logger.info(f"fetch_raw_dataset called with space_id={space_id}, dataset_name={selected_dataset_name}")

        if not selected_dataset_name:
            logger.warning("No dataset selected. Please select a dataset from the dropdown.")
            return pd.DataFrame()

        try:
            dataset_info = dataset_metadata.get(selected_dataset_name)
            if not dataset_info:
                logger.warning(f"No metadata found for dataset: {selected_dataset_name}")
                return pd.DataFrame()

            dataset_id = dataset_info.get("dataset_id")
            logger.info(f"Fetching dataset: {selected_dataset_name} with ID: {dataset_id}")

            # Fetch the specific dataset by ID
            dataset = client.get_dataset(space_id=space_id, dataset_id=dataset_id)
            logger.info(
                f"Raw dataset type: {type(dataset)}, shape: {dataset.shape if hasattr(dataset, 'shape') else 'N/A'}"
            )

            if dataset.empty:
                logger.warning(f"No data found for dataset: {selected_dataset_name}")
                return pd.DataFrame()

            logger.info(f"Successfully fetched dataset with {len(dataset)} rows")
            return dataset  # noqa: TRY300

        except (httpx.RequestError, ValueError) as e:
            logger.exception(f"Error fetching dataset: {e}")
            return pd.DataFrame()
        except (AttributeError, TypeError) as e:
            logger.exception(f"Unexpected error in fetch_raw_dataset: {e}")
            return pd.DataFrame()

    def fetch_datasets(self, client) -> pd.DataFrame:
        """Fetch datasets from the client and return a DataFrame with dataset_id and dataset_name."""
        space_id = getattr(self, "space_id", None)
        try:
            datasets = client.list_datasets(space_id)
            if datasets.empty:
                logger.warning("No datasets found.")
                return pd.DataFrame(columns=["dataset_id", "dataset_name"])

        except (httpx.RequestError, ValueError):
            logger.exception("Error fetching datasets")
            return pd.DataFrame(columns=["dataset_id", "dataset_name"])

        return datasets

    def update_build_config(self, build_config, field_value, field_name=None):
        """Update the build configuration and store datasets in DictInput when the dropdown is updated."""
        if field_name == "dataset_name":
            log_message = f"Fetching datasets and storing them in DictInput {field_value}"
            logger.info(log_message)

            # Check if we have the required credentials
            api_key = getattr(self, "api_key", None)
            space_id = getattr(self, "space_id", None)

            if not api_key or not space_id:
                logger.warning("API key or space ID not provided, cannot fetch datasets")
                build_config["dataset_name"]["options"] = []
                build_config["dataset_metadata"]["value"] = {}
                return build_config

            try:
                # Fetch datasets
                client = self.get_client()
                datasets_df = self.fetch_datasets(client)

                if not datasets_df.empty:
                    # Convert the DataFrame to a dictionary for dropdown and metadata storage
                    datasets_dict = datasets_df.set_index("dataset_name").to_dict("index")

                    # Store metadata for each dataset in DictInput
                    build_config["dataset_metadata"]["value"] = datasets_dict
                    # Populate the dropdown with dataset names
                    build_config["dataset_name"]["options"] = datasets_df["dataset_name"].tolist()
                    logger.info("Successfully loaded %d datasets", len(datasets_df))
                else:
                    # If no datasets, clear the dropdown and DictInput
                    build_config["dataset_name"]["options"] = []
                    build_config["dataset_metadata"]["value"] = {}
                    logger.warning("No datasets found, dropdown options and metadata cleared.")
            except (httpx.RequestError, ValueError, ImportError, AttributeError):
                logger.exception("Error fetching datasets")
                build_config["dataset_name"]["options"] = []
                build_config["dataset_metadata"]["value"] = {}

        return build_config

    def get_client(self):
        """Initialize and return an instance of ArizeDatasetsClient."""
        try:
            api_key = getattr(self, "api_key", None)
            client = ArizeDatasetsClient(api_key=api_key)
            logger.info("Successfully initialized ArizeDatasetsClient.")
        except (ImportError, ValueError, AttributeError):
            logger.exception("Failed to initialize ArizeDatasetsClient")
            raise
        else:
            return client

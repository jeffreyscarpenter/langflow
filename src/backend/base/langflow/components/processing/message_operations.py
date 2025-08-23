from typing import TYPE_CHECKING, Any

from langflow.custom import Component
from langflow.inputs import DictInput, DropdownInput, MessageTextInput, SortableListInput
from langflow.io import MessageInput, Output
from langflow.logging import logger
from langflow.schema import Message
from langflow.schema.dotdict import dotdict
from langflow.utils.component_utils import set_current_fields, set_field_display

if TYPE_CHECKING:
    from collections.abc import Callable

ACTION_CONFIG = {
    "Select Properties": {"is_list": False, "log_msg": "setting property selection fields"},
    "Filter by Property": {"is_list": False, "log_msg": "setting property filtering fields"},
    "Combine Messages": {"is_list": True, "log_msg": "setting message combination fields"},
    "Split Messages": {"is_list": False, "log_msg": "setting message splitting fields"},
    "Transform Properties": {"is_list": False, "log_msg": "setting property transformation fields"},
}

OPERATORS = {
    "equals": lambda a, b: str(a) == str(b),
    "not equals": lambda a, b: str(a) != str(b),
    "contains": lambda a, b: str(b) in str(a),
    "starts with": lambda a, b: str(a).startswith(str(b)),
    "ends with": lambda a, b: str(a).endswith(str(b)),
    "boolean true": lambda a, _: bool(a) is True,
    "boolean false": lambda a, _: bool(a) is False,
}


class MessageOperationsComponent(Component):
    display_name = "Message Operations"
    description = "Perform various operations on Message objects."
    documentation: str = "https://docs.langflow.org/components-processing#message-operations"
    icon = "message-square"
    name = "MessageOperations"
    default_keys = ["operations", "message"]
    metadata = {
        "keywords": [
            "message",
            "operations",
            "property selection",
            "property filtering",
            "message combining",
            "message splitting",
            "property transformation",
            "message routing",
            "conditional routing",
        ],
    }
    actions_data = {
        "Select Properties": ["select_properties_input", "operations"],
        "Filter by Property": ["filter_property", "operations", "operator", "filter_value"],
        "Combine Messages": [],
        "Split Messages": ["split_property", "operations", "split_operator", "split_value"],
        "Transform Properties": ["transform_properties", "operations"],
    }

    inputs = [
        MessageInput(
            name="message", display_name="Message", info="Message object to operate on.", required=True, is_list=True
        ),
        SortableListInput(
            name="operations",
            display_name="Operations",
            placeholder="Select Operation",
            info="List of operations to perform on the message.",
            options=[
                {"name": "Select Properties", "icon": "lasso-select"},
                {"name": "Filter by Property", "icon": "filter"},
                {"name": "Combine Messages", "icon": "merge"},
                {"name": "Split Messages", "icon": "split"},
                {"name": "Transform Properties", "icon": "pencil-line"},
            ],
            real_time_refresh=True,
            limit=1,
        ),
        # Select Properties inputs
        MessageTextInput(
            name="select_properties_input",
            display_name="Select Properties",
            info="List of properties to select from the message.",
            show=False,
            is_list=True,
        ),
        # Filter by Property inputs
        MessageTextInput(
            name="filter_property",
            display_name="Filter Property",
            info="Property to filter by.",
            show=False,
        ),
        DropdownInput(
            name="operator",
            display_name="Comparison Operator",
            options=["equals", "not equals", "contains", "starts with", "ends with", "boolean true", "boolean false"],
            info="The operator to apply for comparing the values.",
            value="equals",
            advanced=False,
            show=False,
        ),
        MessageTextInput(
            name="filter_value",
            display_name="Filter Value",
            info="Value to filter by.",
            show=False,
        ),
        # Split Messages inputs
        MessageTextInput(
            name="split_property",
            display_name="Split Property",
            info="Property to split by.",
            show=False,
        ),
        DropdownInput(
            name="split_operator",
            display_name="Split Operator",
            options=["equals", "not equals", "contains", "starts with", "ends with", "boolean true", "boolean false"],
            info="The operator to apply for splitting.",
            value="equals",
            advanced=False,
            show=False,
        ),
        MessageTextInput(
            name="split_value",
            display_name="Split Value",
            info="Value to split by.",
            show=False,
        ),
        # Transform Properties inputs
        DictInput(
            name="transform_properties",
            display_name="Transform Properties",
            info="Properties to transform in the message.",
            show=False,
            value={"property": "value"},
        ),
    ]

    outputs = [
        Output(display_name="Message", name="message_output", method="as_message"),
        Output(
            display_name="True Output",
            name="true_output",
            method="true_response",
            group_outputs=True,
            show=False,
        ),
        Output(
            display_name="False Output",
            name="false_output",
            method="false_response",
            group_outputs=True,
            show=False,
        ),
    ]

    # Helper methods
    def get_message_list(self) -> list[Message]:
        """Get message list, handling single message case."""
        if isinstance(self.message, list):
            return self.message
        return [self.message] if self.message else []

    def get_single_message(self) -> Message:
        """Get single message, handling list case."""
        messages = self.get_message_list()
        if not messages:
            no_message_error = "No message provided"
            raise ValueError(no_message_error)
        if len(messages) > 1:
            single_message_error = "Operation requires single message"
            raise ValueError(single_message_error)
        return messages[0]

    def compare_values(self, item_value: Any, compare_value: str, operator: str) -> bool:
        """Compare values based on the specified operator."""
        if operator in ["boolean true", "boolean false"]:
            return OPERATORS[operator](item_value, compare_value)

        comparison_func = OPERATORS.get(operator)
        if comparison_func:
            return comparison_func(item_value, compare_value)
        return False

    def get_property_value(self, message: Message, property_name: str) -> Any:
        """Get property value from message, handling nested properties."""
        if hasattr(message, property_name):
            return getattr(message, property_name)
        if hasattr(message, "properties") and hasattr(message.properties, property_name):
            return getattr(message.properties, property_name)
        if hasattr(message, "properties") and isinstance(message.properties, dict):
            return message.properties.get(property_name)
        property_error = f"Property '{property_name}' not found in message"
        raise ValueError(property_error)

    # Message operations
    def select_properties(self) -> Message:
        """Select specific properties from the message."""
        message = self.get_single_message()
        property_names = self.select_properties_input

        if not property_names:
            return message

        # Create new message with only selected properties
        selected_data = {}
        for prop_name in property_names:
            try:
                value = self.get_property_value(message, prop_name)
                selected_data[prop_name] = value
            except ValueError:
                logger.warning(f"Property '{prop_name}' not found in message")

        # Create new message with selected properties
        new_message = Message(text=message.text)
        for prop_name, value in selected_data.items():
            if hasattr(new_message, prop_name):
                setattr(new_message, prop_name, value)

        return new_message

    def filter_by_property(self) -> Message | list[Message]:
        """Filter messages based on property value."""
        messages = self.get_message_list()
        property_name = self.filter_property
        operator = self.operator
        filter_value = self.filter_value

        if not property_name:
            return messages

        filtered_messages = []
        for message in messages:
            try:
                property_value = self.get_property_value(message, property_name)
                if self.compare_values(property_value, filter_value, operator):
                    filtered_messages.append(message)
            except ValueError:
                logger.warning(f"Property '{property_name}' not found in message")

        return filtered_messages[0] if len(filtered_messages) == 1 else filtered_messages

    def combine_messages(self) -> Message:
        """Combine multiple messages into one."""
        messages = self.get_message_list()

        if not messages:
            return Message(text="")

        if len(messages) == 1:
            return messages[0]

        # Combine text from all messages
        combined_text = "\n".join([msg.text for msg in messages if msg.text])

        # Use properties from the first message as base
        base_message = messages[0]
        return Message(
            text=combined_text,
            sender=base_message.sender,
            sender_name=base_message.sender_name,
            session_id=base_message.session_id,
            category=base_message.category,
        )

    def split_messages(self) -> tuple[list[Message], list[Message]]:
        """Split messages based on property value."""
        messages = self.get_message_list()
        property_name = self.split_property
        operator = self.split_operator
        split_value = self.split_value

        if not property_name:
            return messages, []

        true_messages = []
        false_messages = []

        for message in messages:
            try:
                property_value = self.get_property_value(message, property_name)
                if self.compare_values(property_value, split_value, operator):
                    true_messages.append(message)
                else:
                    false_messages.append(message)
            except ValueError:
                logger.warning(f"Property '{property_name}' not found in message")
                false_messages.append(message)

        return true_messages, false_messages

    def transform_properties(self) -> Message:
        """Transform properties in the message."""
        message = self.get_single_message()
        transform_data = self.transform_properties

        if not transform_data:
            return message

        # Create new message with transformed properties
        new_message = Message(
            text=message.text,
            sender=message.sender,
            sender_name=message.sender_name,
            session_id=message.session_id,
            category=message.category,
            error=message.error,
            edit=message.edit,
        )

        # Apply transformations
        for property_name, new_value in transform_data.items():
            if hasattr(new_message, property_name):
                setattr(new_message, property_name, new_value)

        return new_message

    # Output methods
    def true_response(self) -> Message:
        """Return the first message that matches the split condition, or empty message."""
        if not hasattr(self, "operations") or not self.operations:
            return Message(text="")

        selected_actions = [action["name"] for action in self.operations]
        if len(selected_actions) != 1 or selected_actions[0] != "Split Messages":
            return Message(text="")

        true_messages, _ = self.split_messages()
        return true_messages[0] if true_messages else Message(text="")

    def false_response(self) -> Message:
        """Return the first message that doesn't match the split condition, or empty message."""
        if not hasattr(self, "operations") or not self.operations:
            return Message(text="")

        selected_actions = [action["name"] for action in self.operations]
        if len(selected_actions) != 1 or selected_actions[0] != "Split Messages":
            return Message(text="")

        _, false_messages = self.split_messages()
        return false_messages[0] if false_messages else Message(text="")

    # Configuration and execution methods
    def update_build_config(self, build_config: dotdict, field_value: Any, field_name: str | None = None) -> dotdict:
        """Update build configuration based on selected action."""
        if field_name != "operations":
            return build_config

        build_config["operations"]["value"] = field_value
        selected_actions = [action["name"] for action in field_value]

        # Handle single action case
        if len(selected_actions) == 1 and selected_actions[0] in ACTION_CONFIG:
            action = selected_actions[0]
            config = ACTION_CONFIG[action]

            build_config["message"]["is_list"] = config["is_list"]
            logger.info(config["log_msg"])

            return set_current_fields(
                build_config=build_config,
                action_fields=self.actions_data,
                selected_action=action,
                default_fields=self.default_keys,
                func=set_field_display,
            )

        # Handle no operations case
        if not selected_actions:
            logger.info("setting default fields")
            return set_current_fields(
                build_config=build_config,
                action_fields=self.actions_data,
                selected_action=None,
                default_fields=self.default_keys,
                func=set_field_display,
            )

        return build_config

    def update_outputs(self, frontend_node: dict, field_name: str, field_value: Any) -> dict:
        """Control output visibility based on the selected operation."""
        if field_name == "operations" and field_value and len(field_value) == 1:
            selected_action = field_value[0]["name"]

            # Update the show property of existing outputs
            for output in frontend_node.get("outputs", []):
                if output.get("name") == "message_output":
                    output["show"] = selected_action != "Split Messages"
                elif output.get("name") == "true_output" or output.get("name") == "false_output":
                    output["show"] = selected_action == "Split Messages"

        return frontend_node

    def as_message(self) -> Message | list[Message]:
        """Execute the selected action on the message."""
        if not hasattr(self, "operations") or not self.operations:
            return Message(text="")

        selected_actions = [action["name"] for action in self.operations]
        logger.info(f"selected_actions: {selected_actions}")

        # Only handle single action case for now
        if len(selected_actions) != 1:
            return Message(text="")

        action = selected_actions[0]

        # Action mapping
        action_map: dict[str, Callable[[], Message | list[Message]]] = {
            "Select Properties": self.select_properties,
            "Filter by Property": self.filter_by_property,
            "Combine Messages": self.combine_messages,
            "Split Messages": lambda: self.split_messages()[0] if self.split_messages()[0] else Message(text=""),
            "Transform Properties": self.transform_properties,
        }

        handler: Callable[[], Message | list[Message]] | None = action_map.get(action)
        if handler:
            try:
                return handler()
            except Exception as e:
                logger.error(f"Error executing {action}: {e!s}")
                raise

        return Message(text="")

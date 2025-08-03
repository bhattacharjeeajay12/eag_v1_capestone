from typing import List, Tuple, Dict, Optional, Any, Union
import json
import os
from datetime import datetime
import uuid
from langchain.schema import HumanMessage, AIMessage, SystemMessage, BaseMessage


class ConversationMemory:
    """
    Memory class for storing and retrieving conversations in LangChain message format.
    Supports saving to and loading from disk, with enhanced JSON handling.
    """

    def __init__(
        self, conversation_id: Optional[str] = None, save_dir: str = "conversations"
    ):
        """
        Initialize a conversation memory instance.

        Args:
            conversation_id: Unique identifier for the conversation. If None, a UUID will be generated.
            save_dir: Directory to save conversation files.
        """
        self.conversation_id = conversation_id or str(uuid.uuid4())
        self.save_dir = save_dir
        self.messages: List[BaseMessage] = []

        # Create save directory if it doesn't exist
        os.makedirs(save_dir, exist_ok=True)

    def add_message(self, message: BaseMessage) -> None:
        """
        Add a message to the conversation.

        Args:
            message: A LangChain message (SystemMessage, HumanMessage, or AIMessage).
        """
        self.messages.append(message)

    def add_system_message(self, content: str) -> None:
        """Add a system message to the conversation."""
        self.add_message(SystemMessage(content=content))

    def add_human_message(self, content: Union[str, Dict, List]) -> None:
        """
        Add a human message to the conversation.

        Args:
            content: Content of the message. Can be a string or a JSON-serializable object.
                    If a dict or list is provided, it will be converted to a JSON string.
        """
        if isinstance(content, (dict, list)):
            # Convert dict/list to formatted JSON string
            content_str = json.dumps(content, indent=2)
        else:
            content_str = str(content)

        self.add_message(HumanMessage(content=content_str))

    def add_ai_message(self, content: Union[str, Dict, List]) -> None:
        """
        Add an AI message to the conversation.

        Args:
            content: Content of the message. Can be a string or a JSON-serializable object.
                    If a dict or list is provided, it will be converted to a JSON string.
        """
        if isinstance(content, (dict, list)):
            # Convert dict/list to formatted JSON string
            content_str = json.dumps(content, indent=2)
        else:
            content_str = str(content)

        self.add_message(AIMessage(content=content_str))

    def get_messages(self) -> List[BaseMessage]:
        """Get all messages in the conversation."""
        return self.messages

    def get_message_tuples(self) -> List[Tuple[str, str]]:
        """
        Get messages as a list of tuples in the format:
        [("system", "content"), ("human", "content"), ("ai", "content"), ...]
        """
        message_tuples = []
        for message in self.messages:
            if isinstance(message, SystemMessage):
                message_tuples.append(("system", message.content))
            elif isinstance(message, HumanMessage):
                message_tuples.append(("human", message.content))
            elif isinstance(message, AIMessage):
                message_tuples.append(("ai", message.content))
        return message_tuples

    def get_safe_message_tuples(self) -> List[Tuple[str, str]]:
        """
        Get messages as a list of tuples with all curly braces escaped to prevent
        string formatting conflicts. Use this when messages will be inserted into
        prompt templates using .format() or f-strings.

        Returns:
            List of ("role", "content") tuples with escaped content.
        """
        message_tuples = []
        for message in self.messages:
            if isinstance(message, SystemMessage):
                message_tuples.append(("system", self._escape_braces(message.content)))
            elif isinstance(message, HumanMessage):
                message_tuples.append(("human", self._escape_braces(message.content)))
            elif isinstance(message, AIMessage):
                message_tuples.append(("ai", self._escape_braces(message.content)))
        return message_tuples

    def get_message_objects(self) -> List[Dict[str, Any]]:
        """
        Get messages as a list of dictionary objects.
        If a message contains valid JSON, it's parsed into a Python object.

        Returns:
            List of dictionaries with 'role' and 'content' keys.
            The 'content' will be a parsed object if it was valid JSON.
        """
        message_objects = []
        for message in self.messages:
            msg_type = self._get_message_type(message)
            content = message.content

            # Try to parse content as JSON if it looks like JSON
            parsed_content = self._try_parse_json(content)

            message_objects.append(
                {
                    "role": msg_type,
                    "content": (
                        parsed_content if parsed_content is not None else content
                    ),
                }
            )

        return message_objects

    def get_langchain_messages(self) -> List[Dict[str, str]]:
        """
        Get messages in LangChain's expected format for chat history.
        Formats messages for compatible use with LangChain without escaping JSON content.

        Returns:
            List of dictionaries with 'type' and 'content' keys in LangChain format.
        """
        # Convert native LangChain message objects to the format expected by LangChain's MessagesPlaceholder
        messages = []
        for message in self.messages:
            if isinstance(message, SystemMessage):
                messages.append({"type": "system", "content": message.content})
            elif isinstance(message, HumanMessage):
                messages.append({"type": "human", "content": message.content})
            elif isinstance(message, AIMessage):
                messages.append({"type": "ai", "content": message.content})
        return messages

    def _escape_braces(self, content: str) -> str:
        """
        Escape curly braces in a string to prevent conflicts with string formatting.
        Each { becomes {{ and each } becomes }}.

        Args:
            content: String that might contain curly braces.

        Returns:
            String with escaped curly braces.
        """
        return content.replace("{", "{{").replace("}", "}}")

    def _try_parse_json(self, content: str) -> Optional[Union[Dict, List]]:
        """
        Try to parse a string as JSON.

        Args:
            content: String to parse.

        Returns:
            Parsed JSON object if valid, None otherwise.
        """
        if not isinstance(content, str):
            return None

        content = content.strip()
        if (content.startswith("{") and content.endswith("}")) or (
            content.startswith("[") and content.endswith("]")
        ):
            try:
                return json.loads(content)
            except json.JSONDecodeError:
                return None
        return None

    def clear(self) -> None:
        """Clear all messages from memory."""
        self.messages = []

    def to_dict(self) -> Dict[str, Any]:
        """Convert the conversation to a dictionary for serialization."""
        return {
            "conversation_id": self.conversation_id,
            "timestamp": datetime.now().isoformat(),
            "messages": [
                {"type": self._get_message_type(msg), "content": msg.content}
                for msg in self.messages
            ],
        }

    def to_json(self, indent: int = 2) -> str:
        """
        Convert the conversation to a JSON string.

        Args:
            indent: Number of spaces for indentation.

        Returns:
            Conversation as a JSON string.
        """
        return json.dumps(self.to_dict(), indent=indent)

    def _get_message_type(self, message: BaseMessage) -> str:
        """Get the type of a message as a string."""
        if isinstance(message, SystemMessage):
            return "system"
        elif isinstance(message, HumanMessage):
            return "human"
        elif isinstance(message, AIMessage):
            return "ai"
        else:
            return "unknown"

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ConversationMemory":
        """Create a ConversationMemory instance from a dictionary."""
        memory = cls(conversation_id=data.get("conversation_id"))

        for msg_data in data.get("messages", []):
            msg_type = msg_data.get("type")
            content = msg_data.get("content")

            if msg_type == "system":
                memory.add_system_message(content)
            elif msg_type == "human":
                memory.add_human_message(content)
            elif msg_type == "ai":
                memory.add_ai_message(content)

        return memory

    @classmethod
    def from_json(cls, json_str: str) -> "ConversationMemory":
        """
        Create a ConversationMemory instance from a JSON string.

        Args:
            json_str: JSON string representation of a conversation.

        Returns:
            A ConversationMemory instance.
        """
        data = json.loads(json_str)
        return cls.from_dict(data)

    def save(self) -> str:
        """
        Save the conversation to disk.

        Returns:
            Path to the saved file.
        """
        file_path = os.path.join(self.save_dir, f"{self.conversation_id}.json")
        with open(file_path, "w") as f:
            json.dump(self.to_dict(), f, indent=2)
        return file_path

    @classmethod
    def load(
        cls, conversation_id: str, save_dir: str = "conversations"
    ) -> "ConversationMemory":
        """
        Load a conversation from disk.

        Args:
            conversation_id: ID of the conversation to load.
            save_dir: Directory where the conversation file is stored.

        Returns:
            A ConversationMemory instance with loaded messages.

        Raises:
            FileNotFoundError: If the conversation file doesn't exist.
        """
        file_path = os.path.join(save_dir, f"{conversation_id}.json")
        with open(file_path, "r") as f:
            data = json.load(f)
        return cls.from_dict(data)

    @staticmethod
    def list_conversations(save_dir: str = "conversations") -> List[str]:
        """
        List all available conversation IDs.

        Args:
            save_dir: Directory where conversation files are stored.

        Returns:
            A list of conversation IDs (filenames without .json extension).
        """
        if not os.path.exists(save_dir):
            return []

        conversation_ids = []
        for filename in os.listdir(save_dir):
            if filename.endswith(".json"):
                conversation_ids.append(filename[:-5])  # Remove .json extension

        return conversation_ids


if __name__ == "__main__":
    # Basic example
    memory = ConversationMemory()

    # Adding string messages
    memory.add_human_message("Hello, how are you?")
    memory.add_ai_message("I'm good, thank you!")

    # Adding message with curly braces - potential string format conflict
    user_query = "What is the weather in New York? And show me data in {json} format."
    memory.add_human_message(user_query)

    # Adding JSON as an AI message
    ai_response = {
        "temperature": 22,
        "condition": "Sunny",
        "forecast": [
            {"day": "Today", "high": 25, "low": 18},
            {"day": "Tomorrow", "high": 27, "low": 19},
        ],
    }
    memory.add_ai_message(ai_response)

    # Save to disk
    file_path = memory.save()
    print(f"Saved to: {file_path}")

    # Print different representations
    print("\nRaw messages:")
    print(memory.get_messages())

    print("\nNormal message tuples (might cause formatting issues):")
    print(memory.get_message_tuples())

    print("\nSafe message tuples (for use with string formatting):")
    print(memory.get_safe_message_tuples())

    print("\nLangChain format messages (safe for prompts):")
    print(memory.get_langchain_messages())

    print("\nMessage objects (with JSON parsing):")
    for msg in memory.get_message_objects():
        print(f"{msg['role']}: {type(msg['content'])}")
        print(msg["content"])
        print("-" * 40)

    # Show how this works with string formatting
    template = "Content: {}"
    safe_content = memory.get_safe_message_tuples()[2][
        1
    ]  # Get content of third message
    formatted = template.format(safe_content)
    print("\nSafely formatted content:")
    print(formatted)
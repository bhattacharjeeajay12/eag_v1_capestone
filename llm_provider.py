import os
from dotenv import load_dotenv
from typing import Optional, List, Dict
from langchain_openai import ChatOpenAI
from langchain.schema import HumanMessage, SystemMessage, AIMessage
import httpx
import ssl

# Try to import the ssl_helper module for secure client creation
try:
    from ssl_helper import create_insecure_http_client

    use_ssl_helper = True
except ImportError:
    use_ssl_helper = False

# Load environment variables from .env file when module is imported
load_dotenv()

# Set environment variables to disable SSL verification
os.environ["PYTHONHTTPSVERIFY"] = "0"
os.environ["OPENAI_API_SKIP_VERIFY_SSL"] = "true"
os.environ["CURL_CA_BUNDLE"] = ""
os.environ["REQUESTS_CA_BUNDLE"] = ""

# Clear SSL_CERT_FILE environment variable if set
if "SSL_CERT_FILE" in os.environ:
    del os.environ["SSL_CERT_FILE"]

# Disable SSL verification warnings
import urllib3

urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)


class LLMProvider:
    """Provider for LLM interactions using LangChain's OpenAI implementation."""

    def __init__(
        self,
        api_key: Optional[str] = None,
        model: str = "gpt-4o",
        verify_ssl: bool = False,
    ):
        """
        Initialize the LLM provider.

        Args:
            api_key: OpenAI API key. If None, it will be read from environment variable.
            model: The model to use for completions.
            verify_ssl: Whether to verify SSL certificates. Set to False to bypass SSL verification.
        """
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        if not self.api_key:
            raise ValueError(
                "OpenAI API key not provided and not found in environment variables"
            )

        self.model = model
        self.verify_ssl = verify_ssl

        # Create HTTP client with SSL verification disabled
        if use_ssl_helper:
            # Use the helper if available
            http_client = create_insecure_http_client()
        else:
            # Create custom HTTPX client with SSL verification disabled
            ssl_context = ssl.create_default_context()
            if not verify_ssl:
                ssl_context.check_hostname = False
                ssl_context.verify_mode = ssl.CERT_NONE

            # Ensure we're explicitly setting verify=False to disable SSL verification
            http_client = httpx.Client(
                verify=False,  # Force to False
                timeout=120.0,
                http2=False,  # Disable HTTP/2 as it can sometimes cause SSL issues
            )

        # Initialize LangChain's ChatOpenAI with custom HTTP client
        self.chat_model = ChatOpenAI(
            model=self.model,
            openai_api_key=self.api_key,
            temperature=0,
            http_client=http_client,
        )

    def get_completion(
        self,
        prompt: str,
        system_message: str = "You are a helpful assistant that provides concise and accurate information.",
        temperature: float = 0.0,
    ) -> str:
        """
        Get a completion from the LLM.

        Args:
            prompt: The user prompt to send to the model ie the human message
            system_message: The system message to set the context.
            temperature: Controls randomness. Lower is more deterministic.

        Returns:
            The model's response text.
        """
        # Create message objects
        messages = [SystemMessage(content=system_message), HumanMessage(content=prompt)]

        # Override temperature if needed
        if temperature != 0.0:
            self.chat_model.temperature = temperature

        # Get response from the model
        response = self.chat_model.invoke(messages)

        return response.content

    def get_completion_with_messages(
        self, messages: List[Dict[str, str]], temperature: float = 0.0
    ) -> str:
        """
        Get a completion from the LLM with custom message format.

        Args:
            messages: List of message dictionaries with 'role' and 'content' keys.
            temperature: Controls randomness. Lower is more deterministic.

        Returns:
            The model's response text.
        """
        # Convert dict messages to LangChain message objects
        langchain_messages = []
        for msg in messages:
            if msg["role"] == "system":
                langchain_messages.append(SystemMessage(content=msg["content"]))
            elif msg["role"] == "user":
                langchain_messages.append(HumanMessage(content=msg["content"]))
            elif msg["role"] == "assistant":
                langchain_messages.append(AIMessage(content=msg["content"]))

        # Override temperature if needed
        if temperature != 0.0:
            self.chat_model.temperature = temperature

        # Get response from the model
        response = self.chat_model.invoke(langchain_messages)

        return response.content


# Create a default instance for easy import with SSL verification disabled
default_llm = LLMProvider(verify_ssl=False)


def get_llm(
    api_key: Optional[str] = None, model: str = "gpt-4o-mini", verify_ssl: bool = True
) -> LLMProvider:
    """
    Get an LLM provider instance.

    Args:
        api_key: OpenAI API key. If None, it will be read from environment variable.
        model: The model to use for completions.
        verify_ssl: Whether to verify SSL certificates.

    Returns:
        An LLMProvider instance.
    """
    return LLMProvider(api_key=api_key, model=model, verify_ssl=verify_ssl)
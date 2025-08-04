"""
Utility Functions - E-Commerce Multi-Agent System
This module provides common utility functions used throughout the system.
"""

import json
import re
import logging
import asyncio
from datetime import datetime
from typing import Any, Dict, List, Optional, Union
from pathlib import Path
import traceback
import sys

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler('ecommerce_agent.log')
    ]
)

logger = logging.getLogger(__name__)

def log_step(message: str, symbol: str = "🔧"):
    """
    Log a step in the execution process.
    
    Args:
        message: The message to log
        symbol: Emoji symbol to prefix the message
    """
    timestamp = datetime.now().strftime("%H:%M:%S")
    print(f"{symbol} [{timestamp}] {message}")
    logger.info(f"{symbol} {message}")

def log_error(message: str, symbol: str = "❌"):
    """
    Log an error message.
    
    Args:
        message: The error message to log
        symbol: Emoji symbol to prefix the message
    """
    timestamp = datetime.now().strftime("%H:%M:%S")
    print(f"{symbol} [{timestamp}] {message}")
    logger.error(f"{symbol} {message}")

def log_success(message: str, symbol: str = "✅"):
    """
    Log a success message.
    
    Args:
        message: The success message to log
        symbol: Emoji symbol to prefix the message
    """
    timestamp = datetime.now().strftime("%H:%M:%S")
    print(f"{symbol} [{timestamp}] {message}")
    logger.info(f"{symbol} {message}")

def log_warning(message: str, symbol: str = "⚠️"):
    """
    Log a warning message.
    
    Args:
        message: The warning message to log
        symbol: Emoji symbol to prefix the message
    """
    timestamp = datetime.now().strftime("%H:%M:%S")
    print(f"{symbol} [{timestamp}] {message}")
    logger.warning(f"{symbol} {message}")

def parse_llm_json(text: str) -> Dict[str, Any]:
    """
    Parse JSON from LLM response text.
    
    This function handles various JSON formats that LLMs might return,
    including JSON wrapped in markdown code blocks.
    
    Args:
        text: Text containing JSON to parse
        
    Returns:
        Parsed JSON as dictionary
        
    Raises:
        ValueError: If no valid JSON can be extracted
    """
    # Remove leading/trailing whitespace
    text = text.strip()
    
    # Try to parse as direct JSON first
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    
    # Look for JSON in markdown code blocks
    json_patterns = [
        r'```(?:json)?\s*(\{.*?\})\s*```',  # JSON in code blocks
        r'```(?:json)?\s*(\[.*?\])\s*```',  # JSON arrays in code blocks
        r'\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}',  # JSON objects in text
        r'\[[^\[\]]*(?:\[[^\[\]]*\][^\[\]]*)*\]'  # JSON arrays in text
    ]
    
    for pattern in json_patterns:
        matches = re.findall(pattern, text, re.DOTALL)
        for match in matches:
            try:
                return json.loads(match)
            except json.JSONDecodeError:
                continue
    
    # If no JSON found, try to extract key-value pairs
    try:
        return _extract_key_value_pairs(text)
    except Exception:
        pass
    
    raise ValueError(f"Could not extract valid JSON from text: {text[:200]}...")

def _extract_key_value_pairs(text: str) -> Dict[str, Any]:
    """
    Extract key-value pairs from text when JSON parsing fails.
    
    Args:
        text: Text to extract key-value pairs from
        
    Returns:
        Dictionary of extracted key-value pairs
    """
    result = {}
    
    # Look for patterns like "key": value or key: value
    patterns = [
        r'"([^"]+)"\s*:\s*"([^"]*)"',  # "key": "value"
        r'"([^"]+)"\s*:\s*([^,\n]+)',  # "key": value
        r'(\w+)\s*:\s*"([^"]*)"',      # key: "value"
        r'(\w+)\s*:\s*([^,\n]+)'       # key: value
    ]
    
    for pattern in patterns:
        matches = re.findall(pattern, text)
        for key, value in matches:
            # Clean up the value
            value = value.strip()
            if value.lower() in ['true', 'false']:
                value = value.lower() == 'true'
            elif value.isdigit():
                value = int(value)
            elif value.replace('.', '').isdigit():
                value = float(value)
            
            result[key] = value
    
    return result

def safe_json_dumps(obj: Any, **kwargs) -> str:
    """
    Safely serialize an object to JSON string.
    
    Args:
        obj: Object to serialize
        **kwargs: Additional arguments for json.dumps
        
    Returns:
        JSON string representation
    """
    def default_serializer(obj):
        if hasattr(obj, '__dict__'):
            return obj.__dict__
        elif hasattr(obj, 'isoformat'):  # datetime objects
            return obj.isoformat()
        elif hasattr(obj, '__iter__') and not isinstance(obj, (str, bytes, bytearray)):
            return list(obj)
        else:
            return str(obj)
    
    return json.dumps(obj, default=default_serializer, **kwargs)

def validate_file_path(file_path: str) -> bool:
    """
    Validate if a file path exists and is accessible.
    
    Args:
        file_path: Path to validate
        
    Returns:
        True if file exists and is accessible, False otherwise
    """
    try:
        path = Path(file_path)
        return path.exists() and path.is_file()
    except Exception:
        return False

def get_file_info(file_path: str) -> Dict[str, Any]:
    """
    Get information about a file.
    
    Args:
        file_path: Path to the file
        
    Returns:
        Dictionary containing file information
    """
    try:
        path = Path(file_path)
        if not path.exists():
            return {"error": "File does not exist"}
        
        stat = path.stat()
        return {
            "name": path.name,
            "size": stat.st_size,
            "extension": path.suffix.lower(),
            "modified": datetime.fromtimestamp(stat.st_mtime).isoformat(),
            "created": datetime.fromtimestamp(stat.st_ctime).isoformat(),
            "is_file": path.is_file(),
            "is_dir": path.is_dir()
        }
    except Exception as e:
        return {"error": str(e)}

def format_duration(seconds: float) -> str:
    """
    Format duration in seconds to human-readable string.
    
    Args:
        seconds: Duration in seconds
        
    Returns:
        Formatted duration string
    """
    if seconds < 60:
        return f"{seconds:.2f}s"
    elif seconds < 3600:
        minutes = seconds / 60
        return f"{minutes:.1f}m"
    else:
        hours = seconds / 3600
        return f"{hours:.1f}h"

def format_file_size(bytes_size: int) -> str:
    """
    Format file size in bytes to human-readable string.
    
    Args:
        bytes_size: Size in bytes
        
    Returns:
        Formatted size string
    """
    for unit in ['B', 'KB', 'MB', 'GB', 'TB']:
        if bytes_size < 1024.0:
            return f"{bytes_size:.1f}{unit}"
        bytes_size /= 1024.0
    return f"{bytes_size:.1f}PB"

def retry_with_backoff(func, max_retries: int = 3, base_delay: float = 1.0):
    """
    Retry a function with exponential backoff.
    
    Args:
        func: Function to retry
        max_retries: Maximum number of retries
        base_delay: Base delay between retries
        
    Returns:
        Result of the function call
        
    Raises:
        Exception: If all retries fail
    """
    last_exception = None
    
    for attempt in range(max_retries + 1):
        try:
            return func()
        except Exception as e:
            last_exception = e
            if attempt < max_retries:
                delay = base_delay * (2 ** attempt)
                log_warning(f"Attempt {attempt + 1} failed: {str(e)}. Retrying in {delay:.1f}s...")
                asyncio.sleep(delay)
    
    raise last_exception

async def async_retry_with_backoff(func, max_retries: int = 3, base_delay: float = 1.0):
    """
    Retry an async function with exponential backoff.
    
    Args:
        func: Async function to retry
        max_retries: Maximum number of retries
        base_delay: Base delay between retries
        
    Returns:
        Result of the function call
        
    Raises:
        Exception: If all retries fail
    """
    last_exception = None
    
    for attempt in range(max_retries + 1):
        try:
            return await func()
        except Exception as e:
            last_exception = e
            if attempt < max_retries:
                delay = base_delay * (2 ** attempt)
                log_warning(f"Attempt {attempt + 1} failed: {str(e)}. Retrying in {delay:.1f}s...")
                await asyncio.sleep(delay)
    
    raise last_exception

def sanitize_filename(filename: str) -> str:
    """
    Sanitize a filename for safe file system operations.
    
    Args:
        filename: Original filename
        
    Returns:
        Sanitized filename
    """
    # Remove or replace unsafe characters
    unsafe_chars = '<>:"/\\|?*'
    for char in unsafe_chars:
        filename = filename.replace(char, '_')
    
    # Remove leading/trailing spaces and dots
    filename = filename.strip(' .')
    
    # Ensure filename is not empty
    if not filename:
        filename = "unnamed_file"
    
    return filename

def create_safe_directory(path: str) -> bool:
    """
    Create a directory safely.
    
    Args:
        path: Directory path to create
        
    Returns:
        True if directory was created or exists, False otherwise
    """
    try:
        Path(path).mkdir(parents=True, exist_ok=True)
        return True
    except Exception as e:
        log_error(f"Failed to create directory {path}: {str(e)}")
        return False

def get_memory_usage() -> Dict[str, float]:
    """
    Get current memory usage information.
    
    Returns:
        Dictionary containing memory usage information
    """
    try:
        import psutil
        process = psutil.Process()
        memory_info = process.memory_info()
        
        return {
            "rss_mb": memory_info.rss / 1024 / 1024,  # Resident Set Size in MB
            "vms_mb": memory_info.vms / 1024 / 1024,  # Virtual Memory Size in MB
            "percent": process.memory_percent()
        }
    except ImportError:
        return {"error": "psutil not available"}

def format_timestamp(timestamp: Union[str, datetime]) -> str:
    """
    Format timestamp to consistent string format.
    
    Args:
        timestamp: Timestamp to format
        
    Returns:
        Formatted timestamp string
    """
    if isinstance(timestamp, str):
        try:
            timestamp = datetime.fromisoformat(timestamp.replace('Z', '+00:00'))
        except ValueError:
            return timestamp
    
    if isinstance(timestamp, datetime):
        return timestamp.isoformat()
    
    return str(timestamp)

def truncate_text(text: str, max_length: int = 100, suffix: str = "...") -> str:
    """
    Truncate text to specified length.
    
    Args:
        text: Text to truncate
        max_length: Maximum length
        suffix: Suffix to add when truncated
        
    Returns:
        Truncated text
    """
    if len(text) <= max_length:
        return text
    
    return text[:max_length - len(suffix)] + suffix

def extract_emails(text: str) -> List[str]:
    """
    Extract email addresses from text.
    
    Args:
        text: Text to search for emails
        
    Returns:
        List of found email addresses
    """
    email_pattern = r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'
    return re.findall(email_pattern, text)

def extract_phone_numbers(text: str) -> List[str]:
    """
    Extract phone numbers from text.
    
    Args:
        text: Text to search for phone numbers
        
    Returns:
        List of found phone numbers
    """
    phone_pattern = r'(\+\d{1,3}[-.\s]?)?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}'
    return re.findall(phone_pattern, text)

def is_valid_json(text: str) -> bool:
    """
    Check if text is valid JSON.
    
    Args:
        text: Text to validate
        
    Returns:
        True if valid JSON, False otherwise
    """
    try:
        json.loads(text)
        return True
    except (json.JSONDecodeError, TypeError):
        return False

def deep_merge_dicts(dict1: Dict[str, Any], dict2: Dict[str, Any]) -> Dict[str, Any]:
    """
    Deep merge two dictionaries.
    
    Args:
        dict1: First dictionary
        dict2: Second dictionary
        
    Returns:
        Merged dictionary
    """
    result = dict1.copy()
    
    for key, value in dict2.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = deep_merge_dicts(result[key], value)
        else:
            result[key] = value
    
    return result

def get_system_info() -> Dict[str, Any]:
    """
    Get system information.
    
    Returns:
        Dictionary containing system information
    """
    import platform
    import sys
    
    return {
        "platform": platform.platform(),
        "python_version": sys.version,
        "architecture": platform.architecture()[0],
        "processor": platform.processor(),
        "system": platform.system(),
        "release": platform.release()
    } 
"""
AgentRunner - Individual Agent Execution Engine
This module handles the execution of individual agents with LLM integration,
MCP tool access, and comprehensive error handling.
"""

import yaml
import json
import asyncio
from pathlib import Path
from typing import Optional, List, Dict, Any
from datetime import datetime
import time

# Import core components
from .model_manager import ModelManager
from utils.json_parser import parse_llm_json
from utils.utils import log_step, log_error, log_success
from PIL import Image
import os

class AgentRunner:
    """
    Executes individual agents with LLM integration and MCP tool access.
    
    This class is responsible for:
    1. Loading agent configurations and prompts
    2. Managing LLM model interactions
    3. Providing MCP tool access to agents
    4. Handling agent execution with retries and error recovery
    5. Processing agent outputs and responses
    """
    
    def __init__(self, multi_mcp):
        """
        Initialize the AgentRunner.
        
        Args:
            multi_mcp: MultiMCP instance for external tool access
        """
        self.multi_mcp = multi_mcp
        
        # Load agent configurations
        config_path = Path("config/agent_config.yaml")
        if not config_path.exists():
            raise FileNotFoundError(f"Agent config not found: {config_path}")
        
        with open(config_path, "r", encoding="utf-8") as f:
            config = yaml.safe_load(f)
            self.agent_configs = config["agents"]
            self.global_settings = config.get("global_settings", {})
        
        # Initialize model manager
        self.model_manager = ModelManager()
        
        # Performance tracking
        self.execution_stats = {
            "total_agent_runs": 0,
            "successful_runs": 0,
            "failed_runs": 0,
            "total_execution_time": 0.0,
            "total_cost": 0.0
        }
        
        log_step("🚀 AgentRunner initialized", symbol="✅")

    def _analyze_file_strategy(self, uploaded_files: List[str]) -> str:
        """
        Analyze files to determine best upload strategy for MCP tools.
        
        Args:
            uploaded_files: List of file paths
            
        Returns:
            Strategy string for file handling
        """
        if not uploaded_files:
            return "none"
        
        total_size = 0
        file_info = []
        
        for file_path in uploaded_files:
            path = Path(file_path)
            if path.exists():
                size = path.stat().st_size
                total_size += size
                file_info.append({
                    'path': file_path,
                    'size': size,
                    'extension': path.suffix.lower()
                })
        
        # Decision logic based on total size and file count
        if total_size < 15_000_000:  # < 15MB total
            return "inline_batch"  # Send all as inline data
        elif len(file_info) == 1 and total_size < 50_000_000:  # Single file < 50MB
            return "files_api_single"  # Use Files API for single file
        else:
            return "files_api_individual"  # Use Files API for each file

    def _get_mime_type(self, extension: str) -> str:
        """
        Get MIME type for file extension.
        
        Args:
            extension: File extension (e.g., '.pdf', '.txt')
            
        Returns:
            MIME type string
        """
        mime_type_map = {
            # Documents
            '.pdf': 'application/pdf',
            '.txt': 'text/plain',
            '.docx': 'application/vnd.openxmlformats-officedocument.wordprocessingml.document',
            '.doc': 'application/msword',
            '.rtf': 'application/rtf',
            '.json': 'application/json',
            
            # Spreadsheets
            '.xlsx': 'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet',
            '.xls': 'application/vnd.ms-excel',
            '.csv': 'text/csv',
            '.tsv': 'text/tab-separated-values',
            
            # Presentations
            '.pptx': 'application/vnd.openxmlformats-officedocument.presentationml.presentation',
            
            # Images
            '.png': 'image/png',
            '.jpg': 'image/jpeg',
            '.jpeg': 'image/jpeg',
            '.webp': 'image/webp',
            '.heif': 'image/heif',
            
            # Videos
            '.mp4': 'video/mp4',
            '.mpeg': 'video/mpeg',
            '.mov': 'video/quicktime',
            '.avi': 'video/x-msvideo',
            '.mpg': 'video/mpeg',
            '.webm': 'video/webm',
            '.wmv': 'video/x-ms-wmv',
            '.flv': 'video/x-flv',
            '.3gp': 'video/3gpp',
            
            # Code files
            '.c': 'text/x-c',
            '.cpp': 'text/x-c++',
            '.py': 'text/x-python',
            '.java': 'text/x-java',
            '.php': 'text/x-php',
            '.sql': 'application/sql',
            '.html': 'text/html',
            '.htm': 'text/html',
            '.css': 'text/css',
            '.js': 'text/javascript',
            '.xml': 'text/xml',
            '.md': 'text/markdown',
        }
        
        return mime_type_map.get(extension.lower(), 'application/octet-stream')

    def _detect_files_in_inputs(self, input_data: Any) -> List[Dict[str, Any]]:
        """
        Detect file references in input data.
        
        Args:
            input_data: Input data that might contain file references
            
        Returns:
            List of detected file information
        """
        detected_files = []
        
        def _scan_for_files(obj, path=""):
            if isinstance(obj, dict):
                for key, value in obj.items():
                    current_path = f"{path}.{key}" if path else key
                    _scan_for_files(value, current_path)
            elif isinstance(obj, list):
                for i, item in enumerate(obj):
                    current_path = f"{path}[{i}]"
                    _scan_for_files(item, current_path)
            elif isinstance(obj, str):
                # Check if string looks like a file path
                if (obj.startswith('/') or ':' in obj) and Path(obj).exists():
                    detected_files.append({
                        'path': obj,
                        'location': path,
                        'size': Path(obj).stat().st_size if Path(obj).exists() else 0
                    })
        
        _scan_for_files(input_data)
        return detected_files

    async def run_agent(self, agent_type: str, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute an agent with the given input data.
        
        This method handles the complete agent execution lifecycle:
        1. Load agent configuration and prompt
        2. Prepare input data and detect files
        3. Execute LLM with proper error handling
        4. Process and validate agent output
        5. Handle MCP tool calls if needed
        6. Return structured results
        
        Args:
            agent_type: Type of agent to execute (e.g., "CustomerIntentAgent")
            input_data: Input data for the agent
            
        Returns:
            Dictionary containing execution results
        """
        start_time = time.time()
        self.execution_stats["total_agent_runs"] += 1
        
        try:
            # Validate agent type
            if agent_type not in self.agent_configs:
                raise ValueError(f"Unknown agent type: {agent_type}")
            
            agent_config = self.agent_configs[agent_type]
            
            # Load agent prompt
            prompt_file = Path(agent_config["prompt_file"])
            if not prompt_file.exists():
                raise FileNotFoundError(f"Prompt file not found: {prompt_file}")
            
            with open(prompt_file, "r", encoding="utf-8") as f:
                system_prompt = f.read()
            
            # Prepare input data
            prepared_input = await self._prepare_agent_input(input_data, agent_config)
            
            # Build complete prompt
            full_prompt = self._build_prompt(system_prompt, prepared_input)
            
            # Execute LLM
            llm_result = await self._execute_llm(agent_type, full_prompt, agent_config)
            
            # Process agent output
            processed_output = await self._process_agent_output(llm_result, agent_config)
            
            # Handle MCP tool calls if present
            if "tool_calls" in processed_output:
                tool_results = await self._execute_tool_calls(processed_output["tool_calls"], agent_config)
                processed_output["tool_results"] = tool_results
            
            # Update statistics
            execution_time = time.time() - start_time
            self.execution_stats["successful_runs"] += 1
            self.execution_stats["total_execution_time"] += execution_time
            
            log_success(f"✅ {agent_type} completed in {execution_time:.2f}s", symbol="🎉")
            
            return {
                "success": True,
                "output": processed_output,
                "execution_time": execution_time,
                "agent_type": agent_type,
                "input_tokens": llm_result.get("input_tokens", 0),
                "output_tokens": llm_result.get("output_tokens", 0),
                "cost": llm_result.get("cost", 0.0)
            }
            
        except Exception as e:
            execution_time = time.time() - start_time
            self.execution_stats["failed_runs"] += 1
            self.execution_stats["total_execution_time"] += execution_time
            
            log_error(f"❌ {agent_type} failed after {execution_time:.2f}s: {str(e)}", symbol="💥")
            
            return {
                "success": False,
                "error": str(e),
                "execution_time": execution_time,
                "agent_type": agent_type
            }

    async def _prepare_agent_input(self, input_data: Dict[str, Any], agent_config: Dict[str, Any]) -> Dict[str, Any]:
        """
        Prepare input data for agent execution.
        
        Args:
            input_data: Raw input data
            agent_config: Agent configuration
            
        Returns:
            Prepared input data
        """
        prepared_input = input_data.copy()
        
        # Detect files in inputs
        detected_files = self._detect_files_in_inputs(input_data)
        if detected_files:
            prepared_input["detected_files"] = detected_files
            
            # Analyze file strategy
            file_paths = [f["path"] for f in detected_files]
            strategy = self._analyze_file_strategy(file_paths)
            prepared_input["file_strategy"] = strategy
        
        # Add agent capabilities
        prepared_input["agent_capabilities"] = agent_config.get("capabilities", [])
        
        # Add MCP server information
        mcp_servers = agent_config.get("mcp_servers", [])
        if mcp_servers:
            available_tools = []
            for server_id in mcp_servers:
                try:
                    tools = await self.multi_mcp.get_tools_from_servers([server_id])
                    available_tools.extend(tools)
                except Exception as e:
                    log_error(f"❌ Failed to get tools from {server_id}: {str(e)}", symbol="🔧")
            
            prepared_input["available_tools"] = available_tools
        
        return prepared_input

    def _build_prompt(self, system_prompt: str, input_data: Dict[str, Any]) -> str:
        """
        Build the complete prompt for agent execution.
        
        Args:
            system_prompt: Base system prompt
            input_data: Input data for the agent
            
        Returns:
            Complete prompt string
        """
        # Convert input data to JSON for inclusion in prompt
        input_json = json.dumps(input_data, indent=2, default=str)
        
        # Build the complete prompt
        prompt = f"""
{system_prompt}

## INPUT DATA
```json
{input_json}
```

## INSTRUCTIONS
1. Analyze the input data carefully
2. Execute your specific agent responsibilities
3. Return your response in valid JSON format
4. Include any tool calls if needed
5. Provide clear, actionable outputs

## RESPONSE FORMAT
Return your response as a valid JSON object with the following structure:
```json
{{
    "output": {{
        // Your main output data here
    }},
    "tool_calls": {{
        // Optional: Tool calls if needed
    }},
    "metadata": {{
        "reasoning": "Your reasoning process",
        "confidence": 0.95,
        "next_steps": ["suggested next actions"]
    }}
}}
```

Please provide your response now:
"""
        
        return prompt

    async def _execute_llm(self, agent_type: str, prompt: str, agent_config: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute LLM with the given prompt.
        
        Args:
            agent_type: Type of agent being executed
            prompt: Complete prompt for LLM
            agent_config: Agent configuration
            
        Returns:
            LLM execution results
        """
        model_name = agent_config.get("model", self.global_settings.get("default_model", "gemini"))
        max_retries = agent_config.get("max_retries", self.global_settings.get("default_retries", 3))
        timeout = agent_config.get("timeout", self.global_settings.get("default_timeout", 30))
        
        # Set model for this execution
        self.model_manager.set_model(model_name)
        
        last_error = None
        for attempt in range(max_retries):
            try:
                # Execute LLM with timeout
                result = await asyncio.wait_for(
                    self.model_manager.generate_text_with_usage(prompt),
                    timeout=timeout
                )
                
                return result
                
            except asyncio.TimeoutError:
                last_error = f"LLM execution timed out after {timeout}s"
                log_error(f"⏰ {agent_type} attempt {attempt + 1} timed out", symbol="⏰")
                
            except Exception as e:
                last_error = str(e)
                log_error(f"❌ {agent_type} attempt {attempt + 1} failed: {str(e)}", symbol="💥")
            
            # Wait before retry
            if attempt < max_retries - 1:
                wait_time = (2 ** attempt) * self.global_settings.get("retry_delay", 1.0)
                await asyncio.sleep(wait_time)
        
        raise RuntimeError(f"LLM execution failed after {max_retries} attempts. Last error: {last_error}")

    async def _process_agent_output(self, llm_result: Dict[str, Any], agent_config: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process and validate agent output from LLM.
        
        Args:
            llm_result: Raw LLM result
            agent_config: Agent configuration
            
        Returns:
            Processed agent output
        """
        try:
            # Extract text from LLM result
            response_text = llm_result.get("text", "")
            
            # Try to parse as JSON
            try:
                parsed_output = parse_llm_json(response_text)
            except Exception as e:
                # If JSON parsing fails, try to extract JSON from the response
                log_error(f"❌ JSON parsing failed: {str(e)}", symbol="🔧")
                parsed_output = self._extract_json_from_text(response_text)
            
            # Validate output structure
            if not isinstance(parsed_output, dict):
                raise ValueError("Agent output must be a dictionary")
            
            # Ensure required fields
            if "output" not in parsed_output:
                parsed_output["output"] = {}
            
            # Add metadata
            parsed_output["metadata"] = parsed_output.get("metadata", {})
            parsed_output["metadata"].update({
                "agent_type": agent_config.get("description", "Unknown"),
                "execution_timestamp": datetime.utcnow().isoformat(),
                "model_used": llm_result.get("model", "unknown"),
                "input_tokens": llm_result.get("input_tokens", 0),
                "output_tokens": llm_result.get("output_tokens", 0),
                "cost": llm_result.get("cost", 0.0)
            })
            
            return parsed_output
            
        except Exception as e:
            log_error(f"❌ Failed to process agent output: {str(e)}", symbol="💥")
            # Return fallback output
            return {
                "output": {
                    "error": f"Failed to process agent output: {str(e)}",
                    "raw_response": llm_result.get("text", "")
                },
                "metadata": {
                    "processing_error": str(e),
                    "execution_timestamp": datetime.utcnow().isoformat()
                }
            }

    def _extract_json_from_text(self, text: str) -> Dict[str, Any]:
        """
        Extract JSON from text that might contain additional content.
        
        Args:
            text: Text that might contain JSON
            
        Returns:
            Extracted JSON as dictionary
        """
        # Try to find JSON blocks
        import re
        
        # Look for JSON code blocks
        json_pattern = r'```(?:json)?\s*(\{.*?\})\s*```'
        matches = re.findall(json_pattern, text, re.DOTALL)
        
        if matches:
            try:
                return json.loads(matches[0])
            except json.JSONDecodeError:
                pass
        
        # Look for JSON objects in the text
        json_pattern = r'\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}'
        matches = re.findall(json_pattern, text, re.DOTALL)
        
        for match in matches:
            try:
                return json.loads(match)
            except json.JSONDecodeError:
                continue
        
        # If no JSON found, return as text
        return {
            "output": {
                "text_response": text.strip()
            }
        }

    async def _execute_tool_calls(self, tool_calls: Dict[str, Any], agent_config: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute MCP tool calls returned by agents.
        
        Args:
            tool_calls: Tool calls to execute
            agent_config: Agent configuration
            
        Returns:
            Results of tool executions
        """
        results = {}
        
        for tool_name, tool_args in tool_calls.items():
            try:
                # Execute tool through MultiMCP
                result = await self.multi_mcp.call_tool(tool_name, tool_args)
                results[tool_name] = {
                    "success": True,
                    "result": result
                }
                
                log_success(f"✅ Tool {tool_name} executed successfully", symbol="🔧")
                
            except Exception as e:
                results[tool_name] = {
                    "success": False,
                    "error": str(e)
                }
                
                log_error(f"❌ Tool {tool_name} failed: {str(e)}", symbol="🔧")
        
        return results

    def get_execution_stats(self) -> Dict[str, Any]:
        """
        Get current execution statistics.
        
        Returns:
            Dictionary containing execution statistics
        """
        stats = self.execution_stats.copy()
        
        # Calculate success rate
        total_runs = stats["total_agent_runs"]
        if total_runs > 0:
            stats["success_rate"] = (stats["successful_runs"] / total_runs) * 100
            stats["average_execution_time"] = stats["total_execution_time"] / total_runs
        else:
            stats["success_rate"] = 0.0
            stats["average_execution_time"] = 0.0
        
        return stats

    def reset_stats(self):
        """Reset execution statistics."""
        self.execution_stats = {
            "total_agent_runs": 0,
            "successful_runs": 0,
            "failed_runs": 0,
            "total_execution_time": 0.0,
            "total_cost": 0.0
        }

    async def test_agent(self, agent_type: str, test_input: Dict[str, Any]) -> Dict[str, Any]:
        """
        Test an agent with sample input.
        
        Args:
            agent_type: Type of agent to test
            test_input: Test input data
            
        Returns:
            Test results
        """
        log_step(f"🧪 Testing {agent_type}", symbol="🔬")
        
        result = await self.run_agent(agent_type, test_input)
        
        if result["success"]:
            log_success(f"✅ {agent_type} test passed", symbol="🎉")
        else:
            log_error(f"❌ {agent_type} test failed: {result['error']}", symbol="💥")
        
        return result

    def get_agent_info(self, agent_type: str) -> Dict[str, Any]:
        """
        Get information about a specific agent.
        
        Args:
            agent_type: Type of agent
            
        Returns:
            Agent information
        """
        if agent_type not in self.agent_configs:
            raise ValueError(f"Unknown agent type: {agent_type}")
        
        config = self.agent_configs[agent_type]
        
        return {
            "agent_type": agent_type,
            "description": config.get("description", "No description"),
            "model": config.get("model", "default"),
            "mcp_servers": config.get("mcp_servers", []),
            "capabilities": config.get("capabilities", []),
            "max_retries": config.get("max_retries", 3),
            "timeout": config.get("timeout", 30),
            "prompt_file": config.get("prompt_file", "Unknown")
        }

    def list_available_agents(self) -> List[str]:
        """
        Get list of available agent types.
        
        Returns:
            List of agent type names
        """
        return list(self.agent_configs.keys()) 
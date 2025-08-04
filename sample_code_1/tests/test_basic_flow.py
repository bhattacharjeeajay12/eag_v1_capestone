"""
Basic Flow Tests - E-Commerce Multi-Agent System
This module contains basic tests to demonstrate the system functionality.
"""

import pytest
import asyncio
import json
from pathlib import Path
from unittest.mock import Mock, AsyncMock

# Import system components
from agentLoop.flow import AgentLoop4
from agentLoop.contextManager import ExecutionContextManager
from agentLoop.agents import AgentRunner

class TestEcommerceMultiAgentSystem:
    """Test class for the E-Commerce Multi-Agent System."""
    
    @pytest.fixture
    def mock_multi_mcp(self):
        """Create a mock MultiMCP instance."""
        mock = Mock()
        mock.initialize = AsyncMock()
        mock.shutdown = AsyncMock()
        mock.call_tool = AsyncMock(return_value={"success": True, "data": "mock_data"})
        mock.get_tools_from_servers = AsyncMock(return_value=[])
        return mock
    
    @pytest.fixture
    def sample_plan_graph(self):
        """Create a sample execution plan graph."""
        return {
            "nodes": [
                {
                    "id": "T001",
                    "agent": "CustomerIntentAgent",
                    "description": "Analyze customer intent",
                    "reads": [],
                    "writes": ["intent_analysis"]
                },
                {
                    "id": "T002",
                    "agent": "ProductSearchAgent",
                    "description": "Search for products",
                    "reads": ["T001"],
                    "writes": ["product_results"]
                },
                {
                    "id": "T003",
                    "agent": "ResponseFormatterAgent",
                    "description": "Format final response",
                    "reads": ["T002"],
                    "writes": ["final_response"]
                }
            ],
            "edges": [
                {"source": "ROOT", "target": "T001"},
                {"source": "T001", "target": "T002"},
                {"source": "T002", "target": "T003"}
            ]
        }
    
    @pytest.mark.asyncio
    async def test_agent_loop_initialization(self, mock_multi_mcp):
        """Test AgentLoop4 initialization."""
        agent_loop = AgentLoop4(mock_multi_mcp, strategy="conservative")
        
        assert agent_loop.multi_mcp == mock_multi_mcp
        assert agent_loop.strategy == "conservative"
        assert agent_loop.max_concurrent == 4
        assert agent_loop.agent_runner is not None
    
    @pytest.mark.asyncio
    async def test_context_manager_creation(self, sample_plan_graph):
        """Test ExecutionContextManager creation."""
        context = ExecutionContextManager(
            plan_graph=sample_plan_graph,
            session_id="test_session",
            original_query="Test query"
        )
        
        assert context.session_id == "test_session"
        assert context.original_query == "Test query"
        assert context.plan_graph is not None
        assert len(context.plan_graph.nodes) == 4  # ROOT + 3 nodes
    
    @pytest.mark.asyncio
    async def test_get_ready_steps(self, sample_plan_graph):
        """Test getting ready steps from execution context."""
        context = ExecutionContextManager(
            plan_graph=sample_plan_graph,
            session_id="test_session",
            original_query="Test query"
        )
        
        # Initially, only T001 should be ready (depends only on ROOT)
        ready_steps = context.get_ready_steps()
        assert "T001" in ready_steps
        assert len(ready_steps) == 1
        
        # Mark T001 as completed
        await context.mark_done("T001", {"intent": "product_search"})
        
        # Now T002 should be ready
        ready_steps = context.get_ready_steps()
        assert "T002" in ready_steps
        assert len(ready_steps) == 1
    
    @pytest.mark.asyncio
    async def test_get_inputs(self, sample_plan_graph):
        """Test getting inputs for agents."""
        context = ExecutionContextManager(
            plan_graph=sample_plan_graph,
            session_id="test_session",
            original_query="Test query"
        )
        
        # Mark T001 as completed with output
        await context.mark_done("T001", {"intent": "product_search", "entities": ["headphones"]})
        
        # Get inputs for T002 (which reads from T001)
        inputs = context.get_inputs(["T001"])
        assert "T001" in inputs
        assert inputs["T001"]["intent"] == "product_search"
    
    @pytest.mark.asyncio
    async def test_execution_summary(self, sample_plan_graph):
        """Test execution summary generation."""
        context = ExecutionContextManager(
            plan_graph=sample_plan_graph,
            session_id="test_session",
            original_query="Test query"
        )
        
        # Complete all nodes
        await context.mark_done("T001", {"intent": "product_search"})
        await context.mark_done("T002", {"products": ["headphone1", "headphone2"]})
        await context.mark_done("T003", {"response": "Here are your products"})
        
        summary = context.get_execution_summary()
        
        assert summary["session_id"] == "test_session"
        assert summary["original_query"] == "Test query"
        assert summary["status"] == "completed"
        assert summary["total_nodes"] == 3
        assert summary["completed_nodes"] == 3
        assert summary["failed_nodes"] == 0
        assert summary["completion_rate"] == 100.0
    
    @pytest.mark.asyncio
    async def test_agent_runner_initialization(self, mock_multi_mcp):
        """Test AgentRunner initialization."""
        # Mock the config file loading
        with open("config/agent_config.yaml", "w") as f:
            f.write("""
agents:
  CustomerIntentAgent:
    prompt_file: "prompts/customer_intent_prompt.txt"
    model: "gemini"
    mcp_servers: []
global_settings:
  default_model: "gemini"
  max_concurrent_agents: 4
""")
        
        agent_runner = AgentRunner(mock_multi_mcp)
        
        assert agent_runner.multi_mcp == mock_multi_mcp
        assert "CustomerIntentAgent" in agent_runner.agent_configs
        assert agent_runner.model_manager is not None
        
        # Clean up
        Path("config/agent_config.yaml").unlink(missing_ok=True)
    
    def test_utility_functions(self):
        """Test utility functions."""
        from utils.utils import (
            format_duration, format_file_size, sanitize_filename,
            truncate_text, is_valid_json
        )
        
        # Test duration formatting
        assert format_duration(30) == "30.00s"
        assert format_duration(90) == "1.5m"
        assert format_duration(7200) == "2.0h"
        
        # Test file size formatting
        assert format_file_size(1024) == "1.0KB"
        assert format_file_size(1048576) == "1.0MB"
        
        # Test filename sanitization
        assert sanitize_filename("test<file>.txt") == "test_file_.txt"
        assert sanitize_filename("  .  ") == "unnamed_file"
        
        # Test text truncation
        assert truncate_text("short") == "short"
        assert len(truncate_text("very long text that should be truncated", 20)) <= 20
        
        # Test JSON validation
        assert is_valid_json('{"key": "value"}') == True
        assert is_valid_json('invalid json') == False
    
    @pytest.mark.asyncio
    async def test_graph_validation(self, sample_plan_graph):
        """Test graph validation functionality."""
        from agentLoop.graph_validator import GraphValidator
        
        validator = GraphValidator()
        
        # Create a valid graph
        context = ExecutionContextManager(
            plan_graph=sample_plan_graph,
            session_id="test_session",
            original_query="Test query"
        )
        
        # Validate the graph
        validation_results = validator.validate_execution_graph(context.plan_graph)
        
        assert validation_results["is_valid"] == True
        assert validation_results["is_dag"] == True
        assert len(validation_results["errors"]) == 0
    
    def test_configuration_loading(self):
        """Test configuration file loading."""
        # Create test configuration files
        config_dir = Path("config")
        config_dir.mkdir(exist_ok=True)
        
        # Test agent config
        agent_config = {
            "agents": {
                "TestAgent": {
                    "prompt_file": "prompts/test_prompt.txt",
                    "model": "gemini",
                    "mcp_servers": []
                }
            }
        }
        
        with open("config/agent_config.yaml", "w") as f:
            import yaml
            yaml.dump(agent_config, f)
        
        # Test model config
        model_config = {
            "defaults": {"text_generation": "gemini"},
            "models": {
                "gemini": {
                    "type": "gemini",
                    "model": "gemini-2.0-flash"
                }
            }
        }
        
        with open("config/models.json", "w") as f:
            json.dump(model_config, f)
        
        # Test MCP config
        mcp_config = {
            "mcp_servers": [
                {
                    "id": "test_server",
                    "script": "test_server.py",
                    "transport": "stdio"
                }
            ]
        }
        
        with open("config/mcp_server_config.yaml", "w") as f:
            yaml.dump(mcp_config, f)
        
        # Verify files exist
        assert Path("config/agent_config.yaml").exists()
        assert Path("config/models.json").exists()
        assert Path("config/mcp_server_config.yaml").exists()
        
        # Clean up
        for file in config_dir.glob("*"):
            file.unlink()
        config_dir.rmdir()

if __name__ == "__main__":
    # Run basic tests
    pytest.main([__file__, "-v"]) 
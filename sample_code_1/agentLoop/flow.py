"""
AgentLoop4 - Main Execution Orchestrator for E-Commerce Multi-Agent System
This module implements the core NetworkX Graph-First execution engine that coordinates
multiple specialized agents to handle customer queries through a directed acyclic graph (DAG).
"""

import networkx as nx
import asyncio
import json
from typing import Dict, List, Any, Optional
from pathlib import Path
from datetime import datetime

# Import core components
from .contextManager import ExecutionContextManager
from .agents import AgentRunner
from .graph_validator import GraphValidator
from .visualizer import ExecutionVisualizer
from utils.utils import log_step, log_error, log_success
from rich.console import Console
from rich.panel import Panel
from rich.text import Text

class AgentLoop4:
    """
    Main orchestrator for the NetworkX Graph-First Multi-Agent System.
    
    This class manages the complete lifecycle of agent execution:
    1. Planning: Creates execution DAG based on customer intent
    2. Validation: Ensures graph integrity and dependencies
    3. Execution: Runs agents in parallel with dependency management
    4. Monitoring: Tracks progress and handles errors
    5. Completion: Aggregates results and provides final response
    """
    
    def __init__(self, multi_mcp, strategy: str = "conservative", max_concurrent: int = 4):
        """
        Initialize the AgentLoop4 orchestrator.
        
        Args:
            multi_mcp: MultiMCP instance for external tool access
            strategy: Execution strategy ("conservative" or "exploratory")
            max_concurrent: Maximum number of agents to run concurrently
        """
        self.multi_mcp = multi_mcp
        self.strategy = strategy
        self.max_concurrent = max_concurrent
        self.agent_runner = AgentRunner(multi_mcp)
        self.console = Console()
        
        # Performance tracking
        self.execution_stats = {
            "total_queries": 0,
            "successful_executions": 0,
            "failed_executions": 0,
            "average_response_time": 0.0,
            "total_agents_executed": 0
        }
        
        log_step("🚀 AgentLoop4 initialized", symbol="✅")

    async def run(self, query: str, file_manifest: List[Dict] = None, 
                  uploaded_files: List[str] = None) -> ExecutionContextManager:
        """
        Main execution method that processes a customer query through the multi-agent system.
        
        This method implements the complete execution flow:
        1. File Processing (if files provided)
        2. Intent Analysis and Planning
        3. Graph Creation and Validation
        4. Parallel Agent Execution
        5. Result Aggregation and Response
        
        Args:
            query: Customer query string
            file_manifest: List of file metadata (optional)
            uploaded_files: List of file paths (optional)
            
        Returns:
            ExecutionContextManager: Complete execution context with results
            
        Raises:
            RuntimeError: If planning or execution fails
        """
        start_time = datetime.now()
        self.execution_stats["total_queries"] += 1
        
        try:
            log_step("🔄 Starting multi-agent execution", symbol="🚀")
            
            # Phase 1: File Processing (if files exist)
            file_profiles = await self._process_files(uploaded_files, file_manifest)
            
            # Phase 2: Planning - Create execution DAG
            plan_graph = await self._create_execution_plan(query, file_profiles, file_manifest)
            
            # Phase 3: Graph Validation
            await self._validate_execution_graph(plan_graph)
            
            # Phase 4: Context Setup
            context = self._setup_execution_context(plan_graph, query, file_manifest, file_profiles)
            
            # Phase 5: Execute DAG with parallel processing
            await self._execute_dag(context)
            
            # Phase 6: Finalize and return results
            execution_time = (datetime.now() - start_time).total_seconds()
            self._update_execution_stats(context, execution_time)
            
            log_success(f"✅ Execution completed in {execution_time:.2f}s", symbol="🎉")
            return context
            
        except Exception as e:
            self.execution_stats["failed_executions"] += 1
            log_error(f"❌ Execution failed: {str(e)}", symbol="💥")
            raise RuntimeError(f"AgentLoop4 execution failed: {str(e)}")

    async def _process_files(self, uploaded_files: List[str], 
                           file_manifest: List[Dict]) -> Dict[str, Any]:
        """
        Process uploaded files to extract metadata and content profiles.
        
        This method analyzes uploaded files to understand their structure,
        content type, and relevant information that agents might need.
        
        Args:
            uploaded_files: List of file paths
            file_manifest: List of file metadata
            
        Returns:
            Dict containing file profiles and analysis results
        """
        if not uploaded_files:
            return {}
            
        log_step("📁 Processing uploaded files", symbol="📂")
        
        # Create file list for agent consumption
        file_list_text = "\n".join([
            f"- File {i+1}: {Path(f).name} (full path: {f})" 
            for i, f in enumerate(uploaded_files)
        ])
        
        # Build instruction for file analysis
        grounded_instruction = f"""
        Profile and summarize each file's structure, columns, content type, and key information.
        
        IMPORTANT: Use these EXACT file names in your response:
        {file_list_text}
        
        For each file, provide:
        1. File type and format
        2. Content summary
        3. Key data fields/columns
        4. Relevance to customer query
        5. Any special considerations for processing
        """
        
        # Execute file profiling with DistillerAgent
        file_result = await self.agent_runner.run_agent(
            "DistillerAgent",
            {
                "task": "profile_files",
                "files": uploaded_files,
                "instruction": grounded_instruction,
                "writes": ["file_profiles"]
            }
        )
        
        if file_result["success"]:
            log_success(f"✅ Processed {len(uploaded_files)} files", symbol="📄")
            return file_result["output"]
        else:
            log_error(f"❌ File processing failed: {file_result['error']}", symbol="📁")
            return {}

    async def _create_execution_plan(self, query: str, file_profiles: Dict, 
                                   file_manifest: List[Dict]) -> Dict[str, Any]:
        """
        Create execution plan using PlannerAgent to generate NetworkX DAG.
        
        This method uses the PlannerAgent to analyze the customer query and
        create a structured execution plan represented as a directed acyclic graph.
        
        Args:
            query: Customer query string
            file_profiles: Processed file information
            file_manifest: File metadata
            
        Returns:
            Dict containing the execution plan graph
            
        Raises:
            RuntimeError: If planning fails
        """
        log_step("🧠 Creating execution plan", symbol="📋")
        
        # Execute planning with PlannerAgent
        plan_result = await self.agent_runner.run_agent(
            "PlannerAgent",
            {
                "original_query": query,
                "planning_strategy": self.strategy,
                "file_manifest": file_manifest or [],
                "file_profiles": file_profiles,
                "available_agents": self._get_available_agents(),
                "execution_constraints": self._get_execution_constraints()
            }
        )
        
        if not plan_result["success"]:
            raise RuntimeError(f"Planning failed: {plan_result['error']}")
        
        if 'plan_graph' not in plan_result['output']:
            raise RuntimeError("PlannerAgent output missing 'plan_graph' key")
        
        plan_graph = plan_result["output"]["plan_graph"]
        
        # Validate plan structure
        if not isinstance(plan_graph, dict) or 'nodes' not in plan_graph:
            raise RuntimeError("Invalid plan graph structure")
        
        log_success(f"✅ Created plan with {len(plan_graph['nodes'])} nodes", symbol="📊")
        return plan_graph

    async def _validate_execution_graph(self, plan_graph: Dict[str, Any]):
        """
        Validate the execution graph for integrity and correctness.
        
        This method ensures the graph is a valid DAG with proper dependencies,
        no cycles, and all required nodes are present.
        
        Args:
            plan_graph: Execution plan graph
            
        Raises:
            ValueError: If graph validation fails
        """
        log_step("🔍 Validating execution graph", symbol="✅")
        
        # Create temporary NetworkX graph for validation
        temp_graph = nx.DiGraph()
        
        # Add nodes
        for node in plan_graph.get("nodes", []):
            temp_graph.add_node(node["id"], **node)
        
        # Add edges
        for edge in plan_graph.get("edges", []):
            temp_graph.add_edge(edge["source"], edge["target"])
        
        # Validate using GraphValidator
        validator = GraphValidator()
        validation_results = validator.validate_execution_graph(temp_graph, verbose=True)
        
        if not validation_results["is_valid"]:
            errors = validation_results.get("errors", [])
            warnings = validation_results.get("warnings", [])
            
            error_msg = f"Graph validation failed:\nErrors: {errors}\nWarnings: {warnings}"
            raise ValueError(error_msg)
        
        log_success("✅ Graph validation passed", symbol="✅")

    def _setup_execution_context(self, plan_graph: Dict[str, Any], query: str,
                                file_manifest: List[Dict], file_profiles: Dict) -> ExecutionContextManager:
        """
        Set up the execution context with the validated plan graph.
        
        This method creates the ExecutionContextManager that will track
        the state of all agents during execution.
        
        Args:
            plan_graph: Validated execution plan
            query: Original customer query
            file_manifest: File metadata
            file_profiles: Processed file information
            
        Returns:
            ExecutionContextManager: Configured execution context
        """
        log_step("🔧 Setting up execution context", symbol="⚙️")
        
        # Create execution context
        context = ExecutionContextManager(
            plan_graph=plan_graph,
            session_id=None,  # Auto-generated
            original_query=query,
            file_manifest=file_manifest or []
        )
        
        # Set up MCP access
        context.set_multi_mcp(self.multi_mcp)
        
        # Store initial data in output chain
        if file_profiles:
            context.plan_graph.graph['output_chain']['file_profiles'] = file_profiles
        
        # Store uploaded files directly
        if file_manifest:
            for file_info in file_manifest:
                context.plan_graph.graph['output_chain'][file_info['name']] = file_info['path']
        
        log_success("✅ Execution context ready", symbol="🔧")
        return context

    async def _execute_dag(self, context: ExecutionContextManager):
        """
        Execute the DAG with parallel processing and dependency management.
        
        This is the core execution engine that:
        1. Identifies ready-to-execute agents
        2. Runs them in parallel (up to max_concurrent)
        3. Manages dependencies and data flow
        4. Handles errors and retries
        5. Updates execution state
        
        Args:
            context: ExecutionContextManager with the plan graph
        """
        log_step("🚀 Starting DAG execution", symbol="⚡")
        
        visualizer = ExecutionVisualizer(context)
        max_iterations = 50  # Prevent infinite loops
        iteration = 0
        
        while not context.all_done() and iteration < max_iterations:
            iteration += 1
            
            # Display current execution state
            self.console.print(visualizer.get_layout())
            
            # Get agents ready to execute
            ready_steps = context.get_ready_steps()
            
            if not ready_steps:
                # Check if we're stuck due to failures
                failed_nodes = [
                    n for n in context.plan_graph.nodes 
                    if context.plan_graph.nodes[n]['status'] == 'failed'
                ]
                
                if failed_nodes:
                    log_error(f"❌ Execution blocked by failed nodes: {failed_nodes}", symbol="💥")
                    break
                
                # Wait a bit before checking again
                await asyncio.sleep(0.5)
                continue
            
            # Rate limiting - execute in batches
            batch_size = min(len(ready_steps), self.max_concurrent)
            current_batch = ready_steps[:batch_size]
            
            log_step(f"🚀 Executing batch: {current_batch}", symbol="⚡")
            
            # Mark agents as running
            for step_id in current_batch:
                context.mark_running(step_id)
            
            # Execute batch concurrently
            tasks = [self._execute_step(step_id, context) for step_id in current_batch]
            results = await asyncio.gather(*tasks, return_exceptions=True)
            
            # Process results
            for step_id, result in zip(current_batch, results):
                if isinstance(result, Exception):
                    log_error(f"❌ {step_id} failed: {str(result)}", symbol="💥")
                    context.mark_failed(step_id, str(result))
                elif result["success"]:
                    log_success(f"✅ {step_id} completed", symbol="🎉")
                    await context.mark_done(step_id, result["output"])
                else:
                    log_error(f"❌ {step_id} failed: {result['error']}", symbol="💥")
                    context.mark_failed(step_id, result["error"])
            
            # Rate limiting between batches
            if len(ready_steps) > batch_size:
                await asyncio.sleep(2)
        
        if iteration >= max_iterations:
            log_error("❌ Execution exceeded maximum iterations", symbol="⏰")

    async def _execute_step(self, step_id: str, context: ExecutionContextManager) -> Dict[str, Any]:
        """
        Execute a single agent step with input preparation and result processing.
        
        This method handles the execution of individual agents, including:
        1. Input preparation from previous agent outputs
        2. Agent execution with proper error handling
        3. Code execution if agent returns executable code
        4. Result processing and validation
        
        Args:
            step_id: ID of the agent to execute
            context: Execution context with graph state
            
        Returns:
            Dict containing execution result
        """
        step_data = context.get_step_data(step_id)
        agent_type = step_data["agent"]
        
        # Get inputs from previous steps
        inputs = context.get_inputs(step_data.get("reads", []))
        
        # Build agent input
        agent_input = {
            "step_id": step_id,
            "agent_prompt": step_data.get("agent_prompt", step_data["description"]),
            "reads": step_data.get("reads", []),
            "writes": step_data.get("writes", []),
            "inputs": inputs,
            "original_query": context.plan_graph.graph['original_query'],
            "session_context": {
                "session_id": context.plan_graph.graph['session_id'],
                "file_manifest": context.plan_graph.graph['file_manifest']
            }
        }
        
        # Execute agent
        result = await self.agent_runner.run_agent(agent_type, agent_input)
        
        # Handle code execution if agent returned code
        if result["success"] and self._has_executable_code(result["output"]):
            result = await self._handle_code_execution(step_id, result, inputs, context)
        
        return result

    def _has_executable_code(self, output: Dict[str, Any]) -> bool:
        """
        Check if agent output contains executable code.
        
        Args:
            output: Agent output dictionary
            
        Returns:
            bool: True if output contains executable code
        """
        if not isinstance(output, dict):
            return False
        
        code_indicators = [
            "files", "code_variants", "python_code", "javascript_code",
            "html_code", "css_code", "tool_calls", "browser_commands"
        ]
        
        return any(
            key in output or any(k.startswith("CODE_") for k in output.keys())
            for key in code_indicators
        )

    async def _handle_code_execution(self, step_id: str, result: Dict[str, Any], 
                                   inputs: Dict[str, Any], context: ExecutionContextManager) -> Dict[str, Any]:
        """
        Handle execution of code returned by agents.
        
        Args:
            step_id: ID of the agent that returned code
            result: Original agent result
            inputs: Input data for code execution
            context: Execution context
            
        Returns:
            Updated result with code execution outcomes
        """
        log_step(f"🔧 {step_id}: Executing code", symbol="⚙️")
        
        try:
            # Import code executor
            from action.executor import run_user_code
            
            # Prepare executor input
            executor_input = {
                "code_variants": result["output"].get("code", {}),
                "files": result["output"].get("files", {}),
                "tool_calls": result["output"].get("tool_calls", {})
            }
            
            # Execute code
            execution_result = await run_user_code(
                executor_input,
                self.multi_mcp,
                context.plan_graph.graph['session_id'] or "default_session",
                inputs
            )
            
            # Handle execution results
            if execution_result["status"] == "success":
                log_success(f"✅ {step_id}: Code execution succeeded", symbol="🎉")
                
                # Combine agent output with code execution results
                code_output = execution_result.get("code_results", {}).get("result", {})
                combined_output = {
                    **result["output"],
                    **code_output
                }
                
                result["output"] = combined_output
                
            elif execution_result["status"] == "partial_failure":
                log_error(f"⚠️ {step_id}: Code execution partial failure", symbol="⚠️")
                
                # Try to extract any successful results
                code_output = execution_result.get("code_results", {}).get("result", {})
                if code_output:
                    result["output"].update(code_output)
                    
        except Exception as e:
            log_error(f"❌ {step_id}: Code execution failed: {str(e)}", symbol="💥")
            # Continue with original result
        
        return result

    def _get_available_agents(self) -> List[str]:
        """
        Get list of available agent types.
        
        Returns:
            List of agent type names
        """
        return [
            "CustomerIntentAgent",
            "ProductSearchAgent", 
            "OrderStatusAgent",
            "RecommendationAgent",
            "ComplaintHandlerAgent",
            "SentimentAnalysisAgent",
            "ResponseFormatterAgent"
        ]

    def _get_execution_constraints(self) -> Dict[str, Any]:
        """
        Get execution constraints and rules.
        
        Returns:
            Dict containing execution constraints
        """
        return {
            "max_concurrent_agents": self.max_concurrent,
            "required_first_agent": "CustomerIntentAgent",
            "required_last_agent": "ResponseFormatterAgent",
            "parallel_agents": ["ProductSearchAgent", "RecommendationAgent"],
            "max_execution_time": 300,  # 5 minutes
            "retry_policy": {
                "max_retries": 3,
                "backoff_factor": 2.0
            }
        }

    def _update_execution_stats(self, context: ExecutionContextManager, execution_time: float):
        """
        Update execution statistics.
        
        Args:
            context: Execution context
            execution_time: Total execution time in seconds
        """
        completed_nodes = [
            n for n in context.plan_graph.nodes 
            if context.plan_graph.nodes[n]['status'] == 'completed'
        ]
        
        self.execution_stats["total_agents_executed"] += len(completed_nodes)
        
        if context.all_done():
            self.execution_stats["successful_executions"] += 1
        else:
            self.execution_stats["failed_executions"] += 1
        
        # Update average response time
        total_executions = self.execution_stats["successful_executions"] + self.execution_stats["failed_executions"]
        if total_executions > 0:
            current_avg = self.execution_stats["average_response_time"]
            self.execution_stats["average_response_time"] = (
                (current_avg * (total_executions - 1) + execution_time) / total_executions
            )

    def get_execution_stats(self) -> Dict[str, Any]:
        """
        Get current execution statistics.
        
        Returns:
            Dict containing execution statistics
        """
        return self.execution_stats.copy()

    def reset_stats(self):
        """Reset execution statistics."""
        self.execution_stats = {
            "total_queries": 0,
            "successful_executions": 0,
            "failed_executions": 0,
            "average_response_time": 0.0,
            "total_agents_executed": 0
        } 
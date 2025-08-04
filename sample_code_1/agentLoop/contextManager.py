"""
ExecutionContextManager - NetworkX Graph State Management
This module manages the execution state of the multi-agent system using NetworkX graphs.
It handles data flow between agents, execution status tracking, and result aggregation.
"""

import networkx as nx
import json
import time
import asyncio
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any, Optional, Set
import uuid

# Import utilities
from utils.utils import log_step, log_error, log_success
from .graph_validator import GraphValidator
from .session_serializer import SessionSerializer

class ExecutionContextManager:
    """
    Manages the execution context for the NetworkX Graph-First Multi-Agent System.
    
    This class is responsible for:
    1. Maintaining the NetworkX execution graph state
    2. Managing data flow between agents through output chaining
    3. Tracking execution status of all agents
    4. Handling session persistence and recovery
    5. Providing execution statistics and monitoring
    """
    
    def __init__(self, plan_graph: Dict[str, Any], session_id: str = None, 
                 original_query: str = None, file_manifest: List[Dict] = None, 
                 debug_mode: bool = False):
        """
        Initialize the execution context manager.
        
        Args:
            plan_graph: Execution plan as dictionary with nodes and edges
            session_id: Unique session identifier (auto-generated if None)
            original_query: Original customer query
            file_manifest: List of uploaded file metadata
            debug_mode: Enable debug logging and extended error information
        """
        # Generate session ID if not provided
        self.session_id = session_id or f"session_{int(time.time())}_{str(uuid.uuid4())[:8]}"
        
        # Store metadata
        self.original_query = original_query
        self.file_manifest = file_manifest or []
        self.debug_mode = debug_mode
        
        # Initialize NetworkX graph
        self.plan_graph = nx.DiGraph()
        
        # Set graph metadata
        self.plan_graph.graph.update({
            'session_id': self.session_id,
            'original_query': original_query,
            'file_manifest': self.file_manifest,
            'created_at': datetime.utcnow().isoformat(),
            'status': 'running',
            'output_chain': {},  # Simple output chaining for data flow
            'execution_stats': {
                'start_time': datetime.utcnow(),
                'completed_nodes': 0,
                'failed_nodes': 0,
                'total_nodes': 0
            }
        })
        
        # Build the execution DAG
        self._build_execution_graph(plan_graph)
        
        # Validate the graph
        self._validate_graph()
        
        # Initialize session serializer
        self.session_serializer = SessionSerializer()
        
        log_step(f"🔧 ExecutionContextManager initialized for session {self.session_id}", symbol="✅")

    def _build_execution_graph(self, plan_graph: Dict[str, Any]):
        """
        Build the NetworkX execution graph from the plan.
        
        Args:
            plan_graph: Execution plan dictionary
        """
        # Add ROOT node
        self.plan_graph.add_node("ROOT", 
            description="Initial Query", 
            agent="System", 
            status='completed',
            start_time=datetime.utcnow().isoformat(),
            end_time=datetime.utcnow().isoformat(),
            execution_time=0.0,
            output=None,
            error=None,
            cost=0.0
        )
        
        # Add plan nodes
        for node in plan_graph.get("nodes", []):
            self.plan_graph.add_node(node["id"], 
                status='pending',
                output=None,
                error=None,
                cost=0.0,
                start_time=None,
                end_time=None,
                execution_time=0.0,
                **node
            )
        
        # Add plan edges
        for edge in plan_graph.get("edges", []):
            self.plan_graph.add_edge(edge["source"], edge["target"])
        
        # Update total nodes count
        self.plan_graph.graph['execution_stats']['total_nodes'] = len(self.plan_graph.nodes) - 1  # Exclude ROOT

    def _validate_graph(self):
        """
        Validate the execution graph for integrity.
        
        Raises:
            ValueError: If graph validation fails
        """
        validator = GraphValidator()
        validation_results = validator.validate_execution_graph(self.plan_graph, verbose=not self.debug_mode)
        
        if not validation_results["is_valid"]:
            errors = validation_results.get("errors", [])
            warnings = validation_results.get("warnings", [])
            
            error_msg = f"Invalid execution graph:\nErrors: {errors}\nWarnings: {warnings}"
            raise ValueError(error_msg)
        
        # Store validation results
        self.plan_graph.graph['validation_results'] = validation_results
        
        log_success(f"✅ Graph validation passed: {len(self.plan_graph.nodes)} nodes, {len(self.plan_graph.edges)} edges", symbol="✅")

    def get_ready_steps(self) -> List[str]:
        """
        Get list of agents ready to execute based on dependency satisfaction.
        
        Returns:
            List of agent IDs that are ready to run
        """
        try:
            # Get topological order
            topo_order = list(nx.topological_sort(self.plan_graph))
            
            ready_steps = []
            for node in topo_order:
                if node == "ROOT":
                    continue
                
                node_data = self.plan_graph.nodes[node]
                
                # Check if node is pending
                if node_data['status'] != 'pending':
                    continue
                
                # Check if all predecessors are completed
                predecessors = list(self.plan_graph.predecessors(node))
                all_predecessors_completed = all(
                    self.plan_graph.nodes[pred]['status'] == 'completed'
                    for pred in predecessors
                )
                
                if all_predecessors_completed:
                    ready_steps.append(node)
            
            return ready_steps
            
        except nx.NetworkXError as e:
            log_error(f"❌ Error getting ready steps: {str(e)}", symbol="💥")
            return []

    def get_inputs(self, reads: List[str]) -> Dict[str, Any]:
        """
        Get inputs for an agent based on its read dependencies.
        
        This method implements simple output chaining - it directly passes
        the outputs from previous agents as inputs to the current agent.
        
        Args:
            reads: List of agent IDs whose outputs should be read
            
        Returns:
            Dictionary mapping agent IDs to their outputs
        """
        inputs = {}
        output_chain = self.plan_graph.graph['output_chain']
        
        for step_id in reads:
            if step_id in output_chain:
                inputs[step_id] = output_chain[step_id]
            else:
                # Check if the step has completed and has output
                if step_id in self.plan_graph.nodes:
                    node_data = self.plan_graph.nodes[step_id]
                    if node_data['status'] == 'completed' and node_data['output'] is not None:
                        inputs[step_id] = node_data['output']
                    else:
                        log_step(f"⚠️ Missing dependency: '{step_id}' not found or not completed", symbol="❓")
                else:
                    log_step(f"⚠️ Missing dependency: '{step_id}' not found", symbol="❓")
        
        return inputs

    def mark_running(self, step_id: str):
        """
        Mark an agent as running.
        
        Args:
            step_id: ID of the agent to mark as running
        """
        if step_id in self.plan_graph.nodes:
            self.plan_graph.nodes[step_id]['status'] = 'running'
            self.plan_graph.nodes[step_id]['start_time'] = datetime.utcnow().isoformat()
            self._auto_save()
            
            log_step(f"🔄 {step_id} marked as running", symbol="⚡")

    async def mark_done(self, step_id: str, output: Any = None, cost: float = None, 
                       input_tokens: int = None, output_tokens: int = None):
        """
        Mark an agent as completed with its output.
        
        Args:
            step_id: ID of the completed agent
            output: Agent output data
            cost: Execution cost (if applicable)
            input_tokens: Number of input tokens used
            output_tokens: Number of output tokens generated
        """
        if step_id not in self.plan_graph.nodes:
            log_error(f"❌ Cannot mark done: {step_id} not found in graph", symbol="💥")
            return
        
        node_data = self.plan_graph.nodes[step_id]
        end_time = datetime.utcnow()
        
        # Calculate execution time
        if node_data['start_time']:
            start_time = datetime.fromisoformat(node_data['start_time'])
            execution_time = (end_time - start_time).total_seconds()
        else:
            execution_time = 0.0
        
        # Update node data
        node_data.update({
            'status': 'completed',
            'output': output,
            'end_time': end_time.isoformat(),
            'execution_time': execution_time,
            'cost': cost or 0.0,
            'input_tokens': input_tokens,
            'output_tokens': output_tokens
        })
        
        # Store output in output chain for data flow
        if output is not None:
            self.plan_graph.graph['output_chain'][step_id] = output
        
        # Update execution stats
        self.plan_graph.graph['execution_stats']['completed_nodes'] += 1
        
        # Auto-save session
        await self._auto_save()
        
        log_success(f"✅ {step_id} completed in {execution_time:.2f}s", symbol="🎉")

    def mark_failed(self, step_id: str, error: str = None):
        """
        Mark an agent as failed.
        
        Args:
            step_id: ID of the failed agent
            error: Error message or exception
        """
        if step_id not in self.plan_graph.nodes:
            log_error(f"❌ Cannot mark failed: {step_id} not found in graph", symbol="💥")
            return
        
        node_data = self.plan_graph.nodes[step_id]
        end_time = datetime.utcnow()
        
        # Calculate execution time
        if node_data['start_time']:
            start_time = datetime.fromisoformat(node_data['start_time'])
            execution_time = (end_time - start_time).total_seconds()
        else:
            execution_time = 0.0
        
        # Update node data
        node_data.update({
            'status': 'failed',
            'error': str(error) if error else "Unknown error",
            'end_time': end_time.isoformat(),
            'execution_time': execution_time
        })
        
        # Update execution stats
        self.plan_graph.graph['execution_stats']['failed_nodes'] += 1
        
        # Auto-save session
        self._auto_save()
        
        log_error(f"❌ {step_id} failed after {execution_time:.2f}s: {error}", symbol="💥")

    def get_step_data(self, step_id: str) -> Dict[str, Any]:
        """
        Get data for a specific step/agent.
        
        Args:
            step_id: ID of the step
            
        Returns:
            Dictionary containing step data
        """
        if step_id in self.plan_graph.nodes:
            return dict(self.plan_graph.nodes[step_id])
        else:
            raise ValueError(f"Step {step_id} not found in execution graph")

    def all_done(self) -> bool:
        """
        Check if all agents have completed execution.
        
        Returns:
            True if all agents are completed or failed
        """
        for node in self.plan_graph.nodes:
            if node == "ROOT":
                continue
            
            status = self.plan_graph.nodes[node]['status']
            if status not in ['completed', 'failed']:
                return False
        
        return True

    def get_execution_summary(self) -> Dict[str, Any]:
        """
        Get comprehensive execution summary.
        
        Returns:
            Dictionary containing execution statistics and status
        """
        stats = self.plan_graph.graph['execution_stats'].copy()
        
        # Calculate completion statistics
        total_nodes = len(self.plan_graph.nodes) - 1  # Exclude ROOT
        completed_nodes = sum(
            1 for n in self.plan_graph.nodes 
            if n != "ROOT" and self.plan_graph.nodes[n]['status'] == 'completed'
        )
        failed_nodes = sum(
            1 for n in self.plan_graph.nodes 
            if n != "ROOT" and self.plan_graph.nodes[n]['status'] == 'failed'
        )
        running_nodes = sum(
            1 for n in self.plan_graph.nodes 
            if n != "ROOT" and self.plan_graph.nodes[n]['status'] == 'running'
        )
        pending_nodes = sum(
            1 for n in self.plan_graph.nodes 
            if n != "ROOT" and self.plan_graph.nodes[n]['status'] == 'pending'
        )
        
        # Calculate total execution time
        if stats['start_time']:
            total_time = (datetime.utcnow() - stats['start_time']).total_seconds()
        else:
            total_time = 0.0
        
        # Calculate total cost
        total_cost = sum(
            self.plan_graph.nodes[n].get('cost', 0.0)
            for n in self.plan_graph.nodes
            if n != "ROOT"
        )
        
        # Calculate total tokens
        total_input_tokens = sum(
            self.plan_graph.nodes[n].get('input_tokens', 0)
            for n in self.plan_graph.nodes
            if n != "ROOT"
        )
        total_output_tokens = sum(
            self.plan_graph.nodes[n].get('output_tokens', 0)
            for n in self.plan_graph.nodes
            if n != "ROOT"
        )
        
        summary = {
            'session_id': self.session_id,
            'original_query': self.original_query,
            'status': 'completed' if self.all_done() else 'running',
            'total_nodes': total_nodes,
            'completed_nodes': completed_nodes,
            'failed_nodes': failed_nodes,
            'running_nodes': running_nodes,
            'pending_nodes': pending_nodes,
            'completion_rate': (completed_nodes / total_nodes * 100) if total_nodes > 0 else 0,
            'total_execution_time': total_time,
            'total_cost': total_cost,
            'total_input_tokens': total_input_tokens,
            'total_output_tokens': total_output_tokens,
            'created_at': self.plan_graph.graph['created_at'],
            'validation_results': self.plan_graph.graph.get('validation_results', {})
        }
        
        return summary

    def set_multi_mcp(self, multi_mcp):
        """
        Set the MultiMCP instance for external tool access.
        
        Args:
            multi_mcp: MultiMCP instance
        """
        self.multi_mcp = multi_mcp
        self.plan_graph.graph['multi_mcp'] = multi_mcp

    def _auto_save(self):
        """
        Automatically save session state for persistence.
        """
        try:
            if hasattr(self, 'session_serializer'):
                self.session_serializer.save_session(self.session_id, self.get_session_data())
        except Exception as e:
            if self.debug_mode:
                log_error(f"❌ Auto-save failed: {str(e)}", symbol="💾")

    def get_session_data(self) -> Dict[str, Any]:
        """
        Get complete session data for serialization.
        
        Returns:
            Dictionary containing all session data
        """
        # Convert NetworkX graph to serializable format
        graph_data = {
            'nodes': dict(self.plan_graph.nodes(data=True)),
            'edges': list(self.plan_graph.edges(data=True)),
            'graph': dict(self.plan_graph.graph)
        }
        
        session_data = {
            'session_id': self.session_id,
            'original_query': self.original_query,
            'file_manifest': self.file_manifest,
            'debug_mode': self.debug_mode,
            'graph_data': graph_data,
            'saved_at': datetime.utcnow().isoformat()
        }
        
        return session_data

    @classmethod
    def load_session(cls, session_file: Path, debug_mode: bool = False) -> 'ExecutionContextManager':
        """
        Load session from file.
        
        Args:
            session_file: Path to session file
            debug_mode: Enable debug mode
            
        Returns:
            ExecutionContextManager instance with loaded session
        """
        try:
            session_serializer = SessionSerializer()
            session_data = session_serializer.load_session(session_file)
            
            # Reconstruct graph
            graph_data = session_data['graph_data']
            plan_graph = {
                'nodes': [
                    {'id': node_id, **node_data}
                    for node_id, node_data in graph_data['nodes'].items()
                ],
                'edges': [
                    {'source': edge[0], 'target': edge[1], **edge[2]}
                    for edge in graph_data['edges']
                ]
            }
            
            # Create new context manager
            context = cls(
                plan_graph=plan_graph,
                session_id=session_data['session_id'],
                original_query=session_data['original_query'],
                file_manifest=session_data['file_manifest'],
                debug_mode=debug_mode
            )
            
            # Restore graph state
            context.plan_graph.graph.update(graph_data['graph'])
            
            log_success(f"✅ Session loaded from {session_file}", symbol="📂")
            return context
            
        except Exception as e:
            log_error(f"❌ Failed to load session from {session_file}: {str(e)}", symbol="💥")
            raise

    def get_final_output(self) -> Dict[str, Any]:
        """
        Get the final output from the execution.
        
        Returns:
            Dictionary containing final results and metadata
        """
        # Find the final agent (usually ResponseFormatterAgent)
        final_agents = [
            n for n in self.plan_graph.nodes
            if n != "ROOT" and self.plan_graph.out_degree(n) == 0
        ]
        
        final_outputs = {}
        for agent_id in final_agents:
            if self.plan_graph.nodes[agent_id]['status'] == 'completed':
                final_outputs[agent_id] = self.plan_graph.nodes[agent_id]['output']
        
        # Get execution summary
        summary = self.get_execution_summary()
        
        return {
            'session_id': self.session_id,
            'original_query': self.original_query,
            'final_outputs': final_outputs,
            'execution_summary': summary,
            'output_chain': self.plan_graph.graph['output_chain'],
            'completed_at': datetime.utcnow().isoformat()
        }

    def get_graph_visualization_data(self) -> Dict[str, Any]:
        """
        Get data for graph visualization.
        
        Returns:
            Dictionary containing graph data for visualization
        """
        nodes = []
        edges = []
        
        for node_id, node_data in self.plan_graph.nodes(data=True):
            nodes.append({
                'id': node_id,
                'label': node_data.get('description', node_id),
                'agent': node_data.get('agent', 'Unknown'),
                'status': node_data.get('status', 'unknown'),
                'execution_time': node_data.get('execution_time', 0.0),
                'cost': node_data.get('cost', 0.0)
            })
        
        for source, target, edge_data in self.plan_graph.edges(data=True):
            edges.append({
                'source': source,
                'target': target,
                'data': edge_data
            })
        
        return {
            'nodes': nodes,
            'edges': edges,
            'metadata': {
                'session_id': self.session_id,
                'total_nodes': len(nodes),
                'total_edges': len(edges)
            }
        } 
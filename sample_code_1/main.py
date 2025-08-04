"""
Main Entry Point - E-Commerce Multi-Agent Customer Service System
This module provides the main interface for the NetworkX Graph-First Multi-Agent System.
"""

import asyncio
import yaml
import json
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Any
import sys

# Import core components
from agentLoop.flow import AgentLoop4
from mcp_servers.multiMCP import MultiMCP
from utils.utils import log_step, log_error, log_success
from rich.console import Console
from rich.panel import Panel
from rich.text import Text
from rich.table import Table
from rich.prompt import Prompt, Confirm
from rich.layout import Layout
from rich.live import Live
from rich.align import Align

# Banner for the application
BANNER = """
╔══════════════════════════════════════════════════════════════════════════════╗
║                    🛒 E-Commerce Multi-Agent System 🛒                      ║
║                                                                              ║
║  NetworkX Graph-First Multi-Agent Customer Service Platform                 ║
║  Powered by AI Agents with MCP Tool Integration                             ║
║                                                                              ║
║  Type 'help' for commands, 'exit' to quit                                   ║
╚══════════════════════════════════════════════════════════════════════════════╝
"""

class EcommerceMultiAgentSystem:
    """
    Main application class for the E-Commerce Multi-Agent System.
    
    This class provides:
    1. System initialization and configuration loading
    2. Interactive command-line interface
    3. Query processing and agent execution
    4. Result display and session management
    5. System monitoring and statistics
    """
    
    def __init__(self):
        """Initialize the E-Commerce Multi-Agent System."""
        self.console = Console()
        self.multi_mcp = None
        self.agent_loop = None
        self.current_session = None
        self.system_stats = {
            "total_queries": 0,
            "successful_queries": 0,
            "failed_queries": 0,
            "total_execution_time": 0.0,
            "sessions_created": 0
        }
        
        # Load configuration
        self.config = self._load_configuration()
        
    def _load_configuration(self) -> Dict[str, Any]:
        """
        Load system configuration from files.
        
        Returns:
            Dictionary containing system configuration
        """
        try:
            # Load MCP server configuration
            mcp_config_path = Path("config/mcp_server_config.yaml")
            if not mcp_config_path.exists():
                log_error(f"❌ MCP server config not found: {mcp_config_path}", symbol="💥")
                return {}
            
            with open(mcp_config_path, "r", encoding="utf-8") as f:
                mcp_config = yaml.safe_load(f)
            
            # Load agent configuration
            agent_config_path = Path("config/agent_config.yaml")
            if not agent_config_path.exists():
                log_error(f"❌ Agent config not found: {agent_config_path}", symbol="💥")
                return {}
            
            with open(agent_config_path, "r", encoding="utf-8") as f:
                agent_config = yaml.safe_load(f)
            
            # Load model configuration
            model_config_path = Path("config/models.json")
            if not model_config_path.exists():
                log_error(f"❌ Model config not found: {model_config_path}", symbol="💥")
                return {}
            
            with open(model_config_path, "r", encoding="utf-8") as f:
                model_config = json.load(f)
            
            config = {
                "mcp_servers": mcp_config,
                "agents": agent_config,
                "models": model_config
            }
            
            log_success("✅ Configuration loaded successfully", symbol="📋")
            return config
            
        except Exception as e:
            log_error(f"❌ Failed to load configuration: {str(e)}", symbol="💥")
            return {}

    async def initialize(self):
        """Initialize the system components."""
        try:
            log_step("🚀 Initializing E-Commerce Multi-Agent System", symbol="⚡")
            
            # Initialize MultiMCP
            if "mcp_servers" in self.config:
                server_configs = self.config["mcp_servers"].get("mcp_servers", [])
                self.multi_mcp = MultiMCP(server_configs)
                await self.multi_mcp.initialize()
                log_success(f"✅ MultiMCP initialized with {len(server_configs)} servers", symbol="🔧")
            else:
                log_error("❌ No MCP server configuration found", symbol="💥")
                return False
            
            # Initialize AgentLoop4
            self.agent_loop = AgentLoop4(
                multi_mcp=self.multi_mcp,
                strategy="conservative",
                max_concurrent=4
            )
            log_success("✅ AgentLoop4 initialized", symbol="🤖")
            
            # Display system status
            await self._display_system_status()
            
            return True
            
        except Exception as e:
            log_error(f"❌ System initialization failed: {str(e)}", symbol="💥")
            return False

    async def _display_system_status(self):
        """Display current system status and available capabilities."""
        status_table = Table(title="System Status")
        status_table.add_column("Component", style="cyan")
        status_table.add_column("Status", style="green")
        status_table.add_column("Details", style="yellow")
        
        # MCP Servers
        if self.multi_mcp:
            server_count = len(self.multi_mcp.server_configs)
            status_table.add_row("MCP Servers", "✅ Active", f"{server_count} servers configured")
        else:
            status_table.add_row("MCP Servers", "❌ Inactive", "Not initialized")
        
        # Agent Loop
        if self.agent_loop:
            status_table.add_row("Agent Loop", "✅ Active", "Ready for queries")
        else:
            status_table.add_row("Agent Loop", "❌ Inactive", "Not initialized")
        
        # Available Agents
        if "agents" in self.config:
            agent_count = len(self.config["agents"].get("agents", {}))
            status_table.add_row("Available Agents", "✅ Loaded", f"{agent_count} agents configured")
        else:
            status_table.add_row("Available Agents", "❌ Not Loaded", "No configuration found")
        
        # Models
        if "models" in self.config:
            model_count = len(self.config["models"].get("models", {}))
            status_table.add_row("LLM Models", "✅ Available", f"{model_count} models configured")
        else:
            status_table.add_row("LLM Models", "❌ Not Available", "No configuration found")
        
        self.console.print(status_table)

    async def run(self):
        """Main application loop."""
        self.console.print(Panel(BANNER, style="bold blue"))
        
        # Initialize system
        if not await self.initialize():
            self.console.print("[red]❌ System initialization failed. Exiting.[/red]")
            return
        
        # Main command loop
        while True:
            try:
                # Get user command
                command = Prompt.ask("\n[bold cyan]E-Commerce Agent[/bold cyan]", default="help")
                
                if command.lower() in ['exit', 'quit', 'q']:
                    break
                elif command.lower() == 'help':
                    await self._show_help()
                elif command.lower() == 'status':
                    await self._display_system_status()
                elif command.lower() == 'stats':
                    await self._show_statistics()
                elif command.lower() == 'agents':
                    await self._list_agents()
                elif command.lower() == 'tools':
                    await self._list_tools()
                elif command.lower() == 'query':
                    await self._process_query()
                elif command.lower() == 'demo':
                    await self._run_demo()
                else:
                    # Treat as a direct query
                    await self._process_direct_query(command)
                    
            except KeyboardInterrupt:
                self.console.print("\n[yellow]⚠️ Interrupted by user[/yellow]")
                break
            except Exception as e:
                log_error(f"❌ Unexpected error: {str(e)}", symbol="💥")

    async def _show_help(self):
        """Display help information."""
        help_text = """
[bold]Available Commands:[/bold]

[cyan]help[/cyan]     - Show this help message
[cyan]status[/cyan]   - Display system status
[cyan]stats[/cyan]    - Show execution statistics
[cyan]agents[/cyan]   - List available agents
[cyan]tools[/cyan]    - List available MCP tools
[cyan]query[/cyan]    - Process a customer query (interactive)
[cyan]demo[/cyan]     - Run demonstration queries
[cyan]exit[/cyan]     - Exit the application

[bold]Direct Queries:[/bold]
You can also type your query directly, for example:
"I'm looking for wireless headphones under $100"
"What's the status of my order #12345?"
"I have a complaint about my recent purchase"

[bold]Example Queries:[/bold]
• Product search: "Find me wireless headphones under $100"
• Order status: "Check status of order #ORD-2024-001"
• Recommendations: "Suggest products similar to iPhone 15"
• Complaints: "I'm unhappy with my recent purchase"
• General: "What are your return policies?"
        """
        
        self.console.print(Panel(help_text, title="Help", border_style="green"))

    async def _show_statistics(self):
        """Display system statistics."""
        stats_table = Table(title="System Statistics")
        stats_table.add_column("Metric", style="cyan")
        stats_table.add_column("Value", style="green")
        
        stats_table.add_row("Total Queries", str(self.system_stats["total_queries"]))
        stats_table.add_row("Successful Queries", str(self.system_stats["successful_queries"]))
        stats_table.add_row("Failed Queries", str(self.system_stats["failed_queries"]))
        stats_table.add_row("Success Rate", f"{(self.system_stats['successful_queries'] / max(self.system_stats['total_queries'], 1) * 100):.1f}%")
        stats_table.add_row("Total Execution Time", f"{self.system_stats['total_execution_time']:.2f}s")
        stats_table.add_row("Sessions Created", str(self.system_stats["sessions_created"]))
        
        # Agent loop stats
        if self.agent_loop:
            agent_stats = self.agent_loop.get_execution_stats()
            stats_table.add_row("Agents Executed", str(agent_stats.get("total_agents_executed", 0)))
            stats_table.add_row("Average Response Time", f"{agent_stats.get('average_response_time', 0):.2f}s")
        
        self.console.print(stats_table)

    async def _list_agents(self):
        """List available agents."""
        if not self.agent_loop:
            self.console.print("[red]❌ Agent loop not initialized[/red]")
            return
        
        agents_table = Table(title="Available Agents")
        agents_table.add_column("Agent", style="cyan")
        agents_table.add_column("Description", style="yellow")
        agents_table.add_column("Model", style="green")
        agents_table.add_column("Tools", style="blue")
        
        available_agents = self.agent_loop._get_available_agents()
        
        for agent_type in available_agents:
            try:
                agent_info = self.agent_loop.agent_runner.get_agent_info(agent_type)
                tools = ", ".join(agent_info.get("mcp_servers", []))
                agents_table.add_row(
                    agent_type,
                    agent_info.get("description", "No description"),
                    agent_info.get("model", "default"),
                    tools if tools else "None"
                )
            except Exception as e:
                agents_table.add_row(agent_type, "Error loading info", "Unknown", "Unknown")
        
        self.console.print(agents_table)

    async def _list_tools(self):
        """List available MCP tools."""
        if not self.multi_mcp:
            self.console.print("[red]❌ MultiMCP not initialized[/red]")
            return
        
        try:
            tools = await self.multi_mcp.list_all_tools()
            
            tools_table = Table(title="Available MCP Tools")
            tools_table.add_column("Tool", style="cyan")
            tools_table.add_column("Server", style="green")
            tools_table.add_column("Description", style="yellow")
            
            for tool in tools:
                tools_table.add_row(
                    tool.get("name", "Unknown"),
                    tool.get("server", "Unknown"),
                    tool.get("description", "No description")
                )
            
            self.console.print(tools_table)
            
        except Exception as e:
            log_error(f"❌ Failed to list tools: {str(e)}", symbol="🔧")

    async def _process_query(self):
        """Process a customer query interactively."""
        # Get query from user
        query = Prompt.ask("\n[bold yellow]Enter your customer query[/bold yellow]")
        if not query.strip():
            return
        
        # Get file uploads (optional)
        files = []
        if Confirm.ask("Do you want to upload files?"):
            while True:
                file_path = Prompt.ask("Enter file path (or press Enter to finish)")
                if not file_path:
                    break
                
                if Path(file_path).exists():
                    files.append(file_path)
                    self.console.print(f"[green]✅ Added: {Path(file_path).name}[/green]")
                else:
                    self.console.print(f"[red]❌ File not found: {file_path}[/red]")
        
        # Process the query
        await self._execute_query(query, files)

    async def _process_direct_query(self, query: str):
        """Process a direct query from command line."""
        await self._execute_query(query, [])

    async def _execute_query(self, query: str, files: List[str]):
        """Execute a customer query through the multi-agent system."""
        start_time = datetime.now()
        self.system_stats["total_queries"] += 1
        
        try:
            log_step(f"🔄 Processing query: {query[:50]}...", symbol="📝")
            
            # Prepare file manifest
            file_manifest = []
            for file_path in files:
                file_manifest.append({
                    "path": file_path,
                    "name": Path(file_path).name,
                    "size": Path(file_path).stat().st_size
                })
            
            # Execute through agent loop
            execution_context = await self.agent_loop.run(query, file_manifest, files)
            
            # Get final results
            final_output = execution_context.get_final_output()
            
            # Display results
            await self._display_results(final_output)
            
            # Update statistics
            execution_time = (datetime.now() - start_time).total_seconds()
            self.system_stats["successful_queries"] += 1
            self.system_stats["total_execution_time"] += execution_time
            self.system_stats["sessions_created"] += 1
            
            log_success(f"✅ Query completed in {execution_time:.2f}s", symbol="🎉")
            
        except Exception as e:
            execution_time = (datetime.now() - start_time).total_seconds()
            self.system_stats["failed_queries"] += 1
            self.system_stats["total_execution_time"] += execution_time
            
            log_error(f"❌ Query failed after {execution_time:.2f}s: {str(e)}", symbol="💥")
            self.console.print(f"[red]Error: {str(e)}[/red]")

    async def _display_results(self, final_output: Dict[str, Any]):
        """Display query results."""
        # Create results panel
        results_text = f"""
[bold]Query:[/bold] {final_output['original_query']}

[bold]Session ID:[/bold] {final_output['session_id']}

[bold]Execution Summary:[/bold]
• Total Nodes: {final_output['execution_summary']['total_nodes']}
• Completed: {final_output['execution_summary']['completed_nodes']}
• Failed: {final_output['execution_summary']['failed_nodes']}
• Execution Time: {final_output['execution_summary']['total_execution_time']:.2f}s
• Total Cost: ${final_output['execution_summary']['total_cost']:.4f}

[bold]Final Outputs:[/bold]
"""
        
        # Add final outputs
        for agent_id, output in final_output['final_outputs'].items():
            results_text += f"\n[cyan]{agent_id}:[/cyan]\n"
            if isinstance(output, dict):
                results_text += json.dumps(output, indent=2, default=str)
            else:
                results_text += str(output)
            results_text += "\n"
        
        self.console.print(Panel(results_text, title="Query Results", border_style="green"))

    async def _run_demo(self):
        """Run demonstration queries."""
        demo_queries = [
            "I'm looking for wireless headphones under $100",
            "What's the status of my order #ORD-2024-001?",
            "I'm unhappy with my recent purchase of a smartphone",
            "Can you recommend some products similar to iPhone 15?",
            "What are your return policies?"
        ]
        
        self.console.print("[bold yellow]Running demonstration queries...[/bold yellow]")
        
        for i, query in enumerate(demo_queries, 1):
            self.console.print(f"\n[bold cyan]Demo {i}/{len(demo_queries)}:[/bold cyan] {query}")
            
            if Confirm.ask("Run this demo query?"):
                await self._execute_query(query, [])
            else:
                self.console.print("[yellow]Skipped[/yellow]")
            
            if i < len(demo_queries):
                if not Confirm.ask("Continue with next demo?"):
                    break

    async def shutdown(self):
        """Shutdown the system gracefully."""
        try:
            log_step("🔄 Shutting down system...", symbol="⏹️")
            
            if self.multi_mcp:
                await self.multi_mcp.shutdown()
                log_success("✅ MultiMCP shutdown complete", symbol="🔧")
            
            log_success("✅ System shutdown complete", symbol="👋")
            
        except Exception as e:
            log_error(f"❌ Shutdown error: {str(e)}", symbol="💥")

async def main():
    """Main application entry point."""
    system = EcommerceMultiAgentSystem()
    
    try:
        await system.run()
    finally:
        await system.shutdown()

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n👋 Goodbye!")
    except Exception as e:
        print(f"❌ Fatal error: {str(e)}")
        sys.exit(1) 
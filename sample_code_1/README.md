# E-Commerce Multi-Agent Customer Service System

## 🏗️ Architecture Overview

This is a comprehensive implementation of a **NetworkX Graph-First Multi-Agent System** for e-commerce customer service. The system demonstrates how to build a scalable, modular AI system that can handle complex customer queries through specialized agents working in coordination.

### 🎯 Core Architectural Principles

1. **NetworkX Graph-First Design**: All execution plans are represented as directed acyclic graphs (DAGs)
2. **Multi-Agent System (MAS)**: Specialized agents for different tasks with clear responsibilities
3. **Model Context Protocol (MCP)**: External tools accessed through standardized interfaces
4. **Asynchronous Execution**: Full async/await pattern with concurrent agent execution
5. **Modular Design**: Easy to extend with new agents and tools

## 📁 Project Structure

```
sample_code/
├── config/                          # Configuration files
│   ├── agent_config.yaml           # Agent definitions and capabilities
│   ├── mcp_server_config.yaml      # External tool configurations
│   ├── models.json                 # LLM model configurations
│   └── profiles.yaml               # System profiles
├── agentLoop/                      # Core orchestration engine
│   ├── __init__.py
│   ├── flow.py                     # Main execution orchestrator
│   ├── contextManager.py           # NetworkX graph state management
│   ├── agents.py                   # Agent execution engine
│   ├── model_manager.py            # LLM integration layer
│   ├── graph_validator.py          # Graph validation and analysis
│   └── visualizer.py               # Real-time execution visualization
├── prompts/                        # Agent-specific prompts
│   ├── customer_intent_prompt.txt
│   ├── product_search_prompt.txt
│   ├── order_status_prompt.txt
│   ├── recommendation_prompt.txt
│   ├── complaint_handler_prompt.txt
│   ├── sentiment_analysis_prompt.txt
│   └── response_formatter_prompt.txt
├── mcp_servers/                    # External tool providers
│   ├── database_connector.py       # Product/order database operations
│   ├── payment_processor.py        # Payment status and processing
│   ├── inventory_system.py         # Stock and inventory management
│   └── email_system.py             # Customer communication
├── action/                         # Code execution capabilities
│   └── executor.py                 # Sandboxed code execution
├── utils/                          # Utility functions
│   ├── __init__.py
│   └── utils.py                    # Common utilities and logging
├── tests/                          # Test cases
│   ├── test_ecommerce_flow.py
│   └── test_agents.py
├── main.py                         # Application entry point
├── requirements.txt                # Python dependencies
└── README.md                       # This file
```

## 🤖 Agent Architecture

### Agent Types and Responsibilities

| Agent | Purpose | Input | Output | Tools |
|-------|---------|-------|--------|-------|
| **CustomerIntentAgent** | Understands customer queries | Raw customer query | Structured intent + entities | None |
| **ProductSearchAgent** | Searches product database | Search criteria | Product results | Database |
| **OrderStatusAgent** | Checks order status | Order identifier | Order details + status | Database, Payment |
| **RecommendationAgent** | Suggests products | Customer context | Product recommendations | Database |
| **ComplaintHandlerAgent** | Handles complaints | Complaint details | Resolution plan | Database, Email |
| **SentimentAnalysisAgent** | Analyzes customer sentiment | Customer interaction | Sentiment score + analysis | None |
| **ResponseFormatterAgent** | Formats final response | Agent outputs | Formatted response | None |

### Execution Flow Patterns

```
1. Simple Query Flow:
   CustomerIntentAgent → ProductSearchAgent → ResponseFormatterAgent

2. Order Status Flow:
   CustomerIntentAgent → OrderStatusAgent → ResponseFormatterAgent

3. Complex Recommendation Flow:
   CustomerIntentAgent → ProductSearchAgent → RecommendationAgent → ResponseFormatterAgent

4. Complaint Handling Flow:
   CustomerIntentAgent → SentimentAnalysisAgent → ComplaintHandlerAgent → ResponseFormatterAgent
```

## 🔧 Configuration System

### Agent Configuration (`config/agent_config.yaml`)
Defines each agent's capabilities, model preferences, and tool access permissions.

### MCP Server Configuration (`config/mcp_server_config.yaml`)
Configures external tool providers with their connection details and capabilities.

### Model Configuration (`config/models.json`)
Defines available LLM models and their API configurations.

## 🚀 Getting Started

### Prerequisites
- Python 3.11+
- Required API keys (Gemini, OpenAI, etc.)
- Database access (for MCP servers)

### Installation
```bash
# Clone the repository
git clone <repository-url>
cd sample_code_1

# Install dependencies
pip install -r requirements.txt

# Set up environment variables
cp .env.example .env
# Edit .env with your API keys

# Run the system
python main.py
```

### Basic Usage
```python
# Example: Product search query
query = "I'm looking for wireless headphones under $100"

# The system will:
# 1. Analyze intent (product search)
# 2. Search database for matching products
# 3. Generate recommendations
# 4. Format response
```

## 🔍 Key Features

### 1. NetworkX Graph Execution
- **DAG-based planning**: All execution plans are represented as directed acyclic graphs
- **Dependency management**: Automatic resolution of agent dependencies
- **Parallel execution**: Multiple agents can run concurrently when possible
- **Cycle detection**: Prevents infinite loops in execution plans

### 2. Multi-Agent Coordination
- **Specialized agents**: Each agent has a specific, well-defined responsibility
- **Data flow**: Structured data passing between agents
- **Error handling**: Graceful degradation when agents fail
- **Retry mechanisms**: Automatic retry for transient failures

### 3. External Tool Integration
- **MCP Protocol**: Standardized interface for external tools
- **Database operations**: Product search, order status, inventory checks
- **Payment processing**: Payment status and transaction history
- **Communication**: Email notifications and customer updates

### 4. Real-time Monitoring
- **Execution visualization**: Live view of agent execution progress
- **Performance metrics**: Response times, success rates, error tracking
- **Graph analysis**: Critical path analysis and bottleneck identification

## 🧪 Testing

### Running Tests
```bash
# Run all tests
python -m pytest tests/

# Run specific test file
python -m pytest tests/test_ecommerce_flow.py

# Run with verbose output
python -m pytest -v tests/
```

### Test Coverage
- **Unit tests**: Individual agent functionality
- **Integration tests**: Agent coordination and data flow
- **End-to-end tests**: Complete customer query processing
- **Graph validation tests**: NetworkX graph integrity

## 🔧 Customization Guide

### Adding New Agents
1. Create agent prompt in `prompts/`
2. Add agent configuration in `config/agent_config.yaml`
3. Update planning logic in PlannerAgent
4. Add tests for new agent

### Adding New MCP Servers
1. Implement MCP server in `mcp_servers/`
2. Add server configuration in `config/mcp_server_config.yaml`
3. Update agent configurations to use new tools
4. Test integration

### Modifying Execution Flow
1. Update PlannerAgent prompt for new flow patterns
2. Modify graph validation rules if needed
3. Update visualization for new node types
4. Test new execution paths

## 📊 Performance Considerations

### Optimization Strategies
- **Agent caching**: Cache frequently used agent outputs
- **Parallel execution**: Run independent agents concurrently
- **Database connection pooling**: Reuse database connections
- **LLM request batching**: Batch similar LLM requests

### Monitoring Metrics
- **Response time**: End-to-end query processing time
- **Agent success rate**: Percentage of successful agent executions
- **Graph complexity**: Number of nodes and edges in execution plans
- **Resource usage**: CPU, memory, and API call consumption

## 🚨 Error Handling

### Error Types and Recovery
- **Agent failures**: Automatic retry with exponential backoff
- **Network issues**: Connection retry and fallback mechanisms
- **Invalid graphs**: Graph validation and automatic correction
- **API rate limits**: Request throttling and queuing

### Debugging Tools
- **Graph debugger**: Interactive graph analysis and modification
- **Execution logs**: Detailed logging of agent interactions
- **State inspection**: Real-time inspection of execution context
- **Error tracing**: Stack traces and error context preservation

## 🔮 Future Enhancements

### Planned Features
- **Learning capabilities**: Agent performance improvement over time
- **Dynamic agent creation**: Runtime agent generation based on needs
- **Advanced visualization**: 3D graph visualization and analytics
- **Distributed execution**: Multi-node agent execution
- **A/B testing**: Agent strategy comparison and optimization

### Scalability Improvements
- **Microservices architecture**: Decompose into independent services
- **Message queues**: Asynchronous agent communication
- **Load balancing**: Distribute agent execution across nodes
- **Auto-scaling**: Dynamic resource allocation based on demand

## 📚 Learning Resources

### Key Concepts
- **Multi-Agent Systems**: [Wikipedia](https://en.wikipedia.org/wiki/Multi-agent_system)
- **NetworkX**: [Official Documentation](https://networkx.org/)
- **Model Context Protocol**: [MCP Specification](https://modelcontextprotocol.io/)
- **Directed Acyclic Graphs**: [DAG Theory](https://en.wikipedia.org/wiki/Directed_acyclic_graph)

### Related Technologies
- **LangChain**: Alternative agent framework
- **AutoGen**: Microsoft's multi-agent framework
- **CrewAI**: Crew-based agent orchestration
- **Flowise**: Visual agent builder

## 🤝 Contributing

### Development Guidelines
1. **Code style**: Follow PEP 8 and project conventions
2. **Documentation**: Comment all functions and classes
3. **Testing**: Write tests for new features
4. **Graph validation**: Ensure new agents work with graph architecture

### Contribution Areas
- **New agents**: Domain-specific agent implementations
- **MCP servers**: Additional external tool integrations
- **Visualization**: Enhanced execution monitoring
- **Performance**: Optimization and scaling improvements

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgments

- **NetworkX**: Graph theory and analysis library
- **Rich**: Terminal formatting and visualization
- **Pydantic**: Data validation and settings management
- **FastAPI**: Modern web framework for APIs

---

**Note**: This is a comprehensive sample implementation demonstrating the NetworkX Graph-First Multi-Agent architecture. It can be adapted and extended for various domains beyond e-commerce customer service. 
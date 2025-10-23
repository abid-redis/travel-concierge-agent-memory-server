# Developer Quick Reference

## 🚀 Quick Commands

### Setup
```bash
# Install dependencies
uv sync
# or
pip install -r requirements.txt

# Set environment variables
export OPENAI_API_KEY="sk-your-key-here"
export TAVILY_API_KEY="your-key-here"
export MEMORY_SERVER_URL="http://localhost:8000"

# Start Agent Memory Server
agent-memory-server

# Run the application
python gradio_app.py
```

### Testing
```bash
# Test basic functionality
python -c "from agent import TravelAgent; print('✅ Agent imports successfully')"

# Test configuration
python -c "from config import get_config; print('✅ Config loaded:', get_config().memory_server_url)"
```

## 🔧 Key Classes & Methods

### TravelAgent
```python
# Core methods
agent = TravelAgent(config)
await agent.get_client()  # Get memory client
await agent.stream_chat_turn_with_events(user_id, message)  # Main chat method
await agent.get_chat_history(user_id)  # Get chat history
await agent.store_assistant_response(user_id, response)  # Store response

# Memory methods
await agent._get_working_memory(session_id, user_id)
await agent._add_message_to_working_memory(session_id, user_id, role, content)
agent._get_namespace(user_id)  # Get user namespace
```

### TravelAgentUI
```python
# UI methods
ui = TravelAgentUI(config)
await ui.initialize_chat_history()  # Load seed data
await ui.switch_user(user_id)  # Switch user
await ui.clear_chat_history()  # Clear chat
ui.create_interface()  # Create Gradio interface
```

## 🧠 Memory Operations

### Working Memory
```python
# Get working memory
working_memory = await agent._get_working_memory(session_id, user_id)

# Add message
await agent._add_message_to_working_memory(session_id, user_id, "user", "Hello")

# Messages are automatically stored during chat flow
```

### Long-term Memory
```python
# Memory tools available to LLM:
# - search_memory
# - add_memory_to_working_memory  
# - get_or_create_working_memory
# - update_working_memory_data
# - create_long_term_memory
# - search_long_term_memory
# - update_long_term_memory
# - delete_long_term_memory
# - clear_working_memory
```

## 🛠️ Tools

### Travel Tools
```python
# Available to LLM:
search_logistics(query, start_date=None, end_date=None)  # Flights, hotels, transport
search_general(query)  # Activities, attractions, dining
generate_calendar_ics(events, trip_name=None)  # Calendar generation
```

### Memory Tools
All 9 Agent Memory Server tools are automatically available to the LLM.

## 🔍 Debugging

### Common Issues
1. **"Event loop is closed"** - Temporarily disabled memory storage to avoid conflicts
2. **Memory not persisting** - Memory storage is currently disabled (see TODO)
3. **Tool calls failing** - Verify API keys and external services
4. **clear_working_memory not found** - Use `delete_working_memory` instead

### Debug Mode
```bash
export DEBUG=1
export PYTHONPATH=.
```

### Logs
- Memory operations: Check console for memory-related messages
- Tool calls: Monitor agent logs panel in UI
- Errors: Check terminal output for detailed error messages

## 📊 Data Flow

```
User Input → Gradio UI → TravelAgent → Working Memory → AssistantAgent → Tools → Response → UI
```

### Key Points
- User messages stored before processing
- Assistant responses stored after streaming
- Memory tools executed via Agent Memory Server
- Namespace isolation: `travel_agent:{user_id}`

## 🎯 Namespace Strategy

```python
# Each user gets isolated namespace
namespace = f"travel_agent:{user_id}"

# Examples:
# Tyler → "travel_agent:Tyler"
# Purna → "travel_agent:Purna"
```

## 🔄 Session Management

```python
# Session IDs are unique per user context
session_id = f"session_{user_id}_{timestamp}"

# Working memory is session-based
# Long-term memory is user-based (persistent across sessions)
```

## 📝 Configuration

### Required Environment Variables
```bash
OPENAI_API_KEY=sk-your-key-here
TAVILY_API_KEY=your-key-here
MEMORY_SERVER_URL=http://localhost:8000
```

### Optional Configuration
```python
# In config.py
TRAVEL_AGENT_MODEL="gpt-4.1"
MAX_TOOL_ITERATIONS=8
MAX_CHAT_HISTORY_SIZE=6
MAX_SEARCH_RESULTS=5
```

## 🧪 Testing Scenarios

### Basic Chat
```python
"Plan a trip to Japan for 2 weeks"
```

### Memory Recall
```python
"I mentioned I like sushi - what restaurants do you recommend?"
```

### Calendar Generation
```python
"Create a calendar for my Tokyo itinerary"
```

### User Switching
```python
# Switch between users and verify separate memory
```

## 📈 Performance Tips

1. **Memory Client**: Shared instance across users (lazy loading)
2. **Namespace Isolation**: Efficient per-user separation
3. **Background Tasks**: Non-blocking memory operations
4. **Streaming**: Real-time UI updates
5. **Caching**: User contexts cached for performance

## 🔗 External Dependencies

- **Agent Memory Server**: Memory management
- **OpenAI**: Language models
- **Tavily**: Web search
- **Gradio**: Web interface
- **Autogen**: Agent framework
- **Redis**: Caching (optional)

## 📞 Support

1. Check troubleshooting section in README
2. Review logs for error messages
3. Verify all services are running
4. Check API keys and configuration

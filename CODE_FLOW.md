# Code Flow Documentation

## Overview

This document provides detailed code flow documentation for the AI Travel Concierge with Agent Memory Server integration.

## Core Components

### 1. TravelAgent Class (`agent.py`)

The main orchestrator that manages user contexts, memory operations, and agent interactions.

#### Key Methods

```python
class TravelAgent:
    def __init__(self, config: Optional[AppConfig] = None)
    async def get_client(self) -> MemoryAPIClient
    def _get_namespace(self, user_id: str) -> str
    def _get_or_create_user_ctx(self, user_id: str) -> UserCtx
    def _create_agent(self) -> AssistantAgent
    async def _get_working_memory(self, session_id: str, user_id: str) -> WorkingMemory
    async def _add_message_to_working_memory(self, session_id: str, user_id: str, role: str, content: str) -> None
    async def stream_chat_turn_with_events(self, user_id: str, user_message: str) -> AsyncGenerator
    async def get_chat_history(self, user_id: str, n: Optional[int] = None) -> List[Dict[str, str]]
    async def store_assistant_response(self, user_id: str, response: str) -> None
```

### 2. TravelAgentUI Class (`gradio_app.py`)

The Gradio web interface wrapper that handles user interactions and UI updates.

#### Key Methods

```python
class TravelAgentUI:
    def __init__(self, config=None)
    async def initialize_chat_history(self)
    async def switch_user(self, new_user_id: str) -> List[dict]
    async def clear_chat_history(self) -> List[dict]
    def create_interface(self) -> gr.Interface
```

## Detailed Code Flow

### 1. Application Startup

```python
# gradio_app.py - main()
def main():
    # 1. Load configuration
    config = get_config()
    
    # 2. Validate dependencies
    validate_dependencies()
    
    # 3. Create app asynchronously
    app = asyncio.run(create_app(config))
    
    # 4. Launch Gradio interface
    app.queue().launch(server_name=config.server_name, server_port=config.server_port)
```

```python
# gradio_app.py - create_app()
async def create_app(config=None) -> gr.Interface:
    # 1. Create UI instance
    ui = TravelAgentUI(config=config)
    
    # 2. Initialize chat history (loads seed data)
    await ui.initialize_chat_history()
    
    # 3. Create and return interface
    return ui.create_interface()
```

```python
# gradio_app.py - initialize_chat_history()
async def initialize_chat_history(self):
    # 1. Initialize seed data for all users
    await self.agent.initialize_seed_data()
    
    # 2. Get all user IDs and sort them
    users = self.agent.get_all_user_ids()
    self.user_ids = sorted(users)
    
    # 3. Load chat history for current user
    self.initial_history = await self.agent.get_chat_history(self.current_user_id, n=-1)
```

### 2. Agent Initialization

```python
# agent.py - __init__()
def __init__(self, config: Optional[AppConfig] = None):
    # 1. Set up configuration
    self.config = config or get_config()
    
    # 2. Set environment variables
    os.environ["OPENAI_API_KEY"] = config.openai_api_key
    os.environ["TAVILY_API_KEY"] = config.tavily_api_key
    
    # 3. Initialize shared clients
    self.tavily_client = TavilyClient(api_key=config.tavily_api_key)
    self.agent_model = OpenAIChatCompletionClient(model=config.travel_agent_model)
    
    # 4. Initialize memory client (lazy loading)
    self._memory_client: MemoryAPIClient | None = None
    
    # 5. Initialize user context cache
    self._user_ctx_cache = {}
```

### 3. Memory Client Management

```python
# agent.py - get_client()
async def get_client(self) -> MemoryAPIClient:
    # Lazy initialization of memory client
    if not self._memory_client:
        self._memory_client = await create_memory_client(
            base_url=self.config.memory_server_url,
            timeout=30.0,
            default_model_name="gpt-4o",
        )
    return self._memory_client
```

### 4. User Context Management

```python
# agent.py - _get_or_create_user_ctx()
def _get_or_create_user_ctx(self, user_id: str) -> UserCtx:
    # 1. Check cache for existing context
    if user_ctx := self._user_ctx_cache.get(user_id):
        return user_ctx
    
    # 2. Generate unique session ID
    session_id = f"session_{user_id}_{int(datetime.now().timestamp())}"
    
    # 3. Create supervisor agent
    agent = self._create_agent()
    
    # 4. Cache and return user context
    self._user_ctx_cache[user_id] = UserCtx(agent=agent, session_id=session_id)
    return self._user_ctx_cache[user_id]
```

### 5. Agent Creation

```python
# agent.py - _create_agent()
def _create_agent(self) -> AssistantAgent:
    # 1. Create AssistantAgent with tools
    agent = AssistantAgent(
        name="agent",
        model_client=self.agent_model,
        tools=self._get_tools(),  # Includes memory tools
        system_message=self._get_system_message(),
        max_tool_iterations=self.config.max_tool_iterations,
        model_client_stream=True,
    )
    return agent
```

### 6. Tool Integration

```python
# agent.py - _get_tools()
def _get_tools(self) -> List[FunctionTool]:
    tools = []
    
    # 1. Add travel tools
    tools.append(FunctionTool(func=self.search_logistics, ...))
    tools.append(FunctionTool(func=self.search_general, ...))
    tools.append(FunctionTool(func=self.generate_calendar_ics, ...))
    
    # 2. Add memory management tools
    memory_tool_schemas = MemoryAPIClient.get_all_memory_tool_schemas()
    for tool_schema in memory_tool_schemas:
        tools.append(self._create_memory_tool_wrapper(tool_schema))
    
    return tools
```

### 7. Chat Processing Flow

```python
# agent.py - stream_chat_turn_with_events()
async def stream_chat_turn_with_events(self, user_id: str, user_message: str):
    # 1. Get or create user context
    ctx = self._get_or_create_user_ctx(user_id)
    
    # 2. Initialize working memory and store user message
    try:
        working_memory = await self._get_working_memory(ctx.session_id, user_id)
        await self._add_message_to_working_memory(
            ctx.session_id, user_id, "user", user_message
        )
    except Exception as e:
        print(f"⚠️ Failed to initialize working memory: {e}")
    
    # 3. Start streaming with agent
    stream = ctx.agent.run_stream(task=user_message)
    
    # 4. Process streaming events
    async for event in stream:
        # Handle different event types (tool calls, responses, etc.)
        yield partial_response, event_data
    
    # 5. Final yield with complete response
    yield buffer, None
```

### 8. Memory Operations

```python
# agent.py - _get_working_memory()
async def _get_working_memory(self, session_id: str, user_id: str) -> WorkingMemory:
    client = await self.get_client()
    created, result = await client.get_or_create_working_memory(
        session_id=session_id,
        namespace=self._get_namespace(user_id),
        model_name="gpt-4o-mini",
    )
    return WorkingMemory(**result.model_dump())
```

```python
# agent.py - _add_message_to_working_memory()
async def _add_message_to_working_memory(self, session_id: str, user_id: str, role: str, content: str):
    client = await self.get_client()
    await client.get_or_create_working_memory(
        session_id=session_id,
        namespace=self._get_namespace(user_id),
        model_name="gpt-4o-mini",
    )
    await client.append_messages_to_working_memory(
        session_id=session_id,
        messages=[{"role": role, "content": content}],
        namespace=self._get_namespace(user_id),
    )
```

### 9. Gradio Event Handling

```python
# gradio_app.py - handle_streaming_chat()
async def handle_streaming_chat(message, history, events, calendar_file):
    # 1. Add user message to history
    history = history + [{"role": "user", "content": message}]
    
    # 2. Initialize assistant message with thinking animation
    history = history + [{"role": "assistant", "content": '<span class="thinking-animation">●●●</span>'}]
    
    # 3. Stream the response
    async for partial_response, evt in self.agent.stream_chat_turn_with_events(self.current_user_id, message):
        # Update history with partial response
        history[-1] = {"role": "assistant", "content": partial_response}
        
        # Handle events (tool calls, calendar generation, etc.)
        if evt:
            events = events + [evt]
        
        # Yield updates to UI
        yield history, events, events_html, calendar_file, btn_update, status_update
    
    # 4. Store assistant response after streaming completes
    if final_response:
        asyncio.create_task(
            self.agent.store_assistant_response(self.current_user_id, final_response)
        )
```

### 10. Memory Tool Execution

```python
# agent.py - _handle_memory_tool_call()
async def _handle_memory_tool_call(self, function_call: dict, session_id: str, user_id: str) -> str:
    client = await self.get_client()
    
    # Use unified tool call resolver
    result = await client.resolve_tool_call(
        tool_call=function_call,
        session_id=session_id,
        namespace=self._get_namespace(user_id),
    )
    
    return result["formatted_response"]
```

## Error Handling

### Async Operation Safety

```python
# Safe async operations in streaming context
try:
    await self._add_message_to_working_memory(...)
except Exception as e:
    print(f"⚠️ Failed to save user message to memory: {e}")
    # Continue without memory if there's an error
```

### Background Task Management

```python
# Non-blocking memory storage
asyncio.create_task(
    self.agent.store_assistant_response(self.current_user_id, final_response)
)
```

## Data Flow Summary

1. **User Input** → Gradio UI
2. **UI** → TravelAgent.stream_chat_turn_with_events()
3. **Agent** → Get working memory context
4. **Agent** → Store user message
5. **Agent** → AssistantAgent.run_stream()
6. **AssistantAgent** → Process with tools (memory/search)
7. **Tools** → Execute operations (memory server/external APIs)
8. **AssistantAgent** → Stream response back
9. **Agent** → Yield partial responses to UI
10. **UI** → Update display in real-time
11. **Agent** → Store assistant response (background task)

## Key Design Patterns

### 1. Lazy Loading
- Memory client initialized only when needed
- User contexts created on-demand

### 2. Namespace Isolation
- Each user gets unique namespace: `travel_agent:{user_id}`
- Complete memory separation between users

### 3. Async/Await Pattern
- Non-blocking operations throughout
- Background tasks for memory storage

### 4. Event-Driven Architecture
- Streaming responses with real-time events
- Tool call events for UI feedback

### 5. Error Resilience
- Graceful degradation when memory operations fail
- Continue operation without blocking on errors

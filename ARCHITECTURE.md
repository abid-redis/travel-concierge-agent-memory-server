# Architecture Diagrams

## System Architecture

```mermaid
graph TB
    subgraph "Frontend Layer"
        UI[Gradio Web Interface]
        User[User Input]
    end
    
    subgraph "Application Layer"
        Agent[TravelAgent]
        GradioApp[TravelAgentUI]
        Config[Configuration]
    end
    
    subgraph "AI Layer"
        AutogenAgent[AssistantAgent]
        MemoryTools[Memory Tools]
        TravelTools[Travel Tools]
    end
    
    subgraph "Memory Layer"
        MemoryServer[Agent Memory Server]
        WorkingMemory[Working Memory]
        LongTermMemory[Long-term Memory]
    end
    
    subgraph "External Services"
        OpenAI[OpenAI API]
        Tavily[Tavily Search]
        Redis[Redis Cache]
    end
    
    User --> UI
    UI --> GradioApp
    GradioApp --> Agent
    Agent --> AutogenAgent
    AutogenAgent --> MemoryTools
    AutogenAgent --> TravelTools
    MemoryTools --> MemoryServer
    MemoryServer --> WorkingMemory
    MemoryServer --> LongTermMemory
    TravelTools --> Tavily
    AutogenAgent --> OpenAI
    Agent --> Redis
```

## Memory Architecture

```mermaid
graph LR
    subgraph "User Context"
        UserID[User ID]
        SessionID[Session ID]
        Namespace[Namespace: travel_agent:user_id]
    end
    
    subgraph "Memory Types"
        WorkingMem[Working Memory<br/>Session-based conversations]
        LongTermMem[Long-term Memory<br/>User preferences & history]
    end
    
    subgraph "Memory Operations"
        Store[Store Messages]
        Retrieve[Retrieve Context]
        Search[Search Memories]
        Update[Update Preferences]
    end
    
    UserID --> Namespace
    SessionID --> WorkingMem
    Namespace --> LongTermMem
    WorkingMem --> Store
    WorkingMem --> Retrieve
    LongTermMem --> Search
    LongTermMem --> Update
```

## Application Initialization Flow

```mermaid
sequenceDiagram
    participant Main as main()
    participant UI as TravelAgentUI
    participant Agent as TravelAgent
    participant MemoryServer as Agent Memory Server
    
    Main->>UI: Create TravelAgentUI
    UI->>Agent: Initialize TravelAgent
    Agent->>Agent: Setup memory client
    Agent->>MemoryServer: Connect to memory server
    Agent->>Agent: Load seed data
    Agent->>MemoryServer: Create long-term memories
    UI->>UI: Initialize chat history
    UI->>Agent: Get chat history
    Agent->>MemoryServer: Retrieve working memory
```

## Chat Flow

```mermaid
sequenceDiagram
    participant User as User
    participant UI as Gradio UI
    participant Agent as TravelAgent
    participant AutogenAgent as AssistantAgent
    participant MemoryServer as Agent Memory Server
    participant Tools as External Tools
    
    User->>UI: Send message
    UI->>Agent: stream_chat_turn_with_events()
    Agent->>MemoryServer: Get working memory
    Agent->>MemoryServer: Store user message
    Agent->>AutogenAgent: Process with context
    AutogenAgent->>Tools: Call tools (search/memory)
    Tools-->>AutogenAgent: Return results
    AutogenAgent-->>Agent: Stream response
    Agent-->>UI: Yield partial responses
    UI-->>User: Display streaming text
    Agent->>MemoryServer: Store assistant response
```

## Memory Tool Integration

```mermaid
sequenceDiagram
    participant LLM as LLM
    participant Agent as TravelAgent
    participant MemoryServer as Agent Memory Server
    participant Tools as Memory Tools
    
    LLM->>Agent: Function call request
    Agent->>MemoryServer: resolve_tool_call()
    MemoryServer->>Tools: Execute memory operation
    Tools-->>MemoryServer: Return result
    MemoryServer-->>Agent: Formatted response
    Agent-->>LLM: Tool result
```

## Data Flow

```mermaid
flowchart TD
    A[User Input] --> B[Gradio UI]
    B --> C[TravelAgent]
    C --> D[Get Working Memory]
    D --> E[Agent Memory Server]
    E --> F[Store User Message]
    F --> G[AssistantAgent]
    G --> H[Process with Context]
    H --> I{Tool Call?}
    I -->|Yes| J[Execute Tool]
    I -->|No| K[Generate Response]
    J --> L[Memory/Search Tool]
    L --> M[Return Result]
    M --> K
    K --> N[Stream Response]
    N --> O[Update UI]
    O --> P[Store Assistant Response]
    P --> E
```

## Component Relationships

```mermaid
classDiagram
    class TravelAgent {
        +config: AppConfig
        +_memory_client: MemoryAPIClient
        +_user_ctx_cache: Dict
        +get_client()
        +_get_namespace()
        +_get_working_memory()
        +_add_message_to_working_memory()
        +stream_chat_turn_with_events()
        +get_chat_history()
        +store_assistant_response()
    }
    
    class UserCtx {
        +agent: AssistantAgent
        +session_id: str
    }
    
    class TravelAgentUI {
        +agent: TravelAgent
        +current_user_id: str
        +user_ids: List[str]
        +initialize_chat_history()
        +switch_user()
        +clear_chat_history()
        +create_interface()
    }
    
    class AssistantAgent {
        +name: str
        +model_client: OpenAIChatCompletionClient
        +tools: List[FunctionTool]
        +system_message: str
        +run_stream()
    }
    
    class MemoryAPIClient {
        +get_or_create_working_memory()
        +append_messages_to_working_memory()
        +create_long_term_memory()
        +search_memory()
        +resolve_tool_call()
    }
    
    TravelAgent --> UserCtx
    TravelAgent --> MemoryAPIClient
    TravelAgentUI --> TravelAgent
    TravelAgent --> AssistantAgent
    AssistantAgent --> MemoryAPIClient
```

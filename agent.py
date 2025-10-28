# import suppress_warnings  # Must be first to suppress warnings
import warnings

warnings.filterwarnings("ignore")
import os
import json
import re
import traceback
from dataclasses import dataclass
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional, AsyncGenerator
import re
import hashlib

from autogen_agentchat.agents import AssistantAgent
from autogen_agentchat.messages import (
    TextMessage,
    ToolCallRequestEvent,
    ToolCallExecutionEvent,
    MemoryQueryEvent,
    ModelClientStreamingChunkEvent,
    ThoughtEvent,
    UserInputRequestedEvent,
    CodeGenerationEvent,
    CodeExecutionEvent,
    SelectSpeakerEvent,
    ToolCallSummaryMessage,
    MultiModalMessage,
    HandoffMessage,
    StopMessage,
)
from autogen_core.tools import FunctionTool
from autogen_ext.models.openai import OpenAIChatCompletionClient
from tavily import TavilyClient
from ics import Calendar, Event, DisplayAlarm
from ics.grammar.parse import ContentLine
from langchain_openai import ChatOpenAI

from agent_memory_client import (
    MemoryAPIClient,
    MemoryClientConfig,
    create_memory_client,
)
from agent_memory_client.models import WorkingMemory

from config import AppConfig


@dataclass
class UserCtx:
    """User-specific context containing agent instance and session ID.

    Attributes:
        agent: Main assistant agent with tools
        session_id: Unique session identifier for working memory
    """

    agent: AssistantAgent
    session_id: str


class TravelAgent:
    """Travel planning agent with Agent Memory Server-powered personalized memory capabilities.

    This agent provides personalized travel planning services by maintaining
    separate memory contexts for each user using Agent Memory Server. Each user gets their own
    working memory session and supervisor agent that are cached for performance.

    Features:
        - Per-user memory isolation using Agent Memory Server with namespace isolation
        - Semantic memory search and retrieval via Agent Memory Server
        - Working memory for session-based conversation management
        - Web search integration for current travel information
        - Automatic memory extraction and personalized recommendations

    Attributes:
        config: Application configuration containing API keys and model settings
        tavily_client: Web search client for travel information
        agent_model: OpenAI client for the main travel agent
        llm: LangChain OpenAI client for memory tool handling
    """

    def __init__(self, config: Optional[AppConfig] = None):
        """Initialize the TravelAgent with configuration and shared resources.

        Args:
            config: Application configuration. If None, loads default config.
        """
        if config is None:
            from config import get_config

            config = get_config()
        self.config = config

        # Set environment variables for SDK clients
        os.environ["OPENAI_API_KEY"] = config.openai_api_key
        os.environ["TAVILY_API_KEY"] = config.tavily_api_key

        # Initialize shared clients
        self.tavily_client = TavilyClient(api_key=config.tavily_api_key)
        self.agent_model = OpenAIChatCompletionClient(
            model=config.travel_agent_model, parallel_tool_calls=False
        )

        # Initialize memory client
        self._memory_client: MemoryAPIClient | None = None

        # Initialize user context cache
        self._user_ctx_cache = {}

    async def initialize_seed_data(self) -> None:
        """Initialize seed users with their memories. Call this after creating the agent."""
        await self._init_seed_users()

    # ------------------------------
    # Memory Client Management
    # ------------------------------

    async def get_client(self) -> MemoryAPIClient:
        """Get the memory client, initializing it if needed."""
        if not self._memory_client:
            self._memory_client = await create_memory_client(
                base_url=self.config.memory_server_url,
                timeout=30.0,
                default_model_name="gpt-4o",
            )
        return self._memory_client

    def _get_namespace(self, user_id: str) -> str:
        """Generate consistent namespace for a user."""
        return f"travel_agent:{user_id}"

    # ------------------------------
    # User Context Management
    # ------------------------------

    def _get_or_create_user_ctx(self, user_id: str) -> UserCtx:
        """Get or create user-specific context with agent components.

        Creates and caches a complete user context including agent
        and session management for Agent Memory Server integration.

        Args:
            user_id: Unique identifier for the user

        Returns:
            UserCtx: Complete user context with agent initialized
        """
        if user_ctx := self._user_ctx_cache.get(user_id):
            return user_ctx

        # Generate unique session ID for this user
        session_id = f"session_{user_id}_{int(datetime.now().timestamp())}"

        # Create supervisor agent
        agent = self._create_agent()

        # Cache and return user context
        self._user_ctx_cache[user_id] = UserCtx(agent=agent, session_id=session_id)
        return self._user_ctx_cache[user_id]

    def _load_seed_data(self) -> Dict[str, Any]:
        """Load seed data from JSON file."""
        seed_file = Path(__file__).parent / "context" / "seed.json"
        with open(seed_file, "r", encoding="utf-8") as f:
            return json.load(f)

    def get_all_user_ids(self) -> List[str]:
        """Return a unified list of user IDs from currently cached contexts."""
        return self._user_ctx_cache.keys()

    async def _init_seed_users(self) -> None:
        """Initialize seed users with memories from seed.json."""
        seed_data = self._load_seed_data()
        user_memories = seed_data.get("user_memories", {})

        client = await self.get_client()

        for user_id, memories in user_memories.items():
            try:
                ctx = self._get_or_create_user_ctx(str(user_id))
                print(f"🌱 Seeding memory for user: {user_id}")

                from agent_memory_client.models import ClientMemoryRecord

                memory_records = [
                    ClientMemoryRecord(
                        text=memo["insight"],
                        memory_type="semantic",
                        namespace=self._get_namespace(str(user_id)),
                        user_id=str(user_id),
                    )
                    for memo in memories
                ]

                await client.create_long_term_memory(memory_records)
                print(f"✅ Seeded {len(memories)} memories for user: {user_id}")
            except Exception as e:
                print(f"❌ Failed to seed memory for user {user_id}: {e}")
                continue

    def _create_agent(self) -> AssistantAgent:
        """Create supervisor agent with tools.

        Returns:
            AssistantAgent: Configured supervisor with tools
        """
        print("🤖 Creating AssistantAgent with tools...", flush=True)
        try:
            agent = AssistantAgent(
                name="agent",
                model_client=self.agent_model,
                tools=self._get_tools(),
                system_message=self._get_system_message(),
                max_tool_iterations=self.config.max_tool_iterations,
                model_client_stream=True,  # Enable token streaming
            )
            print("✅ AssistantAgent created successfully", flush=True)
            return agent
        except Exception as e:
            print(f"❌ Failed to create AssistantAgent: {e}", flush=True)
            print(f"   Full traceback: {traceback.format_exc()}", flush=True)
            raise

    # ------------------------------
    # Memory Management Methods
    # ------------------------------

    async def _handle_memory_tool_call(
        self,
        function_call: dict,
        session_id: str,
        user_id: str,
    ) -> str:
        """Handle memory tool function calls using the client's unified resolver."""
        client = await self.get_client()

        print("Accessing memory...")
        result = await client.resolve_tool_call(
            tool_call=function_call,
            session_id=session_id,
            namespace=self._get_namespace(user_id),
        )

        if not result["success"]:
            print(f"Memory tool call failed: {result['error']}")
            return result["formatted_response"]

        return result["formatted_response"]

    async def _get_working_memory(self, session_id: str, user_id: str) -> WorkingMemory:
        """Get working memory for a session."""
        client = await self.get_client()
        created, result = await client.get_or_create_working_memory(
            session_id=session_id,
            namespace=self._get_namespace(user_id),
            model_name="gpt-4o-mini",
        )
        return WorkingMemory(**result.model_dump())

    async def _add_message_to_working_memory(
        self, session_id: str, user_id: str, role: str, content: str
    ) -> None:
        """Add a message to working memory."""
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

    def _get_tools(self) -> List[FunctionTool]:
        """Get the list of tools for the travel agent.

        Returns:
            List[FunctionTool]: List of tools available to the agent
        """
        tools = []
        tools.append(
            FunctionTool(
                func=self.search_logistics,
                description=(
                    "Time-aware logistics search ONLY: flights, hotels, and intercity/local transport. "
                    "Use for availability, schedules, prices, carriers/properties, or routes. "
                    "Arguments: query (required), start_date (optional, YYYY-MM-DD), end_date (optional, YYYY-MM-DD). "
                    "Always include dates when the user mentions a travel window; if ambiguous, ask for dates before booking guidance. "
                    "NEVER use this for activities, attractions, neighborhoods, or dining. "
                    "Results are restricted to reputable flight/hotel/transport sources; top URLs are deeply extracted."
                ),
            )
        )
        tools.append(
            FunctionTool(
                func=self.search_general,
                description=(
                    "Time-aware destination research: activities, attractions, neighborhoods, dining, events, local tips. "
                    "Use for up-to-date things to do, cultural context, and planning inspiration. "
                    "Arguments: query (required) "
                    "Scope searches to the relevant season/year when possible and prefer recent sources. "
                    "NEVER use this for flights, hotels, or transport logistics. "
                    "Example: 'things to do in Lisbon in June 2026'."
                ),
            )
        )
        tools.append(
            FunctionTool(
                func=self.generate_calendar_ics,
                description=(
                    "📅 Generate a downloadable calendar file (.ics) from a simple travel itinerary. "
                    "Use when you have a finalized schedule with dates and times. "
                    "Arguments: events (required array of events), trip_name (optional string). "
                    "Each event needs: title, date, start_time (optional), end_time (optional), location (optional), notes (optional). "
                    "Use format: date='2026-06-05', start_time='14:30', end_time='16:00'. "
                    "Returns file_path for user to open. Call this when presenting a complete itinerary."
                ),
            )
        )

        # # Add memory management tools as FunctionTool instances
        # memory_tool_schemas = MemoryAPIClient.get_all_memory_tool_schemas()
        # for tool_schema in memory_tool_schemas:
        #     tools.append(self._create_memory_tool_wrapper(tool_schema))

        config = MemoryClientConfig(
            base_url="http://localhost:8000", session_id="test_session_id"
        )

        memory_client = MemoryAPIClient(config=config)

        tools.append(
            FunctionTool(
                func=memory_client.create_long_term_memory,
                name="create_long_term_memory",
                description=(
                    "Create a long-term memory for the user. "
                    "Arguments: memories (required list of memories). "
                    "Each memory needs: id (optional), user_id (required) = 'Tyler' (default), content (required), topics (optional), entities (optional), memory_type (optional), distance_threshold (optional), limit (optional, default 10), offset (optional, default 0), optimize_query (optional, default false). "
                    "Use format: id='123', content='This is a long-term memory', topics=['travel', 'planning'], entities=['planning', 'travel'], memory_type='semantic', distance_threshold=0.5, limit=10, offset=0, optimize_query=False. "
                    "Returns the created memory."
                ),
            )
        )

        tools.append(
            FunctionTool(
                func=memory_client.search_memory_tool,
                name="search_memory_tool",
                description=(
                    "Search for a long-term memory for the user. "
                    "Arguments: text (required), user_id (required) = 'Tyler' (default), topics (optional), entities (optional), memory_type (optional), distance_threshold (optional), limit (optional, default 10), offset (optional, default 0), optimize_query (optional, default false). "
                    "Use format: text='This is a long-term memory', user_id='123', topics=['travel', 'planning'], entities=['planning', 'travel'], memory_type='semantic', distance_threshold=0.5, limit=10, offset=0, optimize_query=False. "
                    "Returns the searched memory."
                ),
            )
        )

        tools.append(memory_client.get_or_create_working_memory)
        tools.append(memory_client.append_messages_to_working_memory)
        tools.append(memory_client.update_working_memory_data)
        tools.append(memory_client.delete_working_memory)

        print(f"🏁 Tool creation complete. {len(tools)} tools ready.", flush=True)
        return tools

    def _create_memory_tool_wrapper(self, tool_schema: dict) -> FunctionTool:
        """Create a FunctionTool wrapper for Agent Memory Server tools."""
        tool_name = tool_schema["function"]["name"]

        def memory_tool_wrapper():
            """Wrapper function for memory tools."""
            # Return a placeholder - actual execution will be handled by the streaming logic
            return f"Memory tool {tool_name} called"

        return FunctionTool(
            func=memory_tool_wrapper,
            name=tool_name,
            description=tool_schema["function"]["description"],
        )

    def _get_system_message(self) -> str:
        """Get the system message for the travel agent supervisor.

        Returns:
            str: Complete system message with role, responsibilities, and workflow
        """
        today = datetime.utcnow().strftime("%Y-%m-%d")
        return (
            f"You are an expert, time-aware, friendly Travel Concierge AI. Today is {today} (UTC). "
            "Assume your built in knowledge may be outdated; for anything time-sensitive, verify with tools.\n\n"
            "ROLE:\n"
            "- Discover destinations, plan itineraries, recommend accommodations, and organize logistics on behalf of the user.\n"
            "- Research current options, prices, availability, and on-the-ground activities using your tools.\n"
            "- Produce clear, actionable itineraries and booking guidance.\n"
            "- Regardless of your prior knowledge, always use search tools for current or future-state information.\n\n"
            "TOOL USAGE: You have access to the following helpful tools.\n"
            "- Use search_logistics ONLY for flights, hotels, or transport. Include start_date/end_date (YYYY-MM-DD) when known.\n"
            "- Use search_general for activities, attractions, neighborhoods, dining, events, or local tips. Include dates when relevant.\n"
            "- Use generate_calendar_ics when you have a finalized itinerary. Pass simple events array with title, date, optional times/location/notes.\n"
            "- Prefer recent sources (past 12–24 months) and pass explicit dates to tools whenever the user provides a time window.\n"
            "\n"
            "- Use the following memory tools to manage and persist user context across sessions:\n"
            "    - create_long_term_memory\n"
            "    - search_memory_tool\n"
            "    - get_or_create_working_memory\n"
            "    - append_messages_to_working_memory\n"
            "    - update_working_memory_data\n"
            "    - delete_working_memory\n"
            "\n"
            "MEMORY BEHAVIOR:\n"
            "- Always analyze and save important user information or context to long-term memory using create_long_term_memory.\n"
            "- Before responding to each new user query, always search existing long-term memory using search_memory_tool to recall relevant context.\n"
            "- Use working memory (get_or_create_working_memory, append_messages_to_working_memory, etc.) to maintain short-term conversational continuity.\n"
            "- Update or delete working memory as needed when context changes significantly.\n"
            "- Integrate recalled memory context with the current user input to ensure contextually aware, accurate, and personalized responses.\n"
            "DISCOVERY:\n"
            "- If missing details, ask targeted questions (exact dates or window, origin/destination, budget, party size, interests,\n"
            "  lodging preferences, accessibility, loyalty programs).\n\n"
            "OUTPUT STYLE:\n"
            "- Be concise and prescriptive with your suggestions, followups, and recommendations.\n"
            "- Seek to be the best and friendliest travel agent possible. You are the expert after all.\n"
            "- Cite sources with titles and URLs for any tool-based claim.\n"
            "- Normalize to a single currency if prices appear; state assumptions.\n"
            "- For itineraries, list day-by-day with times and logistics.\n\n"
        )

    # -----------------
    # Tools
    # -----------------

    def _perform_search(
        self,
        query: str,
        search_type: str,
        include_domains: Optional[List[str]] = None,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Shared search logic with optional URL extraction.

        Args:
            query: Search query
            search_type: "logistics" or "general" for logging
            include_domains: Optional domain restrictions
            start_date: Optional start date for query enhancement
            end_date: Optional end date for query enhancement

        Returns:
            Dictionary with results and extractions
        """
        print(f"🔧 {search_type.upper()} SEARCH: {query}", flush=True)

        try:
            # Augment query with dates if provided
            enhanced_query = query
            if start_date:
                enhanced_query += f" from {start_date}"
            if end_date and end_date != start_date:
                enhanced_query += f" to {end_date}"

            search_kwargs = {
                "query": enhanced_query,
                "topic": "general",
                "search_depth": "advanced",
                "max_results": self.config.max_search_results,
            }

            if include_domains:
                search_kwargs["include_domains"] = include_domains

            results = self.tavily_client.search(**search_kwargs)

            if not results:
                print(f"⚠️ Empty results from Tavily", flush=True)
                return {"results": [], "extractions": []}

            # Filter results by score
            all_results = results.get("results", [])
            filtered_results = [r for r in all_results if r.get("score", 0) > 0.2]
            print(
                f"📊 Found {len(filtered_results)}/{len(all_results)} quality results",
                flush=True,
            )

            results["results"] = filtered_results

            # Extract top 2 URLs for deeper context
            top_urls = [r.get("url") for r in filtered_results[:2] if r.get("url")]
            extractions: List[Dict[str, Any]] = []

            if top_urls:
                try:
                    extracted = self.tavily_client.extract(urls=top_urls)
                    if isinstance(extracted, dict) and extracted.get("results"):
                        extractions = extracted.get("results", [])
                    elif isinstance(extracted, list):
                        extractions = extracted
                    print(f"📄 Extracted {len(extractions)} content blocks", flush=True)
                except Exception as extract_e:
                    print(f"⚠️ URL extraction failed: {extract_e}", flush=True)

            results["extractions"] = extractions
            print(
                f"✅ {search_type.upper()} COMPLETE: {len(filtered_results)} results + {len(extractions)} extractions",
                flush=True,
            )
            return results

        except Exception as e:
            error_msg = f"❌ {search_type.upper()} ERROR: {str(e)}"
            print(error_msg, flush=True)
            return {"error": error_msg, "results": [], "extractions": []}

    def search_logistics(
        self,
        query: str,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
    ) -> Dict[str, Any]:
        """✈️🏨🚆 Logistics search: flights, hotels, and transport only.

        What it is for
        - Airfare and airline schedules, hotels/stays, and intercity transport (train/bus/ferry/car rental).

        How to use
        - Provide a concise query that includes the route or destination and constraints, e.g.:
          "JFK to LHR, nonstop preferred" or "hotels in Kyoto near Gion, mid-range" or "train Paris to Amsterdam".
        - Optionally include start_date and end_date as YYYY-MM-DD strings to guide availability windows.

        Behavior
        - Restricts sources to reputable flight/hotel/transport providers and aggregators.
        - Returns the strongest matches first and deeply extracts the top URLs for rich context.
        """
        include_domains = [
            # Flights / OTAs
            "expedia.com",
            "kayak.com",
            "travel.google.com",
            # Hotels / stays
            "booking.com",
            "hotels.com",
        ]

        return self._perform_search(
            query=query,
            search_type="logistics",
            include_domains=include_domains,
            start_date=start_date,
            end_date=end_date,
        )

    def search_general(
        self,
        query: str,
    ) -> Dict[str, Any]:
        """📍 General destination research: activities, attractions, neighborhoods, dining, events.

        What it is for
        - Up-to-date things to do, local highlights, neighborhoods to stay, dining ideas, and cultural context.

        How to use
        - Provide a destination/time-focused query, e.g., "things to do in Lisbon in June",
          "Barcelona food tours", "best neighborhoods to stay in Tokyo".

        Behavior
        - Runs an open web search (no logistics domains restriction) with raw content for context.
        """
        return self._perform_search(query=query, search_type="general")

    def generate_calendar_ics(
        self,
        events: List[Dict[str, Any]],
        trip_name: Optional[str] = None,
    ) -> Dict[str, Any]:
        """📅 Generate a simple .ics calendar file from travel events.

        Args:
            events: List of event dictionaries with:
                - title (required): Event name
                - date (required): Date as 'YYYY-MM-DD'
                - start_time (optional): Time as 'HH:MM'
                - end_time (optional): Time as 'HH:MM'
                - location (optional): Where it happens
                - notes (optional): Additional details

        Returns:
            Dictionary with file_path and events_count
        """
        print(f"🔧 CALENDAR GENERATION: Creating simple .ics file", flush=True)

        try:
            if not events:
                return {
                    "error": "No events provided",
                    "file_path": None,
                    "events_count": 0,
                }

            # Create calendar
            calendar = Calendar()
            calendar.extra.append(
                ContentLine("X-WR-CALNAME", value=trip_name or "Travel Itinerary")
            )

            user_id = getattr(self, "_current_user_id", "default")

            for event_data in events:
                event = self._create_simple_event(event_data, user_id)
                if event:
                    calendar.events.add(event)

            # Generate filename
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            safe_name = re.sub(r"[^\w\-_]", "_", (trip_name or "itinerary"))[:30]
            filename = f"{timestamp}_{safe_name}.ics"

            # Save file
            calendar_dir = Path(__file__).parent / "assets" / "calendars" / user_id
            calendar_dir.mkdir(parents=True, exist_ok=True)
            file_path = calendar_dir / filename

            with open(file_path, "w", encoding="utf-8") as f:
                f.write(str(calendar))

            print(
                f"✅ CALENDAR COMPLETE: {len(events)} events in {filename}", flush=True
            )

            return {
                "file_path": str(file_path.absolute()),
                "filename": filename,
                "events_count": len(events),
            }

        except Exception as e:
            error_msg = f"❌ CALENDAR ERROR: {str(e)}"
            print(error_msg, flush=True)
            return {"error": error_msg, "file_path": None, "events_count": 0}

    def _create_simple_event(
        self, event_data: Dict[str, Any], user_id: str
    ) -> Optional[Event]:
        """Create a simple ICS event from basic event data.

        Args:
            event_data: Dict with title, date, optional start_time/end_time/location/notes
            user_id: User ID for UID generation

        Returns:
            Event object or None if creation failed
        """
        try:
            title = event_data.get("title", "").strip()
            date_str = event_data.get("date", "").strip()

            if not title or not date_str:
                print(f"   ⚠️ Skipping event: missing title or date", flush=True)
                return None

            event = Event()
            event.name = title

            # Parse date (YYYY-MM-DD format)
            event_date = datetime.fromisoformat(date_str).date()

            # Check if we have times
            start_time = event_data.get("start_time", "").strip()
            end_time = event_data.get("end_time", "").strip()

            if start_time:
                # Timed event
                start_hour, start_min = map(int, start_time.split(":"))
                event.begin = datetime.combine(
                    event_date,
                    datetime.min.time().replace(hour=start_hour, minute=start_min),
                )

                if end_time:
                    end_hour, end_min = map(int, end_time.split(":"))
                    event.end = datetime.combine(
                        event_date,
                        datetime.min.time().replace(hour=end_hour, minute=end_min),
                    )
                else:
                    # Default 1 hour duration
                    event.end = event.begin + timedelta(hours=1)

                # Add default reminder for timed events (30 minutes before)
                alarm = DisplayAlarm()
                alarm.trigger = event.begin - timedelta(minutes=30)
                alarm.description = f"Reminder: {title}"
                event.alarms.append(alarm)

            else:
                # All-day event
                event.begin = event_date
                event.make_all_day()

            # Add optional fields
            if location := event_data.get("location", "").strip():
                event.location = location

            if notes := event_data.get("notes", "").strip():
                event.description = notes

            # Simple UID
            uid_source = f"{user_id}:{title}:{date_str}:{start_time}"
            uid_hash = hashlib.md5(uid_source.encode()).hexdigest()[:12]
            event.uid = f"{uid_hash}@travel-agent"

            return event

        except Exception as e:
            print(f"   ⚠️ Event creation failed: {e}", flush=True)
            return None

    # -----------------
    # Chat and Memory Interface
    # -----------------

    async def stream_chat_turn_with_events(
        self, user_id: str, user_message: str
    ) -> AsyncGenerator[tuple[str, dict | None], None]:
        """
        Yield (growing assistant reply, normalized event | None) pairs as the agent streams.

        Emits a small set of meaningful events suitable for UI display while
        coalescing token chunk noise:
          - user_message_submitted (when user message is processed)
          - llm_token_stream_start (first token chunk)
          - tool_call (FunctionCall)
          - tool_result (FunctionExecutionResult)
          - llm_message_complete (final assistant text)
        """
        ctx = self._get_or_create_user_ctx(user_id)

        # Store current user ID for calendar generation
        self._current_user_id = user_id

        # # Get working memory for context and add user message
        # try:
        #     working_memory = await self._get_working_memory(
        #         session_id=ctx.session_id,
        #         user_id=user_id
        #     )

        #     # Add user message to working memory
        #     await self._add_message_to_working_memory(
        #         session_id=ctx.session_id,
        #         user_id=user_id,
        #         role="user",
        #         content=user_message
        #     )
        # except Exception as e:
        #     print(f"⚠️ Failed to initialize working memory: {e}")
        #     # Continue without memory if there's an error

        def _html(icon: str, title: str, message: str) -> str:
            safe_icon = icon or ""
            safe_title = title or ""
            safe_msg = message or ""
            return (
                f"<div class='event-card'>"
                f"<div class='event-title'>{safe_icon} {safe_title}</div>"
                f"<div class='event-message'>{safe_msg}</div>"
                f"</div>"
            )

        def _event(event_type: str, icon: str, title: str, message: str) -> dict:
            return {
                "type": event_type,
                "html": _html(icon, title, message),
            }

        stream = ctx.agent.run_stream(task=user_message)

        buffer = ""
        last_yielded = ""
        emitted_stream_start = False
        llm_call_index = 0

        async for event in stream:
            event_name = event.__class__.__name__

            def _get_model_name(ev) -> Optional[str]:
                # Try multiple places for model name; fall back to supervisor's model
                val = getattr(ev, "model", None)
                if isinstance(val, str) and val:
                    return val
                meta = getattr(ev, "metadata", None)
                if isinstance(meta, dict) and meta.get("model"):
                    return meta.get("model")
                if hasattr(meta, "model") and getattr(meta, "model"):
                    return getattr(meta, "model")
                sup_client = getattr(ctx.agent, "model_client", None)
                if sup_client is not None:
                    return getattr(sup_client, "model", None)
                return None

            if isinstance(event, ModelClientStreamingChunkEvent):
                chunk = getattr(event, "content", "") or ""
                if chunk:
                    buffer += chunk
                    if not emitted_stream_start:
                        emitted_stream_start = True
                        llm_call_index += 1
                        yield (
                            buffer,
                            _event(
                                "llm_token_stream_start",
                                "⏳",
                                f"LLM #{llm_call_index}: streaming",
                                f"Model: {_get_model_name(event) or ''}",
                            ),
                        )
                    if buffer != last_yielded:
                        last_yielded = buffer
                        # No event for regular token chunks to reduce UI churn
                        yield (
                            buffer + ' <span class="thinking-animation">●●●</span>',
                            None,
                        )

            elif isinstance(event, ToolCallRequestEvent):
                # content: List[FunctionCall]
                calls = getattr(event, "content", None) or []
                tool_names = []
                tool_args = None
                try:
                    for c in calls:
                        if tool_args is None:
                            tool_args = getattr(c, "arguments", None)
                        name = getattr(c, "name", None)
                        if name:
                            tool_names.append(name)
                except Exception:
                    pass
                tool_name = ", ".join(tool_names) or "tool"
                icon = "🔧"
                title = f"Calling {tool_name}"
                if tool_name == "search_logistics":
                    icon = "✈️"
                    title = "Searching logistics"
                elif tool_name == "search_general":
                    icon = "📍"
                    title = "Searching (general)"
                elif tool_name == "generate_calendar_ics":
                    icon = "📅"
                    title = "Generating calendar"
                elif any(
                    name in tool_names
                    for name in [
                        "search_memory",
                        "add_memory_to_working_memory",
                        "get_or_create_working_memory",
                        "update_working_memory_data",
                    ]
                ):
                    icon = "🧠"
                    title = "Accessing memory"
                yield (
                    buffer,
                    _event(
                        "tool_call",
                        icon,
                        title,
                        f"Invoking {tool_name}",
                    ),
                )

            elif event_name == "ModelClientRequestEvent":
                # Any model request (covers non-streaming LLM calls like memory/tool)
                llm_call_index += 1
                model_name = _get_model_name(event)
                yield (
                    buffer + ' <span class="thinking-animation">● ● ●</span>',
                    _event(
                        "llm_request",
                        "📤",
                        f"LLM #{llm_call_index}: request",
                        f"Model: {model_name or ''}",
                    ),
                )

            elif event_name in (
                "ModelClientResponseEvent",
                "ModelClientStreamEndEvent",
                "ModelClientResponseDoneEvent",
            ):
                model_name = _get_model_name(event)
                yield (
                    buffer,
                    _event(
                        "llm_response_received",
                        "📥",
                        f"LLM #{llm_call_index}: response received",
                        f"Model: {model_name or 'OpenAI'}",
                    ),
                )

            elif isinstance(event, ToolCallRequestEvent):
                # content: List[FunctionCall] with name, arguments, id
                calls = getattr(event, "content", None) or []
                call_infos = []
                try:
                    for c in calls:
                        call_infos.append(
                            {
                                "name": getattr(c, "name", None),
                                "id": getattr(c, "id", None),
                                "arguments": getattr(c, "arguments", None),
                            }
                        )
                except Exception:
                    pass
                # Choose icon/title based on first call
                first_name = next(
                    (ci.get("name") for ci in call_infos if ci.get("name")), None
                )
                icon = "🛠️"
                title = "Tool requested"
                if first_name == "search_logistics":
                    icon = "✈️"
                    title = "Logistics search requested"
                elif first_name == "search_general":
                    icon = "📍"
                    title = "General search requested"
                elif first_name == "generate_calendar_ics":
                    icon = "📅"
                    title = "Calendar generation requested"
                yield (
                    buffer,
                    _event(
                        "tool_request",
                        icon,
                        title,
                        ", ".join(
                            [str(ci.get("name")) for ci in call_infos if ci.get("name")]
                        )
                        or "Tool call",
                    ),
                )

            elif isinstance(event, ToolCallExecutionEvent):
                # content: List[FunctionExecutionResult] with name, call_id, content, is_error
                results = getattr(event, "content", None) or []
                tool_name = "Tool"
                file_path = None
                try:
                    if results:
                        first_result = results[0]
                        tool_name = getattr(first_result, "name", None) or "Tool"

                        # Extract file_path for calendar generation
                        if tool_name == "generate_calendar_ics":
                            # Try multiple shapes for tool result payload
                            result_content = getattr(
                                first_result, "content", None
                            ) or getattr(first_result, "result", None)
                            # Case 1: dict payload
                            if isinstance(result_content, dict):
                                file_path = result_content.get("file_path")
                            # Case 2: list of dicts
                            elif isinstance(result_content, list):
                                for item in result_content:
                                    if isinstance(item, dict) and item.get("file_path"):
                                        file_path = item.get("file_path")
                                        break
                            # Case 3: stringified JSON or text containing a path
                            elif isinstance(result_content, str):
                                # Try JSON decode first
                                try:
                                    data = json.loads(result_content)
                                    if isinstance(data, dict):
                                        file_path = data.get("file_path")
                                except Exception:
                                    pass
                                # Fallback: regex search for any .ics path
                                if not file_path:
                                    try:
                                        m = re.search(
                                            r"(/[^\s\"]+\.ics)", result_content
                                        )
                                        if m:
                                            file_path = m.group(1)
                                    except Exception:
                                        pass
                except Exception:
                    pass

                # Set icon based on tool name
                icon = "✅"
                if tool_name == "search_logistics":
                    icon = "✈️"
                elif tool_name == "search_general":
                    icon = "📍"
                elif tool_name == "generate_calendar_ics":
                    icon = "📅"
                elif any(
                    name in tool_name
                    for name in [
                        "search_memory",
                        "add_memory_to_working_memory",
                        "get_or_create_working_memory",
                        "update_working_memory_data",
                    ]
                ):
                    icon = "🧠"

                event_data = {
                    "type": "tool_result",
                    "html": _html(
                        icon, f"{tool_name} finished", "Tool execution completed"
                    ),
                    "tool_name": tool_name,
                }

                # Add file_path for calendar generation
                if file_path:
                    event_data["file_path"] = file_path

                yield buffer, event_data

            elif isinstance(event, MemoryQueryEvent):
                # content: List[MemoryContent]
                mems = getattr(event, "content", None) or []
                insights = []
                try:
                    for m in mems:
                        insights.append(getattr(m, "content", None) or str(m))
                except Exception:
                    pass
                yield (
                    buffer,
                    _event(
                        "memory_injected",
                        "🧠",
                        "Insights applied",
                        "\n".join([f"• {i}" for i in insights if i])
                        or "Memory context injected",
                    ),
                )

            elif isinstance(event, UserInputRequestedEvent):
                yield (
                    buffer,
                    _event(
                        "user_input_requested",
                        "⌛",
                        "Awaiting input",
                        "Agent requested user input",
                    ),
                )

            elif isinstance(event, ThoughtEvent):
                yield (
                    buffer,
                    _event(
                        "thought",
                        "💭",
                        "Agent thought",
                        getattr(event, "content", "") or "",
                    ),
                )

            elif isinstance(event, SelectSpeakerEvent):
                speakers = getattr(event, "content", None) or []
                try:
                    speakers = list(speakers)
                except Exception:
                    speakers = [str(speakers)] if speakers else []
                yield (
                    buffer,
                    _event(
                        "select_speaker",
                        "🎙️",
                        "Speaker selected",
                        ", ".join([str(s) for s in speakers]) or "",
                    ),
                )

            elif isinstance(event, CodeGenerationEvent):
                msg = getattr(event, "content", "") or ""
                blocks = getattr(event, "code_blocks", None)
                yield (
                    buffer,
                    _event(
                        "code_generated",
                        "🧩",
                        "Code generated",
                        msg,
                    ),
                )

            elif isinstance(event, CodeExecutionEvent):
                result = getattr(event, "result", None)
                output = None
                exit_code = None
                if result is not None:
                    output = getattr(result, "output", None)
                    exit_code = getattr(result, "exit_code", None)
                yield (
                    buffer,
                    _event(
                        "code_executed",
                        "▶️",
                        "Code executed",
                        (output[:500] + "…")
                        if isinstance(output, str) and len(output) > 500
                        else (output or ""),
                    ),
                )

            elif isinstance(
                event,
                (
                    ToolCallSummaryMessage,
                    MultiModalMessage,
                    HandoffMessage,
                    StopMessage,
                ),
            ):
                # General message-like events
                summary_content = getattr(event, "content", "") or ""
                yield (
                    buffer,
                    _event(
                        event.__class__.__name__,
                        "ℹ️",
                        event.__class__.__name__,
                        summary_content,
                    ),
                )

            elif isinstance(event, TextMessage):
                source = getattr(event, "source", "") or ""
                content = getattr(event, "content", "") or ""
                icon = "👤" if source == "user" else "🤖"
                title = f"{source.title()} message" if source else "Message"
                yield (
                    buffer,
                    _event(
                        "text_message",
                        icon,
                        title,
                        "",  # Keep message empty for concise logs
                    ),
                )

            elif hasattr(event, "messages") and hasattr(event, "stop_reason"):
                # Final TaskResult
                yield (
                    buffer,
                    _event(
                        "run_complete",
                        "🏁",
                        "Run complete",
                        f"Stop reason: {getattr(event, 'stop_reason', '')}",
                    ),
                )

            else:
                # Catch-all for unknown events
                yield (
                    buffer + ' <span class="thinking-animation">● ● ●</span>',
                    _event(
                        "unknown_event",
                        "❓",
                        f"Unknown: {event_name}",
                        "Unhandled event type encountered.",
                    ),
                )

        # Final yield to ensure thinking animation is removed
        if buffer:
            yield buffer, None

    # -----------------
    # Utility Methods
    # -----------------

    async def store_assistant_response(self, user_id: str, response: str) -> None:
        """Store assistant response in working memory after streaming completes."""
        try:
            ctx = self._get_or_create_user_ctx(user_id)
            print(f"👉👉👉 Context: {ctx}")
            await self._add_message_to_working_memory(
                session_id=ctx.session_id,
                user_id=user_id,
                role="assistant",
                content=response,
            )
        except Exception as e:
            print(f"⚠️ Failed to store assistant response: {e}")

    async def cleanup(self):
        """Clean up resources."""
        if self._memory_client:
            await self._memory_client.close()
            print("Memory client closed")

    async def get_chat_history(
        self, user_id: str, n: Optional[int] = None
    ) -> List[Dict[str, str]]:
        """Retrieve chat history from Agent Memory Server working memory.

        Converts working memory messages to Gradio-compatible format,
        filtering for user and assistant messages with text content.

        Args:
            user_id: User identifier to get history for
            n: Number of messages to retrieve. If None, retrieves all.
               If -1, retrieves all messages.

        Returns:
            List[Dict[str, str]]: Message dictionaries with 'role' and 'content'
                                keys suitable for Gradio chat interface
        """
        ctx = self._get_or_create_user_ctx(user_id)
        working_memory = await self._get_working_memory(
            session_id=ctx.session_id, user_id=user_id
        )

        messages = working_memory.messages
        if n and n > 0:
            messages = messages[-n:]

        # Convert to Gradio format
        return [
            {"role": msg.role, "content": msg.content}
            for msg in messages
            if msg.role in ["user", "assistant"]
        ]

    def user_exists(self, user_id: str) -> bool:
        """Check if a user context exists in the cache.

        Args:
            user_id: User identifier to check

        Returns:
            bool: True if user context exists, False otherwise
        """
        return user_id in self._user_ctx_cache

    def reset_user_memory(self, user_id: str) -> None:
        """Reset a user's memory by removing their cached context.

        This clears the user's cached context and forces recreation of
        a fresh agent instance on next interaction. Note that Agent Memory Server
        long-term memories are persistent and not cleared by this method.

        Args:
            user_id: User identifier whose memory should be reset
        """
        if user_id in self._user_ctx_cache:
            print(f"🗑️  Resetting cached context for user: {user_id}")
            self._user_ctx_cache.pop(user_id, None)

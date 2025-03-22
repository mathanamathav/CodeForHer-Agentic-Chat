from datetime import datetime
from typing import Literal

from langchain_community.tools import (DuckDuckGoSearchResults,
                                       OpenWeatherMapQueryRun)
from langchain_community.utilities import OpenWeatherMapAPIWrapper
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, SystemMessage
from langchain_core.runnables import (RunnableConfig, RunnableLambda,
                                      RunnableSerializable)
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, MessagesState, StateGraph
from langgraph.managed import RemainingSteps
from langgraph.prebuilt import ToolNode

from agents.llama_guard import LlamaGuard, LlamaGuardOutput, SafetyAssessment
from agents.tools import (calculator, current_location, nearby_places, route,
                          sos_alert)
from core import get_model, settings


class AgentState(MessagesState, total=False):
    """`total=False` is PEP589 specs."""

    safety: LlamaGuardOutput
    remaining_steps: RemainingSteps
    current_agent: str  # Track which agent is currently active
    first_run: bool  # Track if this is the first run


web_search = DuckDuckGoSearchResults(name="WebSearch")
tools = [web_search, calculator, nearby_places, current_location, route, sos_alert]

current_date = datetime.now().strftime("%B %d, %Y")
base_instructions = f"""
    You are a supportive and empathetic travel safety companion for women. Today's date is {current_date}.
    
    Core principles:
    1. Always maintain a friendly, supportive, and respectful tone
    2. Prioritize safety and well-being in all responses
    3. Be sensitive to the user's concerns and emotions
    4. Provide practical, actionable safety advice
    5. Use clear, simple language
    6. Never make assumptions about the user's situation
    7. Always offer to help further if needed
"""

location_agent_instructions = (
    base_instructions
    + """
    You are the Location Safety Agent. Your role is to:
    1. Help users understand the safety profile of their destination
    2. Provide specific safety tips for the location
    3. Share information about safe areas and areas to avoid
    4. Suggest safe transportation options
    5. Recommend emergency contacts and resources
    6. Be mindful of cultural context and local customs
"""
)

emergency_agent_instructions = (
    base_instructions
    + """
    You are the Emergency Response Agent. Your role is to:
    1. Provide immediate guidance in emergency situations
    2. Share emergency contact numbers and resources
    3. Offer step-by-step safety protocols
    4. Maintain calm and clear communication
    5. Guide users to safe locations or help
    6. Connect users with appropriate emergency services
"""
)

companion_agent_instructions = (
    base_instructions
    + """
    You are the Travel Companion Agent. Your role is to:
    1. Engage in friendly, supportive conversation
    2. Offer emotional support and reassurance
    3. Share general travel tips and best practices
    4. Help users plan safe travel routes
    5. Suggest safety-focused travel companions
    6. Provide check-in reminders and safety checklists
"""
)

nearby_places_agent_instructions = (
    base_instructions
    + """
    You are the Nearby Places Agent. Your role is to:
    1. Provide information about nearby safe places
    2. Share information about safe areas and areas to avoid
    """
)

get_route_agent_instructions = (
    base_instructions
    + """
    You are the route Agent. Your role is to:
    1. route with start and destination inputs without country , state , pincode just city and area name is enough
    2. provide the route with the best safety tips and information along the routes details
    """
)


def wrap_model(
    model: BaseChatModel, agent_type: str
) -> RunnableSerializable[AgentState, AIMessage]:
    model = model.bind_tools(tools)
    instructions = {
        "location": location_agent_instructions,
        "emergency": emergency_agent_instructions,
        "companion": companion_agent_instructions,
        "nearby_places": nearby_places_agent_instructions,
        "get_route": get_route_agent_instructions,
    }.get(agent_type, base_instructions)

    preprocessor = RunnableLambda(
        lambda state: [SystemMessage(content=instructions)] + state["messages"],
        name="StateModifier",
    )
    return preprocessor | model


def format_safety_message(safety: LlamaGuardOutput) -> AIMessage:
    content = f"This conversation was flagged for unsafe content: {', '.join(safety.unsafe_categories)}"
    return AIMessage(content=content)


def determine_agent(state: AgentState) -> Literal["location", "emergency", "companion"]:
    """Determine which agent should handle the conversation based on the user's message."""
    last_message = state["messages"][-1].content.lower()

    # Emergency keywords
    emergency_keywords = [
        "emergency",
        "help",
        "danger",
        "unsafe",
        "threat",
        "attack",
        "harassment",
    ]
    if any(keyword in last_message for keyword in emergency_keywords):
        return "emergency"

    # Location keywords
    location_keywords = [
        "location",
        "place",
        "area",
        "destination",
        "neighborhood",
        "region",
    ]
    if any(keyword in last_message for keyword in location_keywords):
        return "location"

    # Default to companion agent
    return "companion"


async def supervisor(state: AgentState, config: RunnableConfig) -> AgentState:
    """Supervisor node that manages the conversation flow and handles responses."""
    # Initialize the conversation on first run
    if not state.get("messages") or state.get("first_run", True):
        return {
            "messages": [
                AIMessage(
                    content="""Hello! I'm your travel safety companion. I'm here to support you and help ensure your safety during your travels. 
                    I can help you with:
                    - Location safety information and tips
                    - Emergency guidance and resources
                    - General travel safety advice and companionship
                    
                    How can I assist you today? Feel free to ask any questions about your travel safety concerns."""
                )
            ],
            "current_agent": "companion",
            "first_run": False,
        }

    # Determine which agent should handle the message
    current_agent = determine_agent(state)
    state["current_agent"] = current_agent

    # Get model and generate response
    m = get_model(config["configurable"].get("model", settings.DEFAULT_MODEL))
    model_runnable = wrap_model(m, current_agent)
    response = await model_runnable.ainvoke(state, config)

    # Run llama guard check to avoid returning the message if it's unsafe
    llama_guard = LlamaGuard()
    safety_output = await llama_guard.ainvoke("Agent", state["messages"] + [response])
    if safety_output.safety_assessment == SafetyAssessment.UNSAFE:
        return {
            "messages": [format_safety_message(safety_output)],
            "safety": safety_output,
        }

    if state["remaining_steps"] < 2 and response.tool_calls:
        return {
            "messages": [
                AIMessage(
                    id=response.id,
                    content="I apologize, but I need more steps to process this request. Would you like me to help you with something else?",
                )
            ]
        }

    return {"messages": [response]}


# Define the routing logic
def route_next(state: AgentState) -> Literal["supervisor", "tools", "done"]:
    last_message = state["messages"][-1]
    if not isinstance(last_message, AIMessage):
        return "supervisor"

    if last_message.tool_calls:
        return "tools"
    return "done"


# Define the graph
agent = StateGraph(AgentState)
agent.add_node("supervisor", supervisor)
agent.add_node("tools", ToolNode(tools))
agent.set_entry_point("supervisor")
agent.add_conditional_edges(
    "supervisor",
    route_next,
    {"supervisor": "supervisor", "tools": "tools", "done": END},
)
agent.add_edge("tools", "supervisor")

research_assistant = agent.compile(checkpointer=MemorySaver())

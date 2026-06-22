from typing import TypedDict, Annotated
from langchain_core.messages import AnyMessage, SystemMessage
from langchain_openai import ChatOpenAI
from langgraph.graph.message import add_messages

# Import the custom tools we defined inside our agent directory
from agent.tools import (
    get_n_random_words
)


# =====================================================================
# 1. STATE DEFINITION
# =====================================================================
# The agent state acts like the short-term memory of the agent.
# It tracks the ongoing conversation history. 'add_messages' ensures
# that new messages are appended to the list rather than overwriting it.
class AgentState(TypedDict):
    messages: Annotated[list[AnyMessage], add_messages]


# =====================================================================
# 2. TOOLS CONFIGURATION
# =====================================================================
# These are the additional capabilities the agent can use to achieve a goal.
local_tools = [
    get_n_random_words,
]


# =====================================================================
# 3. ASSISTANT NODE (The Central Brain)
# =====================================================================
# This acts like the central planner of the agent, allowing the LLM
# to decompose a problem, evaluate the steps already carried out, and
# select which tools to use.
def assistant(state: AgentState):
    # Define the core identity and guardrails for the model
    sys_msg = SystemMessage(content=f"""
        You are a helpful language learning assistant.

        The user is going to give you a command.
    """)

    # This line checks if this function node has been decorated or bound with tools.
    # It reads from 'assistant.tools' if it exists, ensuring the graph knows which
    # capabilities to expose to the LLM core during execution.
    tools = assistant.tools if hasattr(assistant, 'tools') else []

    # Initialize the primary reasoning LLM engine (GPT-4o)
    llm = ChatOpenAI(model="gpt-4o")

    # Bind the allowed tools directly to the model configuration.
    # We set parallel_tool_calls=False to force the agent to reason step-by-step
    # rather than firing off multiple external tool commands at the exact same time.
    llm_with_tools = llm.bind_tools(tools, parallel_tool_calls=False)

    # Invoke the model by feeding it the structural system message followed by
    # the entire historical chat conversation state, then return the updated message list.
    return {
        "messages": [llm_with_tools.invoke([sys_msg] + state["messages"])]
    }
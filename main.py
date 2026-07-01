import asyncio
from typing import TypedDict, Annotated, Optional
from langchain_core.messages import AnyMessage, SystemMessage
#from langchain_openai import ChatOpenAI
from langchain_ollama import ChatOllama
from langgraph.graph.message import add_messages
from langgraph.graph import StateGraph, START
from langgraph.prebuilt import ToolNode, tools_condition
from dotenv import load_dotenv
from langchain_mcp_adapters.client import MultiServerMCPClient

# Import the custom tools we defined inside our agent directory
from agent.tools import (
    get_n_random_words,
    get_n_random_words_by_difficulty_level,
    translate_words
)

CLANKI_JS = r"C:\Users\nasri\source\PyCharmProject\clanki\build\index.js"
# =====================================================================
# 1. STATE DEFINITION
# =====================================================================
# The agent state acts like the short-term memory of the agent.
# It tracks the ongoing conversation history. 'add_messages' ensures
# that new messages are appended to the list rather than overwriting it.
class AgentState(TypedDict):
    messages: Annotated[list[AnyMessage], add_messages]
    source_language: Optional[str]
    number_of_words: Optional[int]
    word_difficulty: Optional[str]
    target_language: Optional[str]
    deck_name: Optional[str]


# =====================================================================
# 2. TOOLS CONFIGURATION
# =====================================================================
# These are the additional capabilities the agent can use to achieve a goal.
# Cache container and asynchronous lock to prevent multiple concurrent initializations
_cached_tools = None
_tools_lock = asyncio.Lock()
local_tools = [
    get_n_random_words,
    get_n_random_words_by_difficulty_level,
    translate_words
]


async def setup_tools():
    global _cached_tools

    # Fast path: Return cached tools immediately
    if _cached_tools is not None:
        return _cached_tools

    async with _tools_lock:
        if _cached_tools is not None:
            return _cached_tools

        client = MultiServerMCPClient(
            {
                "clanki": {
                    "command": "node",
                    "args": [CLANKI_JS],
                    "transport": "stdio"
                }
            }
        )
        raw_mcp_tools = await client.get_tools()

        _cached_tools = [*local_tools, *raw_mcp_tools]
        return _cached_tools

# =====================================================================
# 3. ASSISTANT NODE (The Central Brain)
# =====================================================================

# This acts like the central planner of the agent, allowing the LLM
# to decompose a problem, evaluate the steps already carried out, and
# select which tools to use.
def assistant(state: AgentState):
    textual_description_of_tools = """
    get_n_random_words(language: str, n: int) -> list
    Dynamically generates a specified number of highly common, everyday
    conversational words in a target language using a local LLM.

    get_n_random_words_by_difficulty_level(language: str, difficulty_level: str, n: int) -> list
    Dynamically generates everyday conversational words in a target language,
    filtered strictly by the requested difficulty level
    ('beginner', 'intermediate', or 'advanced').

    translate_words(random_words: list, source_language: str, target_language: str) -> list
    Translates a list of words from a source language to a target language,
    returning a list of {"original": ..., "translation": ...} pairs in the
    same order as the input.
    """

    # Define the core identity and guardrails for the model
    sys_msg = SystemMessage(content=f"""
    You are an expert language learning assistant tightly integrated with Anki via MCP tools.
    You have access to the following tools:
    {textual_description_of_tools}

    YOUR ABSOLUTE CORE TASK IS TO EXECUTE THIS MULTI-STEP WORKFLOW IN A SINGLE RUN:
    1. Generate the required words using your vocabulary generation tool.
    2. If translation is required, pass those words to the translation tool.
    3. MANDATORY: Call `create-deck` with the Deck Name '{state.get('deck_name', 'Language::Vocabulary')}'.
    4. MANDATORY: ONLY AFTER calling `create-deck` and receiving confirmation, proceed to call `create-card` for every single word pair. 

    CRITICAL STATE DATA:
    - Target Deck Name: {state.get('deck_name', 'Language::Vocabulary')}
    - Number of Words: {state.get('number_of_words', '5')}
    - Source Language: {state.get('source_language', 'English')}
    - Target Language: {state.get('target_language', 'None')}
    - Difficulty: {state.get('word_difficulty', 'any')}

    CRITICAL EXECUTION RULES:
    - Chain these tool calls automatically back-to-back. Do not stop until all cards are created.
    - Always prefer using the dynamic LLM-based vocabulary tools (`get_n_random_words` or `get_n_random_words_by_difficulty_level`) over file-reading tools.
    - STRICT DECK ENFORCEMENT: Never add a card without verifying the target deck exists. If adding a card fails, stop and re-create the deck.
    - MANDATORY DUPLICATE HANDLING: You will likely encounter an 'AnkiConnect error: cannot create note because it is a duplicate' error. This is EXPECTED. When you see this error:
    1. DO NOT stop the process.
    2. DO NOT interpret this as a failure of your entire task.
    3. IMMEDIATELY move to the next word in your generated list and attempt to add it.
    4. Continue your loop until you have processed EVERY word in your generated list, regardless of how many duplicates you skip.
    - SELF-VERIFICATION: Before you send any tool call, verify in your own internal thought process that the `deckName` argument matches '{state.get('deck_name', 'Language::Vocabulary')}'.

        Here are some example workflows to follow:
        input: Get 20 random words in Spanish.
        source language: Spanish
        number of words: 20

        input: Get 10 hard words in German.
        source language: German
        number of words: 10
        word difficulty: advanced

        input: Get 15 random words in English and translate them to Spanish.
        source language: English
        number of words: 15
        word difficulty: beginner
        target language: Spanish

        input: Get 20 easy words in Spanish, translate them to English, and create a new Anki deck with them called Spanish::Easy
        source language: Spanish
        target language: English
        number of words: 20
        word difficulty: beginner
        tools workflow : get_n_random_words_by_difficulty_level -> translate_words -> create-deck -> create-card

        input: Get 20 easy words in Spanish, translate them to English, and create a new Anki deck with them called Spanish::Easy
        source language: Spanish
        target language: English
        number of words: 20
        word difficulty: beginner
        tools workflow : get_n_random_words -> create-deck -> create-card

        FORMATTING RULES:
        - Always put the numbered words to the left.
        - Keep definitions, meanings, and examples on completely separate lines (do not put them on the same line as the word).
        """)

    # This line checks if this function node has been decorated or bound with tools.
    # It reads from 'assistant.tools' if it exists, ensuring the graph knows which
    # capabilities to expose to the LLM core during execution.
    tools = assistant.tools if hasattr(assistant, 'tools') else []


    # Initialise the primary reasoning LLM engine (GPT-4o) -  not free :-\
    #llm = ChatOpenAI(model="gpt-4o")

    llm = ChatOllama(
        model= "qwen3:8b",
        temperature= 0
    )

    # Bind the allowed tools directly to the model configuration.
    # We set parallel_tool_calls=False to force the agent to reason step-by-step
    # rather than firing off multiple external tool commands at the exact same time.
    llm_with_tools = llm.bind_tools(tools)

    # Invoke the model by feeding it the structural system message followed by
    # the entire historical chat conversation state, then return the updated message list.
    return {
        "messages": [llm_with_tools.invoke([sys_msg] + state["messages"])],
        "source_language": state ["source_language"],
        "number_of_words": state["number_of_words"],
        "word_difficulty": state["word_difficulty"],
        "target_language": state.get("target_language"),
        "deck_name": state.get("deck_name")

    }

# =====================================================================
# 4-GRAPH BUILDING & COMPILATION
# =====================================================================

async def build_graph():

    # Dynamically bind our tool array onto the function object itself.
    # This fulfills the 'hasattr(assistant, "tools")' check inside the node.
    tools = await setup_tools()
    assistant.tools = tools

    # Initialise the graph builder configuration with our strict message state schem
    # noinspection PyTypeChecker
    builder = StateGraph(AgentState)

    # Define our two processing checkpoints (Nodes)
    # noinspection PyTypeChecker
    builder.add_node("assistant", assistant)
    builder.add_node("tools", ToolNode(tools))

    # Define execution paths (Edges)
    # Entry Point: Send the conversation directly into the assistant brain first
    builder.add_edge(START, "assistant")

    # Conditional Routing: After the LLM runs, check the output payload.
    # If the LLM requested a tool invocation, route to the "tools" node.
    # If the LLM provided a direct answer to the user, stop and return the output.
    builder.add_conditional_edges(
         "assistant",
        tools_condition
    )

    # Loopback Edge: Once a tool executes and fetches data, route back to
    # the assistant so it can interpret the results and speak to the user.
    builder.add_edge( "tools",  "assistant")

    # Lock down the structure into a compiled executable application loop
    # and attach the default execution configuration (increasing recursion limit to 50)
    return builder.compile().with_config({"recursion_limit": 50})


if __name__ == "__main__":
    from langchain_core.messages import HumanMessage

    async def visualise_and_test():
        print("🎉 Graph compiling...")
        try:
            # 1. Compile the graph
            app = await build_graph()

            # 2. Generate and save your flowchart visualisation
            image_data = app.get_graph().draw_mermaid_png()
            with open("graph_flowchart.png", "wb") as f:
                f.write(image_data)
            print("💾 Success! Open 'graph_flowchart.png' in your project tree.")

            print("\n🚀 Running interactive graph execution test...")

            # 3. Dynamically capture your request directly from the terminal console
            user_prompt = input("Type your request (e.g., 'Give me 5 French beginner words'): ")

            messages = [HumanMessage(content=user_prompt)]

            # 4. Invoke the graph asynchronously with your custom initial state map
            result = await app.ainvoke({ # type: ignore
                "messages": messages,
                "source_language": "",
                "number_of_words": 0,
                "word_difficulty": "",
                "target_language":""
            })

            # 5. Extract and display the terminal output response from the graph agent
            print("\n📥 Agent Response:")
            print(result["messages"][-1].content)

        except Exception as e:
            print(f"⚠️ An error occurred during execution: {e}")

    asyncio.run(visualise_and_test())

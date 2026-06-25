import asyncio

from typing import TypedDict, Annotated, Optional
from langchain_core.messages import AnyMessage, SystemMessage
#from langchain_openai import ChatOpenAI
from langchain_ollama import ChatOllama
from langgraph.graph.message import add_messages
from langgraph.graph import StateGraph, START
from langgraph.prebuilt import ToolNode, tools_condition
from dotenv import load_dotenv


# Import the custom tools we defined inside our agent directory
from agent.tools import (
    get_n_random_words,
    get_n_random_words_by_difficulty_level
)

load_dotenv()

# =====================================================================
# 1. STATE DEFINITION
# =====================================================================
# The agent state acts like the short-term memory of the agent.
# It tracks the ongoing conversation history. 'add_messages' ensures
# that new messages are appended to the list rather than overwriting it.
class AgentState(TypedDict):
    messages: Annotated[list[AnyMessage], add_messages]
    source_language: Optional[str]
    number_of_words: Optional[str]
    word_difficulty: Optional[str]


# =====================================================================
# 2. TOOLS CONFIGURATION
# =====================================================================
# These are the additional capabilities the agent can use to achieve a goal.
local_tools = [
    get_n_random_words,
    get_n_random_words_by_difficulty_level
]

# This mimics the setup to cleanly bridge tools onto the node attributes.
async def setup_tools():
    return [*local_tools]

# =====================================================================
# 3. ASSISTANT NODE (The Central Brain)
# =====================================================================

# This acts like the central planner of the agent, allowing the LLM
# to decompose a problem, evaluate the steps already carried out, and
# select which tools to use.
def assistant(state: AgentState):
    tesxtual_description_of_tools = """
    Selects a specified number of random words from a language-specific word list.

    The function reads a JSON file containing words for the specified language from
    a predefined directory. It then selects `n` random words from the file and
    returns them in a list.

    :param language: A string representing the language for which to fetch the word list.
    :param n: An integer specifying the number of random words to retrieve.
    :return: A list containing `n` randomly selected words.
    
    def get_n_random_words_by_difficulty_level(language: str,
                                           difficulty_level: str,
                                           n:int) -> list:
    
    Retrieves a specified number of random words filtered by a given difficulty level
    from a word list corresponding to a specific language. The function reads the
    word list from a JSON file located in the directory `data/{language}/word-list-cleaned.json`.

    :param language: The language of the word list to be used.
    :type language: str
    :param difficulty_level: The difficulty level to filter words by. Possible values
        depend on the data structure in the JSON file.The only valid values are 'beginner',
        'intermediate', and 'advanced'.
    :type difficulty_level: str
    :param n: The number of random words to retrieve.
    :type n: int
    :return: A list containing `n` random words filtered by the specified difficulty level.
    :rtype: list
    """

    # Define the core identity and guardrails for the model
    sys_msg = SystemMessage(content=f"""
    You are a helpful language learning assistant. You have access to the following tools
        :{tesxtual_description_of_tools}

    CRITICAL REQUIREMENT: You must pull exactly {state.get('number_of_words', '5')} words 
    for the language: {state.get('source_language', 'English')} with a difficulty level of: {state.get('word_difficulty', 'any')}.
    Ignore any conflicting numbers, languages, or difficulties mentioned in the human's chat message.

   FORMATTING RULES:
  - Always put the numbered words to the left.
  - Keep definitions, meanings, and examples on completely separate lines (do not put them on the same line as the word).
""")

    # This line checks if this function node has been decorated or bound with tools.
    # It reads from 'assistant.tools' if it exists, ensuring the graph knows which
    # capabilities to expose to the LLM core during execution.
    #tools = assistant.tools if hasattr(assistant, 'tools') else []

    tools = local_tools

    # Initialise the primary reasoning LLM engine (GPT-4o) -  not free :-\
    #llm = ChatOpenAI(model="gpt-4o")

    llm = ChatOllama(
        model= "qwen3:8b",
        temperature= 0
    )

    # Bind the allowed tools directly to the model configuration.
    # We set parallel_tool_calls=False to force the agent to reason step-by-step
    # rather than firing off multiple external tool commands at the exact same time.

   # llm_with_tools = llm.bind_tools(tools, parallel_tool_calls=False)

    llm_with_tools = llm.bind_tools(tools)

    # Invoke the model by feeding it the structural system message followed by
    # the entire historical chat conversation state, then return the updated message list.
    return {
        "messages": [llm_with_tools.invoke([sys_msg] + state["messages"])],
        "source_language": state ["source_language"],
        "number_of_words": state["number_of_words"],
        "number_difficulty": state["word_difficulty"]
    }

# =====================================================================
# 4-GRAPH BUILDING & COMPILATION
# =====================================================================

async def build_graph():

    # Dynamically bind our tool array onto the function object itself.
    # This fulfills the 'hasattr(assistant, "tools")' check inside the node.
    #tools = await setup_tools()
    #assistant.tools = tools

    # Initialise the graph builder configuration with our strict message state schem
    builder = StateGraph(AgentState)

    # Define our two processing checkpoints (Nodes)
    builder.add_node("assistant", assistant)
    builder.add_node("tools", ToolNode(local_tools))

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
    return builder.compile()


if __name__ == "__main__":
    import asyncio
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
                "word_difficulty": ""
            })

            # 5. Extract and display the terminal output response from the graph agent
            print("\n📥 Agent Response:")
            print(result["messages"][-1].content)

        except Exception as e:
            print(f"⚠️ An error occurred during execution: {e}")

    asyncio.run(visualise_and_test())

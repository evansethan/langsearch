import streamlit as st
import os
from typing import TypedDict
try:
    from typing import Annotated
except ImportError:
    from typing_extensions import Annotated
from langgraph.graph import StateGraph, END
from langgraph.graph.message import add_messages
from langchain_openai import ChatOpenAI
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_core.messages import HumanMessage, AIMessage, ToolMessage, BaseMessage

os.environ["OPENAI_API_KEY"] = st.secrets["OPENAI_API_KEY"]
os.environ["TAVILY_API_KEY"] = st.secrets["TAVILY_API_KEY"]

class AgentState(TypedDict):
    messages: Annotated[list, add_messages]

@st.cache_resource
def get_agent():
    def calculator(expression: str) -> str:
        """A simple calculator that evaluates a string expression."""
        try:
            allowed_chars = "0123456789+-*/.() "
            if all(c in allowed_chars for c in expression):
                return str(eval(expression))
            else:
                return "Error: Invalid characters in expression."
        except Exception as e:
            return f"Error evaluating expression: {e}"

    tavily = TavilySearchResults(max_results=3)
    # Manually create a dictionary for the calculator tool
    calculator_tool_dict = {
        "name": "calculator",
        "description": "A simple calculator that evaluates a string expression.",
        "parameters": {
            "type": "object",
            "properties": {
                "expression": {
                    "type": "string",
                    "description": "The mathematical expression to evaluate."
                }
            },
            "required": ["expression"]
        }
    }
    llm = ChatOpenAI(model="gpt-4o").bind_tools([tavily, calculator_tool_dict])
    
    def agent(state):
        return {"messages": [llm.invoke(state["messages"])]}
    
    def tools(state):
        tool_calls = state["messages"][-1].tool_calls
        tool_messages = []
        for tc in tool_calls:
            if tc["name"] == "tavily_search_results_json":
                result = tavily.invoke(tc["args"])
            elif tc["name"] == "calculator":
                result = calculator(tc["args"]["expression"])
            tool_messages.append(ToolMessage(content=str(result), tool_call_id=tc["id"]))
        return {"messages": tool_messages}
    
    def route(state):
        return "tools" if state["messages"][-1].tool_calls else "end"
    
    g = StateGraph(AgentState)
    g.add_node("agent", agent)
    g.add_node("tools", tools)
    g.set_entry_point("agent")
    g.add_conditional_edges("agent", route, {"tools": "tools", "end": END})
    g.add_edge("tools", "agent")
    return g.compile()

st.title("AI Chat (Web Search and Calculator enabled)")

if "msgs" not in st.session_state:
    st.session_state.msgs = []

for m in st.session_state.msgs:
    st.chat_message(m["role"]).markdown(m["content"])

if q := st.chat_input("Ask..."):
    st.session_state.msgs.append({"role": "user", "content": q})
    st.chat_message("user").markdown(q)
    r = get_agent().invoke({"messages": [HumanMessage(content=q)]})
    a = [m.content for m in reversed(r["messages"]) if isinstance(m, AIMessage)][0]
    st.chat_message("assistant").markdown(a)
    st.session_state.msgs.append({"role": "assistant", "content": a})
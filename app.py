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
from langchain_core.messages import HumanMessage, AIMessage, ToolMessage

os.environ["OPENAI_API_KEY"] = st.secrets["OPENAI_API_KEY"]
os.environ["TAVILY_API_KEY"] = st.secrets["TAVILY_API_KEY"]

class AgentState(TypedDict):
    messages: Annotated[list, add_messages]

@st.cache_resource
def get_agent():
    tavily = TavilySearchResults(max_results=3)
    llm = ChatOpenAI(model="gpt-4o").bind_tools([tavily])
    
    def agent(state):
        return {"messages": [llm.invoke(state["messages"])]}
    
    def tools(state):
        return {"messages": [ToolMessage(content=str(tavily.invoke(tc["args"])),
                                         tool_call_id=tc["id"]) for tc in state["messages"][-1].tool_calls]}
    
    def route(state):
        return "tools" if state["messages"][-1].tool_calls else "end"
    
    g = StateGraph(AgentState)
    g.add_node("agent", agent)
    g.add_node("tools", tools)
    g.set_entry_point("agent")
    g.add_conditional_edges("agent", route, {"tools": "tools", "end": END})
    g.add_edge("tools", "agent")
    return g.compile()

st.title("LangGraph Search Agent")

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
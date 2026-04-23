from __future__ import annotations

from langgraph.graph import END, StateGraph

from app.agents.nodes.critic_agent import critic_agent
from app.agents.nodes.formatter_agent import formatter_agent
from app.agents.nodes.reasoning_agent import reasoning_agent
from app.agents.nodes.retrieval_agent import retrieval_agent
from app.agents.state import GraphState


def build_graph():
    graph = StateGraph(GraphState)
    graph.add_node("retrieval_agent", retrieval_agent)
    graph.add_node("reasoning_agent", reasoning_agent)
    graph.add_node("critic_agent", critic_agent)
    graph.add_node("formatter_agent", formatter_agent)

    graph.set_entry_point("retrieval_agent")
    graph.add_edge("retrieval_agent", "reasoning_agent")
    graph.add_edge("reasoning_agent", "critic_agent")
    graph.add_edge("critic_agent", "formatter_agent")
    graph.add_edge("formatter_agent", END)

    return graph.compile()

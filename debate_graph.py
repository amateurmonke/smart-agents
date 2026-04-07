from __future__ import annotations
from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver

from debate_engine import DebateState, HistoryEntry
from agents import DebateAgent
from adjudicator import Adjudicator


def build_debate_graph(
    agents: list[DebateAgent],
    adjudicator: Adjudicator,
    max_rounds: int,
):
    """
    Build and compile a LangGraph StateGraph for the debate.

    The graph topology is:
        opening → [rebuttal]* → closing → verdict → END

    Conditional routing after opening/rebuttal:
      - consensus reached OR current_round >= max_rounds  →  closing
      - otherwise                                         →  rebuttal (loop)

    Nodes close over *agents*, *adjudicator*, and *max_rounds* so they
    don't need to be serialised into the graph state.
    """

    # ------------------------------------------------------------------
    # Shared helpers
    # ------------------------------------------------------------------

    def _make_entry(agent: DebateAgent, content: str, round_num: int, round_type: str) -> HistoryEntry:
        return {
            "agent": agent.name,
            "persona": agent.persona,
            "content": content,
            "round": round_num,
            "round_type": round_type,
            "stance": agent.current_stance,
            "tool_calls": list(agent.last_tool_calls),
        }

    def _consensus(stances: dict[str, str]) -> bool:
        return len(set(stances.values())) == 1

    # ------------------------------------------------------------------
    # Nodes — each returns a *partial* state update dict
    # ------------------------------------------------------------------

    def opening_node(state: DebateState) -> dict:
        history = list(state["history"])
        stances = dict(state["agent_stances"])

        for agent in agents:
            text = agent.respond(state["topic"], history, "opening statement")
            history.append(_make_entry(agent, text, round_num=0, round_type="opening"))
            stances[agent.name] = agent.current_stance

        return {
            "history": history,
            "agent_stances": stances,
            "consensus_met": _consensus(stances),
        }

    def rebuttal_node(state: DebateState) -> dict:
        round_num = state["current_round"] + 1
        history = list(state["history"])
        stances = dict(state["agent_stances"])

        for agent in agents:
            text = agent.respond(state["topic"], history, "rebuttal")
            history.append(_make_entry(agent, text, round_num=round_num, round_type="rebuttal"))
            stances[agent.name] = agent.current_stance

        return {
            "history": history,
            "agent_stances": stances,
            "current_round": round_num,
            "consensus_met": _consensus(stances),
        }

    def closing_node(state: DebateState) -> dict:
        closing_round = state["current_round"] + 1
        history = list(state["history"])
        stances = dict(state["agent_stances"])

        for agent in agents:
            text = agent.respond(state["topic"], history, "closing statement")
            history.append(_make_entry(agent, text, round_num=closing_round, round_type="closing"))
            stances[agent.name] = agent.current_stance

        return {"history": history, "agent_stances": stances}

    def verdict_node(state: DebateState) -> dict:
        verdict = adjudicator.final_verdict(
            state["topic"],
            state["history"],
            [a.name for a in agents],
        )
        return {"final_verdict": verdict}

    # ------------------------------------------------------------------
    # Routing functions
    # ------------------------------------------------------------------

    def route_after_opening(state: DebateState) -> str:
        if max_rounds == 0 or state["consensus_met"]:
            return "closing"
        return "rebuttal"

    def route_after_rebuttal(state: DebateState) -> str:
        if state["consensus_met"] or state["current_round"] >= max_rounds:
            return "closing"
        return "rebuttal"

    # ------------------------------------------------------------------
    # Graph assembly
    # ------------------------------------------------------------------

    graph = StateGraph(DebateState)

    graph.add_node("opening", opening_node)
    graph.add_node("rebuttal", rebuttal_node)
    graph.add_node("closing", closing_node)
    graph.add_node("verdict", verdict_node)

    graph.set_entry_point("opening")

    graph.add_conditional_edges(
        "opening",
        route_after_opening,
        {"rebuttal": "rebuttal", "closing": "closing"},
    )
    graph.add_conditional_edges(
        "rebuttal",
        route_after_rebuttal,
        {"rebuttal": "rebuttal", "closing": "closing"},
    )
    graph.add_edge("closing", "verdict")
    graph.add_edge("verdict", END)

    return graph.compile(checkpointer=MemorySaver())

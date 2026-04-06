from __future__ import annotations
from typing import TypedDict
from agents import DebateAgent


class HistoryEntry(TypedDict):
    agent: str
    persona: str
    content: str
    round: int
    round_type: str
    stance: str


class DebateState(TypedDict):
    topic: str
    history: list[HistoryEntry]
    current_round: int
    agent_stances: dict[str, str]
    consensus_met: bool
    final_verdict: str


class DebateOrchestrator:
    def __init__(
        self,
        topic: str,
        agents: list[DebateAgent],
        max_rounds: int = 3,
    ) -> None:
        self.topic = topic
        self.agents = agents
        self.max_rounds = max_rounds
        self.state: DebateState = {
            "topic": topic,
            "history": [],
            "current_round": 0,
            "agent_stances": {a.name: a.initial_stance for a in agents},
            "consensus_met": False,
            "final_verdict": "",
        }

    # --- Phase runners ---

    def run_opening(self) -> list[tuple[DebateAgent, str]]:
        results = []
        for agent in self.agents:
            response = agent.respond(self.topic, self.state["history"], "opening statement")
            self._record(agent, response, round_num=0, round_type="opening")
            results.append((agent, response))
        self._check_consensus()
        return results

    def run_rebuttal(self, round_num: int) -> list[tuple[DebateAgent, str]]:
        results = []
        for agent in self.agents:
            response = agent.respond(self.topic, self.state["history"], "rebuttal")
            self._record(agent, response, round_num=round_num, round_type="rebuttal")
            results.append((agent, response))
        self.state["current_round"] = round_num
        self._check_consensus()
        return results

    def run_closing(self) -> list[tuple[DebateAgent, str]]:
        closing_round = self.state["current_round"] + 1
        results = []
        for agent in self.agents:
            response = agent.respond(self.topic, self.state["history"], "closing statement")
            self._record(agent, response, round_num=closing_round, round_type="closing")
            results.append((agent, response))
        return results

    # --- Helpers ---

    def _record(
        self,
        agent: DebateAgent,
        content: str,
        round_num: int,
        round_type: str,
    ) -> None:
        entry: HistoryEntry = {
            "agent": agent.name,
            "persona": agent.persona,
            "content": content,
            "round": round_num,
            "round_type": round_type,
            "stance": agent.current_stance,
        }
        self.state["history"].append(entry)
        self.state["agent_stances"][agent.name] = agent.current_stance

    def _check_consensus(self) -> None:
        stances = list(self.state["agent_stances"].values())
        if len(set(stances)) == 1:
            self.state["consensus_met"] = True

    def history_for_round(self, round_num: int) -> list[HistoryEntry]:
        return [h for h in self.state["history"] if h["round"] == round_num]

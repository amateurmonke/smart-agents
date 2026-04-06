from __future__ import annotations
import json
from openai import OpenAI

JUDGE_SYSTEM = """\
You are an impartial debate judge. Evaluate the arguments presented in the round.

Score each participant on three criteria (each 0 to 10):
- logical_consistency: Is the argument internally coherent and free of fallacies?
- use_of_evidence: Are claims supported by evidence or well-reasoned examples?
- responsiveness: Does the participant engage directly with opposing arguments?

Return a JSON object where each key is a participant name and the value is a dict with:
  "logical_consistency": <int>,
  "use_of_evidence": <int>,
  "responsiveness": <int>,
  "note": "<one sentence justification>"

Return valid JSON only. Do not include markdown fences.\
"""

VERDICT_SYSTEM = """\
You are an impartial debate judge delivering a final verdict.
Write 3 to 5 sentences. Identify who made the most logically sound and evidence-based case.
Acknowledge any meaningful concessions or position changes. Be direct and fair.
Do not use em dashes.\
"""


class Adjudicator:
    def __init__(self, client: OpenAI, model: str = "gpt-4o-mini") -> None:
        self.client = client
        self.model = model
        self._round_scores: list[dict] = []

    def score_round(self, topic: str, round_entries: list[dict], round_num: int) -> dict:
        if not round_entries:
            return {}

        transcript = "\n\n".join(
            f"{h['agent']}: {h['content']}" for h in round_entries
        )

        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": JUDGE_SYSTEM},
                {
                    "role": "user",
                    "content": (
                        f"Topic: {topic}\n\n"
                        f"Round {round_num} transcript:\n{transcript}"
                    ),
                },
            ],
            response_format={"type": "json_object"},
        )

        try:
            scores = json.loads(response.choices[0].message.content or "{}")
            self._round_scores.append(scores)
            return scores
        except json.JSONDecodeError:
            return {}

    def borda_count(self) -> dict[str, dict]:
        if not self._round_scores:
            return {}

        totals: dict[str, dict] = {}

        for round_score in self._round_scores:
            for agent, data in round_score.items():
                if not isinstance(data, dict):
                    continue
                if agent not in totals:
                    totals[agent] = {
                        "logical_consistency": 0,
                        "use_of_evidence": 0,
                        "responsiveness": 0,
                        "rounds_scored": 0,
                    }
                totals[agent]["logical_consistency"] += data.get("logical_consistency", 0)
                totals[agent]["use_of_evidence"] += data.get("use_of_evidence", 0)
                totals[agent]["responsiveness"] += data.get("responsiveness", 0)
                totals[agent]["rounds_scored"] += 1

        for agent, t in totals.items():
            rounds = t["rounds_scored"] or 1
            raw_total = t["logical_consistency"] + t["use_of_evidence"] + t["responsiveness"]
            t["total"] = raw_total
            t["average"] = raw_total / (rounds * 3)

        return totals

    def final_verdict(self, topic: str, history: list[dict], agent_names: list[str]) -> str:
        transcript = "\n\n".join(
            f"[{h['round_type'].upper()} R{h['round']}] {h['agent']}: {h['content']}"
            for h in history
        )

        borda = self.borda_count()
        if borda:
            score_lines = "\n".join(
                f"  {name}: {vals['average']:.1f}/10"
                for name, vals in sorted(borda.items(), key=lambda x: x[1]["average"], reverse=True)
            )
            score_summary = f"Aggregated Borda scores:\n{score_lines}"
        else:
            score_summary = "No round scores available."

        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": VERDICT_SYSTEM},
                {
                    "role": "user",
                    "content": (
                        f"Topic: {topic}\n\n"
                        f"Full debate transcript:\n{transcript}\n\n"
                        f"{score_summary}\n\n"
                        "Deliver your final verdict."
                    ),
                },
            ],
        )
        return response.choices[0].message.content or ""

    @property
    def round_scores(self) -> list[dict]:
        return self._round_scores

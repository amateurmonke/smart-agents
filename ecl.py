from __future__ import annotations

_DEFAULT_TRUST = 5.0
_RECENCY_WEIGHT = 0.6  # weight given to the latest round vs the running average


class TrustProfile:
    """
    Epistemic Context Learning — Stage 1: Trust Estimation.

    Maintains a per-agent reliability score (0–10) derived from adjudicator
    round scores (logical_consistency, use_of_evidence, responsiveness).

    Scores are updated with an exponential moving average so that recent
    performance matters more than early-round behaviour.

    Usage
    -----
    profile = TrustProfile()

    # after each adjudicated round:
    profile.update(round_scores)   # round_scores: {agent_name: {logical_consistency, ...}}

    # when building agent context:
    context_str = profile.format_trust_context(history, for_agent="Alice")
    """

    def __init__(self) -> None:
        # agent_name → current trust score (float 0–10)
        self._scores: dict[str, float] = {}
        # number of rounds each agent has been scored
        self._rounds: dict[str, int] = {}

    # ------------------------------------------------------------------
    # Stage 1: Trust Estimation
    # ------------------------------------------------------------------

    def update(self, round_scores: dict) -> None:
        """
        Ingest one round of adjudicator scores and update trust estimates.

        *round_scores* has the shape returned by Adjudicator.score_round():
            { agent_name: { "logical_consistency": int,
                            "use_of_evidence": int,
                            "responsiveness": int, ... } }
        """
        for agent, data in round_scores.items():
            if not isinstance(data, dict):
                continue

            lc = data.get("logical_consistency", 0)
            ue = data.get("use_of_evidence", 0)
            rs = data.get("responsiveness", 0)
            round_avg = (lc + ue + rs) / 3.0

            if agent not in self._scores:
                self._scores[agent] = _DEFAULT_TRUST
                self._rounds[agent] = 0

            # Exponential moving average — recent rounds weighted more.
            self._scores[agent] = (
                _RECENCY_WEIGHT * round_avg
                + (1 - _RECENCY_WEIGHT) * self._scores[agent]
            )
            self._rounds[agent] += 1

    def get_score(self, agent_name: str) -> float:
        """Return the current trust score for *agent_name* (0–10, default 5.0)."""
        return self._scores.get(agent_name, _DEFAULT_TRUST)

    def ranked_peers(self, exclude: str) -> list[tuple[str, float]]:
        """Return all known agents except *exclude*, sorted by trust descending."""
        peers = [
            (name, score)
            for name, score in self._scores.items()
            if name != exclude
        ]
        return sorted(peers, key=lambda x: x[1], reverse=True)

    def has_data(self) -> bool:
        """True once at least one round has been scored."""
        return bool(self._scores)

    # ------------------------------------------------------------------
    # Stage 2: Trust-Informed Aggregation — context formatter
    # ------------------------------------------------------------------

    def format_trust_context(
        self,
        history: list[dict],
        for_agent: str,
        window: int = 8,
    ) -> str:
        """
        Format recent debate history with trust annotations for *for_agent*.

        Each peer contribution is prefixed with a trust badge so the agent
        can give appropriate epistemic weight to each speaker's arguments.

        Badge scale
        -----------
        8.0+  →  [TRUST: HIGH  X.X]
        5.5–7.9 →  [TRUST: MED   X.X]
        <5.5  →  [TRUST: LOW   X.X]
        """
        recent = history[-window:]
        lines: list[str] = []

        for entry in recent:
            speaker = entry["agent"]
            round_type = entry["round_type"]
            content = entry["content"]

            if speaker == for_agent:
                prefix = f"{speaker} (you, {round_type})"
            else:
                score = self.get_score(speaker)
                tier = _trust_tier(score)
                prefix = f"{speaker} [{tier} {score:.1f}] ({round_type})"

            lines.append(f"{prefix}:\n{content}")

        return "\n\n".join(lines)

    def trust_summary(self) -> dict[str, float]:
        """Return a copy of the current trust scores for display."""
        return dict(self._scores)


# ------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------

def _trust_tier(score: float) -> str:
    if score >= 8.0:
        return "TRUST: HIGH "
    if score >= 5.5:
        return "TRUST: MED  "
    return "TRUST: LOW  "

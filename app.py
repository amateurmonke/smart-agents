from __future__ import annotations
import os
import streamlit as st
from openai import OpenAI
from agents import DebateAgent, PERSONAS
from debate_engine import DebateOrchestrator
from adjudicator import Adjudicator
from metrics import compute_metrics, disagreement_collapse_rate

# ---------------------------------------------------------------------------
# Page config
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="Multi-Agent Debate System",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

STANCE_COLORS = {
    "FOR": "#2e7d32",
    "AGAINST": "#c62828",
    "NEUTRAL": "#555555",
}


def stance_badge(stance: str) -> str:
    color = STANCE_COLORS.get(stance, "#555555")
    return (
        f'<span style="background:{color};color:white;'
        f'padding:2px 10px;border-radius:4px;font-size:0.8em;'
        f'font-weight:600;">{stance}</span>'
    )


def render_agent_card(agent_name: str, persona: str, stance: str, content: str, tool_calls: list[dict]) -> None:
    badge = stance_badge(stance)
    st.markdown(
        f"**{agent_name}** &nbsp; <span style='color:#888;font-size:0.85em;'>{persona}</span>"
        f" &nbsp; {badge}",
        unsafe_allow_html=True,
    )
    if tool_calls:
        for tc in tool_calls:
            import json
            try:
                args = json.loads(tc["args"])
                query = args.get("query", tc["args"])
            except Exception:
                query = tc["args"]
            st.caption(f"[web search] {query}")
    st.markdown(content)
    st.divider()


# ---------------------------------------------------------------------------
# Sidebar: configuration
# ---------------------------------------------------------------------------
with st.sidebar:
    st.title("Configuration")

    api_key = st.text_input(
        "OpenAI API Key",
        type="password",
        value=os.getenv("OPENAI_API_KEY", ""),
        help="Your key is never stored outside this session.",
    )

    st.subheader("Topic")
    topic = st.text_area(
        "Debate topic",
        value="Artificial intelligence will do more harm than good to society.",
        height=80,
    )

    st.subheader("Agents")
    num_agents = st.slider("Number of agents", min_value=2, max_value=4, value=2)

    persona_options = list(PERSONAS.keys())
    stance_options = ["FOR", "AGAINST", "NEUTRAL"]
    default_stances = ["FOR", "AGAINST", "NEUTRAL", "NEUTRAL"]

    agent_configs: list[dict] = []
    for i in range(num_agents):
        with st.expander(f"Agent {i + 1}", expanded=True):
            name = st.text_input("Name", value=f"Agent {i + 1}", key=f"name_{i}")
            persona = st.selectbox(
                "Persona",
                persona_options,
                index=i % len(persona_options),
                key=f"persona_{i}",
            )
            stance = st.selectbox(
                "Stance",
                stance_options,
                index=i % 2,
                key=f"stance_{i}",
            )
            agent_configs.append({"name": name, "persona": persona, "stance": stance})

    st.subheader("Debate Settings")
    max_rounds = st.slider("Rebuttal rounds", min_value=1, max_value=5, value=2)
    use_tools = st.toggle("Enable web search", value=True)

    start = st.button(
        "Start Debate",
        type="primary",
        disabled=not (api_key and topic.strip()),
        use_container_width=True,
    )

# ---------------------------------------------------------------------------
# Main area
# ---------------------------------------------------------------------------
st.title("Multi-Agent Debate System")

if not start:
    st.info("Configure the debate in the sidebar and press Start Debate.")
    st.stop()

if not api_key:
    st.error("An OpenAI API key is required.")
    st.stop()

# ---------------------------------------------------------------------------
# Initialise components
# ---------------------------------------------------------------------------
client = OpenAI(api_key=api_key)

agents = [
    DebateAgent(
        name=cfg["name"],
        persona=cfg["persona"],
        initial_stance=cfg["stance"],
        client=client,
        use_tools=use_tools,
    )
    for cfg in agent_configs
]

orchestrator = DebateOrchestrator(topic=topic, agents=agents, max_rounds=max_rounds)
judge = Adjudicator(client=client)

st.markdown(f"**Topic:** {topic}")
participant_list = " | ".join(
    f"{a.name} ({a.persona}, {a.initial_stance})" for a in agents
)
st.caption(f"Participants: {participant_list}")

# ---------------------------------------------------------------------------
# Opening statements
# ---------------------------------------------------------------------------
st.header("Opening Statements")

with st.spinner("Agents preparing opening statements..."):
    opening_results = orchestrator.run_opening()

for agent, response in opening_results:
    render_agent_card(agent.name, agent.persona, agent.current_stance, response, agent.last_tool_calls)

opening_scores = judge.score_round(topic, orchestrator.history_for_round(0), round_num=0)

# ---------------------------------------------------------------------------
# Rebuttal rounds
# ---------------------------------------------------------------------------
for round_num in range(1, max_rounds + 1):
    if orchestrator.state["consensus_met"]:
        st.info(f"All agents reached consensus after round {round_num - 1}. Skipping remaining rebuttals.")
        break

    st.header(f"Rebuttal Round {round_num}")

    with st.spinner(f"Running rebuttal round {round_num}..."):
        rebuttal_results = orchestrator.run_rebuttal(round_num)

    for agent, response in rebuttal_results:
        render_agent_card(agent.name, agent.persona, agent.current_stance, response, agent.last_tool_calls)

    round_scores = judge.score_round(topic, orchestrator.history_for_round(round_num), round_num=round_num)

    if round_scores:
        with st.expander(f"Round {round_num} scores", expanded=False):
            cols = st.columns(len(round_scores))
            for col, (agent_name, data) in zip(cols, round_scores.items()):
                if not isinstance(data, dict):
                    continue
                with col:
                    st.markdown(f"**{agent_name}**")
                    st.metric("Logic", f"{data.get('logical_consistency', 'N/A')}/10")
                    st.metric("Evidence", f"{data.get('use_of_evidence', 'N/A')}/10")
                    st.metric("Responsiveness", f"{data.get('responsiveness', 'N/A')}/10")
                    if data.get("note"):
                        st.caption(data["note"])

# ---------------------------------------------------------------------------
# Closing statements
# ---------------------------------------------------------------------------
st.header("Closing Statements")

with st.spinner("Agents preparing closing statements..."):
    closing_results = orchestrator.run_closing()

for agent, response in closing_results:
    render_agent_card(agent.name, agent.persona, agent.current_stance, response, agent.last_tool_calls)

# ---------------------------------------------------------------------------
# Final verdict
# ---------------------------------------------------------------------------
st.header("Final Verdict")

with st.spinner("Judge deliberating..."):
    verdict = judge.final_verdict(topic, orchestrator.state["history"], [a.name for a in agents])

st.markdown(verdict)

# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------
st.header("Debate Metrics")

all_metrics = compute_metrics(agents)
dcr = disagreement_collapse_rate(all_metrics)
borda = judge.borda_count()

col_left, col_right = st.columns(2)

with col_left:
    st.subheader("Agent Metrics")
    for m in all_metrics:
        with st.container(border=True):
            st.markdown(f"**{m.name}**")
            cols = st.columns(2)
            with cols[0]:
                st.markdown(f"Initial stance: **{m.initial_stance}**")
                st.markdown(f"Final stance: **{m.final_stance}**")
            with cols[1]:
                st.metric("Sycophancy Score", f"{m.sycophancy_score:.2f}")
                tof = m.turn_of_flip
                st.metric(
                    "Turn of Flip",
                    f"Round {tof}" if tof >= 0 else "None",
                    help="The round index at which the agent first changed stance.",
                )

with col_right:
    st.subheader("Aggregate Metrics")

    st.metric(
        "Disagreement Collapse Rate",
        f"{dcr:.2f}",
        help="1.0 = agents converged immediately. 0.0 = agents never converged.",
    )

    if borda:
        st.subheader("Borda Count (avg score /10)")
        sorted_borda = sorted(borda.items(), key=lambda x: x[1]["average"], reverse=True)
        for rank, (agent_name, scores) in enumerate(sorted_borda, start=1):
            label = f"#{rank} {agent_name}"
            st.metric(
                label=label,
                value=f"{scores['average']:.1f}/10",
                delta=f"total: {scores['total']} pts",
                delta_color="off",
            )
            detail_cols = st.columns(3)
            with detail_cols[0]:
                st.caption(f"Logic: {scores['logical_consistency']}")
            with detail_cols[1]:
                st.caption(f"Evidence: {scores['use_of_evidence']}")
            with detail_cols[2]:
                st.caption(f"Responsiveness: {scores['responsiveness']}")

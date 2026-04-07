from __future__ import annotations
import os
import uuid
import streamlit as st
from openai import OpenAI
from agents import DebateAgent, PERSONAS
from debate_engine import DebateOrchestrator
from adjudicator import Adjudicator
from metrics import compute_metrics, disagreement_collapse_rate
import tools as tools_module
from ecl import TrustProfile

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
    import json
    badge = stance_badge(stance)
    st.markdown(
        f"**{agent_name}** &nbsp; <span style='color:#888;font-size:0.85em;'>{persona}</span>"
        f" &nbsp; {badge}",
        unsafe_allow_html=True,
    )
    for tc in tool_calls:
        try:
            args = json.loads(tc["args"])
            query = args.get("query", tc["args"])
        except Exception:
            query = tc["args"]
        label = "[rag]" if tc["name"] == "rag_search" else "[web search]"
        st.caption(f"{label} {query}")
    st.markdown(content)
    st.divider()


def render_round_scores(round_scores: dict, round_label: str) -> None:
    if not round_scores:
        return
    with st.expander(f"{round_label} scores", expanded=False):
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


def render_trust_scores(trust_profile: TrustProfile) -> None:
    if not trust_profile.has_data():
        return
    st.subheader("ECL Trust Scores")
    scores = trust_profile.trust_summary()
    cols = st.columns(len(scores))
    for col, (name, score) in zip(cols, sorted(scores.items(), key=lambda x: x[1], reverse=True)):
        tier = "HIGH" if score >= 8.0 else ("MED" if score >= 5.5 else "LOW")
        col.metric(label=name, value=f"{score:.1f}/10", delta=tier, delta_color="off")


def render_metrics(agents, judge, trust_profile: TrustProfile | None = None) -> None:
    st.header("Debate Metrics")
    all_metrics = compute_metrics(agents)
    dcr = disagreement_collapse_rate(all_metrics)
    borda = judge.borda_count()

    col_left, col_right = st.columns(2)

    if trust_profile is not None:
        render_trust_scores(trust_profile)

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
                st.metric(
                    label=f"#{rank} {agent_name}",
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

    st.subheader("RAG")
    uploaded_file = st.file_uploader(
        "Upload document (.txt)",
        type=["txt"],
        help="Agents can retrieve passages from this document during the debate.",
    )
    use_rag = st.toggle("Enable RAG search", value=False, disabled=uploaded_file is None)

    st.subheader("Orchestration")
    use_langgraph = st.toggle(
        "Use LangGraph",
        value=False,
        help="Run the debate as a compiled LangGraph StateGraph with MemorySaver checkpointing.",
    )
    use_ecl = st.toggle(
        "Enable ECL (Epistemic Context Learning)",
        value=False,
        help=(
            "Agents track peer reliability across rounds. "
            "High-trust peers' arguments are weighted more heavily when an agent is uncertain."
        ),
    )

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
# Initialise shared components
# ---------------------------------------------------------------------------
client = OpenAI(api_key=api_key)

# RAG: load uploaded document into the store once per session upload.
if uploaded_file is not None and use_rag:
    if "rag_file_id" not in st.session_state or st.session_state.rag_file_id != uploaded_file.file_id:
        from rag import DocumentStore
        store = DocumentStore(client)
        text = uploaded_file.read().decode("utf-8")
        n_chunks = store.add_text(text, metadata={"filename": uploaded_file.name})
        tools_module.set_document_store(store)
        st.session_state.rag_file_id = uploaded_file.file_id
        st.sidebar.success(f"Loaded {n_chunks} chunks from {uploaded_file.name}")

trust_profile = TrustProfile() if use_ecl else None

agents = [
    DebateAgent(
        name=cfg["name"],
        persona=cfg["persona"],
        initial_stance=cfg["stance"],
        client=client,
        use_tools=use_tools,
        use_rag=use_rag,
        trust_profile=trust_profile,
    )
    for cfg in agent_configs
]

judge = Adjudicator(client=client)

st.markdown(f"**Topic:** {topic}")
participant_list = " | ".join(
    f"{a.name} ({a.persona}, {a.initial_stance})" for a in agents
)
st.caption(f"Participants: {participant_list}")
if use_langgraph:
    st.caption("Orchestration: LangGraph StateGraph")
if use_ecl:
    st.caption("ECL: enabled — agents will weight peer arguments by trust tier")

# ---------------------------------------------------------------------------
# Run via LangGraph
# ---------------------------------------------------------------------------
if use_langgraph:
    from debate_graph import build_debate_graph

    graph = build_debate_graph(agents, judge, max_rounds, trust_profile=trust_profile)
    initial_state = {
        "topic": topic,
        "history": [],
        "current_round": 0,
        "agent_stances": {a.name: a.initial_stance for a in agents},
        "consensus_met": False,
        "final_verdict": "",
    }
    thread_config = {"configurable": {"thread_id": str(uuid.uuid4())}}

    displayed_len = 0  # tracks how many history entries have been rendered

    for step in graph.stream(initial_state, config=thread_config):
        node_name, node_output = next(iter(step.items()))

        new_history: list = node_output.get("history", [])
        new_entries = new_history[displayed_len:]
        displayed_len = len(new_history)

        if node_name == "opening" and new_entries:
            st.header("Opening Statements")
            for entry in new_entries:
                render_agent_card(
                    entry["agent"], entry["persona"], entry["stance"],
                    entry["content"], entry.get("tool_calls", []),
                )
            opening_scores = judge.score_round(topic, new_entries, round_num=0)
            render_round_scores(opening_scores, "Opening")

        elif node_name == "rebuttal" and new_entries:
            round_num = node_output.get("current_round", "?")
            st.header(f"Rebuttal Round {round_num}")
            if node_output.get("consensus_met"):
                st.info("Agents reached consensus.")
            for entry in new_entries:
                render_agent_card(
                    entry["agent"], entry["persona"], entry["stance"],
                    entry["content"], entry.get("tool_calls", []),
                )
            round_scores = judge.score_round(topic, new_entries, round_num=round_num)
            render_round_scores(round_scores, f"Rebuttal Round {round_num}")

        elif node_name == "closing" and new_entries:
            st.header("Closing Statements")
            for entry in new_entries:
                render_agent_card(
                    entry["agent"], entry["persona"], entry["stance"],
                    entry["content"], entry.get("tool_calls", []),
                )

        elif node_name == "verdict":
            st.header("Final Verdict")
            st.markdown(node_output.get("final_verdict", ""))

    render_metrics(agents, judge, trust_profile=trust_profile)
    st.stop()

# ---------------------------------------------------------------------------
# Run via original DebateOrchestrator (default path)
# ---------------------------------------------------------------------------
orchestrator = DebateOrchestrator(topic=topic, agents=agents, max_rounds=max_rounds, trust_profile=trust_profile)

# Opening statements
st.header("Opening Statements")
with st.spinner("Agents preparing opening statements..."):
    opening_results = orchestrator.run_opening()

for agent, response in opening_results:
    render_agent_card(
        agent.name, agent.persona, agent.current_stance,
        response, agent.last_tool_calls,
    )

opening_scores = judge.score_round(topic, orchestrator.history_for_round(0), round_num=0)
orchestrator.update_trust(opening_scores)

# Rebuttal rounds
for round_num in range(1, max_rounds + 1):
    if orchestrator.state["consensus_met"]:
        st.info(f"All agents reached consensus after round {round_num - 1}. Skipping remaining rebuttals.")
        break

    st.header(f"Rebuttal Round {round_num}")
    with st.spinner(f"Running rebuttal round {round_num}..."):
        rebuttal_results = orchestrator.run_rebuttal(round_num)

    for agent, response in rebuttal_results:
        render_agent_card(
            agent.name, agent.persona, agent.current_stance,
            response, agent.last_tool_calls,
        )

    round_scores = judge.score_round(topic, orchestrator.history_for_round(round_num), round_num=round_num)
    orchestrator.update_trust(round_scores)
    render_round_scores(round_scores, f"Rebuttal Round {round_num}")

# Closing statements
st.header("Closing Statements")
with st.spinner("Agents preparing closing statements..."):
    closing_results = orchestrator.run_closing()

for agent, response in closing_results:
    render_agent_card(
        agent.name, agent.persona, agent.current_stance,
        response, agent.last_tool_calls,
    )

# Final verdict
st.header("Final Verdict")
with st.spinner("Judge deliberating..."):
    verdict = judge.final_verdict(topic, orchestrator.state["history"], [a.name for a in agents])
st.markdown(verdict)

render_metrics(agents, judge, trust_profile=trust_profile)

# Multi-Agent Debate System

A structured multi-agent debate framework built with LangGraph, OpenAI, and Streamlit. Agents with distinct reasoning personas argue a topic over multiple rounds, grounded by web search and a local RAG corpus, while an adjudicator scores each round and delivers a final verdict with a recommended stance and confidence rating.

---

## Features

- **Multi-agent debate** — 2–4 agents, each with a distinct reasoning persona (Analytical, Philosophical, Empiricist, Devil's Advocate) and an assigned stance (FOR / AGAINST / NEUTRAL)
- **LangGraph orchestration** — optional graph-based execution with `MemorySaver` checkpointing; debate resumes from the last node on failure
- **Tool-augmented agents** — web search via DuckDuckGo; agents cite sources inline
- **Mini RAG** — upload a `.txt` document; agents retrieve relevant passages via OpenAI embeddings and cosine similarity before falling back to web search
- **Epistemic Context Learning (ECL)** — agents track peer reliability across rounds using an exponential moving average; high-trust peers' arguments are annotated and weighted more heavily
- **Round-by-round adjudication** — judge scores each round on logical consistency, use of evidence, and responsiveness; aggregated via Borda count
- **Structured final verdict** — recommended stance (FOR / AGAINST / NEUTRAL) with a confidence rating (0–100)
- **Debate metrics** — Sycophancy Score, Turn of Flip, Disagreement Collapse Rate per agent

---

## Architecture

```
app.py                  Streamlit UI — configuration, rendering, orchestration dispatch
├── debate_engine.py    DebateOrchestrator — imperative phase runner (default path)
├── debate_graph.py     LangGraph StateGraph — opening → rebuttal* → closing → verdict
├── agents.py           DebateAgent — persona, tool use, ECL-annotated context
├── adjudicator.py      Adjudicator — round scoring, Borda count, structured verdict
├── metrics.py          AgentMetrics, sycophancy score, turn of flip, DCR
├── ecl.py              TrustProfile — EMA trust estimation, context annotation
├── rag.py              DocumentStore — chunking, OpenAI embeddings, cosine retrieval
└── tools.py            web_search (DuckDuckGo), rag_search, tool registry
```

### LangGraph topology

```
[opening] ──→ [rebuttal] ──(loop)──→ [rebuttal]
                  │
                  └──(consensus or max_rounds)──→ [closing] ──→ [verdict] ──→ END
```

Each node scores the previous round and updates ECL trust before agents respond.

---

## Setup

### Prerequisites

- Python 3.10+
- An OpenAI API key

### Install

```bash
git clone <repo-url>
cd smart-agents
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### Environment

```bash
cp .env.example .env
# add your key:
# OPENAI_API_KEY=sk-...
```

### Run

```bash
streamlit run app.py
```

---

## Usage

1. Enter your OpenAI API key in the sidebar (or set `OPENAI_API_KEY` in `.env`)
2. Set the debate topic
3. Configure 2–4 agents — name, persona, initial stance
4. Choose the number of rebuttal rounds (1–5)
5. Toggle optional features: web search, RAG (upload a `.txt` first), LangGraph, ECL
6. Press **Start Debate**

### RAG

Upload a plain `.txt` file in the sidebar. The document is chunked by paragraph, embedded with `text-embedding-3-small`, and stored in memory. Agents with "Enable RAG search" on will retrieve the top-3 passages per query before falling back to the web.

### ECL

When enabled, a shared `TrustProfile` is updated after each scored round. Trust scores (0–10) are displayed in the metrics section. Agents see peer contributions annotated with `[TRUST: HIGH 8.2]` / `[TRUST: MED 5.7]` / `[TRUST: LOW 3.1]` in their context.

---

## Configuration reference

| Sidebar option | Default | Description |
|---|---|---|
| Number of agents | 2 | 2–4 debate participants |
| Rebuttal rounds | 2 | Number of back-and-forth rounds |
| Enable web search | On | Agents can call DuckDuckGo mid-argument |
| Upload document | — | Populates the local RAG corpus |
| Enable RAG search | Off | Requires a document to be uploaded |
| Use LangGraph | Off | Graph-based execution with checkpointing |
| Enable ECL | Off | Peer trust tracking across rounds |

---

## Project structure

```
smart-agents/
├── app.py
├── agents.py
├── debate_engine.py
├── debate_graph.py
├── adjudicator.py
├── metrics.py
├── ecl.py
├── rag.py
├── tools.py
├── requirements.txt
├── .env.example
└── README.md
```

---

## Dependencies

| Package | Purpose |
|---|---|
| `openai` | LLM calls, embeddings |
| `streamlit` | Web UI |
| `langgraph` | Graph-based debate orchestration |
| `duckduckgo-search` | Web search tool |
| `numpy` | Cosine similarity for RAG |
| `python-dotenv` | `.env` loading |

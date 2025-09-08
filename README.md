# RAG + Jira/Bitbucket MCP Integration

This repository demonstrates how to combine **Retrieval‑Augmented Generation (RAG)** with **MCP (Model Context Protocol)** servers running in Docker. The setup allows an LLM agent to:

* Query your own document database via RAG (Chroma + OpenAI embeddings).
* Use MCP‑based tools (Jira, Bitbucket, etc.) via Dockerized MCP servers.
* Orchestrate workflow using **LangGraph**.

---

## Features

* **RAG Retriever**: Search your story/knowledge base using OpenAI embeddings + Chroma.
* **MCP Tools**: Jira + Bitbucket adapters running as Docker containers.
* **LangGraph Orchestration**: State machine routing between agent and tools.
* **Interactive Mode**: Chat with your agent in terminal, combining RAG and MCP.

---

## Repository Structure

```
.
├─ prepareRAG.py          # Script to embed and persist local documents
├─ RAG_MCP.py             # Example with one‑MCP setup (Jira)
├─ RAG_muiltiMCP.py       # Example with multi‑MCP setup (Jira + Bitbucket)
├─ requirements.txt       # Python dependencies
└─ data/                  # (optional) folder to store your documents
```

---

## Requirements

* Python 3.10+
* Docker Desktop / Engine
* OpenAI API key
* Jira / Bitbucket credentials (for MCP servers)

Install dependencies:

```bash
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

---

## Environment Setup

Copy `.env.example` to `.env` and fill values:

```dotenv
OPENAI_API_KEY=sk-...
CHROMA_DIR=./first_Aid_embeddings

# Jira MCP
JIRA_URL=https://your-domain.atlassian.net
JIRA_USERNAME=email@example.com
JIRA_API_TOKEN=token

# Bitbucket MCP
BITBUCKET_USERNAME=your-username
BITBUCKET_APP_PASSWORD=your-app-password
```

---

## Running MCP Servers in Docker

Example (Jira MCP):

```bash
docker run -i --rm \
  -e JIRA_URL=$JIRA_URL \
  -e JIRA_USERNAME=$JIRA_USERNAME \
  -e JIRA_API_TOKEN=$JIRA_API_TOKEN \
  ghcr.io/sooperset/mcp-atlassian:latest
```

Example (Bitbucket MCP):

```bash
docker run -i --rm \
  -e BITBUCKET_USERNAME=$BITBUCKET_USERNAME \
  -e BITBUCKET_APP_PASSWORD=$BITBUCKET_APP_PASSWORD \
  mcp-bitbucket
```

---

## Usage

### 1. Prepare embeddings

```bash
python prepareRAG.py --data ./data --persist ./first_Aid_embeddings
```

### 2. Run the agent

```bash
# Run interactive agent with Jira + Bitbucket MCP
ython RAG_MCP.py
```

### 3. Ask questions interactively

```text
ASK: List Jira issues for project TEST
Agent: Here are the open tickets...

ASK: Search Bitbucket repos in workspace "ahmadalawwad"
Agent: Found 12 repositories...

ASK: What themes of persistence appear in Daniel’s story?
Agent: Found relevant story excerpts...
```

---

## Notes

* `retriever_tool` uses persisted Chroma embeddings under `./first_Aid_embeddings`.
* MCP tools are merged with retriever and bound to LLM (`gpt-4o-mini`).
* Workflow uses LangGraph `StateGraph` with conditional routing.

---

## Troubleshooting

* **MCP tools not loading?** Ensure Docker is running and env vars are correct.
* **No story results?** Run `prepareRAG.py` to embed documents first.
* **Permission denied (Bitbucket/Jira)?** Re‑check credentials and API tokens.

---

## License

MIT

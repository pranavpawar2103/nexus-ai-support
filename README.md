# NexusAI Support

> Multi-Agent Customer Intelligence Platform — query customer data, search policy documents, and get AI-assisted answers through a natural language interface.

[![Python 3.11+](https://img.shields.io/badge/Python-3.11%2B-blue?style=flat-square)](https://python.org)
[![LangChain](https://img.shields.io/badge/LangChain-0.3.7-green?style=flat-square)](https://langchain.com)
[![LangGraph](https://img.shields.io/badge/LangGraph-0.2.45-purple?style=flat-square)](https://langchain-ai.github.io/langgraph/)
[![ChromaDB](https://img.shields.io/badge/ChromaDB-0.5.18-orange?style=flat-square)](https://trychroma.com)
[![Gradio](https://img.shields.io/badge/Gradio-4.42.0-ff7c00?style=flat-square)](https://gradio.app)
---

## Overview

NexusAI Support is a multi-agent AI system built for customer support teams. It connects a natural language interface to a live SQLite database and a vector-based policy knowledge base, routing queries automatically to the right agent.

---

## Architecture

```
User Query
    │
    ▼
Gradio UI (port 7860)
    │
    ▼
LangGraph Supervisor
    │
    ├── SQL_QUERY    ──► SQL Agent    ──► SQLite DB
    ├── POLICY_QUERY ──► RAG Agent    ──► ChromaDB (PDF chunks)
    ├── HYBRID_QUERY ──► Both agents  ──► Synthesized response
    └── GENERAL      ──► Conversational fallback
```

The supervisor classifies every incoming query and routes it to the appropriate agent. Hybrid queries run both agents in parallel and synthesize the results.

---

## Features

- **Natural language SQL** — ask questions about customers, tickets, and billing in plain English
- **Policy document Q&A** — upload PDFs and ask questions; answers cite the source page
- **Hybrid queries** — combine database facts with policy context in a single response
- **MCP server** — exposes all capabilities to MCP-compatible clients (Claude Desktop, etc.)
- **Live stats sidebar** — customer count, open tickets, and knowledge base docs updated in real time

---

## Project Structure

```
nexus-ai-support/
├── main.py                    # Entry point
├── requirements.txt           # Pinned dependencies
├── .env.example               # Environment variable template
├── core/
│   ├── config.py              # Settings + LLM/embedding factories
│   ├── database.py            # SQLAlchemy ORM models
│   └── vector_store.py        # ChromaDB manager + PDF ingestion
├── agents/
│   ├── tools.py               # LangChain tool definitions
│   ├── sql_agent.py           # Text-to-SQL agent
│   ├── rag_agent.py           # RAG agent for policy Q&A
│   └── orchestrator.py        # LangGraph supervisor
├── mcp_server/
│   └── server.py              # FastMCP server
├── ui/
│   └── app.py                 # Gradio interface
└── data/
    ├── seed_database.py        # Generates sample data
    └── sample_policies/        # Place PDF policy documents here
```

---

## Quick Start

**Prerequisites:** Python 3.11+, OpenAI API key with GPT-4o access

```bash
# 1. Clone and enter the project
git clone https://github.com/pranavpawar2103/nexus-ai-support.git
cd nexus-ai-support

# 2. Create virtual environment
python -m venv venv
venv\Scripts\activate        # Windows
source venv/bin/activate     # macOS/Linux

# 3. Install dependencies
pip install -r requirements.txt

# 4. Configure environment
cp .env.example .env
# Add your OPENAI_API_KEY to .env

# 5. Run
python main.py
```

Open **http://localhost:7860** in your browser.

---

## Configuration

```env
OPENAI_API_KEY=sk-your-key-here
OPENAI_MODEL=gpt-4o
EMBEDDING_MODEL=text-embedding-3-small
DB_PATH=./data/nexus_support.db
CHROMA_PATH=./data/chroma_db
MCP_HOST=localhost
MCP_PORT=8765
```

---

## Example Queries

**Customer lookups**
```
Find customer Sarah and show her account details
List all Enterprise plan customers
Show customers who have been with us the longest
```

**Support tickets**
```
Show all critical open tickets with customer names
What is the average resolution time by category?
Which tickets have been open the longest?
```

**Billing**
```
Which customers have overdue invoices?
Show total outstanding balance across all accounts
List Enterprise customers with billing issues
```

**Policy questions** *(requires a PDF to be ingested)*
```
What is the refund policy for annual plans?
What are the SLA response time requirements?
What happens when a customer cancels their subscription?
```

**Hybrid**
```
Show our Enterprise customers and what the policy says about enterprise support obligations
Which customers are eligible for a refund based on our policy?
```

---

## Agent Routing

| Query type | Routed to | Data source |
|------------|-----------|-------------|
| `SQL_QUERY` | SQL Agent | SQLite database |
| `POLICY_QUERY` | RAG Agent | ChromaDB (PDF chunks) |
| `HYBRID_QUERY` | Both agents | Database + PDFs |
| `GENERAL` | Fallback | GPT-4o general knowledge |

---

## MCP Server

The MCP server exposes NexusAI to any MCP-compatible client such as Claude Desktop.

| Tool | Description |
|------|-------------|
| `query_support_system` | Query the full multi-agent system |
| `ingest_pdf_document` | Add a PDF to the knowledge base |
| `list_knowledge_base` | List all ingested documents |
| `get_system_stats` | System-wide statistics |
| `system://health` | Health check resource |

---

## Tech Stack

| Component | Technology | Version |
|-----------|------------|---------|
| LLM | OpenAI GPT-4o | via API |
| Agent framework | LangChain + LangGraph | 0.3.7 / 0.2.45 |
| Vector store | ChromaDB | 0.5.18 |
| Database | SQLite + SQLAlchemy | 2.0.35 |
| UI | Gradio | 4.42.0 |
| MCP server | FastMCP | 0.4.1 |
| PDF processing | PyPDF | 5.1.0 |
| Embeddings | text-embedding-3-small | via API |

---

## Security

- SQL agent only executes `SELECT` statements — all DML and DDL is blocked
- PDF uploads validate the `%PDF` magic byte signature before processing
- Startup halts with a clear error if `OPENAI_API_KEY` is missing

---

## Synthetic Database

The project ships with a seed script that generates a realistic but entirely fictional dataset using the [Faker](https://faker.readthedocs.io/) library. No real customer data is used anywhere.

**What gets generated (`data/seed_database.py`)**

| Table | Rows | Details |
|-------|------|---------|
| `customers` | 50 | Names, emails, company names, plan type, account status, join date |
| `support_tickets` | 150–300 | Status, priority, category, assigned agent, timestamps, resolution time |
| `billing_records` | 100–250 | Invoice amounts, due dates, payment status, linked to customers |

**How the data is shaped**

- Plan distribution: 40% Basic, 35% Pro, 25% Enterprise
- Ticket statuses: Open (25%), In Progress (20%), Resolved (35%), Closed (20%)
- Ticket priorities: Low (25%), Medium (40%), High (25%), Critical (10%)
- Ticket categories: Technical, Billing, General, Refund, Account
- Billing statuses: weighted toward Paid, with a realistic mix of Pending and Overdue
- 8 named support agents assigned to resolved/closed tickets
- All dates are generated relative to `datetime.now()` so the data always feels current

**Reproducibility**

The seed is fixed (`Faker.seed(42)`, `random.seed(42)`), so every run produces the same dataset. To regenerate:

```bash
python data/seed_database.py
```

This drops and recreates all tables. The database is stored at `./data/nexus_support.db` (SQLite).

---

## Development

```bash
# Reseed the database
python data/seed_database.py

# Run only the UI
python ui/app.py

# Run only the MCP server
python mcp_server/server.py

# Use the SQL agent directly
python -c "
from agents.sql_agent import SQLAgent
r = SQLAgent().query('How many open tickets are there?')
print(r.answer)
"
```

---

## Adding Policy Documents

1. Drop a PDF into `data/sample_policies/`
2. Use the **Ingest Document** panel in the UI, or call the MCP tool:
   ```
   ingest_pdf_document(file_path="./data/sample_policies/your_policy.pdf")
   ```

Documents are split into 1000-character chunks with 200-character overlap, embedded using `text-embedding-3-small`, and stored persistently in ChromaDB.

---

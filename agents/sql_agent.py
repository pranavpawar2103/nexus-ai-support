"""SQL Agent for NexusAI Support.

Uses LangChain's SQLDatabase + direct LLM calls to avoid create_sql_agent
compatibility issues with newer langchain-community versions.
"""

import logging
import time
from dataclasses import dataclass, field
from typing import List, Optional

logger = logging.getLogger(__name__)

SQL_SYSTEM_PROMPT = """You are NexusSQL, an expert database analyst for a customer support platform.
Translate the user's question into a single valid SQLite SELECT query, execute it, and return a clear answer.

DATABASE SCHEMA:
- customers: id, name, email, phone, plan_type (Basic/Pro/Enterprise),
  account_status (Active/Suspended/Cancelled), created_at, location, company_name
- support_tickets: id, customer_id (FK→customers.id), title, description,
  status (Open/In Progress/Resolved/Closed), priority (Low/Medium/High/Critical),
  category (Billing/Technical/General/Refund/Account), created_at, resolved_at,
  resolution_notes, agent_name
- billing_records: id, customer_id (FK→customers.id), amount (USD),
  status (Paid/Pending/Overdue), invoice_date, due_date, description

SQL RULES:
1. ALWAYS use LOWER(name) LIKE LOWER('%term%') for name searches
2. Only write SELECT statements — never INSERT, UPDATE, DELETE
3. Use strftime('%Y-%m', created_at) for date filtering
4. JOIN example: SELECT c.name, t.title FROM customers c JOIN support_tickets t ON c.id = t.customer_id

FEW-SHOT EXAMPLES:
Q: How many customers do we have?
SQL: SELECT COUNT(*) as total_customers FROM customers

Q: Find customer named Sarah
SQL: SELECT * FROM customers WHERE LOWER(name) LIKE LOWER('%sarah%')

Q: Show all critical open tickets
SQL: SELECT t.id, t.title, t.priority, t.status, c.name as customer_name FROM support_tickets t JOIN customers c ON t.customer_id = c.id WHERE t.priority = 'Critical' AND t.status IN ('Open', 'In Progress') ORDER BY t.created_at DESC

Q: Which customers have overdue billing?
SQL: SELECT DISTINCT c.name, c.email, c.plan_type, b.amount, b.due_date FROM customers c JOIN billing_records b ON c.id = b.customer_id WHERE b.status = 'Overdue' ORDER BY b.due_date

Q: Ticket resolution statistics
SQL: SELECT status, priority, COUNT(*) as count FROM support_tickets GROUP BY status, priority ORDER BY status, priority

Q: Average resolution time by category
SQL: SELECT category, ROUND(AVG((julianday(resolved_at) - julianday(created_at)) * 24), 1) as avg_hours FROM support_tickets WHERE resolved_at IS NOT NULL GROUP BY category ORDER BY avg_hours DESC

Respond ONLY with the raw SQL query — no explanation, no markdown, no backticks.
"""

ANSWER_PROMPT = """You are a helpful customer support analyst. 
The user asked: {question}

The SQL query used was: {sql}

The query returned these results:
{results}

Write a clear, friendly answer in 2-4 sentences summarizing what was found.
Use markdown tables if there are multiple rows of data. Be concise and professional."""


@dataclass
class AgentResponse:
    """Response from the SQL agent."""
    answer: str
    sql_used: str = ""
    sources: List[str] = field(default_factory=list)
    execution_time: float = 0.0
    error: Optional[str] = None


class SQLAgent:
    """Direct SQL agent using LLM for query generation + SQLAlchemy for execution."""

    def __init__(self, db_path: Optional[str] = None) -> None:
        from core.config import get_settings
        if db_path is None:
            settings = get_settings()
            db_path = str(settings.get_db_abs_path())
        self.db_path = db_path
        self._llm = None
        logger.info("SQLAgent initialized with db: %s", db_path)

    def _get_llm(self):
        if self._llm is None:
            from core.config import get_llm
            self._llm = get_llm(temperature=0.0)
        return self._llm

    def _generate_sql(self, question: str) -> str:
        """Ask LLM to generate SQL for the question."""
        from langchain_core.messages import HumanMessage, SystemMessage
        llm = self._get_llm()
        response = llm.invoke([
            SystemMessage(content=SQL_SYSTEM_PROMPT),
            HumanMessage(content=f"Question: {question}"),
        ])
        sql = response.content.strip()
        # Clean up any accidental markdown
        sql = sql.replace("```sql", "").replace("```", "").strip()
        return sql

    def _execute_sql(self, sql: str) -> str:
        """Execute SQL and return formatted results."""
        import pandas as pd
        from sqlalchemy import text
        from core.database import get_engine

        # Safety check
        if not sql.strip().upper().startswith("SELECT"):
            raise ValueError("Only SELECT queries are permitted.")

        engine = get_engine(self.db_path)
        with engine.connect() as conn:
            result = conn.execute(text(sql))
            rows = result.fetchmany(50)
            columns = list(result.keys())

        if not rows:
            return "No results found."

        df = pd.DataFrame(rows, columns=columns)
        return df.to_markdown(index=False)

    def _generate_answer(self, question: str, sql: str, results: str) -> str:
        """Ask LLM to write a friendly answer from the results."""
        from langchain_core.messages import HumanMessage
        llm = self._get_llm()
        prompt = ANSWER_PROMPT.format(question=question, sql=sql, results=results)
        response = llm.invoke([HumanMessage(content=prompt)])
        return response.content

    def query(self, user_question: str) -> AgentResponse:
        """Process a natural language question against the database."""
        start_time = time.time()
        logger.info("SQLAgent processing query: '%s...'", user_question[:80])

        try:
            # Step 1: Generate SQL
            sql = self._generate_sql(user_question)
            logger.info("Generated SQL: %s", sql[:200])

            # Step 2: Execute SQL
            results = self._execute_sql(sql)

            # Step 3: Generate natural language answer
            answer = self._generate_answer(user_question, sql, results)

            execution_time = time.time() - start_time
            logger.info("SQLAgent completed in %.2fs", execution_time)

            return AgentResponse(
                answer=answer,
                sql_used=sql,
                sources=["SQLite Database: customers, support_tickets, billing_records"],
                execution_time=execution_time,
            )

        except Exception as e:
            execution_time = time.time() - start_time
            error_msg = str(e)
            logger.error("SQLAgent error after %.2fs: %s", execution_time, error_msg)

            # Fallback: try direct tool approach
            try:
                return self._fallback_query(user_question, start_time)
            except Exception as e2:
                logger.error("SQLAgent fallback also failed: %s", e2)

            return AgentResponse(
                answer=(
                    f"I encountered an error querying the database: {error_msg}\n\n"
                    "Please try rephrasing your question."
                ),
                execution_time=execution_time,
                error=error_msg,
            )

    def _fallback_query(self, question: str, start_time: float) -> AgentResponse:
        """Fallback using tools, always formatted by LLM into clean prose."""
        from agents.tools import get_ticket_statistics, run_custom_sql

        q_lower = question.lower()

        if any(w in q_lower for w in ["statistic", "stat", "resolution", "top agent", "how many", "total", "count", "breakdown", "agent"]):
            raw = get_ticket_statistics.invoke({})
            sql_used = "get_ticket_statistics() tool"
        elif any(w in q_lower for w in ["overdue", "billing", "invoice", "payment"]):
            raw = run_custom_sql.invoke({
                "query": "SELECT c.name, c.email, c.plan_type, b.amount, b.due_date FROM customers c JOIN billing_records b ON c.id = b.customer_id WHERE b.status = 'Overdue' ORDER BY b.due_date LIMIT 20"
            })
            sql_used = "overdue billing query"
        elif any(w in q_lower for w in ["open", "critical", "high", "priority", "unresolved"]):
            raw = run_custom_sql.invoke({
                "query": "SELECT t.id, t.title, t.priority, t.status, c.name as customer FROM support_tickets t JOIN customers c ON t.customer_id = c.id WHERE t.status IN ('Open','In Progress') ORDER BY t.priority DESC LIMIT 20"
            })
            sql_used = "open tickets query"
        else:
            raw = run_custom_sql.invoke({
                "query": "SELECT COUNT(*) as total_customers FROM customers"
            })
            sql_used = "customer count query"

        # Pass raw data through LLM to get clean formatted prose instead of raw JSON
        answer = self._generate_answer(question, sql_used, str(raw))

        return AgentResponse(
            answer=answer,
            sql_used=sql_used,
            sources=["SQLite Database"],
            execution_time=time.time() - start_time,
        )
"""LangChain tools for NexusAI Support agents.

Defines all @tool-decorated functions that agents use to query the database,
search policy documents, and retrieve billing information.
"""

import json
import logging
import re
from functools import lru_cache
from typing import Any, Dict, List, Optional

from langchain_core.tools import tool

logger = logging.getLogger(__name__)

# ─── SQL Injection Guard ───────────────────────────────────────────────────────

_FORBIDDEN_SQL_KEYWORDS = re.compile(
    r"\b(INSERT|UPDATE|DELETE|DROP|CREATE|ALTER|TRUNCATE|REPLACE|EXEC|EXECUTE|"
    r"PRAGMA|ATTACH|DETACH|VACUUM|REINDEX|GRANT|REVOKE)\b",
    re.IGNORECASE,
)


def _validate_select_only(query: str) -> None:
    """Ensure SQL query is a SELECT statement only.

    Args:
        query: SQL query string to validate.

    Raises:
        ValueError: If query contains forbidden DML/DDL keywords.
    """
    stripped = query.strip()
    if not stripped.upper().startswith("SELECT"):
        raise ValueError("Only SELECT queries are allowed for security reasons.")
    if _FORBIDDEN_SQL_KEYWORDS.search(stripped):
        raise ValueError(
            "Query contains forbidden SQL keywords. Only SELECT statements are permitted."
        )


# ─── Database Query Helpers ────────────────────────────────────────────────────

def _format_customer(customer) -> Dict[str, Any]:
    """Convert a Customer ORM object to a JSON-serializable dict."""
    return {
        "id": customer.id,
        "name": customer.name,
        "email": customer.email,
        "phone": customer.phone or "N/A",
        "plan_type": customer.plan_type,
        "account_status": customer.account_status,
        "created_at": customer.created_at.isoformat() if customer.created_at else None,
        "location": customer.location or "N/A",
        "company_name": customer.company_name or "N/A",
    }


def _format_ticket(ticket) -> Dict[str, Any]:
    """Convert a SupportTicket ORM object to a JSON-serializable dict."""
    return {
        "id": ticket.id,
        "title": ticket.title,
        "status": ticket.status,
        "priority": ticket.priority,
        "category": ticket.category,
        "created_at": ticket.created_at.isoformat() if ticket.created_at else None,
        "resolved_at": ticket.resolved_at.isoformat() if ticket.resolved_at else None,
        "agent_name": ticket.agent_name or "Unassigned",
        "resolution_notes": ticket.resolution_notes or "",
    }


# ─── LangChain Tools ──────────────────────────────────────────────────────────

@tool
def search_customer_by_name(name: str) -> str:
    """Search for a customer by name and return their full profile.

    Performs a case-insensitive partial name match. Returns customer details
    including plan type, account status, location, and company.

    Args:
        name: Customer name or partial name to search for.

    Returns:
        str: JSON-formatted customer profile(s), or error message if not found.
    """
    try:
        from core.database import Customer, get_session

        with get_session() as session:
            customers = (
                session.query(Customer)
                .filter(Customer.name.ilike(f"%{name}%"))
                .limit(10)
                .all()
            )

            if not customers:
                return f"No customers found matching name: '{name}'"

            results = [_format_customer(c) for c in customers]
            return json.dumps(results, indent=2, default=str)

    except Exception as e:
        logger.error("search_customer_by_name error: %s", e)
        return f"Error searching for customer: {str(e)}"


@tool
def get_customer_tickets(customer_id: int) -> str:
    """Retrieve all support tickets for a specific customer.

    Returns tickets sorted by creation date (newest first), including
    status, priority, category, and resolution details.

    Args:
        customer_id: The numeric customer ID.

    Returns:
        str: Formatted ticket list with summary statistics.
    """
    try:
        from core.database import Customer, SupportTicket, get_session

        with get_session() as session:
            customer = session.query(Customer).filter(Customer.id == customer_id).first()
            if not customer:
                return f"No customer found with ID: {customer_id}"

            tickets = (
                session.query(SupportTicket)
                .filter(SupportTicket.customer_id == customer_id)
                .order_by(SupportTicket.created_at.desc())
                .all()
            )

            if not tickets:
                return f"No tickets found for customer '{customer.name}' (ID: {customer_id})"

            ticket_list = [_format_ticket(t) for t in tickets]
            open_count = sum(1 for t in tickets if t.status in ("Open", "In Progress"))
            critical_count = sum(1 for t in tickets if t.priority == "Critical")

            result = {
                "customer": {"id": customer.id, "name": customer.name, "email": customer.email},
                "summary": {
                    "total_tickets": len(tickets),
                    "open_tickets": open_count,
                    "critical_tickets": critical_count,
                },
                "tickets": ticket_list,
            }
            return json.dumps(result, indent=2, default=str)

    except Exception as e:
        logger.error("get_customer_tickets error: %s", e)
        return f"Error retrieving tickets: {str(e)}"


@tool
def get_ticket_statistics() -> str:
    """Get aggregated statistics across all support tickets.

    Returns metrics including open tickets by priority, average resolution time,
    tickets by category, and agent performance overview.

    Returns:
        str: JSON-formatted statistics report.
    """
    try:
        from sqlalchemy import func
        from core.database import SupportTicket, Customer, get_session

        with get_session() as session:
            total = session.query(func.count(SupportTicket.id)).scalar()

            # Status breakdown
            status_counts = (
                session.query(SupportTicket.status, func.count(SupportTicket.id))
                .group_by(SupportTicket.status)
                .all()
            )

            # Priority breakdown for open tickets
            open_by_priority = (
                session.query(SupportTicket.priority, func.count(SupportTicket.id))
                .filter(SupportTicket.status.in_(["Open", "In Progress"]))
                .group_by(SupportTicket.priority)
                .all()
            )

            # Category breakdown
            category_counts = (
                session.query(SupportTicket.category, func.count(SupportTicket.id))
                .group_by(SupportTicket.category)
                .all()
            )

            # Average resolution time (hours) for resolved tickets
            resolved_tickets = (
                session.query(SupportTicket)
                .filter(
                    SupportTicket.status.in_(["Resolved", "Closed"]),
                    SupportTicket.resolved_at.isnot(None),
                )
                .all()
            )

            avg_resolution_hours = 0.0
            if resolved_tickets:
                resolution_times = [
                    (t.resolved_at - t.created_at).total_seconds() / 3600
                    for t in resolved_tickets
                    if t.resolved_at and t.created_at
                ]
                if resolution_times:
                    avg_resolution_hours = round(sum(resolution_times) / len(resolution_times), 1)

            # Top agents by resolved tickets
            top_agents = (
                session.query(SupportTicket.agent_name, func.count(SupportTicket.id).label("count"))
                .filter(
                    SupportTicket.agent_name.isnot(None),
                    SupportTicket.status.in_(["Resolved", "Closed"]),
                )
                .group_by(SupportTicket.agent_name)
                .order_by(func.count(SupportTicket.id).desc())
                .limit(5)
                .all()
            )

            stats = {
                "total_tickets": total,
                "by_status": dict(status_counts),
                "open_by_priority": dict(open_by_priority),
                "by_category": dict(category_counts),
                "avg_resolution_hours": avg_resolution_hours,
                "top_agents": [{"name": a, "resolved_count": c} for a, c in top_agents],
                "resolution_rate_pct": round(
                    sum(1 for s, c in status_counts if s in ("Resolved", "Closed"))
                    / max(len(status_counts), 1)
                    * 100,
                    1,
                ),
            }

            return json.dumps(stats, indent=2, default=str)

    except Exception as e:
        logger.error("get_ticket_statistics error: %s", e)
        return f"Error retrieving ticket statistics: {str(e)}"


@tool
def search_policy_documents(query: str) -> str:
    """Search policy documents using semantic similarity search.

    Retrieves the most relevant policy document sections matching the query.
    Returns content with source document name and page number citations.

    Args:
        query: Natural language question about company policies.

    Returns:
        str: Top matching policy document sections with source citations.
    """
    try:
        from core.vector_store import get_vector_store

        store = get_vector_store()
        results = store.similarity_search(query, k=5)

        if not results:
            return (
                "No policy documents found in the knowledge base. "
                "Please upload policy PDFs using the document upload feature."
            )

        output_parts = [f"**Policy Search Results for:** '{query}'\n"]
        for result in results:
            source = result.metadata.get("source", "Unknown source")
            filename = result.metadata.get("filename", "unknown.pdf")
            page = result.metadata.get("page", "?")
            score = result.score

            output_parts.append(
                f"\n---\n"
                f" **Source:** {filename} (Page {page})\n"
                f"**Relevance Score:** {1 - score:.2%}\n\n"
                f"{result.content}\n"
            )

        return "\n".join(output_parts)

    except Exception as e:
        logger.error("search_policy_documents error: %s", e)
        return f"Error searching policy documents: {str(e)}"


@tool
def get_billing_summary(customer_id: int) -> str:
    """Get billing history and outstanding amounts for a customer.

    Returns all billing records with payment status, amounts, and dates.
    Includes total outstanding balance calculation.

    Args:
        customer_id: The numeric customer ID.

    Returns:
        str: Formatted billing summary with outstanding balance.
    """
    try:
        from core.database import BillingRecord, Customer, get_session

        with get_session() as session:
            customer = session.query(Customer).filter(Customer.id == customer_id).first()
            if not customer:
                return f"No customer found with ID: {customer_id}"

            records = (
                session.query(BillingRecord)
                .filter(BillingRecord.customer_id == customer_id)
                .order_by(BillingRecord.invoice_date.desc())
                .all()
            )

            if not records:
                return f"No billing records found for customer '{customer.name}' (ID: {customer_id})"

            total_paid = sum(r.amount for r in records if r.status == "Paid")
            total_pending = sum(r.amount for r in records if r.status == "Pending")
            total_overdue = sum(r.amount for r in records if r.status == "Overdue")

            billing_list = [
                {
                    "id": r.id,
                    "amount": f"${r.amount:.2f}",
                    "status": r.status,
                    "invoice_date": r.invoice_date.strftime("%Y-%m-%d") if r.invoice_date else "N/A",
                    "due_date": r.due_date.strftime("%Y-%m-%d") if r.due_date else "N/A",
                    "description": r.description or "",
                }
                for r in records
            ]

            result = {
                "customer": {
                    "id": customer.id,
                    "name": customer.name,
                    "plan": customer.plan_type,
                },
                "billing_summary": {
                    "total_records": len(records),
                    "total_paid": f"${total_paid:.2f}",
                    "total_pending": f"${total_pending:.2f}",
                    "total_overdue": f"${total_overdue:.2f}",
                    "outstanding_balance": f"${total_pending + total_overdue:.2f}",
                },
                "records": billing_list,
            }

            return json.dumps(result, indent=2, default=str)

    except Exception as e:
        logger.error("get_billing_summary error: %s", e)
        return f"Error retrieving billing summary: {str(e)}"


@tool
def run_custom_sql(query: str) -> str:
    """Execute a custom read-only SQL query against the support database.

    Only SELECT statements are permitted. SQL injection is prevented by
    blocking all DML/DDL operations. Returns results as a formatted table.

    Args:
        query: A valid SQL SELECT statement.

    Returns:
        str: Query results formatted as a markdown table, or error message.
    """
    try:
        _validate_select_only(query)

        import pandas as pd
        from sqlalchemy import text
        from core.database import get_engine

        engine = get_engine()
        with engine.connect() as conn:
            result = conn.execute(text(query))
            rows = result.fetchmany(100)  # Limit results
            columns = list(result.keys())

        if not rows:
            return "Query returned no results."

        df = pd.DataFrame(rows, columns=columns)
        row_count = len(df)

        # Format as markdown table
        table_str = df.to_markdown(index=False)
        footer = f"\n*{row_count} row(s) returned*"
        if row_count == 100:
            footer += " *(limited to 100 rows)*"

        return f"```\n{table_str}\n```\n{footer}"

    except ValueError as ve:
        return f"Security Error: {str(ve)}"
    except Exception as e:
        logger.error("run_custom_sql error: %s", e)
        return f"Error executing SQL query: {str(e)}"


# ─── Tool Registry ─────────────────────────────────────────────────────────────

ALL_TOOLS = [
    search_customer_by_name,
    get_customer_tickets,
    get_ticket_statistics,
    search_policy_documents,
    get_billing_summary,
    run_custom_sql,
]

SQL_TOOLS = [
    search_customer_by_name,
    get_customer_tickets,
    get_ticket_statistics,
    get_billing_summary,
    run_custom_sql,
]

RAG_TOOLS = [
    search_policy_documents,
]
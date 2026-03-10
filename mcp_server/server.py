"""FastMCP Server for NexusAI Support.

Exposes the multi-agent system as an MCP (Model Context Protocol) server,
allowing external AI systems and tools to query the support platform.
"""

import logging
import sys
from pathlib import Path
from typing import Optional

# Ensure parent package is on path
sys.path.insert(0, str(Path(__file__).parent.parent))

logger = logging.getLogger(__name__)


def create_mcp_server():
    """Create and configure the FastMCP server instance.

    Returns:
        FastMCP: Configured MCP server with all tools and resources registered.
    """
    from fastmcp import FastMCP

    mcp = FastMCP(
        name="NexusAI Support",
        version="1.0.0",
        description=(
            "Multi-Agent Customer Support AI System. Provides natural language access to "
            "customer profiles, support tickets, billing records, and policy documents."
        ),
    )

    # ─── Tool: Query the Support System ──────────────────────────────────────

    @mcp.tool()
    async def query_support_system(query: str) -> str:
        """Query the NexusAI Support multi-agent system with a natural language question.

        Routes the query to the appropriate specialized agent:
        - SQL Agent: customer data, tickets, billing statistics
        - RAG Agent: policy documents, procedures, terms
        - Hybrid: questions requiring both data and policy context

        Args:
            query: Natural language question about customers, tickets, billing, or policies.

        Returns:
            str: Formatted response with answer, agent used, and source citations.
        """
        logger.info("MCP query_support_system: '%s...'", query[:60])
        try:
            from agents.orchestrator import get_supervisor

            supervisor = get_supervisor()
            response = supervisor.run(query)

            sources_text = ""
            if response.sources:
                sources_text = "\n\n**Sources:**\n" + "\n".join(f"- {s}" for s in response.sources)

            result = (
                f"{response.answer}"
                f"{sources_text}\n\n"
                f"---\n"
                f"*Agent: {response.agent_used} | "
                f"Type: {response.query_type} | "
                f"Confidence: {response.confidence:.0%} | "
                f"Time: {response.execution_time:.2f}s*"
            )
            return result

        except Exception as e:
            logger.error("MCP query_support_system error: %s", e)
            return f"Error processing query: {str(e)}"

    # ─── Tool: Ingest PDF Document ────────────────────────────────────────────

    @mcp.tool()
    async def ingest_pdf_document(file_path: str) -> str:
        """Ingest a PDF document into the knowledge base vector store.

        Loads, splits, embeds, and stores the PDF for semantic search.
        The document will be available for policy Q&A immediately after ingestion.

        Args:
            file_path: Absolute or relative path to the PDF file to ingest.

        Returns:
            str: Success message with chunk count, or error description.
        """
        logger.info("MCP ingest_pdf_document: %s", file_path)
        try:
            from core.vector_store import PDFIngestionPipeline, get_vector_store

            pipeline = PDFIngestionPipeline()
            chunks = pipeline.load_and_split(file_path)

            store = get_vector_store()
            added = store.add_documents(chunks)

            return (
                f"✅ Successfully ingested '{Path(file_path).name}'\n"
                f"- Chunks created: {added}\n"
                f"- Document is now searchable in the knowledge base"
            )

        except FileNotFoundError as e:
            return f"❌ File not found: {str(e)}"
        except ValueError as e:
            return f"❌ Invalid file: {str(e)}"
        except Exception as e:
            logger.error("MCP ingest_pdf_document error: %s", e)
            return f"❌ Ingestion failed: {str(e)}"

    # ─── Tool: List Knowledge Base ────────────────────────────────────────────

    @mcp.tool()
    async def list_knowledge_base() -> str:
        """List all documents currently in the knowledge base vector store.

        Returns:
            str: Formatted list of ingested documents with metadata.
        """
        logger.info("MCP list_knowledge_base")
        try:
            from core.vector_store import get_vector_store

            store = get_vector_store()
            documents = store.list_documents()

            if not documents:
                return "📚 Knowledge base is empty. No documents have been ingested yet."

            lines = [f"📚 Knowledge Base ({len(documents)} document(s)):\n"]
            for i, doc in enumerate(documents, 1):
                lines.append(
                    f"{i}. **{doc['filename']}**\n"
                    f"   - Chunks: {doc['chunk_count']}\n"
                    f"   - Pages: {doc.get('total_pages', 'N/A')}\n"
                    f"   - Uploaded: {doc.get('upload_timestamp', 'N/A')[:10]}\n"
                )

            return "\n".join(lines)

        except Exception as e:
            logger.error("MCP list_knowledge_base error: %s", e)
            return f"Error listing knowledge base: {str(e)}"

    # ─── Tool: Get System Stats ────────────────────────────────────────────────

    @mcp.tool()
    async def get_system_stats() -> str:
        """Get comprehensive statistics about the NexusAI Support system.

        Returns database record counts, vector store statistics, and system health.

        Returns:
            str: Formatted statistics report.
        """
        logger.info("MCP get_system_stats")
        try:
            from sqlalchemy import func
            from core.database import Customer, SupportTicket, BillingRecord, get_session
            from core.vector_store import get_vector_store

            stats = {}

            # Database stats
            with get_session() as session:
                stats["total_customers"] = session.query(func.count(Customer.id)).scalar()
                stats["total_tickets"] = session.query(func.count(SupportTicket.id)).scalar()
                stats["open_tickets"] = (
                    session.query(func.count(SupportTicket.id))
                    .filter(SupportTicket.status.in_(["Open", "In Progress"]))
                    .scalar()
                )
                stats["critical_tickets"] = (
                    session.query(func.count(SupportTicket.id))
                    .filter(
                        SupportTicket.priority == "Critical",
                        SupportTicket.status.in_(["Open", "In Progress"]),
                    )
                    .scalar()
                )
                stats["total_billing"] = session.query(func.count(BillingRecord.id)).scalar()
                stats["overdue_billing"] = (
                    session.query(func.count(BillingRecord.id))
                    .filter(BillingRecord.status == "Overdue")
                    .scalar()
                )

            # Vector store stats
            vs_stats = get_vector_store().get_stats()
            stats.update(vs_stats)

            return (
                f"📊 **NexusAI Support System Statistics**\n\n"
                f"**Database:**\n"
                f"- Customers: {stats['total_customers']}\n"
                f"- Total Tickets: {stats['total_tickets']}\n"
                f"- Open Tickets: {stats['open_tickets']}\n"
                f"- Critical Open: {stats['critical_tickets']}\n"
                f"- Billing Records: {stats['total_billing']}\n"
                f"- Overdue Invoices: {stats['overdue_billing']}\n\n"
                f"**Knowledge Base:**\n"
                f"- Documents: {stats['total_documents']}\n"
                f"- Indexed Chunks: {stats['total_chunks']}\n"
                f"- Storage: {stats['storage_path']}"
            )

        except Exception as e:
            logger.error("MCP get_system_stats error: %s", e)
            return f"Error retrieving system stats: {str(e)}"

    # ─── Resource: Health Check ───────────────────────────────────────────────

    @mcp.resource("system://health")
    async def health_check() -> str:
        """Health check endpoint for the NexusAI Support system.

        Returns:
            str: System health status.
        """
        try:
            from core.database import get_session, Customer
            from sqlalchemy import func

            with get_session() as session:
                customer_count = session.query(func.count(Customer.id)).scalar()

            return (
                f'{{"status": "healthy", "database": "connected", '
                f'"customers": {customer_count}, "service": "NexusAI Support MCP Server"}}'
            )
        except Exception as e:
            return f'{{"status": "unhealthy", "error": "{str(e)}"}}'

    return mcp


def run_server(host: Optional[str] = None, port: Optional[int] = None) -> None:
    """Start the MCP server.

    Args:
        host: Server host (defaults to config value).
        port: Server port (defaults to config value).
    """
    from core.config import get_settings, setup_logging

    setup_logging()

    settings = get_settings()
    server_host = host or settings.mcp_host
    server_port = port or settings.mcp_port

    logger.info("Starting NexusAI Support MCP Server on %s:%d", server_host, server_port)
    mcp = create_mcp_server()
    # FastMCP 0.4.x may not accept host/port — try progressively simpler calls
    try:
        mcp.run(transport="sse", host=server_host, port=server_port)
    except TypeError:
        try:
            mcp.run(transport="sse")
        except TypeError:
            mcp.run()


if __name__ == "__main__":
    run_server()
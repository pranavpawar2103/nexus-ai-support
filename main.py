"""NexusAI Support — Main Entry Point.

Bootstraps the entire application:
1. Validates configuration
2. Seeds the database if needed
3. Starts the MCP server in a background thread
4. Launches the Gradio UI
"""

import logging
import os
import sys
import threading
import time
from pathlib import Path

# ─── Colored terminal output ──────────────────────────────────────────────────

RESET = "\033[0m"
BOLD = "\033[1m"
CYAN = "\033[96m"
GREEN = "\033[92m"
YELLOW = "\033[93m"
RED = "\033[91m"
PURPLE = "\033[95m"
DIM = "\033[2m"


def cprint(text: str, color: str = RESET) -> None:
    """Print colored text to stdout."""
    print(f"{color}{text}{RESET}")


def print_banner() -> None:
    """Print the NexusAI Support startup banner."""
    banner = f"""
{CYAN}{BOLD}
  ███╗   ██╗███████╗██╗  ██╗██╗   ██╗███████╗ █████╗ ██╗
  ████╗  ██║██╔════╝╚██╗██╔╝██║   ██║██╔════╝██╔══██╗██║
  ██╔██╗ ██║█████╗   ╚███╔╝ ██║   ██║███████╗███████║██║
  ██║╚██╗██║██╔══╝   ██╔██╗ ██║   ██║╚════██║██╔══██║██║
  ██║ ╚████║███████╗██╔╝ ██╗╚██████╔╝███████║██║  ██║██║
  ╚═╝  ╚═══╝╚══════╝╚═╝  ╚═╝ ╚═════╝ ╚══════╝╚═╝  ╚═╝╚═╝
{RESET}
{PURPLE}  ⚡ NexusAI Support — Multi-Agent Customer Intelligence Platform{RESET}
{DIM}  Powered by LangChain · LangGraph · ChromaDB · OpenAI · FastMCP{RESET}
  {"─" * 60}"""
    print(banner)


def check_env() -> bool:
    """Validate that required environment variables are set.

    Returns:
        bool: True if all required vars are present, False otherwise.
    """
    env_file = Path(".env")
    env_example = Path(".env.example")

    if not env_file.exists():
        if env_example.exists():
            cprint("\n  ⚠️  No .env file found. Copying from .env.example...", YELLOW)
            import shutil
            shutil.copy(env_example, env_file)
            cprint("  📝 Created .env — please update OPENAI_API_KEY before continuing.", YELLOW)
            return False
        else:
            cprint("\n  ❌ No .env file found. Please create one from .env.example", RED)
            return False

    # Load dotenv
    from dotenv import load_dotenv
    load_dotenv(override=True)

    api_key = os.getenv("OPENAI_API_KEY", "")
    if not api_key or api_key == "your_key_here":
        cprint("\n  ❌ OPENAI_API_KEY is not set in .env", RED)
        cprint("  Please add your OpenAI API key to the .env file.", YELLOW)
        return False

    return True


def ensure_database() -> dict:
    """Ensure the database exists and is seeded with data.

    Returns:
        dict: Stats about the database (customer_count, ticket_count).
    """
    from core.config import get_settings
    from core.database import create_tables, Customer, SupportTicket, BillingRecord, get_session
    from sqlalchemy import func

    settings = get_settings()
    db_path = settings.get_db_abs_path()

    needs_seeding = not db_path.exists()

    # Create tables regardless
    create_tables()

    # Check if data exists
    with get_session() as session:
        customer_count = session.query(func.count(Customer.id)).scalar() or 0

    if needs_seeding or customer_count == 0:
        cprint("  🌱 Seeding database with synthetic data...", YELLOW)
        sys.path.insert(0, str(Path(__file__).parent / "data"))
        from seed_database import main as seed_main
        seed_main()
        # Re-check counts
        with get_session() as session:
            customer_count = session.query(func.count(Customer.id)).scalar() or 0
            ticket_count = session.query(func.count(SupportTicket.id)).scalar() or 0
            billing_count = session.query(func.count(BillingRecord.id)).scalar() or 0
    else:
        with get_session() as session:
            ticket_count = session.query(func.count(SupportTicket.id)).scalar() or 0
            billing_count = session.query(func.count(BillingRecord.id)).scalar() or 0

    return {
        "customers": customer_count,
        "tickets": ticket_count,
        "billing": billing_count,
    }


def ensure_vector_store() -> dict:
    """Ensure the ChromaDB vector store is ready.

    Returns:
        dict: Stats about the vector store.
    """
    from core.vector_store import get_vector_store

    store = get_vector_store()
    stats = store.get_stats()
    return stats


def start_mcp_server_background(host: str, port: int) -> threading.Thread:
    """Start the MCP server in a background daemon thread.

    Args:
        host: MCP server host.
        port: MCP server port.

    Returns:
        threading.Thread: The background thread.
    """
    def _run():
        try:
            from mcp_server.server import create_mcp_server
            mcp = create_mcp_server()
            # Try progressively simpler signatures for FastMCP compatibility
            try:
                mcp.run(transport="sse", host=host, port=port)
            except TypeError:
                try:
                    mcp.run(transport="sse")
                except TypeError:
                    mcp.run()
        except Exception as e:
            logging.getLogger(__name__).warning("MCP server unavailable (non-fatal): %s", e)

    thread = threading.Thread(target=_run, daemon=True, name="mcp-server")
    thread.start()
    return thread


def print_checklist(db_stats: dict, vs_stats: dict, mcp_host: str, mcp_port: int) -> None:
    """Print the startup checklist to the terminal.

    Args:
        db_stats: Database statistics.
        vs_stats: Vector store statistics.
        mcp_host: MCP server host.
        mcp_port: MCP server port.
    """
    print()
    cprint("  Startup Checklist:", BOLD)
    print()
    cprint(
        f"  {GREEN}✅{RESET} Database initialized  "
        f"{DIM}({db_stats['customers']} customers, {db_stats['tickets']} tickets, "
        f"{db_stats['billing']} billing records){RESET}"
    )
    cprint(
        f"  {GREEN}✅{RESET} Vector store ready    "
        f"{DIM}({vs_stats['total_documents']} documents, {vs_stats['total_chunks']} chunks){RESET}"
    )
    cprint(
        f"  {YELLOW}○{RESET}  MCP server           "
        f"{DIM}(optional, http://{mcp_host}:{mcp_port}){RESET}"
    )
    cprint(
        f"  {GREEN}✅{RESET} Gradio UI launching   "
        f"{DIM}(http://0.0.0.0:7860){RESET}"
    )
    print()
    cprint(f"  {'─' * 60}", DIM)
    cprint(f"\n  🌐 Open in browser: {CYAN}http://localhost:7860{RESET}\n")


def main() -> None:
    """Main application entry point."""
    print_banner()

    # ── Step 1: Validate environment ─────────────────────────────────────
    if not check_env():
        sys.exit(1)

    # ── Step 2: Setup logging ─────────────────────────────────────────────
    from core.config import setup_logging, get_settings
    setup_logging()
    logger = logging.getLogger(__name__)

    settings = get_settings()

    # ── Step 3: Initialize database ───────────────────────────────────────
    cprint("  Initializing systems...", DIM)
    try:
        db_stats = ensure_database()
    except Exception as e:
        cprint(f"\n Database initialization failed: {e}", RED)
        logger.exception("Database init failed")
        sys.exit(1)

    # ── Step 4: Initialize vector store ──────────────────────────────────
    try:
        vs_stats = ensure_vector_store()
    except Exception as e:
        cprint(f"\n Vector store warning: {e}", YELLOW)
        vs_stats = {"total_documents": 0, "total_chunks": 0}

    # ── Step 5: Start MCP server in background ────────────────────────────
    mcp_thread = start_mcp_server_background(settings.mcp_host, settings.mcp_port)
    time.sleep(0.5)  # Brief pause to let MCP server initialize

    # ── Step 6: Print checklist ───────────────────────────────────────────
    print_checklist(db_stats, vs_stats, settings.mcp_host, settings.mcp_port)

    # ── Step 7: Launch Gradio UI ──────────────────────────────────────────
    try:
        from ui.app import build_ui

        app = build_ui()
        app.launch(
            server_name="0.0.0.0",
            server_port=7860,
            share=False,
            show_error=True,
            inbrowser=False,
        )
    except KeyboardInterrupt:
        cprint("\n\n NexusAI Support shutting down gracefully.", CYAN)
    except Exception as e:
        cprint(f"\n  Failed to launch UI: {e}", RED)
        logger.exception("UI launch failed")
        sys.exit(1)


if __name__ == "__main__":
    main()
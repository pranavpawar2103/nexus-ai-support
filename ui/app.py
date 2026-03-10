"""NexusAI Support - Clean light UI."""

import json
import logging
import sys
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import gradio as gr

logger = logging.getLogger(__name__)

# Minimal CSS — only custom classes, no overrides of Gradio internals
CUSTOM_CSS = """
footer { display: none !important; }

.stat-card {
    border: 1px solid #e5e7eb;
    border-radius: 8px;
    padding: 12px 14px;
    margin-bottom: 6px;
    background: #ffffff;
}
.stat-label {
    font-size: 11px;
    color: #9ca3af;
    text-transform: uppercase;
    letter-spacing: 0.5px;
    margin-bottom: 4px;
}
.stat-value       { font-size: 22px; font-weight: 600; color: #111827; line-height: 1; }
.stat-value.blue  { color: #2563eb; }
.stat-value.amber { color: #d97706; }
.stat-value.green { color: #16a34a; }
.stat-sub         { font-size: 11px; color: #ef4444; margin-top: 3px; }

.meta-bar {
    padding: 7px 12px;
    background: #f9fafb;
    border: 1px solid #e5e7eb;
    border-radius: 6px;
    font-size: 12px;
    color: #6b7280;
    display: flex;
    align-items: center;
    gap: 8px;
    flex-wrap: wrap;
    min-height: 32px;
}
.atag {
    font-size: 11px;
    font-weight: 500;
    padding: 1px 7px;
    border-radius: 4px;
    font-family: monospace;
}
.atag-blue   { background:#eff6ff; color:#2563eb; }
.atag-amber  { background:#fffbeb; color:#d97706; }
.atag-purple { background:#f5f3ff; color:#7c3aed; }
.atag-green  { background:#f0fdf4; color:#16a34a; }
.atag-red    { background:#fef2f2; color:#dc2626; }

.msg-ok  { color: #16a34a; font-size: 12px; }
.msg-err { color: #dc2626; font-size: 12px; }
"""


def get_stats():
    try:
        from sqlalchemy import func
        from core.database import Customer, SupportTicket, get_session
        from core.vector_store import get_vector_store

        with get_session() as session:
            customers = session.query(func.count(Customer.id)).scalar() or 0
            open_t    = session.query(func.count(SupportTicket.id)).filter(
                SupportTicket.status.in_(["Open","In Progress"])).scalar() or 0
            critical  = session.query(func.count(SupportTicket.id)).filter(
                SupportTicket.priority=="Critical",
                SupportTicket.status.in_(["Open","In Progress"])).scalar() or 0

        docs = get_vector_store().get_stats().get("total_documents", 0)

        c = (f'<div class="stat-card"><div class="stat-label">Customers</div>'
             f'<div class="stat-value blue">{customers}</div></div>')
        t = (f'<div class="stat-card"><div class="stat-label">Open Tickets</div>'
             f'<div class="stat-value amber">{open_t}</div>'
             f'<div class="stat-sub">{critical} critical</div></div>')
        d = (f'<div class="stat-card"><div class="stat-label">Knowledge Docs</div>'
             f'<div class="stat-value green">{docs}</div></div>')
        return c, t, d
    except Exception as e:
        logger.error("get_stats: %s", e)
        blank = '<div class="stat-card"><div class="stat-value" style="color:#d1d5db">—</div></div>'
        return blank, blank, blank


def _tag(qt):
    m = {
        "SQL_QUERY":    '<span class="atag atag-blue">database</span>',
        "POLICY_QUERY": '<span class="atag atag-amber">policy</span>',
        "HYBRID_QUERY": '<span class="atag atag-purple">hybrid</span>',
        "GENERAL":      '<span class="atag atag-green">general</span>',
        "ERROR":        '<span class="atag atag-red">error</span>',
    }
    return m.get(qt, m["GENERAL"])


def process_query(user_message, chat_history, status_state):
    if not user_message.strip():
        c, t, d = get_stats()
        return chat_history, "", c, t, d, status_state

    chat_history = chat_history or []
    try:
        from agents.orchestrator import get_supervisor
        response = get_supervisor().run(user_message)

        sources = ""
        if response.sources:
            sources = "\n\n**Sources:** " + " · ".join(response.sources)
        chat_history.append([user_message, response.answer + sources])

        conf_color = "#16a34a" if response.confidence >= 0.8 else "#d97706" if response.confidence >= 0.6 else "#dc2626"
        meta = (
            f'<div class="meta-bar">'
            f'{_tag(response.query_type)}'
            f'<span style="color:#d1d5db">·</span>'
            f'<span>{response.agent_used}</span>'
            f'<span style="color:#d1d5db">·</span>'
            f'<span>{response.execution_time:.1f}s</span>'
            f'<span style="color:#d1d5db">·</span>'
            f'<span style="color:{conf_color}">{response.confidence:.0%} confidence</span>'
            f'</div>'
        )
        status_state["last"] = response.query_type
    except Exception as e:
        logger.error("process_query: %s", e)
        chat_history.append([user_message, f"**Error:** {e}"])
        meta = '<div class="meta-bar"><span class="atag atag-red">error</span></div>'

    c, t, d = get_stats()
    return chat_history, meta, c, t, d, status_state


def upload_pdf(file, status_state):
    if file is None:
        c, t, d = get_stats()
        return "No file selected.", c, t, d, status_state
    try:
        from core.vector_store import PDFIngestionPipeline, get_vector_store
        filename = Path(file.name).name
        if not filename.lower().endswith(".pdf"):
            c, t, d = get_stats()
            return '<span class="msg-err">Only PDF files accepted.</span>', c, t, d, status_state
        chunks = PDFIngestionPipeline().load_and_split(file.name)
        added  = get_vector_store().add_documents(chunks)
        msg    = f'<span class="msg-ok">{filename} — {added} chunks indexed.</span>'
        status_state["last_upload"] = filename
    except Exception as e:
        logger.error("upload_pdf: %s", e)
        msg = f'<span class="msg-err">Failed: {e}</span>'
    c, t, d = get_stats()
    return msg, c, t, d, status_state


def export_chat(chat_history):
    if not chat_history:
        return "Nothing to export."
    try:
        data = {"exported_at": datetime.utcnow().isoformat(),
                "messages": [{"user": u, "assistant": a} for u, a in chat_history]}
        p = Path("./nexus_chat_export.json")
        p.write_text(json.dumps(data, indent=2, default=str))
        return f"Saved to {p.name}"
    except Exception as e:
        return f"Failed: {e}"


def build_ui():
    theme = gr.themes.Soft(
        primary_hue=gr.themes.colors.blue,
        neutral_hue=gr.themes.colors.gray,
        font=[gr.themes.GoogleFont("Inter"), "ui-sans-serif", "sans-serif"],
        font_mono=[gr.themes.GoogleFont("JetBrains Mono"), "ui-monospace", "monospace"],
    )

    with gr.Blocks(title="NexusAI Support", css=CUSTOM_CSS, theme=theme) as app:

        status_state = gr.State({})

        gr.Markdown("## NexusAI Support &nbsp;·&nbsp; Customer Intelligence Platform")

        with gr.Row():

            # Sidebar
            with gr.Column(scale=3, min_width=210):
                gr.Markdown("**Overview**")
                customers_html = gr.HTML()
                tickets_html   = gr.HTML()
                docs_html      = gr.HTML()

                gr.Markdown("**Knowledge Base**")
                pdf_upload    = gr.File(label="Upload PDF", file_types=[".pdf"])
                upload_btn    = gr.Button("Ingest Document", size="sm", variant="secondary")
                upload_status = gr.HTML()

                gr.Markdown("**Quick Queries**")
                chip1 = gr.Button("Customer profile lookup",  size="sm", variant="secondary")
                chip2 = gr.Button("Critical open tickets",    size="sm", variant="secondary")
                chip3 = gr.Button("Overdue billing accounts", size="sm", variant="secondary")
                chip4 = gr.Button("Ticket statistics",        size="sm", variant="secondary")
                chip5 = gr.Button("Refund policy",            size="sm", variant="secondary")

            # Chat panel
            with gr.Column(scale=9):
                chatbot = gr.Chatbot(
                    label="",
                    height=460,
                    bubble_full_width=False,
                    show_copy_button=True,
                    show_label=False,
                )

                metadata_display = gr.HTML(
                    value='<div class="meta-bar" style="min-height:32px"></div>'
                )

                with gr.Row():
                    query_input = gr.Textbox(
                        placeholder="Ask about customers, tickets, billing, or policies...",
                        lines=2,
                        max_lines=6,
                        show_label=False,
                        scale=9,
                        container=False,
                    )
                    submit_btn = gr.Button("Send", variant="primary", scale=1, min_width=72)

                with gr.Row():
                    gr.Button("How many customers?", size="sm", variant="secondary").click(
                        fn=lambda: "How many customers do we have?", outputs=query_input)
                    gr.Button("Critical tickets",    size="sm", variant="secondary").click(
                        fn=lambda: "Show all critical open tickets", outputs=query_input)
                    gr.Button("Refund policy",       size="sm", variant="secondary").click(
                        fn=lambda: "What is the refund policy?", outputs=query_input)
                    gr.Button("Ticket stats",        size="sm", variant="secondary").click(
                        fn=lambda: "Full breakdown of ticket statistics by priority and category", outputs=query_input)

                with gr.Row():
                    clear_btn     = gr.Button("Clear",  size="sm", variant="secondary")
                    export_btn    = gr.Button("Export", size="sm", variant="secondary")
                    export_status = gr.Textbox(show_label=False, interactive=False,
                                               max_lines=1, scale=4, container=False)

        # Event wiring
        outs = [chatbot, metadata_display, customers_html, tickets_html, docs_html, status_state]

        submit_btn.click(
            fn=process_query, inputs=[query_input, chatbot, status_state], outputs=outs
        ).then(fn=lambda: "", outputs=query_input)

        query_input.submit(
            fn=process_query, inputs=[query_input, chatbot, status_state], outputs=outs
        ).then(fn=lambda: "", outputs=query_input)

        chip1.click(fn=lambda: "Find a customer named Sarah and show her full profile", outputs=query_input)
        chip2.click(fn=lambda: "Show all critical priority open tickets with customer names", outputs=query_input)
        chip3.click(fn=lambda: "List customers with overdue billing and amounts owed", outputs=query_input)
        chip4.click(fn=lambda: "Show full ticket resolution statistics and top agents", outputs=query_input)
        chip5.click(fn=lambda: "What is the refund policy?", outputs=query_input)

        upload_btn.click(
            fn=upload_pdf, inputs=[pdf_upload, status_state],
            outputs=[upload_status, customers_html, tickets_html, docs_html, status_state],
        )

        clear_btn.click(
            fn=lambda: ([], '<div class="meta-bar" style="min-height:32px"></div>'),
            outputs=[chatbot, metadata_display],
        )
        export_btn.click(fn=export_chat, inputs=[chatbot], outputs=export_status)
        app.load(fn=get_stats, outputs=[customers_html, tickets_html, docs_html])

    return app


def main():
    from core.config import setup_logging
    setup_logging()
    build_ui().launch(server_name="0.0.0.0", server_port=7860, share=False, show_error=True)


if __name__ == "__main__":
    main()
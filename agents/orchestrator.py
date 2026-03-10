"""LangGraph Orchestrator for NexusAI Support.

Implements a supervisor agent that classifies incoming queries and routes
them to the appropriate specialized agent (SQL, RAG, or hybrid).
"""

import logging
import time
from dataclasses import dataclass, field
from typing import Annotated, Any, Dict, List, Literal, Optional, TypedDict

from langchain_core.messages import AIMessage, HumanMessage

logger = logging.getLogger(__name__)

# ─── State Definition ─────────────────────────────────────────────────────────

QueryType = Literal["SQL_QUERY", "POLICY_QUERY", "HYBRID_QUERY", "GENERAL"]


class OrchestratorState(TypedDict):
    """LangGraph state for the orchestrator workflow.

    Attributes:
        messages: Conversation history.
        query: Original user query.
        query_type: Classified query type.
        agent_used: Which agent(s) handled the query.
        sql_response: Response from SQL agent (if used).
        rag_response: Response from RAG agent (if used).
        final_answer: Synthesized final response.
        sources: All source references.
        confidence: Confidence score 0.0-1.0.
        metadata: Additional metadata dict.
    """

    messages: List
    query: str
    query_type: str
    agent_used: str
    sql_response: Optional[str]
    rag_response: Optional[str]
    final_answer: str
    sources: List[str]
    confidence: float
    metadata: Dict[str, Any]


CLASSIFIER_PROMPT = """You are a query classifier for a customer support AI system.

Classify the user query into exactly one of these categories:

SQL_QUERY: Questions requiring database lookups (customer profiles, tickets, billing data, statistics)
  Examples: "Show me Emma's profile", "How many open tickets?", "List overdue invoices", 
            "What are the critical tickets?", "Customer billing history"

POLICY_QUERY: Questions about company policies, procedures, terms of service, refunds, SLAs
  Examples: "What is the refund policy?", "How long is the SLA for critical tickets?",
            "What does the privacy policy say?", "What are the cancellation terms?"

HYBRID_QUERY: Questions needing BOTH database data AND policy information
  Examples: "Is Emma eligible for a refund based on our policy?",
            "Check John's billing status and explain our overdue policy",
            "Show the SLA breach tickets and our escalation procedure"

GENERAL: Greetings, system questions, clarifications, help requests
  Examples: "Hello", "What can you do?", "Help", "Thank you"

Query: {query}

Respond with ONLY the category name (SQL_QUERY, POLICY_QUERY, HYBRID_QUERY, or GENERAL).
Do not include any explanation."""


SYNTHESIS_PROMPT = """You are NexusAI Support assistant, synthesizing information from multiple sources.

You have data from two specialized systems:

DATABASE RESPONSE (customer data, tickets, billing):
{sql_response}

POLICY RESPONSE (company policies, procedures):
{rag_response}

Original Question: {query}

Provide a comprehensive, unified answer that:
1. Directly addresses the user's question
2. Combines relevant database facts with applicable policy information
3. Is clearly formatted with headers if multiple sections are needed
4. Highlights any policy-relevant customer situations (e.g., SLA breaches, refund eligibility)
5. Ends with actionable next steps if appropriate

Response:"""

GENERAL_PROMPT = """You are NexusAI Support, an intelligent customer support intelligence platform.

Respond helpfully to this message: {query}

You can help with:
- Looking up customer profiles, tickets, and billing information
- Answering questions about company policies and procedures  
- Providing support statistics and analytics
- Searching policy documents

Keep your response friendly, professional, and concise."""


@dataclass
class FinalResponse:
    """Final response from the orchestrator.

    Attributes:
        answer: The complete, formatted answer.
        agent_used: Which agent(s) handled the query.
        query_type: The classified query type.
        sources: List of source references.
        confidence: Confidence score 0.0-1.0.
        execution_time: Total processing time in seconds.
        metadata: Additional response metadata.
    """

    answer: str
    agent_used: str
    query_type: str
    sources: List[str] = field(default_factory=list)
    confidence: float = 0.8
    execution_time: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)


class SupervisorAgent:
    """LangGraph-based supervisor that routes queries to specialized agents.

    Classifies each incoming query and routes to SQL agent, RAG agent,
    or both (hybrid), then synthesizes a final response.
    """

    def __init__(self) -> None:
        """Initialize the SupervisorAgent with lazy-loaded sub-agents."""
        self._sql_agent = None
        self._rag_agent = None
        self._llm = None
        self._graph = None
        logger.info("SupervisorAgent initialized")

    def _get_llm(self):
        """Get or create the classifier LLM."""
        if self._llm is None:
            from core.config import get_llm
            self._llm = get_llm(temperature=0.0)
        return self._llm

    def _get_sql_agent(self):
        """Get or create the SQL agent."""
        if self._sql_agent is None:
            from agents.sql_agent import SQLAgent
            self._sql_agent = SQLAgent()
        return self._sql_agent

    def _get_rag_agent(self):
        """Get or create the RAG agent."""
        if self._rag_agent is None:
            from agents.rag_agent import RAGAgent
            self._rag_agent = RAGAgent()
        return self._rag_agent

    # ─── LangGraph Nodes ──────────────────────────────────────────────────────

    def _classify_query(self, state: OrchestratorState) -> OrchestratorState:
        """Classify the query type using LLM.

        Args:
            state: Current orchestrator state.

        Returns:
            Updated state with query_type set.
        """
        query = state["query"]
        logger.info("Classifying query: '%s...'", query[:60])

        try:
            llm = self._get_llm()
            prompt = CLASSIFIER_PROMPT.format(query=query)
            response = llm.invoke([HumanMessage(content=prompt)])
            raw = response.content.strip().upper()

            # Validate response
            valid_types = {"SQL_QUERY", "POLICY_QUERY", "HYBRID_QUERY", "GENERAL"}
            if raw in valid_types:
                query_type = raw
            else:
                # Fallback classification based on keywords
                lower_query = query.lower()
                if any(w in lower_query for w in ["policy", "refund", "sla", "terms", "procedure", "cancel"]):
                    query_type = "POLICY_QUERY"
                elif any(w in lower_query for w in ["customer", "ticket", "billing", "invoice", "show", "list", "find"]):
                    query_type = "SQL_QUERY"
                else:
                    query_type = "GENERAL"

            logger.info("Query classified as: %s", query_type)
            state["query_type"] = query_type
            return state

        except Exception as e:
            logger.error("Classification error: %s", e)
            state["query_type"] = "GENERAL"
            return state

    def _sql_node(self, state: OrchestratorState) -> OrchestratorState:
        """Execute SQL agent for database queries.

        Args:
            state: Current orchestrator state.

        Returns:
            Updated state with sql_response populated.
        """
        try:
            agent = self._get_sql_agent()
            response = agent.query(state["query"])
            state["sql_response"] = response.answer
            state["agent_used"] = "SQL Agent"
            if response.sources:
                state["sources"].extend(response.sources)
            state["metadata"]["sql_execution_time"] = response.execution_time
        except Exception as e:
            logger.error("SQL node error: %s", e)
            state["sql_response"] = f"Database query failed: {str(e)}"
            state["agent_used"] = "SQL Agent (error)"
        return state

    def _rag_node(self, state: OrchestratorState) -> OrchestratorState:
        """Execute RAG agent for policy document queries.

        Args:
            state: Current orchestrator state.

        Returns:
            Updated state with rag_response populated.
        """
        try:
            agent = self._get_rag_agent()
            response = agent.query(state["query"])
            state["rag_response"] = response.answer
            state["agent_used"] = "RAG Agent"

            # Parse the References block the LLM wrote.
            # Supports: "Page 6", "page 6", "p. 6", "p.6", bold markdown **[1]...**
            import re
            answer_text = response.answer
            cited_sources = set()

            for m in re.finditer(
                r'\[\d+\]\s+\*{0,2}(.+?),\s*(?:[Pp]age|[Pp]\.)\ *(\d+)\*{0,2}',
                answer_text
            ):
                fname = m.group(1).strip().strip('*')
                page = int(m.group(2))
                cited_sources.add((fname, page))

            # Cross-validate: only show sources that are BOTH cited by the LLM
            # AND actually present in the retrieved chunks. This blocks any
            # hallucinated page numbers the LLM may have invented.
            retrieved = {(doc.filename, doc.page) for doc in response.source_documents}
            verified_sources = cited_sources & retrieved

            # Detect "not found" answers — no sources should be shown
            NOT_FOUND_PHRASES = [
                "could not find information",
                "not found in the available",
                "no documents have been uploaded",
            ]
            is_not_found = any(p in answer_text.lower() for p in NOT_FOUND_PHRASES)

            for doc in response.source_documents:
                if is_not_found:
                    # Answer is "not found" — suppress all sources, nothing to cite
                    break
                elif verified_sources:
                    # Normal case: only show verified cited sources
                    if (doc.filename, doc.page) in verified_sources:
                        source_ref = f"📄 {doc.filename} (Page {doc.page})"
                        if source_ref not in state["sources"]:
                            state["sources"].append(source_ref)
                else:
                    # Fallback: LLM gave an answer but wrote no References block.
                    # Show all retrieved docs so the user has some provenance.
                    source_ref = f"📄 {doc.filename} (Page {doc.page})"
                    if source_ref not in state["sources"]:
                        state["sources"].append(source_ref)

            state["metadata"]["rag_execution_time"] = response.execution_time
            state["metadata"]["found_in_context"] = response.found_in_context
            state["confidence"] = 0.9 if response.found_in_context else 0.4

        except Exception as e:
            logger.error("RAG node error: %s", e)
            state["rag_response"] = f"Policy search failed: {str(e)}"
            state["agent_used"] = "RAG Agent (error)"
        return state

    def _hybrid_node(self, state: OrchestratorState) -> OrchestratorState:
        """Execute both SQL and RAG agents, then synthesize results.

        Args:
            state: Current orchestrator state.

        Returns:
            Updated state with both responses and synthesis metadata.
        """
        # Run both agents
        state = self._sql_node(state)
        state = self._rag_node(state)
        state["agent_used"] = "SQL + RAG (Hybrid)"
        return state

    def _general_node(self, state: OrchestratorState) -> OrchestratorState:
        """Handle general/conversational queries.

        Args:
            state: Current orchestrator state.

        Returns:
            Updated state with a helpful general response.
        """
        try:
            llm = self._get_llm()
            prompt = GENERAL_PROMPT.format(query=state["query"])
            response = llm.invoke([HumanMessage(content=prompt)])
            state["final_answer"] = response.content
            state["agent_used"] = "General Assistant"
            state["confidence"] = 0.95
        except Exception as e:
            logger.error("General node error: %s", e)
            state["final_answer"] = (
                "Hello! I'm NexusAI Support. I can help you look up customer information, "
                "search support tickets, check billing records, and answer policy questions. "
                "What would you like to know?"
            )
            state["agent_used"] = "General Assistant"
        return state

    def _format_response(self, state: OrchestratorState) -> OrchestratorState:
        """Format the final response based on which agents were used.

        Args:
            state: Current orchestrator state.

        Returns:
            Updated state with final_answer formatted.
        """
        if state.get("final_answer"):
            # Already set (general queries)
            return state

        query_type = state["query_type"]

        if query_type == "SQL_QUERY":
            answer = state.get("sql_response", "No data found.")
            state["confidence"] = 0.85

        elif query_type == "POLICY_QUERY":
            answer = state.get("rag_response", "Policy information not available.")

        elif query_type == "HYBRID_QUERY":
            sql_resp = state.get("sql_response", "")
            rag_resp = state.get("rag_response", "")

            if sql_resp and rag_resp:
                try:
                    llm = self._get_llm()
                    synthesis_prompt = SYNTHESIS_PROMPT.format(
                        sql_response=sql_resp,
                        rag_response=rag_resp,
                        query=state["query"],
                    )
                    response = llm.invoke([HumanMessage(content=synthesis_prompt)])
                    answer = response.content
                    state["confidence"] = 0.80
                except Exception as e:
                    logger.error("Synthesis error: %s", e)
                    answer = f"**Database Information:**\n{sql_resp}\n\n**Policy Information:**\n{rag_resp}"
            else:
                answer = sql_resp or rag_resp or "Unable to retrieve information."

        else:
            answer = state.get("final_answer", "I couldn't process your request.")

        # Low confidence warning
        if state["confidence"] < 0.7:
            answer = (
                " *I'm not fully confident in this response. Please verify with a supervisor.*\n\n"
                + answer
            )

        state["final_answer"] = answer
        return state

    # ─── Graph Construction & Execution ──────────────────────────────────────

    def _build_graph(self):
        """Build the LangGraph workflow.

        Returns:
            Compiled LangGraph StateGraph.
        """
        from langgraph.graph import StateGraph, START, END

        workflow = StateGraph(OrchestratorState)

        # Add nodes
        workflow.add_node("classify", self._classify_query)
        workflow.add_node("sql_node", self._sql_node)
        workflow.add_node("rag_node", self._rag_node)
        workflow.add_node("hybrid_node", self._hybrid_node)
        workflow.add_node("general_node", self._general_node)
        workflow.add_node("format_response", self._format_response)

        # Entry point
        workflow.add_edge(START, "classify")

        # Conditional routing based on classification
        def route_query(state: OrchestratorState) -> str:
            return {
                "SQL_QUERY": "sql_node",
                "POLICY_QUERY": "rag_node",
                "HYBRID_QUERY": "hybrid_node",
                "GENERAL": "general_node",
            }.get(state["query_type"], "general_node")

        workflow.add_conditional_edges(
            "classify",
            route_query,
            {
                "sql_node": "sql_node",
                "rag_node": "rag_node",
                "hybrid_node": "hybrid_node",
                "general_node": "general_node",
            },
        )

        # All agent nodes route to format_response
        workflow.add_edge("sql_node", "format_response")
        workflow.add_edge("rag_node", "format_response")
        workflow.add_edge("hybrid_node", "format_response")
        workflow.add_edge("general_node", "format_response")
        workflow.add_edge("format_response", END)

        return workflow.compile()

    def _get_graph(self):
        """Get or create the compiled LangGraph workflow."""
        if self._graph is None:
            self._graph = self._build_graph()
        return self._graph

    def run(self, query: str) -> FinalResponse:
        """Process a user query through the multi-agent orchestration pipeline.

        Args:
            query: Natural language query from the user.

        Returns:
            FinalResponse: Complete response with answer, metadata, and sources.
        """
        start_time = time.time()
        logger.info("SupervisorAgent processing: '%s...'", query[:80])

        initial_state: OrchestratorState = {
            "messages": [HumanMessage(content=query)],
            "query": query,
            "query_type": "",
            "agent_used": "",
            "sql_response": None,
            "rag_response": None,
            "final_answer": "",
            "sources": [],
            "confidence": 0.8,
            "metadata": {},
        }

        try:
            graph = self._get_graph()
            final_state = graph.invoke(initial_state)

            execution_time = time.time() - start_time
            logger.info(
                "SupervisorAgent completed in %.2fs (type=%s, agent=%s, confidence=%.2f)",
                execution_time,
                final_state.get("query_type"),
                final_state.get("agent_used"),
                final_state.get("confidence", 0.8),
            )

            return FinalResponse(
                answer=final_state.get("final_answer", "No response generated."),
                agent_used=final_state.get("agent_used", "Unknown"),
                query_type=final_state.get("query_type", "GENERAL"),
                sources=final_state.get("sources", []),
                confidence=final_state.get("confidence", 0.8),
                execution_time=execution_time,
                metadata=final_state.get("metadata", {}),
            )

        except Exception as e:
            execution_time = time.time() - start_time
            error_msg = str(e)
            logger.error("SupervisorAgent error after %.2fs: %s", execution_time, error_msg)

            return FinalResponse(
                answer=(
                    f"I encountered an unexpected error while processing your request.\n\n"
                    f"**Error:** {error_msg}\n\n"
                    "Please try again or rephrase your question. If the issue persists, "
                    "contact the system administrator."
                ),
                agent_used="Error Handler",
                query_type="ERROR",
                confidence=0.0,
                execution_time=execution_time,
                metadata={"error": error_msg},
            )


# ─── Module singleton ─────────────────────────────────────────────────────────

_supervisor: Optional[SupervisorAgent] = None


def get_supervisor() -> SupervisorAgent:
    """Get or create the singleton SupervisorAgent.

    Returns:
        SupervisorAgent: The global supervisor instance.
    """
    global _supervisor
    if _supervisor is None:
        _supervisor = SupervisorAgent()
    return _supervisor
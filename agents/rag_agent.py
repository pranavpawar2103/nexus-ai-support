"""RAG Agent for NexusAI Support.

Provides Retrieval-Augmented Generation for policy document Q&A.
Uses ChromaDB vector store with OpenAI embeddings for semantic retrieval.
"""

import logging
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

RAG_SYSTEM_PROMPT = """You are NexusPolicy, an expert customer support policy analyst for NexusAI Support.

Your role is to answer questions EXCLUSIVELY based on the provided policy document context.

STRICT RULES:
1. ONLY answer from the provided context. Do NOT use general knowledge.
2. If the answer is not found in the context, respond EXACTLY with:
   "I could not find information about this topic in the available policy documents.
   Please consult your supervisor or the official policy portal for guidance."
3. Paraphrase policy content in plain English — do NOT reproduce long verbatim quotes.
4. Use numbered superscript citations [1], [2], etc. inline after each claim.
5. End with a compact reference list mapping each number to its source.
6. Be precise and factual. Do not speculate beyond what the documents say.
7. If partial information is available, share what you found and clearly state what is missing.

RESPONSE FORMAT:
Write a clear, concise answer in plain prose or short bullet points.
Place a citation marker like [1] immediately after each claim that comes from a source.
Then end with a "**References**" section listing each source once:

**References**
[1] filename.pdf, Page N
[2] filename.pdf, Page N

Context:
{context}

Question: {question}

Answer:"""


@dataclass
class SourceDocument:
    """A source document used to generate a RAG response.

    Attributes:
        content: The retrieved text chunk.
        filename: Source PDF filename.
        page: Page number within the document.
        score: Relevance score (lower = more similar).
    """

    content: str
    filename: str
    page: int
    score: float


@dataclass
class RAGResponse:
    """Response from the RAG agent.

    Attributes:
        answer: Generated answer based on retrieved context.
        source_documents: List of source documents used.
        query: The original question asked.
        execution_time: Time taken to generate response in seconds.
        found_in_context: Whether an answer was found in the documents.
        error: Error message if the query failed.
    """

    answer: str
    source_documents: List[SourceDocument] = field(default_factory=list)
    query: str = ""
    execution_time: float = 0.0
    found_in_context: bool = True
    error: Optional[str] = None


class RAGAgent:
    """Retrieval-Augmented Generation agent for policy document Q&A.

    Combines ChromaDB semantic search with GPT-4 for accurate, cited
    answers to policy questions. Uses temperature=0 for factual accuracy.

    Attributes:
        k: Number of documents to retrieve for each query.
    """

    def __init__(self, k: int = 5) -> None:
        """Initialize the RAG agent.

        Args:
            k: Number of top documents to retrieve per query.
        """
        self.k = k
        self._chain = None
        logger.info("RAGAgent initialized (k=%d)", k)

    def _build_context(self, search_results) -> tuple[str, List[SourceDocument]]:
        """Build context string and source list from search results.

        Args:
            search_results: List of SearchResult objects from vector store.

        Returns:
            Tuple of (context_string, source_documents_list).
        """
        context_parts = []
        sources = []

        for result in search_results:
            filename = result.metadata.get("filename", "unknown.pdf")
            page = result.metadata.get("page", 0)

            context_parts.append(
                f"[Document: {filename}, Page {page}]\n{result.content}"
            )

            sources.append(
                SourceDocument(
                    content=result.content,
                    filename=filename,
                    page=page,
                    score=result.score,
                )
            )

        return "\n\n---\n\n".join(context_parts), sources

    def query(self, question: str) -> RAGResponse:
        """Answer a policy question using retrieved document context.

        Retrieves relevant document chunks, builds context, and generates
        a precise, cited answer using the configured LLM.

        Args:
            question: Natural language policy question.

        Returns:
            RAGResponse: Contains answer, source documents, and metadata.
        """
        start_time = time.time()
        logger.info("RAGAgent processing: '%s...'", question[:80])

        try:
            from core.vector_store import get_vector_store
            from core.config import get_llm
            from langchain_core.messages import HumanMessage, SystemMessage

            # Retrieve relevant documents
            store = get_vector_store()
            search_results = store.similarity_search(question, k=self.k)

            if not search_results:
                execution_time = time.time() - start_time
                return RAGResponse(
                    answer=(
                        "I could not find information about this topic in the available policy documents. "
                        "No documents have been uploaded to the knowledge base yet. "
                        "Please upload policy PDF documents using the document upload feature, "
                        "then ask your question again."
                    ),
                    query=question,
                    execution_time=execution_time,
                    found_in_context=False,
                )

            context, sources = self._build_context(search_results)

            # Build prompt with retrieved context
            prompt_text = RAG_SYSTEM_PROMPT.format(
                context=context,
                question=question,
            )

            # Generate answer with LLM
            llm = get_llm(temperature=0.0)
            response = llm.invoke([HumanMessage(content=prompt_text)])
            answer = response.content

            execution_time = time.time() - start_time
            found_in_context = "could not find" not in answer.lower()

            logger.info(
                "RAGAgent completed in %.2fs (found_in_context=%s, sources=%d)",
                execution_time,
                found_in_context,
                len(sources),
            )

            return RAGResponse(
                answer=answer,
                source_documents=sources,
                query=question,
                execution_time=execution_time,
                found_in_context=found_in_context,
            )

        except Exception as e:
            execution_time = time.time() - start_time
            error_msg = str(e)
            logger.error("RAGAgent error after %.2fs: %s", execution_time, error_msg)

            return RAGResponse(
                answer=(
                    f"I encountered an error while searching the policy documents: {error_msg}\n\n"
                    "Please try rephrasing your question or contact support if the issue persists."
                ),
                query=question,
                execution_time=execution_time,
                found_in_context=False,
                error=error_msg,
            )
"""Vector store management for NexusAI Support.

Provides PDF ingestion pipeline and ChromaDB vector store management
for semantic search over policy documents and knowledge base content.
"""

import logging
import os
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)

COLLECTION_NAME = "nexus_policy_docs"


@dataclass
class DocumentChunk:
    """Represents a chunk of text from an ingested document.

    Attributes:
        content: The text content of the chunk.
        metadata: Associated metadata (filename, page, etc.).
        chunk_id: Unique identifier for this chunk.
    """

    content: str
    metadata: Dict
    chunk_id: str


@dataclass
class SearchResult:
    """Result from a similarity search.

    Attributes:
        content: Matching text chunk.
        metadata: Source document metadata.
        score: Similarity score (lower is more similar in L2).
        rank: Position in result set.
    """

    content: str
    metadata: Dict
    score: float
    rank: int


class PDFIngestionPipeline:
    """Handles loading, splitting, embedding, and storing PDF documents.

    This pipeline takes a PDF file, splits it into overlapping chunks,
    generates embeddings via OpenAI, and stores them in ChromaDB with
    rich metadata for retrieval.

    Attributes:
        chunk_size: Maximum characters per chunk.
        chunk_overlap: Character overlap between consecutive chunks.
    """

    def __init__(self, chunk_size: int = 1000, chunk_overlap: int = 200) -> None:
        """Initialize the PDF ingestion pipeline.

        Args:
            chunk_size: Maximum size of each text chunk in characters.
            chunk_overlap: Number of overlapping characters between chunks.
        """
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        logger.info(
            "PDFIngestionPipeline initialized (chunk_size=%d, overlap=%d)",
            chunk_size,
            chunk_overlap,
        )

    def load_and_split(self, pdf_path: str) -> List[DocumentChunk]:
        """Load a PDF file and split it into overlapping chunks.

        Args:
            pdf_path: Path to the PDF file to ingest.

        Returns:
            List[DocumentChunk]: List of document chunks with metadata.

        Raises:
            FileNotFoundError: If the PDF file does not exist.
            ValueError: If the file is not a valid PDF.
        """
        from langchain_community.document_loaders import PyPDFLoader
        from langchain.text_splitter import RecursiveCharacterTextSplitter

        path = Path(pdf_path)
        if not path.exists():
            raise FileNotFoundError(f"PDF file not found: {pdf_path}")

        if not path.suffix.lower() == ".pdf":
            raise ValueError(f"File must be a PDF: {pdf_path}")

        # Validate magic bytes (PDF signature: %PDF)
        with open(path, "rb") as f:
            header = f.read(4)
            if header != b"%PDF":
                raise ValueError(f"File is not a valid PDF (invalid magic bytes): {pdf_path}")

        logger.info("Loading PDF: %s", path.name)
        loader = PyPDFLoader(str(path))
        pages = loader.load()

        splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            separators=["\n\n", "\n", ". ", " ", ""],
            length_function=len,
        )

        split_docs = splitter.split_documents(pages)
        upload_timestamp = datetime.utcnow().isoformat()

        chunks = []
        for i, doc in enumerate(split_docs):
            page_num = doc.metadata.get("page", 0) + 1  # 1-indexed
            chunk = DocumentChunk(
                content=doc.page_content,
                metadata={
                    "filename": path.name,
                    "filepath": str(path.resolve()),
                    "page": page_num,
                    "upload_timestamp": upload_timestamp,
                    "chunk_index": i,
                    "total_chunks": len(split_docs),
                    "source": f"{path.name} (page {page_num})",
                },
                chunk_id=f"{path.stem}__chunk_{i:04d}",
            )
            chunks.append(chunk)

        logger.info(
            "Split '%s' into %d chunks across %d pages",
            path.name,
            len(chunks),
            len(pages),
        )
        return chunks


class VectorStoreManager:
    """Manages a ChromaDB vector store for document storage and retrieval.

    Handles collection lifecycle, document addition, similarity search,
    and document management. Uses persistent storage for session continuity.

    Attributes:
        chroma_path: Filesystem path for ChromaDB persistence.
        collection_name: Name of the ChromaDB collection.
    """

    def __init__(
        self,
        chroma_path: Optional[str] = None,
        collection_name: str = COLLECTION_NAME,
    ) -> None:
        """Initialize the vector store manager.

        Args:
            chroma_path: Path for ChromaDB persistence. Uses config default if None.
            collection_name: Name of the ChromaDB collection to use.
        """
        from core.config import get_settings

        if chroma_path is None:
            settings = get_settings()
            chroma_path = str(settings.get_chroma_abs_path())

        self.chroma_path = chroma_path
        self.collection_name = collection_name
        self._client = None
        self._collection = None
        self._embeddings = None

        # Ensure storage directory exists
        os.makedirs(chroma_path, exist_ok=True)
        logger.info("VectorStoreManager initialized at: %s", chroma_path)

    def _get_client(self):
        """Get or create the ChromaDB persistent client."""
        if self._client is None:
            import chromadb

            self._client = chromadb.PersistentClient(path=self.chroma_path)
        return self._client

    def _get_embeddings(self):
        """Get or create the embeddings model."""
        if self._embeddings is None:
            from core.config import get_embeddings
            self._embeddings = get_embeddings()
        return self._embeddings

    def get_or_create_collection(self):
        """Get the existing collection or create a new one.

        Returns:
            chromadb.Collection: The active ChromaDB collection.
        """
        if self._collection is None:
            client = self._get_client()
            self._collection = client.get_or_create_collection(
                name=self.collection_name,
                metadata={"hnsw:space": "cosine"},
            )
            logger.info(
                "Collection '%s' ready (%d documents)",
                self.collection_name,
                self._collection.count(),
            )
        return self._collection

    def add_documents(self, chunks: List[DocumentChunk]) -> int:
        """Add document chunks to the vector store.

        Args:
            chunks: List of DocumentChunk objects to embed and store.

        Returns:
            int: Number of chunks successfully added.

        Raises:
            RuntimeError: If embedding or storage fails.
        """
        if not chunks:
            logger.warning("No chunks provided to add_documents")
            return 0

        collection = self.get_or_create_collection()
        embeddings_model = self._get_embeddings()

        texts = [chunk.content for chunk in chunks]
        metadatas = [chunk.metadata for chunk in chunks]
        ids = [chunk.chunk_id for chunk in chunks]

        # Generate embeddings
        logger.info("Generating embeddings for %d chunks...", len(chunks))
        embeddings = embeddings_model.embed_documents(texts)

        # Store in ChromaDB (upsert to handle re-ingestion)
        collection.upsert(
            embeddings=embeddings,
            documents=texts,
            metadatas=metadatas,
            ids=ids,
        )

        count = collection.count()
        logger.info(
            "Added %d chunks. Collection now has %d total documents.",
            len(chunks),
            count,
        )
        return len(chunks)

    def similarity_search(self, query: str, k: int = 5) -> List[SearchResult]:
        """Perform semantic similarity search over stored documents.

        Args:
            query: Natural language query to search for.
            k: Number of top results to return.

        Returns:
            List[SearchResult]: Ranked list of matching document chunks.
        """
        collection = self.get_or_create_collection()
        embeddings_model = self._get_embeddings()

        if collection.count() == 0:
            logger.warning("Vector store is empty. No documents to search.")
            return []

        query_embedding = embeddings_model.embed_query(query)

        results = collection.query(
            query_embeddings=[query_embedding],
            n_results=min(k, collection.count()),
            include=["documents", "metadatas", "distances"],
        )

        search_results = []
        if results and results["documents"]:
            for i, (doc, meta, dist) in enumerate(
                zip(
                    results["documents"][0],
                    results["metadatas"][0],
                    results["distances"][0],
                )
            ):
                search_results.append(
                    SearchResult(
                        content=doc,
                        metadata=meta,
                        score=float(dist),
                        rank=i + 1,
                    )
                )

        logger.info(
            "Similarity search returned %d results for query: '%s...'",
            len(search_results),
            query[:50],
        )
        return search_results

    def list_documents(self) -> List[Dict]:
        """List all unique documents in the vector store.

        Returns:
            List[Dict]: List of document metadata dicts (filename, chunks, timestamp).
        """
        collection = self.get_or_create_collection()

        if collection.count() == 0:
            return []

        all_items = collection.get(include=["metadatas"])
        metadatas = all_items.get("metadatas", [])

        # Aggregate by filename
        docs: Dict[str, Dict] = {}
        for meta in metadatas:
            filename = meta.get("filename", "unknown")
            if filename not in docs:
                docs[filename] = {
                    "filename": filename,
                    "chunk_count": 0,
                    "upload_timestamp": meta.get("upload_timestamp", ""),
                    "total_pages": 0,
                }
            docs[filename]["chunk_count"] += 1
            page = meta.get("page", 0)
            if page > docs[filename]["total_pages"]:
                docs[filename]["total_pages"] = page

        return list(docs.values())

    def delete_document(self, filename: str) -> int:
        """Delete all chunks belonging to a specific document.

        Args:
            filename: Name of the file to delete from the store.

        Returns:
            int: Number of chunks deleted.
        """
        collection = self.get_or_create_collection()

        # Get all chunk IDs for this filename
        results = collection.get(where={"filename": filename}, include=["metadatas"])
        ids_to_delete = results.get("ids", [])

        if ids_to_delete:
            collection.delete(ids=ids_to_delete)
            logger.info("Deleted %d chunks for document: %s", len(ids_to_delete), filename)
        else:
            logger.warning("No chunks found for document: %s", filename)

        return len(ids_to_delete)

    def get_stats(self) -> Dict:
        """Get statistics about the vector store.

        Returns:
            Dict: Statistics including total chunks and document count.
        """
        collection = self.get_or_create_collection()
        total_chunks = collection.count()
        documents = self.list_documents()

        return {
            "total_chunks": total_chunks,
            "total_documents": len(documents),
            "collection_name": self.collection_name,
            "storage_path": self.chroma_path,
        }


# ─── Module-level singleton ────────────────────────────────────────────────────

_vector_store_manager: Optional[VectorStoreManager] = None


def get_vector_store() -> VectorStoreManager:
    """Get or create the singleton VectorStoreManager.

    Returns:
        VectorStoreManager: The global vector store manager instance.
    """
    global _vector_store_manager
    if _vector_store_manager is None:
        _vector_store_manager = VectorStoreManager()
    return _vector_store_manager
"""
retriever.py - Dense Retrieval Module for the Local RAG System

This module handles the retrieval step of the RAG pipeline:
1. Takes a user query
2. Generates an embedding for the query using the same model used during ingestion
3. Searches the FAISS index for the most similar document chunks
4. Returns the top-k chunks with their metadata and similarity scores

Uses dense retrieval via FAISS inner product search (equivalent to
cosine similarity when embeddings are L2-normalized).
"""

from typing import List, Dict, Any, Optional

import numpy as np
import faiss
from sentence_transformers import SentenceTransformer

from ingest import EMBEDDING_MODEL_NAME, load_embedding_model


# ─── Retrieval Function ─────────────────────────────────────────────────────

def retrieve(
    query: str,
    index: faiss.IndexFlatIP,
    metadata: List[Dict[str, Any]],
    embedding_model: SentenceTransformer,
    top_k: int = 5
) -> List[Dict[str, Any]]:
    """
    Retrieve the top-k most relevant document chunks for a given query.
    
    Process:
    1. Encode the query into an embedding using the same model used for documents
    2. Search the FAISS index using inner product similarity
    3. Return chunks ranked by similarity score (highest first)
    
    Args:
        query: The user's question
        index: FAISS index containing document chunk embeddings
        metadata: List of metadata dicts (one per chunk in the index)
        embedding_model: The SentenceTransformer model (must match ingestion model)
        top_k: Number of top results to return (default: 5)
        
    Returns:
        List of dicts, each containing:
        - "text": The chunk text
        - "source": Source PDF filename
        - "chunk_index": Position of this chunk in the original document
        - "score": Cosine similarity score (higher is better)
    """
    # Validate inputs
    if index is None or index.ntotal == 0:
        print("⚠️  No documents in the index. Please upload PDFs first.")
        return []
    
    if not query.strip():
        return []
    
    # Step 1: Encode the query using the SAME embedding model
    # normalize_embeddings=True ensures cosine similarity via inner product
    query_embedding = embedding_model.encode(
        [query],
        convert_to_numpy=True,
        normalize_embeddings=True
    ).astype(np.float32)
    
    # Step 2: Search FAISS index
    # Ensure we don't request more results than available
    actual_k = min(top_k, index.ntotal)
    
    # scores = similarity scores, indices = positions in the index
    scores, indices = index.search(query_embedding, actual_k)
    
    # Step 3: Build results with metadata
    results = []
    for rank, (score, idx) in enumerate(zip(scores[0], indices[0])):
        # FAISS returns -1 for indices when there aren't enough results
        if idx == -1:
            continue
        
        chunk_metadata = metadata[idx]
        results.append({
            "text": chunk_metadata["text"],
            "source": chunk_metadata["source"],
            "chunk_index": chunk_metadata["chunk_index"],
            "total_chunks": chunk_metadata.get("total_chunks", 0),
            "token_count": chunk_metadata.get("token_count", 0),
            "score": float(score),  # Convert numpy float to Python float
            "rank": rank + 1        # 1-indexed rank for display
        })
    
    return results


# ─── Multi-Query Retrieval (Optional Enhancement) ───────────────────────────

def retrieve_with_dedup(
    query: str,
    index: faiss.IndexFlatIP,
    metadata: List[Dict[str, Any]],
    embedding_model: SentenceTransformer,
    top_k: int = 5
) -> List[Dict[str, Any]]:
    """
    Retrieve top-k chunks with deduplication.
    
    Removes duplicate chunks that may appear when the same content
    exists in overlapping chunks. Uses text hash for deduplication.
    
    Args:
        query: The user's question
        index: FAISS index
        metadata: Chunk metadata list
        embedding_model: SentenceTransformer model
        top_k: Desired number of unique results
        
    Returns:
        List of unique result dicts
    """
    # Retrieve more than needed to account for duplicates
    raw_results = retrieve(
        query=query,
        index=index,
        metadata=metadata,
        embedding_model=embedding_model,
        top_k=top_k * 2  # Fetch extra to handle dedup
    )
    
    # Deduplicate by text content
    seen_texts = set()
    unique_results = []
    
    for result in raw_results:
        # Use first 200 chars as dedup key (handles minor variations)
        text_key = result["text"][:200]
        if text_key not in seen_texts:
            seen_texts.add(text_key)
            unique_results.append(result)
        
        if len(unique_results) >= top_k:
            break
    
    return unique_results


# ─── Context Formatting ─────────────────────────────────────────────────────

def format_context(retrieved_chunks: List[Dict[str, Any]]) -> str:
    """
    Format retrieved chunks into a context string for the LLM prompt.
    
    Each chunk is clearly labeled with its source and index to help
    the LLM attribute information correctly.
    
    Args:
        retrieved_chunks: List of retrieved chunk dicts
        
    Returns:
        Formatted context string ready for prompt injection
    """
    if not retrieved_chunks:
        return "No relevant context found."
    
    context_parts = []
    for i, chunk in enumerate(retrieved_chunks, 1):
        source = chunk.get("source", "Unknown")
        chunk_idx = chunk.get("chunk_index", 0) + 1  # 1-indexed for display
        score = chunk.get("score", 0.0)
        
        context_parts.append(
            f"[Source: {source}, Chunk {chunk_idx}, Relevance: {score:.4f}]\n"
            f"{chunk['text']}"
        )
    
    return "\n\n---\n\n".join(context_parts)

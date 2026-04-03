"""
utils.py - Utility functions for the Local RAG System

This module provides helper functions used across the application:
- Token counting for chunk size experiments
- Logging queries and responses
- Timing utilities
- Text cleaning
"""

import os
import json
import time
from datetime import datetime
from typing import List, Dict, Any, Optional


# ─── Constants ───────────────────────────────────────────────────────────────

LOG_DIR = "logs"
LOG_FILE = os.path.join(LOG_DIR, "query_log.jsonl")


# ─── Token Counting ─────────────────────────────────────────────────────────

def count_tokens(text: str) -> int:
    """
    Estimate token count for a piece of text.
    Uses a simple whitespace-based approximation (1 token ≈ 0.75 words).
    This avoids heavy dependencies while being reasonably accurate.
    
    Args:
        text: Input text string
        
    Returns:
        Estimated number of tokens
    """
    # A good approximation: split on whitespace, multiply by ~1.33
    # (since tokens are typically shorter than words)
    words = text.split()
    return int(len(words) * 1.33)


def count_tokens_tiktoken(text: str) -> int:
    """
    Count tokens using tiktoken for more accurate measurement.
    Falls back to simple estimation if tiktoken is not available.
    
    Args:
        text: Input text string
        
    Returns:
        Number of tokens
    """
    try:
        import tiktoken
        # Use cl100k_base encoding (used by GPT-4, similar tokenization)
        encoding = tiktoken.get_encoding("cl100k_base")
        return len(encoding.encode(text))
    except ImportError:
        return count_tokens(text)


# ─── Text Cleaning ──────────────────────────────────────────────────────────

def clean_text(text: str) -> str:
    """
    Clean extracted PDF text by removing excess whitespace and artifacts.
    
    Args:
        text: Raw text from PDF extraction
        
    Returns:
        Cleaned text string
    """
    # Replace multiple newlines with single newline
    import re
    text = re.sub(r'\n{3,}', '\n\n', text)
    
    # Replace multiple spaces with single space
    text = re.sub(r' {2,}', ' ', text)
    
    # Remove leading/trailing whitespace from each line
    lines = [line.strip() for line in text.split('\n')]
    text = '\n'.join(lines)
    
    # Remove empty lines at start and end
    text = text.strip()
    
    return text


# ─── Query Logging ──────────────────────────────────────────────────────────

def ensure_log_dir():
    """Create the logs directory if it doesn't exist."""
    os.makedirs(LOG_DIR, exist_ok=True)


def log_query(
    query: str,
    response: str,
    retrieved_chunks: List[Dict[str, Any]],
    response_time: float,
    chunk_size: int,
    top_k: int,
    model_name: str = "llama3"
) -> None:
    """
    Log a query and its response to a JSONL file for evaluation.
    
    Each log entry contains:
    - Timestamp
    - User query
    - LLM response
    - Retrieved chunks with metadata
    - Performance metrics (response time, chunk size, top_k)
    
    Args:
        query: The user's question
        response: The LLM's answer
        retrieved_chunks: List of retrieved chunk dicts with text and metadata
        response_time: Time taken to generate response (seconds)
        chunk_size: Chunk size used for this query
        top_k: Number of chunks retrieved
        model_name: Name of the LLM model used
    """
    ensure_log_dir()
    
    log_entry = {
        "timestamp": datetime.now().isoformat(),
        "query": query,
        "response": response,
        "retrieved_chunks": [
            {
                "text": chunk.get("text", "")[:200] + "...",  # Truncate for log
                "source": chunk.get("source", "unknown"),
                "chunk_index": chunk.get("chunk_index", -1),
                "score": chunk.get("score", 0.0)
            }
            for chunk in retrieved_chunks
        ],
        "metrics": {
            "response_time_seconds": round(response_time, 3),
            "chunk_size_tokens": chunk_size,
            "top_k": top_k,
            "num_chunks_retrieved": len(retrieved_chunks),
            "model": model_name
        }
    }
    
    # Append to JSONL file (one JSON object per line)
    with open(LOG_FILE, "a") as f:
        f.write(json.dumps(log_entry) + "\n")


def load_query_logs() -> List[Dict[str, Any]]:
    """
    Load all query logs from the log file.
    
    Returns:
        List of log entry dictionaries
    """
    if not os.path.exists(LOG_FILE):
        return []
    
    logs = []
    with open(LOG_FILE, "r") as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    logs.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
    return logs


# ─── Precision@K Calculation ────────────────────────────────────────────────

def calculate_precision_at_k(
    retrieved_sources: List[str],
    relevant_sources: List[str],
    k: int
) -> float:
    """
    Calculate Precision@K - the fraction of retrieved documents that are relevant.
    
    This is a basic evaluation metric for retrieval quality.
    
    Args:
        retrieved_sources: List of source filenames from retrieval
        relevant_sources: List of source filenames marked as relevant (ground truth)
        k: Number of top results to consider
        
    Returns:
        Precision@K score between 0.0 and 1.0
    """
    if k == 0 or not retrieved_sources:
        return 0.0
    
    # Only consider top-k results
    top_k_sources = retrieved_sources[:k]
    
    # Count how many retrieved sources are in the relevant set
    relevant_count = sum(
        1 for source in top_k_sources 
        if source in relevant_sources
    )
    
    return relevant_count / k


# ─── Timer Context Manager ──────────────────────────────────────────────────

class Timer:
    """
    Simple timer context manager for measuring execution time.
    
    Usage:
        with Timer() as t:
            # do work
        print(f"Took {t.elapsed:.2f}s")
    """
    
    def __init__(self):
        self.elapsed = 0.0
        self._start = None
    
    def __enter__(self):
        self._start = time.time()
        return self
    
    def __exit__(self, *args):
        self.elapsed = time.time() - self._start


# ─── Formatting Helpers ─────────────────────────────────────────────────────

def format_source_display(source: str, chunk_index: int, score: float) -> str:
    """
    Format a source reference for display in the UI.
    
    Args:
        source: Source filename
        chunk_index: Index of the chunk within the document
        score: Similarity score
        
    Returns:
        Formatted string for display
    """
    return f"📄 **{source}** (Chunk #{chunk_index + 1}) — Similarity: {score:.4f}"

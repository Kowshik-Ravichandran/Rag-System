"""
ingest.py - Document Ingestion Pipeline for the Local RAG System

This module handles the complete document processing pipeline:
1. PDF text extraction using PyMuPDF (fitz)
2. Text chunking with configurable chunk sizes (200-500 tokens)
3. Embedding generation using SentenceTransformers (all-MiniLM-L6-v2)
4. FAISS vector index creation and persistence

The pipeline stores metadata (source filename, chunk index) alongside
each chunk for source attribution in responses.
"""

import os
import json
import hashlib
from typing import List, Dict, Tuple, Optional

import fitz  # PyMuPDF - fast PDF text extraction
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer

from utils import clean_text, count_tokens, count_tokens_tiktoken


# ─── Constants ───────────────────────────────────────────────────────────────

# Directory to store the FAISS index and metadata
INDEX_DIR = "vector_store"

# Default embedding model - runs locally, no API needed
# all-MiniLM-L6-v2: Fast, lightweight, 384-dimensional embeddings
EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"

# Default chunk configuration
DEFAULT_CHUNK_SIZE = 300      # tokens per chunk (sweet spot between 200-500)
DEFAULT_CHUNK_OVERLAP = 50    # tokens of overlap between consecutive chunks


# ─── PDF Text Extraction ────────────────────────────────────────────────────

def extract_text_from_pdf(pdf_path: str) -> str:
    """
    Extract all text content from a PDF file using PyMuPDF.
    
    PyMuPDF (fitz) is chosen for its:
    - Speed: Fastest Python PDF library
    - Apple Silicon compatibility
    - Reliable text extraction
    
    Args:
        pdf_path: Path to the PDF file
        
    Returns:
        Extracted text as a single string
    """
    doc = fitz.open(pdf_path)
    text_parts = []
    
    for page_num in range(len(doc)):
        page = doc[page_num]
        # Extract text while preserving reading order
        text = page.get_text("text")
        if text.strip():
            text_parts.append(text)
    
    doc.close()
    
    # Join all pages and clean the text
    full_text = "\n\n".join(text_parts)
    return clean_text(full_text)


def extract_text_from_uploaded_pdf(uploaded_file) -> str:
    """
    Extract text from a Streamlit UploadedFile object.
    
    Args:
        uploaded_file: Streamlit UploadedFile (file-like object)
        
    Returns:
        Extracted text as a string
    """
    # Read the bytes from the uploaded file
    pdf_bytes = uploaded_file.read()
    
    # Open PDF from bytes using PyMuPDF
    doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    text_parts = []
    
    for page_num in range(len(doc)):
        page = doc[page_num]
        text = page.get_text("text")
        if text.strip():
            text_parts.append(text)
    
    doc.close()
    
    full_text = "\n\n".join(text_parts)
    return clean_text(full_text)


# ─── Text Chunking ──────────────────────────────────────────────────────────

def chunk_text(
    text: str,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    chunk_overlap: int = DEFAULT_CHUNK_OVERLAP
) -> List[str]:
    """
    Split text into overlapping chunks based on token count.
    
    Strategy:
    - Split text into sentences (by period/newline boundaries)
    - Group sentences into chunks of approximately `chunk_size` tokens
    - Add `chunk_overlap` tokens of overlap between consecutive chunks
    - This ensures context continuity across chunk boundaries
    
    Args:
        text: Full document text
        chunk_size: Target number of tokens per chunk (200-500 range)
        chunk_overlap: Number of overlapping tokens between chunks
        
    Returns:
        List of text chunks
    """
    if not text.strip():
        return []
    
    # Split into sentences using multiple delimiters
    import re
    # Split on sentence boundaries (., !, ?) followed by whitespace
    # Also split on double newlines (paragraph boundaries)
    sentences = re.split(r'(?<=[.!?])\s+|\n\n+', text)
    sentences = [s.strip() for s in sentences if s.strip()]
    
    chunks = []
    current_chunk_sentences = []
    current_token_count = 0
    
    for sentence in sentences:
        sentence_tokens = count_tokens(sentence)
        
        # If adding this sentence would exceed chunk_size, finalize current chunk
        if current_token_count + sentence_tokens > chunk_size and current_chunk_sentences:
            chunk_text_str = " ".join(current_chunk_sentences)
            chunks.append(chunk_text_str)
            
            # Calculate overlap: keep last few sentences that fit in overlap window
            overlap_sentences = []
            overlap_tokens = 0
            for s in reversed(current_chunk_sentences):
                s_tokens = count_tokens(s)
                if overlap_tokens + s_tokens <= chunk_overlap:
                    overlap_sentences.insert(0, s)
                    overlap_tokens += s_tokens
                else:
                    break
            
            current_chunk_sentences = overlap_sentences
            current_token_count = overlap_tokens
        
        current_chunk_sentences.append(sentence)
        current_token_count += sentence_tokens
    
    # Don't forget the last chunk
    if current_chunk_sentences:
        chunk_text_str = " ".join(current_chunk_sentences)
        chunks.append(chunk_text_str)
    
    return chunks


# ─── Embedding Generation ───────────────────────────────────────────────────

def load_embedding_model(model_name: str = EMBEDDING_MODEL_NAME) -> SentenceTransformer:
    """
    Load the SentenceTransformer embedding model.
    
    all-MiniLM-L6-v2 details:
    - Output dimension: 384
    - Speed: Very fast on CPU/Apple Silicon
    - Quality: Good for semantic similarity tasks
    - Size: ~80MB (downloads once, cached locally)
    
    Args:
        model_name: Name of the SentenceTransformer model
        
    Returns:
        Loaded SentenceTransformer model
    """
    print(f"Loading embedding model: {model_name}...")
    model = SentenceTransformer(model_name)
    print("Embedding model loaded successfully!")
    return model


def generate_embeddings(
    texts: List[str],
    model: SentenceTransformer,
    batch_size: int = 32
) -> np.ndarray:
    """
    Generate embeddings for a list of text chunks.
    
    Args:
        texts: List of text strings to embed
        model: Loaded SentenceTransformer model
        batch_size: Number of texts to process at once (for memory efficiency)
        
    Returns:
        NumPy array of embeddings, shape (n_texts, embedding_dim)
    """
    print(f"Generating embeddings for {len(texts)} chunks...")
    embeddings = model.encode(
        texts,
        batch_size=batch_size,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=True  # L2 normalize for cosine similarity
    )
    print(f"Generated {len(embeddings)} embeddings of dimension {embeddings.shape[1]}")
    return embeddings


# ─── FAISS Index Management ─────────────────────────────────────────────────

def create_faiss_index(embeddings: np.ndarray) -> faiss.IndexFlatIP:
    """
    Create a FAISS index for similarity search.
    
    Uses IndexFlatIP (Inner Product) since embeddings are L2-normalized,
    making inner product equivalent to cosine similarity.
    
    Args:
        embeddings: NumPy array of embeddings, shape (n, dim)
        
    Returns:
        FAISS index with embeddings added
    """
    dimension = embeddings.shape[1]
    
    # IndexFlatIP = exact inner product search (cosine sim with normalized vectors)
    index = faiss.IndexFlatIP(dimension)
    
    # Add all embeddings to the index
    index.add(embeddings.astype(np.float32))
    
    print(f"FAISS index created with {index.ntotal} vectors of dimension {dimension}")
    return index


def save_index(
    index: faiss.IndexFlatIP,
    metadata: List[Dict],
    index_dir: str = INDEX_DIR
) -> None:
    """
    Save the FAISS index and chunk metadata to disk.
    
    Args:
        index: FAISS index to save
        metadata: List of metadata dicts for each chunk
        index_dir: Directory to save files in
    """
    os.makedirs(index_dir, exist_ok=True)
    
    # Save FAISS index
    index_path = os.path.join(index_dir, "faiss_index.bin")
    faiss.write_index(index, index_path)
    
    # Save metadata as JSON
    metadata_path = os.path.join(index_dir, "metadata.json")
    with open(metadata_path, "w") as f:
        json.dump(metadata, f, indent=2)
    
    print(f"Index saved to {index_dir}/ ({index.ntotal} vectors)")


def load_index(index_dir: str = INDEX_DIR) -> Tuple[Optional[faiss.IndexFlatIP], Optional[List[Dict]]]:
    """
    Load a previously saved FAISS index and metadata from disk.
    
    Args:
        index_dir: Directory containing saved index files
        
    Returns:
        Tuple of (FAISS index, metadata list) or (None, None) if not found
    """
    index_path = os.path.join(index_dir, "faiss_index.bin")
    metadata_path = os.path.join(index_dir, "metadata.json")
    
    if not os.path.exists(index_path) or not os.path.exists(metadata_path):
        return None, None
    
    index = faiss.read_index(index_path)
    with open(metadata_path, "r") as f:
        metadata = json.load(f)
    
    print(f"Loaded index with {index.ntotal} vectors")
    return index, metadata


# ─── Complete Ingestion Pipeline ─────────────────────────────────────────────

def ingest_documents(
    uploaded_files: list,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    chunk_overlap: int = DEFAULT_CHUNK_OVERLAP,
    embedding_model: Optional[SentenceTransformer] = None
) -> Tuple[faiss.IndexFlatIP, List[Dict]]:
    """
    Complete ingestion pipeline: PDF → Text → Chunks → Embeddings → FAISS Index.
    
    This is the main function called when users upload PDFs.
    
    Args:
        uploaded_files: List of Streamlit UploadedFile objects
        chunk_size: Token count per chunk (200-500)
        chunk_overlap: Token overlap between chunks
        embedding_model: Pre-loaded embedding model (loads one if not provided)
        
    Returns:
        Tuple of (FAISS index, metadata list)
    """
    # Step 1: Load embedding model if not provided
    if embedding_model is None:
        embedding_model = load_embedding_model()
    
    all_chunks = []     # List of chunk text strings
    all_metadata = []   # List of metadata dicts per chunk
    
    # Step 2: Process each uploaded PDF
    for uploaded_file in uploaded_files:
        filename = uploaded_file.name
        print(f"\n📄 Processing: {filename}")
        
        # Extract text from PDF
        text = extract_text_from_uploaded_pdf(uploaded_file)
        
        if not text.strip():
            print(f"  ⚠️  No text extracted from {filename}, skipping.")
            continue
        
        print(f"  Extracted {len(text)} characters")
        
        # Chunk the text
        chunks = chunk_text(text, chunk_size=chunk_size, chunk_overlap=chunk_overlap)
        print(f"  Split into {len(chunks)} chunks (size={chunk_size}, overlap={chunk_overlap})")
        
        # Create metadata for each chunk
        for i, chunk in enumerate(chunks):
            all_chunks.append(chunk)
            all_metadata.append({
                "text": chunk,
                "source": filename,
                "chunk_index": i,
                "total_chunks": len(chunks),
                "chunk_size_setting": chunk_size,
                "token_count": count_tokens(chunk)
            })
    
    if not all_chunks:
        raise ValueError("No text could be extracted from any uploaded PDF.")
    
    # Step 3: Generate embeddings for all chunks
    embeddings = generate_embeddings(all_chunks, embedding_model)
    
    # Step 4: Create FAISS index
    index = create_faiss_index(embeddings)
    
    # Step 5: Save to disk for persistence
    save_index(index, all_metadata)
    
    print(f"\n✅ Ingestion complete: {len(all_chunks)} chunks from {len(uploaded_files)} file(s)")
    
    return index, all_metadata


def get_file_hash(uploaded_file) -> str:
    """
    Generate a hash for an uploaded file to detect duplicates.
    
    Args:
        uploaded_file: Streamlit UploadedFile object
        
    Returns:
        MD5 hash string of the file content
    """
    content = uploaded_file.read()
    uploaded_file.seek(0)  # Reset file pointer
    return hashlib.md5(content).hexdigest()

"""
app.py - Streamlit Chat Interface for the Local RAG System

This is the main application file that provides:
- PDF document upload interface
- Chat-based question answering
- Retrieved source document display
- Evaluation dashboard (query logs, response times, chunk size comparison)

Run with: streamlit run app.py
"""

import os
import time
import streamlit as st
from typing import List, Dict, Any

from ingest import (
    ingest_documents,
    load_embedding_model,
    load_index,
    DEFAULT_CHUNK_SIZE,
    DEFAULT_CHUNK_OVERLAP,
    EMBEDDING_MODEL_NAME
)
from retriever import retrieve_with_dedup
from generator import (
    generate_response,
    generate_response_stream,
    check_ollama_connection,
    DEFAULT_MODEL
)
from utils import (
    log_query,
    load_query_logs,
    format_source_display,
    Timer,
    calculate_precision_at_k
)


# ─── Page Configuration ─────────────────────────────────────────────────────

st.set_page_config(
    page_title="📚 Local RAG System",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="expanded"
)


# ─── Custom CSS for Premium Dark Theme ──────────────────────────────────────

st.markdown("""
<style>
    /* Import modern font */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
    
    /* Global styles */
    .stApp {
        font-family: 'Inter', sans-serif;
    }
    
    /* Header styling */
    .main-header {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        font-size: 2.5rem;
        font-weight: 700;
        text-align: center;
        padding: 1rem 0;
        margin-bottom: 0.5rem;
    }
    
    .sub-header {
        text-align: center;
        color: #8b8fa3;
        font-size: 1rem;
        font-weight: 300;
        margin-bottom: 2rem;
    }
    
    /* Source card styling */
    .source-card {
        background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%);
        border: 1px solid rgba(102, 126, 234, 0.2);
        border-radius: 12px;
        padding: 1rem 1.2rem;
        margin: 0.5rem 0;
        transition: all 0.3s ease;
    }
    
    .source-card:hover {
        border-color: rgba(102, 126, 234, 0.5);
        box-shadow: 0 4px 15px rgba(102, 126, 234, 0.1);
    }
    
    .source-header {
        color: #667eea;
        font-weight: 600;
        font-size: 0.9rem;
        margin-bottom: 0.5rem;
    }
    
    .source-text {
        color: #c4c7d4;
        font-size: 0.85rem;
        line-height: 1.5;
    }
    
    .score-badge {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 2px 10px;
        border-radius: 20px;
        font-size: 0.75rem;
        font-weight: 600;
    }
    
    /* Status indicator */
    .status-connected {
        color: #00d26a;
        font-weight: 500;
    }
    
    .status-disconnected {
        color: #f44336;
        font-weight: 500;
    }
    
    /* Metric card */
    .metric-card {
        background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%);
        border: 1px solid rgba(102, 126, 234, 0.15);
        border-radius: 12px;
        padding: 1.2rem;
        text-align: center;
    }
    
    .metric-value {
        font-size: 2rem;
        font-weight: 700;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
    }
    
    .metric-label {
        color: #8b8fa3;
        font-size: 0.85rem;
        margin-top: 0.3rem;
    }
    
    /* Chat message styling */
    .stChatMessage {
        border-radius: 12px !important;
    }
    
    /* Sidebar styling */
    .sidebar-section {
        background: rgba(102, 126, 234, 0.05);
        border: 1px solid rgba(102, 126, 234, 0.1);
        border-radius: 10px;
        padding: 1rem;
        margin: 0.5rem 0;
    }
    
    /* Hide Streamlit branding */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
</style>
""", unsafe_allow_html=True)


# ─── Session State Initialization ────────────────────────────────────────────

def init_session_state():
    """Initialize all session state variables on first run."""
    defaults = {
        "chat_history": [],           # List of {"role": ..., "content": ...}
        "faiss_index": None,          # FAISS index object
        "metadata": None,             # Chunk metadata list
        "embedding_model": None,      # SentenceTransformer model
        "documents_loaded": False,    # Whether docs have been indexed
        "uploaded_files_hashes": [],  # Track uploaded file hashes
        "chunk_size": DEFAULT_CHUNK_SIZE,
        "chunk_overlap": DEFAULT_CHUNK_OVERLAP,
        "top_k": 5,
        "show_sources": True,
        "stream_responses": True,
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


init_session_state()


# ─── Sidebar ─────────────────────────────────────────────────────────────────

def render_sidebar():
    """Render the sidebar with system status, settings, and file upload."""
    
    with st.sidebar:
        st.markdown("## ⚙️ System Status")
        
        # Check Ollama connection
        with st.spinner("Checking Ollama..."):
            status = check_ollama_connection()
        
        if status["connected"]:
            st.markdown(
                '<p class="status-connected">● Ollama Connected</p>',
                unsafe_allow_html=True
            )
            if status["model_available"]:
                st.success(f"✅ Model `{DEFAULT_MODEL}` ready")
            else:
                st.warning(
                    f"⚠️ Model `{DEFAULT_MODEL}` not found.\n\n"
                    f"Pull it with:\n```\nollama pull {DEFAULT_MODEL}\n```\n\n"
                    f"Available models: {', '.join(status['models']) or 'None'}"
                )
        else:
            st.markdown(
                '<p class="status-disconnected">● Ollama Disconnected</p>',
                unsafe_allow_html=True
            )
            st.error(status.get("error", "Cannot connect to Ollama"))
            st.info(
                "**Start Ollama:**\n"
                "1. Install from [ollama.com](https://ollama.com)\n"
                "2. Run `ollama serve` in terminal\n"
                "3. Pull model: `ollama pull llama3`"
            )
        
        st.divider()
        
        # ── Document Upload ──
        st.markdown("## 📄 Upload Documents")
        
        uploaded_files = st.file_uploader(
            "Upload PDF files",
            type=["pdf"],
            accept_multiple_files=True,
            help="Upload one or more PDF documents to query"
        )
        
        if uploaded_files:
            st.info(f"📁 {len(uploaded_files)} file(s) selected")
            for f in uploaded_files:
                st.caption(f"  • {f.name} ({f.size / 1024:.0f} KB)")
        
        st.divider()
        
        # ── Retrieval Settings ──
        st.markdown("## 🔧 Settings")
        
        with st.expander("Chunk Settings", expanded=False):
            st.session_state.chunk_size = st.slider(
                "Chunk Size (tokens)",
                min_value=100,
                max_value=800,
                value=st.session_state.chunk_size,
                step=50,
                help="Number of tokens per document chunk. Experiment with 200-500 for best results."
            )
            
            st.session_state.chunk_overlap = st.slider(
                "Chunk Overlap (tokens)",
                min_value=0,
                max_value=200,
                value=st.session_state.chunk_overlap,
                step=10,
                help="Overlapping tokens between consecutive chunks for context continuity."
            )
        
        with st.expander("Retrieval Settings", expanded=False):
            st.session_state.top_k = st.slider(
                "Top-K Results",
                min_value=1,
                max_value=10,
                value=st.session_state.top_k,
                help="Number of relevant chunks to retrieve for each query."
            )
            
            st.session_state.show_sources = st.checkbox(
                "Show Retrieved Sources",
                value=st.session_state.show_sources,
                help="Display the document chunks used to generate each answer."
            )
            
            st.session_state.stream_responses = st.checkbox(
                "Stream Responses",
                value=st.session_state.stream_responses,
                help="Show tokens as they are generated (real-time)."
            )
        
        st.divider()
        
        # ── Process Documents Button ──
        if uploaded_files:
            process_col1, process_col2 = st.columns(2)
            
            with process_col1:
                if st.button("📥 Process Docs", use_container_width=True, type="primary"):
                    process_documents(uploaded_files)
            
            with process_col2:
                if st.button("🗑️ Clear Index", use_container_width=True):
                    clear_index()
        
        # Show index status
        if st.session_state.documents_loaded and st.session_state.metadata:
            sources = set(m["source"] for m in st.session_state.metadata)
            st.success(
                f"📊 **Index Active**\n\n"
                f"• {len(st.session_state.metadata)} chunks\n"
                f"• {len(sources)} document(s)\n"
                f"• Chunk size: {st.session_state.chunk_size} tokens"
            )
        
        # ── Clear Chat ──
        st.divider()
        if st.button("🧹 Clear Chat History", use_container_width=True):
            st.session_state.chat_history = []
            st.rerun()


# ─── Document Processing ────────────────────────────────────────────────────

def process_documents(uploaded_files):
    """Process uploaded PDFs through the ingestion pipeline."""
    
    with st.spinner("🔄 Loading embedding model..."):
        if st.session_state.embedding_model is None:
            st.session_state.embedding_model = load_embedding_model()
    
    progress_bar = st.sidebar.progress(0, text="Processing documents...")
    
    try:
        progress_bar.progress(20, text="Extracting text from PDFs...")
        
        index, metadata = ingest_documents(
            uploaded_files=uploaded_files,
            chunk_size=st.session_state.chunk_size,
            chunk_overlap=st.session_state.chunk_overlap,
            embedding_model=st.session_state.embedding_model
        )
        
        progress_bar.progress(90, text="Finalizing index...")
        
        st.session_state.faiss_index = index
        st.session_state.metadata = metadata
        st.session_state.documents_loaded = True
        
        progress_bar.progress(100, text="✅ Done!")
        time.sleep(0.5)
        progress_bar.empty()
        
        st.sidebar.success(f"✅ Processed {len(uploaded_files)} file(s), {len(metadata)} chunks created!")
        st.rerun()
        
    except Exception as e:
        progress_bar.empty()
        st.sidebar.error(f"❌ Error processing documents: {str(e)}")


def clear_index():
    """Clear the current document index."""
    st.session_state.faiss_index = None
    st.session_state.metadata = None
    st.session_state.documents_loaded = False
    st.session_state.chat_history = []
    
    # Remove saved index files
    import shutil
    if os.path.exists("vector_store"):
        shutil.rmtree("vector_store")
    
    st.rerun()


# ─── Main Chat Interface ────────────────────────────────────────────────────

def render_chat():
    """Render the main chat interface."""
    
    # Header
    st.markdown('<h1 class="main-header">📚 Local RAG System</h1>', unsafe_allow_html=True)
    st.markdown(
        '<p class="sub-header">Ask questions about your documents · Powered by Llama 3 & FAISS · 100% Local</p>',
        unsafe_allow_html=True
    )
    
    # Tabs for Chat and Evaluation
    tab_chat, tab_eval = st.tabs(["💬 Chat", "📊 Evaluation"])
    
    with tab_chat:
        render_chat_tab()
    
    with tab_eval:
        render_evaluation_tab()


def render_chat_tab():
    """Render the chat messages and input."""
    
    # Show onboarding if no documents loaded
    if not st.session_state.documents_loaded:
        st.markdown("---")
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown("""
            <div class="metric-card">
                <div style="font-size: 2rem;">📄</div>
                <div class="metric-label" style="margin-top: 0.5rem; font-weight: 500;">Step 1</div>
                <div class="metric-label">Upload PDFs</div>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            st.markdown("""
            <div class="metric-card">
                <div style="font-size: 2rem;">⚡</div>
                <div class="metric-label" style="margin-top: 0.5rem; font-weight: 500;">Step 2</div>
                <div class="metric-label">Process Documents</div>
            </div>
            """, unsafe_allow_html=True)
        
        with col3:
            st.markdown("""
            <div class="metric-card">
                <div style="font-size: 2rem;">💬</div>
                <div class="metric-label" style="margin-top: 0.5rem; font-weight: 500;">Step 3</div>
                <div class="metric-label">Ask Questions</div>
            </div>
            """, unsafe_allow_html=True)
        
        st.markdown("---")
        st.info(
            "👈 **Get started:** Upload PDF documents in the sidebar, "
            "then click **Process Docs** to build the search index."
        )
        return
    
    # Display chat history
    for message in st.session_state.chat_history:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            
            # Show sources if they exist in the message
            if message["role"] == "assistant" and "sources" in message:
                render_sources(message["sources"])
    
    # Chat input
    if prompt := st.chat_input("Ask a question about your documents..."):
        handle_user_query(prompt)


def handle_user_query(query: str):
    """Process a user query through the RAG pipeline."""
    
    # Display user message
    with st.chat_message("user"):
        st.markdown(query)
    
    # Add to chat history
    st.session_state.chat_history.append({"role": "user", "content": query})
    
    # Generate response
    with st.chat_message("assistant"):
        with Timer() as timer:
            # Step 1: Retrieve relevant chunks
            with st.spinner("🔍 Searching documents..."):
                retrieved_chunks = retrieve_with_dedup(
                    query=query,
                    index=st.session_state.faiss_index,
                    metadata=st.session_state.metadata,
                    embedding_model=st.session_state.embedding_model,
                    top_k=st.session_state.top_k
                )
            
            if not retrieved_chunks:
                response = "I couldn't find any relevant information in the uploaded documents."
                st.markdown(response)
            else:
                # Step 2: Generate response with LLM
                if st.session_state.stream_responses:
                    # Streaming mode - show tokens as they arrive
                    response = st.write_stream(
                        generate_response_stream(
                            query=query,
                            retrieved_chunks=retrieved_chunks
                        )
                    )
                else:
                    # Non-streaming mode
                    with st.spinner("🤖 Generating answer..."):
                        response = generate_response(
                            query=query,
                            retrieved_chunks=retrieved_chunks
                        )
                    st.markdown(response)
        
        # Show response time
        st.caption(f"⏱️ Response time: {timer.elapsed:.2f}s")
        
        # Show sources
        if st.session_state.show_sources and retrieved_chunks:
            render_sources(retrieved_chunks)
    
    # Add assistant response to history
    st.session_state.chat_history.append({
        "role": "assistant",
        "content": response,
        "sources": retrieved_chunks if retrieved_chunks else []
    })
    
    # Log the query for evaluation
    log_query(
        query=query,
        response=response,
        retrieved_chunks=retrieved_chunks,
        response_time=timer.elapsed,
        chunk_size=st.session_state.chunk_size,
        top_k=st.session_state.top_k
    )


def render_sources(chunks: List[Dict[str, Any]]):
    """Render retrieved source chunks in an expandable section."""
    
    with st.expander(f"📑 Retrieved Sources ({len(chunks)} chunks)", expanded=False):
        for i, chunk in enumerate(chunks):
            source = chunk.get("source", "Unknown")
            chunk_idx = chunk.get("chunk_index", 0)
            score = chunk.get("score", 0.0)
            text = chunk.get("text", "")
            
            st.markdown(
                f"""<div class="source-card">
                    <div class="source-header">
                        📄 {source} · Chunk #{chunk_idx + 1} 
                        <span class="score-badge">{score:.4f}</span>
                    </div>
                    <div class="source-text">{text[:300]}{'...' if len(text) > 300 else ''}</div>
                </div>""",
                unsafe_allow_html=True
            )


# ─── Evaluation Dashboard ───────────────────────────────────────────────────

def render_evaluation_tab():
    """Render the evaluation and analytics dashboard."""
    
    st.markdown("### 📊 Evaluation Dashboard")
    st.caption("Track query performance, response times, and compare chunk size effectiveness.")
    
    logs = load_query_logs()
    
    if not logs:
        st.info("No queries logged yet. Start chatting to see evaluation metrics!")
        return
    
    # ── Summary Metrics ──
    st.markdown("#### Key Metrics")
    
    col1, col2, col3, col4 = st.columns(4)
    
    total_queries = len(logs)
    avg_response_time = sum(
        l["metrics"]["response_time_seconds"] for l in logs
    ) / total_queries
    avg_chunks = sum(
        l["metrics"]["num_chunks_retrieved"] for l in logs
    ) / total_queries
    
    # Get unique chunk sizes used
    chunk_sizes_used = set(l["metrics"]["chunk_size_tokens"] for l in logs)
    
    with col1:
        st.metric("Total Queries", total_queries)
    with col2:
        st.metric("Avg Response Time", f"{avg_response_time:.2f}s")
    with col3:
        st.metric("Avg Chunks/Query", f"{avg_chunks:.1f}")
    with col4:
        st.metric("Chunk Sizes Tested", len(chunk_sizes_used))
    
    st.divider()
    
    # ── Response Time Chart ──
    st.markdown("#### ⏱️ Response Times")
    
    response_times = [l["metrics"]["response_time_seconds"] for l in logs]
    query_labels = [f"Q{i+1}" for i in range(len(logs))]
    
    chart_data = {"Query": query_labels, "Response Time (s)": response_times}
    st.bar_chart(chart_data, x="Query", y="Response Time (s)", color="#667eea")
    
    # ── Chunk Size Comparison ──
    if len(chunk_sizes_used) > 1:
        st.markdown("#### 📏 Chunk Size Comparison")
        
        # Group by chunk size
        chunk_size_data = {}
        for log in logs:
            cs = log["metrics"]["chunk_size_tokens"]
            if cs not in chunk_size_data:
                chunk_size_data[cs] = {"times": [], "count": 0}
            chunk_size_data[cs]["times"].append(log["metrics"]["response_time_seconds"])
            chunk_size_data[cs]["count"] += 1
        
        comparison_cols = st.columns(len(chunk_size_data))
        for i, (cs, data) in enumerate(sorted(chunk_size_data.items())):
            with comparison_cols[i]:
                avg_time = sum(data["times"]) / len(data["times"])
                st.metric(
                    f"Chunk Size: {cs}",
                    f"{avg_time:.2f}s avg",
                    f"{data['count']} queries"
                )
    
    st.divider()
    
    # ── Query Log Table ──
    st.markdown("#### 📝 Query Log")
    
    for i, log in enumerate(reversed(logs)):
        with st.expander(
            f"Q{total_queries - i}: {log['query'][:80]}... "
            f"({log['metrics']['response_time_seconds']:.2f}s)",
            expanded=False
        ):
            st.markdown(f"**Query:** {log['query']}")
            st.markdown(f"**Response:** {log['response'][:500]}...")
            st.markdown(f"**Time:** {log['metrics']['response_time_seconds']:.2f}s")
            st.markdown(f"**Chunk Size:** {log['metrics']['chunk_size_tokens']} tokens")
            st.markdown(f"**Top-K:** {log['metrics']['top_k']}")
            st.markdown(f"**Chunks Retrieved:** {log['metrics']['num_chunks_retrieved']}")
            st.markdown(f"**Timestamp:** {log['timestamp']}")
            
            if log.get("retrieved_chunks"):
                st.markdown("**Sources:**")
                for chunk in log["retrieved_chunks"]:
                    st.caption(
                        f"  📄 {chunk['source']} (Chunk #{chunk['chunk_index'] + 1}) "
                        f"— Score: {chunk['score']:.4f}"
                    )
    
    # ── Precision@K Calculator ──
    st.divider()
    st.markdown("#### 🎯 Precision@K Calculator")
    st.caption(
        "Manually evaluate retrieval quality by marking which sources are relevant."
    )
    
    if logs:
        latest_log = logs[-1]
        
        st.markdown(f"**Latest Query:** {latest_log['query']}")
        
        if latest_log.get("retrieved_chunks"):
            relevant_selections = []
            
            for chunk in latest_log["retrieved_chunks"]:
                is_relevant = st.checkbox(
                    f"📄 {chunk['source']} (Chunk #{chunk['chunk_index'] + 1}) — "
                    f"Score: {chunk['score']:.4f}",
                    key=f"rel_{chunk['source']}_{chunk['chunk_index']}"
                )
                if is_relevant:
                    relevant_selections.append(chunk["source"])
            
            if st.button("Calculate Precision@K"):
                retrieved_sources = [c["source"] for c in latest_log["retrieved_chunks"]]
                k = len(retrieved_sources)
                precision = calculate_precision_at_k(
                    retrieved_sources, relevant_selections, k
                )
                st.metric(f"Precision@{k}", f"{precision:.2%}")


# ─── Main Entry Point ───────────────────────────────────────────────────────

def main():
    """Main application entry point."""
    render_sidebar()
    render_chat()


if __name__ == "__main__":
    main()

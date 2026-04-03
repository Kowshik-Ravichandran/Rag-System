"""
generator.py - LLM Response Generation for the Local RAG System

This module handles the generation step of the RAG pipeline:
1. Constructs a prompt with the system instruction and retrieved context
2. Sends the prompt to the local Ollama server (Llama 3 8B)
3. Returns the generated response

Uses the Ollama REST API (http://localhost:11434) which requires:
- Ollama installed on the machine
- Llama 3 model pulled: `ollama pull llama3`

No paid APIs are used - everything runs locally on Apple Silicon.
"""

import json
from typing import List, Dict, Any, Optional, Generator

import requests

from retriever import format_context


# ─── Constants ───────────────────────────────────────────────────────────────

# Ollama API endpoint (runs locally)
OLLAMA_BASE_URL = "http://localhost:11434"
OLLAMA_GENERATE_URL = f"{OLLAMA_BASE_URL}/api/generate"
OLLAMA_CHAT_URL = f"{OLLAMA_BASE_URL}/api/chat"

# Default model - Llama 3 8B Instruct (quantized for Apple Silicon)
DEFAULT_MODEL = "llama3"

# System prompt that constrains the LLM to answer from context only
SYSTEM_PROMPT = """You are a helpful and accurate assistant. Your task is to answer questions based ONLY on the provided context from uploaded documents.

STRICT RULES:
1. Answer ONLY using information found in the provided context below.
2. If the answer is NOT found in the context, clearly state: "I don't have enough information in the uploaded documents to answer this question."
3. Do NOT use any prior knowledge or make assumptions beyond what is in the context.
4. When possible, cite which source document the information comes from.
5. Be concise but thorough in your answers.
6. If the context is partially relevant, explain what you found and what is missing."""


# ─── Ollama Connection Check ────────────────────────────────────────────────

def check_ollama_connection() -> Dict[str, Any]:
    """
    Check if Ollama is running and the required model is available.
    
    Returns:
        Dict with keys:
        - "connected": bool - Whether Ollama server is reachable
        - "model_available": bool - Whether the required model is pulled
        - "models": list - Available model names
        - "error": str - Error message if any
    """
    result = {
        "connected": False,
        "model_available": False,
        "models": [],
        "error": None
    }
    
    try:
        # Check if Ollama is running
        response = requests.get(f"{OLLAMA_BASE_URL}/api/tags", timeout=5)
        
        if response.status_code == 200:
            result["connected"] = True
            data = response.json()
            models = [m["name"] for m in data.get("models", [])]
            result["models"] = models
            
            # Check if our required model is available
            # Match both "llama3" and "llama3:latest" format
            result["model_available"] = any(
                DEFAULT_MODEL in model_name 
                for model_name in models
            )
        else:
            result["error"] = f"Ollama returned status code {response.status_code}"
            
    except requests.ConnectionError:
        result["error"] = (
            "Cannot connect to Ollama. Please ensure Ollama is running.\n"
            "Start it with: `ollama serve` in a terminal."
        )
    except requests.Timeout:
        result["error"] = "Connection to Ollama timed out."
    except Exception as e:
        result["error"] = f"Unexpected error: {str(e)}"
    
    return result


# ─── Prompt Construction ────────────────────────────────────────────────────

def build_prompt(
    query: str,
    retrieved_chunks: List[Dict[str, Any]],
    system_prompt: str = SYSTEM_PROMPT
) -> str:
    """
    Construct the full prompt for the LLM by combining:
    1. System instruction (answer from context only)
    2. Retrieved document chunks (the context)
    3. User's question
    
    Args:
        query: The user's question
        retrieved_chunks: List of retrieved chunk dicts from the retriever
        system_prompt: System-level instruction for the LLM
        
    Returns:
        Complete prompt string
    """
    # Format the retrieved chunks into a context block
    context = format_context(retrieved_chunks)
    
    # Build the full prompt
    prompt = f"""CONTEXT FROM UPLOADED DOCUMENTS:
{context}

USER QUESTION:
{query}

Please answer the question based ONLY on the context provided above."""
    
    return prompt


# ─── LLM Generation ─────────────────────────────────────────────────────────

def generate_response(
    query: str,
    retrieved_chunks: List[Dict[str, Any]],
    model: str = DEFAULT_MODEL,
    temperature: float = 0.1,
    max_tokens: int = 1024
) -> str:
    """
    Generate a response using the local Ollama LLM.
    
    Uses the Ollama Chat API with the system prompt and context-injected
    user message. Low temperature (0.1) ensures factual, consistent responses.
    
    Args:
        query: User's question
        retrieved_chunks: Retrieved document chunks for context
        model: Ollama model name (default: "llama3")
        temperature: Sampling temperature (lower = more deterministic)
        max_tokens: Maximum tokens in the response
        
    Returns:
        Generated response text
    """
    # Build the prompt with context
    user_message = build_prompt(query, retrieved_chunks)
    
    # Prepare the request payload for Ollama Chat API
    payload = {
        "model": model,
        "messages": [
            {
                "role": "system",
                "content": SYSTEM_PROMPT
            },
            {
                "role": "user",
                "content": user_message
            }
        ],
        "stream": False,  # Get complete response at once
        "options": {
            "temperature": temperature,
            "num_predict": max_tokens,
            # Optimize for Apple Silicon
            "num_thread": 4,
        }
    }
    
    try:
        response = requests.post(
            OLLAMA_CHAT_URL,
            json=payload,
            timeout=120  # Allow up to 2 minutes for generation
        )
        
        if response.status_code == 200:
            result = response.json()
            return result.get("message", {}).get("content", "No response generated.")
        else:
            return f"Error: Ollama returned status code {response.status_code}. Response: {response.text}"
            
    except requests.ConnectionError:
        return (
            "❌ Cannot connect to Ollama. Please ensure:\n"
            "1. Ollama is installed: https://ollama.com\n"
            "2. Ollama is running: `ollama serve`\n"
            "3. The model is pulled: `ollama pull llama3`"
        )
    except requests.Timeout:
        return "⏰ Request timed out. The model may be loading. Please try again."
    except Exception as e:
        return f"❌ Error generating response: {str(e)}"


def generate_response_stream(
    query: str,
    retrieved_chunks: List[Dict[str, Any]],
    model: str = DEFAULT_MODEL,
    temperature: float = 0.1,
    max_tokens: int = 1024
) -> Generator[str, None, None]:
    """
    Generate a streaming response from the local Ollama LLM.
    
    Yields tokens as they are generated, enabling real-time display
    in the Streamlit chat interface.
    
    Args:
        query: User's question
        retrieved_chunks: Retrieved document chunks
        model: Ollama model name
        temperature: Sampling temperature
        max_tokens: Maximum tokens in response
        
    Yields:
        Token strings as they are generated
    """
    user_message = build_prompt(query, retrieved_chunks)
    
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_message}
        ],
        "stream": True,
        "options": {
            "temperature": temperature,
            "num_predict": max_tokens,
            "num_thread": 4,
        }
    }
    
    try:
        response = requests.post(
            OLLAMA_CHAT_URL,
            json=payload,
            stream=True,
            timeout=120
        )
        
        if response.status_code == 200:
            for line in response.iter_lines():
                if line:
                    try:
                        data = json.loads(line)
                        token = data.get("message", {}).get("content", "")
                        if token:
                            yield token
                    except json.JSONDecodeError:
                        continue
        else:
            yield f"Error: Ollama returned status {response.status_code}"
            
    except requests.ConnectionError:
        yield "❌ Cannot connect to Ollama. Is it running? (`ollama serve`)"
    except requests.Timeout:
        yield "⏰ Request timed out. Please try again."
    except Exception as e:
        yield f"❌ Error: {str(e)}"


# ─── Chat with History ──────────────────────────────────────────────────────

def generate_response_with_history(
    query: str,
    retrieved_chunks: List[Dict[str, Any]],
    chat_history: List[Dict[str, str]],
    model: str = DEFAULT_MODEL,
    temperature: float = 0.1,
    max_tokens: int = 1024
) -> str:
    """
    Generate a response considering previous chat history.
    
    Includes the last few messages for conversational context,
    while still grounding answers in the retrieved documents.
    
    Args:
        query: Current user question
        retrieved_chunks: Retrieved document chunks
        chat_history: List of {"role": "user"|"assistant", "content": "..."} dicts
        model: Ollama model name
        temperature: Sampling temperature
        max_tokens: Maximum tokens
        
    Returns:
        Generated response text
    """
    user_message = build_prompt(query, retrieved_chunks)
    
    # Build messages list with history (keep last 6 messages for context)
    messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    
    # Add recent chat history (limit to avoid context overflow)
    recent_history = chat_history[-6:] if len(chat_history) > 6 else chat_history
    messages.extend(recent_history)
    
    # Add current query with context
    messages.append({"role": "user", "content": user_message})
    
    payload = {
        "model": model,
        "messages": messages,
        "stream": False,
        "options": {
            "temperature": temperature,
            "num_predict": max_tokens,
            "num_thread": 4,
        }
    }
    
    try:
        response = requests.post(OLLAMA_CHAT_URL, json=payload, timeout=120)
        
        if response.status_code == 200:
            result = response.json()
            return result.get("message", {}).get("content", "No response generated.")
        else:
            return f"Error: Ollama returned status {response.status_code}"
            
    except requests.ConnectionError:
        return "❌ Cannot connect to Ollama. Is it running?"
    except requests.Timeout:
        return "⏰ Request timed out. Please try again."
    except Exception as e:
        return f"❌ Error: {str(e)}"

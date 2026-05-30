import os
import json
import requests
from fastapi import FastAPI, HTTPException, Body, Depends, Query, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import List, Dict, Any, Optional
from dotenv import load_dotenv

# Import the RAG Engine from local backend module
from rag_engine import RAGEngine

# Load local environment files if available
load_dotenv()

app = FastAPI(
    title="ISO 9001:2015 QMS RAG API",
    description="Backend API serving the QMS semantic search and auditor generator (Gemini & Ollama).",
    version="1.1.0"
)

# Enable CORS (Cross-Origin Resource Sharing)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global API Key storage
API_KEY_FILE = os.path.join(os.path.dirname(__file__), ".env")

def get_gemini_key() -> str:
    """Helper to retrieve the configured Gemini API key."""
    # 1. Try reading from environment variable
    api_key = os.environ.get("GEMINI_API_KEY", "")
    
    # 2. Try loading from local .env if env var is empty
    if not api_key and os.path.exists(API_KEY_FILE):
        with open(API_KEY_FILE, "r") as f:
            for line in f:
                if line.startswith("GEMINI_API_KEY="):
                    api_key = line.split("=", 1)[1].strip()
                    break
    return api_key


def get_cohere_key() -> str:
    """Helper to retrieve the configured Cohere API key."""
    # 1. Try reading from environment variable
    api_key = os.environ.get("COHERE_API_KEY", "")
    
    # 2. Try loading from local .env if env var is empty
    if not api_key and os.path.exists(API_KEY_FILE):
        with open(API_KEY_FILE, "r") as f:
            for line in f:
                if line.startswith("COHERE_API_KEY="):
                    api_key = line.split("=", 1)[1].strip()
                    break
    return api_key


def save_env_key(key_name: str, key_val: str):
    """Safely updates or inserts a key-value pair in the local .env file without erasing other variables."""
    lines = []
    key_written = False
    try:
        if os.path.exists(API_KEY_FILE):
            with open(API_KEY_FILE, "r") as f:
                for line in f:
                    if line.strip().startswith(f"{key_name}="):
                        lines.append(f"{key_name}={key_val}\n")
                        key_written = True
                    else:
                        lines.append(line)
        if not key_written:
            lines.append(f"{key_name}={key_val}\n")
            
        with open(API_KEY_FILE, "w") as f:
            f.writelines(lines)
    except Exception as e:
        print(f"Warning: Failed to write {key_name} to local .env (might be read-only filesystem): {e}")
        
    os.environ[key_name] = key_val


def get_engine(provider: str = "gemini") -> RAGEngine:
    """Factory helper to initialize RAGEngine on the fly depending on provider."""
    provider = provider.lower().strip()
    if provider == "gemini":
        api_key = get_gemini_key()
        if not api_key:
            raise HTTPException(
                status_code=401, 
                detail="Gemini API Key is not configured. Please supply a key in the settings panel."
            )
        return RAGEngine(api_key=api_key, provider="gemini")
    elif provider == "cohere":
        api_key = get_cohere_key()
        if not api_key:
            raise HTTPException(
                status_code=401,
                detail="Cohere API Key is not configured. Please supply a key in the settings panel."
            )
        return RAGEngine(api_key=api_key, provider="cohere")
    else:
        # Verify Ollama service is active
        try:
            resp = requests.get("http://127.0.0.1:11434/api/tags", timeout=3)
            resp.raise_for_status()
        except Exception:
            raise HTTPException(
                status_code=503,
                detail="Local Ollama service is offline. Please launch the Ollama app on your computer."
            )
        return RAGEngine(api_key="", provider="ollama")


# Pydantic Schemas for Requests & Responses
class QueryRequest(BaseModel):
    query: str = Field(..., description="The QMS/ISO-related query to ask.")
    top_k: int = Field(5, ge=1, le=10, description="Number of source passages to retrieve.")
    temperature: float = Field(0.1, ge=0.0, le=1.0, description="Generative model temperature.")
    provider: str = Field("gemini", description="Selected provider pathway: 'gemini', 'ollama', or 'cohere'.")
    ollama_model: str = Field("llama3", description="Specific local Ollama model to use for chat.")
    gemini_model: str = Field("gemini-2.5-flash", description="Specific Gemini model to use for chat.")
    cohere_model: str = Field("command-r-08-2024", description="Specific Cohere model to use for chat.")

class APIKeyRequest(BaseModel):
    api_key: str = Field(..., description="The Gemini API key to configure.")

# Structured Clause Database (For Tab 2: Clause Navigator)
CLAUSES_DATABASE = {
    "Clause 4: Context of the Organization": {
        "focus": "Understanding organization environment, needs of interested parties, scope, and process maps.",
        "sub_clauses": [
            "4.1 Understanding the organization and its context",
            "4.2 Understanding the needs and expectations of interested parties",
            "4.3 Determining the scope of the quality management system",
            "4.4 Quality management system and its processes"
        ],
        "recommended_query": "Explain what documented information is required for Clause 4 Context of the Organization."
    },
    "Clause 5: Leadership": {
        "focus": "Top management leadership commitment, QMS integration, customer focus, quality policies, and organizational roles.",
        "sub_clauses": [
            "5.1 Leadership and commitment (5.1.1 General, 5.1.2 Customer focus)",
            "5.2 Policy (5.2.1 Establishing, 5.2.2 Communicating)",
            "5.3 Organizational roles, responsibilities and authorities"
        ],
        "recommended_query": "What are the specific leadership duties expected of top management under Clause 5.1?"
    },
    "Clause 6: Planning": {
        "focus": "Risk-based thinking, actions to address risks and opportunities, Quality Objectives, and planning process changes.",
        "sub_clauses": [
            "6.1 Actions to address risks and opportunities",
            "6.2 Quality objectives and planning to achieve them",
            "6.3 Planning of changes"
        ],
        "recommended_query": "How does ISO 9001:2015 describe 'Actions to address risks and opportunities' in Clause 6.1?"
    },
    "Clause 7: Support": {
        "focus": "Resources (people, infrastructure, environment, calibration), competency, awareness, internal/external communication, and documented info controls.",
        "sub_clauses": [
            "7.1 Resources (7.1.5 Monitoring/measuring resources, 7.1.6 Organizational knowledge)",
            "7.2 Competence",
            "7.3 Awareness",
            "7.4 Communication",
            "7.5 Documented information (7.5.3 Control of documented information)"
        ],
        "recommended_query": "What are the requirements for controlling documented information under Clause 7.5.3?"
    },
    "Clause 8: Operation": {
        "focus": "Core service/product design, customer requirements, supplier/vendor procurement controls, production control, product release, and nonconforming outputs.",
        "sub_clauses": [
            "8.1 Operational planning and control",
            "8.2 Requirements for products and services (customer communication, reviews)",
            "8.3 Design and development of products and services",
            "8.4 Control of externally provided processes, products and services (Suppliers)",
            "8.5 Production and service provision (8.5.1 Control, 8.5.2 Identification/traceability)",
            "8.6 Release of products and services",
            "8.7 Control of nonconforming outputs"
        ],
        "recommended_query": "Detail the full operational requirements for controlling externally provided suppliers/vendors under Clause 8.4."
    },
    "Clause 9: Performance Evaluation": {
        "focus": "Customer satisfaction tracking, monitoring metrics, internal audits, and top management reviews.",
        "sub_clauses": [
            "9.1 Monitoring, measurement, analysis and evaluation",
            "9.2 Internal audit",
            "9.3 Management review"
        ],
        "recommended_query": "Provide a comprehensive template checklist for conducting a management review under Clause 9.3."
    },
    "Clause 10: Improvement": {
        "focus": "Reacting to nonconformities, implementing corrective actions, root cause analyses, and continuous improvement metrics.",
        "sub_clauses": [
            "10.1 General requirements",
            "10.2 Nonconformity and corrective action",
            "10.3 Continual improvement"
        ],
        "recommended_query": "What is the structured response required when a nonconformity occurs under Clause 10.2?"
    }
}


# --- REST ENDPOINTS ---

@app.get("/api/status")
def get_status(provider: str = "gemini"):
    """Returns the current state of selected provider, keys, and isolated index files."""
    provider = provider.lower().strip()
    
    # 1. Evaluate corresponding provider key status
    api_key_set = False
    if provider == "gemini":
        api_key_set = bool(get_gemini_key())
    elif provider == "cohere":
        api_key_set = bool(get_cohere_key())
        
    # 2. Evaluate local Ollama status
    ollama_active = False
    try:
        resp = requests.get("http://127.0.0.1:11434/api/tags", timeout=2)
        ollama_active = (resp.status_code == 200)
    except:
        pass
        
    # 3. Evaluate matching index existence
    if provider == "gemini":
        index_name = "vector_store_gemini.json"
    elif provider == "cohere":
        index_name = "vector_store_cohere.json"
    else:
        index_name = "vector_store_ollama.json"
        
    index_file = os.path.join(os.path.dirname(__file__), index_name)
    index_exists = os.path.exists(index_file)
    
    num_chunks = 0
    if index_exists:
        try:
            with open(index_file, "r", encoding="utf-8") as f:
                index_data = json.load(f)
                num_chunks = len(index_data.get("chunks", []))
        except:
            pass
            
    return {
        "api_key_configured": api_key_set,
        "ollama_active": ollama_active,
        "index_ready": index_exists,
        "indexed_chunks": num_chunks
    }


@app.post("/api/configure-key")
def configure_key(payload: APIKeyRequest):
    """Saves a user-submitted Gemini API Key locally into the backend's .env file."""
    try:
        save_env_key("GEMINI_API_KEY", payload.api_key)
        return {"status": "success", "message": "Gemini API Key configured successfully."}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to write key: {e}")


@app.post("/api/configure-cohere-key")
def configure_cohere_key(payload: APIKeyRequest):
    """Saves a user-submitted Cohere API Key locally into the backend's .env file."""
    try:
        save_env_key("COHERE_API_KEY", payload.api_key)
        return {"status": "success", "message": "Cohere API Key configured successfully."}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to write key: {e}")


@app.get("/api/ollama/models")
def get_ollama_models():
    """Queries the local Ollama daemon and lists currently pulled model names."""
    try:
        response = requests.get("http://127.0.0.1:11434/api/tags", timeout=3)
        response.raise_for_status()
        data = response.json()
        models = [item["name"] for item in data.get("models", [])]
        return {"status": "success", "models": models}
    except Exception as e:
        # If server is offline, return success but empty list so UI doesn't crash
        return {"status": "offline", "models": [], "error": str(e)}


@app.post("/api/ingest")
def trigger_ingest(
    file: UploadFile = File(...),
    force: bool = False,
    provider: str = "gemini"
):
    """Triggers the PDF standard extraction, chunking, and embedding creation on selected provider."""
    # Check if the ISO 9001 PDF standard file exists in the workspace
    workspace_pdf_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "ISO-9001-2015-Fifth-Edition.pdf"))
    if not os.path.exists(workspace_pdf_path):
        raise HTTPException(
            status_code=400,
            detail="ISO 9001 PDF standard file not found in the workspace."
        )

    # Resolve engine dynamically
    engine = get_engine(provider=provider)
    
    # Save the uploaded file to a temporary location inside the backend directory
    temp_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "temp_ingest.pdf"))
    
    try:
        with open(temp_path, "wb") as buffer:
            # Read and write file contents
            content = file.file.read()
            buffer.write(content)
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to write uploaded file to temporary directory: {e}"
        )
        
    try:
        chunks_count, loaded_from_cache = engine.ingest_pdf(temp_path, force_rebuild=force)
        return {
            "status": "success",
            "chunks_count": chunks_count,
            "loaded_from_cache": loaded_from_cache,
            "message": f"Successfully processed PDF into {chunks_count} vector chunks using {provider.upper()}."
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed PDF Ingestion: {e}")
    finally:
        # Clean up the temporary file immediately
        if os.path.exists(temp_path):
            try:
                os.remove(temp_path)
            except Exception as e:
                print(f"Error removing temporary file {temp_path}: {e}")


TRACE_LOG_FILE = os.path.join(os.path.dirname(__file__), "trace_log.json")

def save_trace_log(trace_data: Dict[str, Any]):
    """Appends a new decision row to the local trace_log.json file."""
    history = []
    try:
        if os.path.exists(TRACE_LOG_FILE):
            with open(TRACE_LOG_FILE, "r", encoding="utf-8") as f:
                history = json.load(f)
                if not isinstance(history, list):
                    history = []
    except Exception as e:
        print(f"Error loading trace history: {e}")
        
    history.append(trace_data)
    
    try:
        with open(TRACE_LOG_FILE, "w", encoding="utf-8") as f:
            json.dump(history, f, ensure_ascii=False, indent=2)
    except Exception as e:
        print(f"Error writing trace log: {e}")


@app.post("/api/query")
def trigger_query(payload: QueryRequest):
    """Executes a semantic vector Q&A query and returns a citation-backed LLM response."""
    # Resolve engine dynamically
    engine = get_engine(provider=payload.provider)
    
    # Ensure matching index exists before query
    index_file = os.path.join(os.path.dirname(__file__), engine.index_name)
    if not os.path.exists(index_file):
        raise HTTPException(
            status_code=400, 
            detail=f"The QMS vector store index has not been built yet for {payload.provider.upper()} mode. Please run Ingestion first."
        )
        
    # Lazy load vector index if it hasn't been loaded in engine yet
    if not engine.vector_store.chunks:
        engine.vector_store.load(index_file)
        
    try:
        result = engine.query(
            user_query=payload.query, 
            top_k=payload.top_k, 
            temperature=payload.temperature,
            ollama_model=payload.ollama_model,
            gemini_model=payload.gemini_model,
            cohere_model=payload.cohere_model
        )
        
        # Save trace decision log row
        from datetime import datetime
        trace_row = {
            "timestamp": datetime.now().isoformat(),
            "query": payload.query,
            "provider": payload.provider,
            "gemini_model": payload.gemini_model if payload.provider == "gemini" else "",
            "cohere_model": payload.cohere_model if payload.provider == "cohere" else "",
            "ollama_model": payload.ollama_model if payload.provider == "ollama" else "",
            "query_attempts": result.get("query_attempts", []),
            "answer_summary": result.get("answer", "")[:120] + "..."
        }
        save_trace_log(trace_row)
        
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed generation: {e}")


@app.get("/api/trace")
def get_trace_log():
    """Returns the query decision trace log containing search, rewrite, and rerank metrics."""
    if not os.path.exists(TRACE_LOG_FILE):
        return []
    try:
        with open(TRACE_LOG_FILE, "r", encoding="utf-8") as f:
            data = json.load(f)
            return data
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to read trace log: {e}")


@app.post("/api/trace/clear")
def clear_trace_log():
    """Clears the query decision trace log history."""
    try:
        if os.path.exists(TRACE_LOG_FILE):
            os.remove(TRACE_LOG_FILE)
        return {"status": "success", "message": "Decision trace log cleared."}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to clear trace log: {e}")


@app.get("/api/clauses")
def get_clauses():
    """Returns the structured dictionary of ISO 9001:2015 clauses for navigator panel."""
    return CLAUSES_DATABASE


@app.get("/api/stats")
def get_stats(provider: str = "gemini"):
    """Extracts diagnostic statistics from the selected provider's indexed vector store."""
    index_name = "vector_store_gemini.json" if provider.lower().strip() == "gemini" else "vector_store_ollama.json"
    index_file = os.path.join(os.path.dirname(__file__), index_name)
    
    if not os.path.exists(index_file):
        raise HTTPException(status_code=400, detail="No diagnostic index found. Index is empty.")
        
    try:
        with open(index_file, "r", encoding="utf-8") as f:
            data = json.load(f)
            
        chunks = data.get("chunks", [])
        if not chunks:
            return {"chunks_count": 0}
            
        total_chunks = len(chunks)
        total_chars = sum(len(c["text"]) for c in chunks)
        avg_chunk = total_chars / total_chunks
        
        pages_set = set(c["page"] for c in chunks)
        
        return {
            "total_pages": len(pages_set),
            "chunks_count": total_chunks,
            "avg_chunk_length": round(avg_chunk),
            "total_characters": total_chars,
            "chunks": chunks
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed reading diagnostics: {e}")

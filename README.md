# ISO 9001:2015 QMS AI Auditor - Architecture & Process Flow

Welcome to the **ISO 9001:2015 QMS AI Auditor**, a state-of-the-art Quality Management Systems (QMS) auditing assistant. This application leverages Semantic Vector Search (RAG) and Google Gemini (or local offline Ollama models) to query standards, generate compliance checklists, and audit organizational processes with full page-level and clause-level citations.

---

## 🏛️ System Architecture

The application is structured into a modular, decoupled architecture consisting of:
1. **Vanilla Web Frontend (HTML5/CSS3/ES6)**: A premium-grade responsive dashboard optimized for low latency and rich user interaction. Contains panels for Q&A, clause indexing, and vector store diagnostics.
2. **FastAPI Backend (Python)**: A high-performance REST API supplying the semantic engine, managing configuration keys, indexing vectors, and executing queries.
3. **Semantic RAG Engine (Numpy/Gemini/Ollama)**: A self-contained RAG pipeline coordinating document parsing, sliding-window chunking, embedding serialization, and augmented response generation.

---

## 🔄 End-to-End Process Flow

The system operates via two core pipelines: **Knowledge Ingestion & Vector Indexing** and **Semantic Retrieval & Augmented Generation**.

```
[ PHASE A: KNOWLEDGE BASE INGESTION ]
   📄 ISO 9001 PDF Standard
           │
           ▼
   🧹 Text Extraction & Cleaning (pypdf & regex layout cleaning)
           │
           ▼
   ✂️ Sliding-Window Chunking (800 chars / 150 chars overlap + boundary alignment)
           │
           ▼
   🧠 Embedding Generation (Google embedding-001 or Local nomic-embed-text)
           │
           ▼
   💾 Index Serialization (JSON isolated vector databases)

---------------------------------------------------------------------------------

[ PHASE B: SEMANTIC RETRIEVAL & Q&A GENERATION ]
   👤 User Search Query
           │
           ▼
   🧠 Query Embedding Generation (Embed query dynamically)
           │
           ▼
   🔍 Cosine Similarity Search (Numpy dot-product scan on active Vector Store)
           │
           ▼
   📑 Context & System Prompt Assembly (Inject top-k chunks + citations + system rules)
           │
           ▼
   🤖 LLM Generative Evaluation (Gemini 2.5/3.5 or Local Llama3 chat models)
           │
           ▼
   🛡️ Citation-Backed Auditor Response (Rendered markdown response with source badges)
```

---

### 📥 Phase A: Knowledge Base Ingestion Flow

1. **PDF Selection**: The user selects or drags the standard `ISO-9001-2015-Fifth-Edition.pdf` file into the sidebar uploader.
2. **Workspace Presence Verification**: The backend validates that the authentic PDF exists within the workspace directory. If missing, it raises an explicit alert.
3. **Text Extraction**: The `PDFProcessor` parses the document page-by-page, executing cleanups to strip running headers (`ISO 9001:2015(E)`) and page counters to avoid context noise.
4. **Sliding-Window Chunking**: The text is chunked into standard windows of **800 characters** with a **150-character overlap**. Chunk boundaries are dynamically adjusted to end at complete sentences or paragraphs rather than splitting mid-word. The processor extracts active clauses (e.g. `Clause 5.1`) appearing in the text as metadata tags.
5. **Vector Embedding**: Text chunks are dispatched to the selected provider:
   - **Gemini**: Sent in rate-limit safe batches (max 40 per batch with sleep cooldowns) using `models/gemini-embedding-001`.
   - **Ollama**: Dispatched locally via `nomic-embed-text`.
6. **Isolated Serialization**: Stored along with metadata as `vector_store_gemini.json` or `vector_store_ollama.json` within the backend directory. This prevents matrix dimension mismatches between providers.

---

### 💬 Phase B: Semantic Q&A Retrieval Flow

1. **Query Submission**: The user submits a query (e.g., *"What documented information is required for Clause 4?"*).
2. **Dynamic Query Embedding**: The query is converted into a vector using the corresponding provider's embedding pipeline.
3. **Vector Similarity Scan**: An in-memory Numpy dot-product is calculated against the active database:
   $$\text{Cosine Similarity} = \frac{A \cdot B}{\|A\| \|B\|}$$
   The top-$k$ (default 5) highest-scoring chunks are retrieved.
4. **Context Injection**: The text content of the matching chunks is combined with original page numbers, clause tags, and similarity scores.
5. **Auditor Persona Grounding**: The system creates a strict instruction payload prompting the LLM to behave as a professional Lead QMS Auditor, answer *strictly* using the retrieved context, and cite sources in bracketed `[Source #1]` formats.
6. **Inference & Streaming**: Generates the final compliance evaluation using active modern models (such as `gemini-2.5-flash` or local `llama3`) and renders the citation badges and checklist boxes dynamically in the chat UI.

---

## 🚀 Running the System

### 1. Launching the Servers
Run the PowerShell launcher script at the workspace root to start both backend and frontend servers in parallel and open your default browser:
```powershell
./run_qms.ps1
```

### 2. Running Automated Tests
Run the self-contained integration test suite to verify frontend-backend ports, file routing, HTML DOM element integrity, and endpoint status schema:
```powershell
python test_ui.py
```

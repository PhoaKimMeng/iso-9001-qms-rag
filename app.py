import streamlit as st
import os
import time
from backend.rag_engine import RAGEngine, PDFProcessor
from backend.main import CLAUSES_DATABASE, get_gemini_key, get_cohere_key, save_env_key

# Page configurations
st.set_page_config(
    page_title="ISO 9001:2015 QMS AI Auditor",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom premium styling
st.markdown("""
<style>
    /* Premium dark mode theme tweaks */
    .stApp {
        background: linear-gradient(135deg, #0f172a 0%, #1e1b4b 100%);
        color: #f8fafc;
    }
    
    /* Gilded Headers */
    .glow-text {
        font-family: 'Inter', sans-serif;
        font-weight: 800;
        background: linear-gradient(to right, #6366f1, #a855f7, #ec4899);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 20px;
    }
    
    /* Custom Card Design */
    .premium-card {
        background: rgba(30, 41, 59, 0.7);
        border: 1px solid rgba(99, 102, 241, 0.2);
        border-radius: 12px;
        padding: 20px;
        box-shadow: 0 4px 20px rgba(0, 0, 0, 0.3);
        margin-bottom: 15px;
    }
    
    /* Source badges */
    .source-badge {
        display: inline-block;
        background: rgba(99, 102, 241, 0.2);
        border: 1px solid rgba(99, 102, 241, 0.4);
        border-radius: 4px;
        padding: 2px 8px;
        font-size: 0.8rem;
        font-weight: 600;
        color: #c7d2fe;
        margin-right: 8px;
    }
    
    .score-badge {
        display: inline-block;
        background: rgba(16, 185, 129, 0.2);
        border: 1px solid rgba(16, 185, 129, 0.4);
        border-radius: 4px;
        padding: 2px 8px;
        font-size: 0.8rem;
        font-weight: 600;
        color: #a7f3d0;
        margin-right: 8px;
    }
    
    /* Diagnostics metrics */
    .metric-val {
        font-size: 2.2rem;
        font-weight: 800;
        color: #818cf8;
        line-height: 1;
        margin-bottom: 4px;
    }
</style>
""", unsafe_value_html=True)

# ----------------- SESSION STATE INITIALIZATION -----------------
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "active_provider" not in st.session_state:
    st.session_state.active_provider = "gemini"

# ----------------- SIDEBAR CONFIGURATIONS -----------------
st.sidebar.markdown("<h2 style='text-align: center; margin-bottom: 0;'>🛡️ System Config</h2>", unsafe_allow_html=True)
st.sidebar.markdown("<p style='text-align: center; color: #94a3b8; font-size: 0.85rem;'>ISO 9001:2015 QMS Auditor Settings</p>", unsafe_allow_html=True)
st.sidebar.markdown("<hr style='margin: 10px 0; opacity: 0.25;' />", unsafe_allow_html=True)

# 1. Provider select
provider = st.sidebar.selectbox(
    "🔌 RAG Engine Provider",
    options=["gemini", "cohere", "ollama"],
    format_func=lambda x: {
        "gemini": "Google Gemini (Cloud)",
        "cohere": "Cohere (Cloud)",
        "ollama": "Ollama (Local Offline)"
    }[x]
)
st.session_state.active_provider = provider

st.sidebar.markdown("<br>", unsafe_allow_html=True)

# 2. Provider API Authentication Settings
if provider == "gemini":
    st.sidebar.markdown("### 🔑 API Authentication")
    current_key = get_gemini_key()
    placeholder = "••••••••••••••••••••••••" if current_key else "Enter Gemini API Key"
    gemini_key_input = st.sidebar.text_input("Enter Gemini API Key", type="password", placeholder=placeholder, label_visibility="collapsed")
    if st.sidebar.button("Save API Key", key="save_gem_btn", use_container_width=True):
        if gemini_key_input:
            save_env_key("GEMINI_API_KEY", gemini_key_input)
            st.sidebar.success("Gemini API Key configured successfully.")
            time.sleep(1)
            st.rerun()
        else:
            st.sidebar.error("Please enter a valid API Key.")
            
elif provider == "cohere":
    st.sidebar.markdown("### 🔑 Cohere Authentication")
    current_key = get_cohere_key()
    placeholder = "••••••••••••••••••••••••" if current_key else "Enter Cohere API Key"
    cohere_key_input = st.sidebar.text_input("Enter Cohere API Key", type="password", placeholder=placeholder, label_visibility="collapsed")
    if st.sidebar.button("Save API Key", key="save_coh_btn", use_container_width=True):
        if cohere_key_input:
            save_env_key("COHERE_API_KEY", cohere_key_input)
            st.sidebar.success("Cohere API Key configured successfully.")
            time.sleep(1)
            st.rerun()
        else:
            st.sidebar.error("Please enter a valid API Key.")
            
else:
    st.sidebar.markdown("### 🦙 Local Ollama Server")
    # Verify local Ollama status
    import requests
    ollama_connected = False
    local_models = []
    try:
        resp = requests.get("http://127.0.0.1:11434/api/tags", timeout=2)
        if resp.status_code == 200:
            ollama_connected = True
            local_models = [m["name"] for m in resp.json().get("models", []) if "embed" not in m["name"]]
    except:
        pass
        
    if ollama_connected:
        st.sidebar.markdown("🟢 **Ollama: Connected**")
    else:
        st.sidebar.markdown("🔴 **Ollama: Offline**")
        st.sidebar.error("Ollama service offline. Please launch the Ollama application.")
        st.sidebar.info("Run in terminal:\n`ollama pull llama3 nomic-embed-text`")

st.sidebar.markdown("<hr style='margin: 15px 0; opacity: 0.25;' />", unsafe_allow_html=True)

# 3. Knowledge Base Ingestion Section
st.sidebar.markdown("### 📚 Knowledge Base Ingestion")

# Check vector index readiness
index_name = f"vector_store_{provider}.json"
backend_dir = os.path.join(os.path.dirname(__file__), "backend")
index_file = os.path.join(backend_dir, index_name)
index_ready = os.path.exists(index_file)

if index_ready:
    st.sidebar.markdown(f"🟢 **Index: Ready ({provider.upper()})**")
else:
    st.sidebar.markdown(f"🔴 **Index: Not Found ({provider.upper()})**")

# File Selection (Uploader)
uploaded_file = st.sidebar.file_uploader("Select QMS / ISO 9001 PDF", type=["pdf"])

# Enforce workspace presence safety check
workspace_pdf_name = "ISO-9001-2015-Fifth-Edition.pdf"
workspace_pdf_path = os.path.abspath(workspace_pdf_name)
workspace_pdf_exists = os.path.exists(workspace_pdf_path)

if st.sidebar.button("Ingest Selected Document", use_container_width=True, disabled=not uploaded_file):
    if not workspace_pdf_exists:
        st.sidebar.error("Ingestion Error: ISO 9001 PDF standard file not found in the workspace.")
    else:
        with st.spinner(f"Ingesting PDF into {provider.upper()} vector store..."):
            # Save uploaded file temporarily to backend
            temp_path = os.path.join(backend_dir, "temp_streamlit_ingest.pdf")
            try:
                with open(temp_path, "wb") as f:
                    f.write(uploaded_file.getbuffer())
                    
                # Instantiate RAGEngine dynamically
                if provider == "gemini":
                    engine = RAGEngine(api_key=get_gemini_key(), provider="gemini")
                elif provider == "cohere":
                    engine = RAGEngine(api_key=get_cohere_key(), provider="cohere")
                else:
                    engine = RAGEngine(api_key="", provider="ollama")
                    
                chunks_count, loaded_from_cache = engine.ingest_pdf(temp_path, force_rebuild=True)
                st.sidebar.success(f"Ingestion Completed! {chunks_count} vector chunks generated and indexed in {provider.upper()} mode.")
                time.sleep(1.5)
                st.rerun()
            except Exception as e:
                st.sidebar.error(f"Ingestion Error: {e}")
            finally:
                if os.path.exists(temp_path):
                    os.remove(temp_path)

st.sidebar.markdown("<hr style='margin: 15px 0; opacity: 0.25;' />", unsafe_allow_html=True)

# 4. Auditor Tuning Section
st.sidebar.markdown("### 🎛️ Auditor Tuning")

# Model selectors
if provider == "gemini":
    model_choice = st.sidebar.selectbox(
        "Generative Model",
        options=["gemini-2.5-flash", "gemini-2.0-flash", "gemini-2.5-pro", "gemini-3.5-flash"],
        format_func=lambda x: {
            "gemini-2.5-flash": "gemini-2.5-flash (Balanced)",
            "gemini-2.0-flash": "gemini-2.0-flash (Fast)",
            "gemini-2.5-pro": "gemini-2.5-pro (Analytical)",
            "gemini-3.5-flash": "gemini-3.5-flash (Next-Gen)"
        }[x]
    )
elif provider == "cohere":
    model_choice = st.sidebar.selectbox(
        "Generative Model",
        options=["command-r-08-2024", "command-r-plus-08-2024", "command-r7b-12-2024"],
        format_func=lambda x: {
            "command-r-08-2024": "command-r-08-2024 (Balanced)",
            "command-r-plus-08-2024": "command-r-plus-08-2024 (Analytical)",
            "command-r7b-12-2024": "command-r7b-12-2024 (Fast)"
        }[x]
    )
else:
    # Ollama models select
    if local_models:
        model_choice = st.sidebar.selectbox("Choose Local Model", options=local_models)
    else:
        model_choice = st.sidebar.selectbox("Choose Local Model", options=[""], format_func=lambda x: "No local models found")

top_k = st.sidebar.slider("Retrieved Source Count (Top-k)", min_value=3, max_value=8, value=5)
temperature = st.sidebar.slider("Auditor Temperature", min_value=0.0, max_value=1.0, value=0.1, step=0.05)


# ----------------- MAIN APP WORKSPACE -----------------

# Header Section
st.markdown("<h1 class='glow-text'>🛡️ ISO 9001:2015 QMS AI Auditor</h1>", unsafe_allow_html=True)
st.markdown("<p style='color:#94a3b8; font-size:1.1rem; margin-top:-15px; margin-bottom:25px;'>A premium desktop Quality Management Systems auditor. Powered by semantic vector search.</p>", unsafe_allow_html=True)

# Navigation tabs
tab_chat, tab_nav, tab_diag = st.tabs([
    "💬 Q&A Auditor Chat", 
    "📑 Document Clause Navigator", 
    "📊 Ingestion Stats & Diagnostics"
])

# ----------------- TAB 1: Chat Q&A interface -----------------
with tab_chat:
    # Quick Queries Section
    st.markdown("### ⚡ Quick Audit Queries")
    q_col1, q_col2, q_col3, q_col4 = st.columns(4)
    
    quick_queries = [
        ("What is the scope of Clause 4.1?", "What is the scope of Clause 4.1 regarding understanding the organization and its context?"),
        ("Audit checklist for Clause 5.2?", "Generate an auditor checklist for Clause 5.2 (establishing and communicating the quality policy)."),
        ("Explain Clause 8.4 control?", "Explain Clause 8.4 requirements for control of externally provided processes, products and services."),
        ("Requirements for Clause 9.2?", "What are the requirements for internal audits according to Clause 9.2?")
    ]
    
    selected_quick_query = None
    if q_col1.button(quick_queries[0][0], key="qq1", use_container_width=True):
        selected_quick_query = quick_queries[0][1]
    if q_col2.button(quick_queries[1][0], key="qq2", use_container_width=True):
        selected_quick_query = quick_queries[1][1]
    if q_col3.button(quick_queries[2][0], key="qq3", use_container_width=True):
        selected_quick_query = quick_queries[2][1]
    if q_col4.button(quick_queries[3][0], key="qq4", use_container_width=True):
        selected_quick_query = quick_queries[3][1]

    st.markdown("<hr style='opacity: 0.15;' />", unsafe_allow_html=True)

    # Render Chat History
    chat_container = st.container()
    with chat_container:
        if not st.session_state.chat_history:
            # Render Welcome Message
            with st.chat_message("assistant", avatar="🛡️"):
                st.markdown(f"""
                Welcome to the **ISO 9001:2015 QMS AI Auditor**.
                
                System mode is active on **{provider.upper()}**. 
                
                * Please verify that your API Key is configured and your PDF index status shows **Ready** in the sidebar.
                * Ask compliance questions, request checklist drafts, or examine clause citations!
                """)
        else:
            for msg in st.session_state.chat_history:
                with st.chat_message(msg["role"], avatar="👤" if msg["role"] == "user" else "🛡️"):
                    st.markdown(msg["content"])
                    if msg.get("sources"):
                        with st.expander("🔍 Verified Sources Used (Click to toggle)"):
                            for i, src in enumerate(msg["sources"]):
                                st.markdown(f"""
                                <div class='premium-card' style='margin-bottom: 10px; padding: 12px; background: rgba(30, 41, 59, 0.4);'>
                                    <div>
                                        <span class='source-badge'>Page {src['page']}</span>
                                        <span class='score-badge'>Sim: {src['score']:.3f}</span>
                                        <span style='color: #818cf8; font-weight: 600;'>Clauses: {", ".join(src['clauses'])}</span>
                                    </div>
                                    <p style='margin-top: 8px; font-style: italic; color: #cbd5e1;'>"{src['text']}"</p>
                                </div>
                                """, unsafe_allow_html=True)

    # Chat Input handling
    user_query = st.chat_input("Ask about ISO 9001:2015 requirements, audits, or implementation...")
    
    # Trigger query if either quick query or chat input is activated
    active_query = selected_quick_query or user_query
    
    if active_query:
        # Enforce index readiness check
        if not index_ready:
            st.error("⚠️ Complete PDF Ingestion in the sidebar first before querying.")
        else:
            # Append User message
            st.session_state.chat_history.append({"role": "user", "content": active_query})
            with st.chat_message("user", avatar="👤"):
                st.markdown(active_query)
                
            # Perform query call
            with st.chat_message("assistant", avatar="🛡️"):
                with st.spinner("Auditing QMS standards database..."):
                    try:
                        # Instantiate dynamically
                        if provider == "gemini":
                            engine = RAGEngine(api_key=get_gemini_key(), provider="gemini")
                        elif provider == "cohere":
                            engine = RAGEngine(api_key=get_cohere_key(), provider="cohere")
                        else:
                            engine = RAGEngine(api_key="", provider="ollama")
                            
                        # Load vector index in engine
                        engine.vector_store.load(index_file)
                        
                        # Trigger RAG Query
                        result = engine.query(
                            user_query=active_query,
                            top_k=top_k,
                            temperature=temperature,
                            ollama_model=model_choice if provider == "ollama" else "llama3",
                            gemini_model=model_choice if provider == "gemini" else "gemini-2.5-flash",
                            cohere_model=model_choice if provider == "cohere" else "command-r-08-2024"
                        )
                        
                        st.markdown(result["answer"])
                        
                        # Store in history
                        st.session_state.chat_history.append({
                            "role": "assistant",
                            "content": result["answer"],
                            "sources": result["sources"]
                        })
                        
                        # Show sources under expander
                        with st.expander("🔍 Verified Sources Used (Click to toggle)"):
                            for i, src in enumerate(result["sources"]):
                                st.markdown(f"""
                                <div class='premium-card' style='margin-bottom: 10px; padding: 12px; background: rgba(30, 41, 59, 0.4);'>
                                    <div>
                                        <span class='source-badge'>Page {src['page']}</span>
                                        <span class='score-badge'>Sim: {src['score']:.3f}</span>
                                        <span style='color: #818cf8; font-weight: 600;'>Clauses: {", ".join(src['clauses'])}</span>
                                    </div>
                                    <p style='margin-top: 8px; font-style: italic; color: #cbd5e1;'>"{src['text']}"</p>
                                </div>
                                """, unsafe_allow_html=True)
                    except Exception as err:
                        st.error(f"Error executing audit search: {err}")

# ----------------- TAB 2: Clause Navigator -----------------
with tab_nav:
    st.markdown("### 📑 Interactive ISO 9001:2015 Clause Index")
    st.markdown("<p style='color: #94a3b8;'>Select a clause to explore its core focus and sub-clauses. Click to ask the Auditor directly about it.</p>", unsafe_allow_html=True)
    st.markdown("<br>", unsafe_allow_html=True)
    
    for title, details in CLAUSES_DATABASE.items():
        with st.expander(f"📘 {title}"):
            st.markdown(f"**Core Focus**: {details['focus']}")
            st.markdown("**Sub-Clauses**:")
            for sub in details["sub_clauses"]:
                st.markdown(f"- {sub}")
            
            st.markdown("<br>", unsafe_allow_html=True)
            if st.button(f"Ask Auditor about {title.split(':')[0]}", key=f"nav_btn_{title}"):
                st.session_state.chat_history.append({"role": "user", "content": details["recommended_query"]})
                # Rerun to Tab 1 chat logic
                st.rerun()

# ----------------- TAB 3: Diagnostics Panel -----------------
with tab_diag:
    st.markdown("### 📊 Knowledge Base Diagnostics")
    st.markdown("<p style='color: #94a3b8;'>Detailed breakdown of the processed document vector store index.</p>", unsafe_allow_html=True)
    st.markdown("<br>", unsafe_allow_html=True)
    
    if not index_ready:
        st.info("Please perform the PDF standard ingestion in the sidebar to view diagnostics metrics.")
    else:
        # Load stats
        try:
            import json
            with open(index_file, "r", encoding="utf-8") as f:
                data = json.load(f)
            chunks = data.get("chunks", [])
            
            if not chunks:
                st.warning("Diagnostics: The loaded index is empty.")
            else:
                total_chunks = len(chunks)
                total_chars = sum(len(c["text"]) for c in chunks)
                avg_chunk = round(total_chars / total_chunks)
                pages_set = set(c["page"] for c in chunks)
                total_pages = len(pages_set)
                
                # Metrics layout
                d_col1, d_col2, d_col3, d_col4 = st.columns(4)
                d_col1.markdown(f"<div class='premium-card'><div class='metric-val'>{total_pages}</div><div style='color:#94a3b8;'>Total Pages Parsed</div></div>", unsafe_allow_html=True)
                d_col2.markdown(f"<div class='premium-card'><div class='metric-val'>{total_chunks}</div><div style='color:#94a3b8;'>Text Chunks Generated</div></div>", unsafe_allow_html=True)
                d_col3.markdown(f"<div class='premium-card'><div class='metric-val'>{avg_chunk}</div><div style='color:#94a3b8;'>Avg Characters / Chunk</div></div>", unsafe_allow_html=True)
                d_col4.markdown(f"<div class='premium-card'><div class='metric-val'>{total_chars:,}</div><div style='color:#94a3b8;'>Total Characters Indexed</div></div>", unsafe_allow_html=True)
                
                st.markdown("<br><hr style='opacity: 0.15;' /><br>", unsafe_allow_html=True)
                
                # Chunk Inspector Slider
                st.markdown("### 📝 Vector Store Chunks Sample")
                chunk_idx = st.slider("Select chunk index to inspect details:", min_value=0, max_value=total_chunks - 1, value=0)
                
                selected_chunk = chunks[chunk_idx]
                c_col1, c_col2, c_col3 = st.columns(3)
                c_col1.markdown(f"**Chunk ID:** `{selected_chunk['id']}`")
                c_col2.markdown(f"**Source Page:** `{selected_chunk['page']}`")
                c_col3.markdown(f"**Clause Tags:** `{', '.join(selected_chunk['clauses'])}`")
                
                st.markdown("<br>", unsafe_allow_html=True)
                st.markdown("**Chunk Text Content**:")
                st.code(selected_chunk["text"], language="text")
        except Exception as e:
            st.error(f"Failed to load vector store details for diagnostics: {e}")

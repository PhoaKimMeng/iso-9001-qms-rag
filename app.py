import os
import streamlit as st
import time
from dotenv import load_dotenv
from rag_engine import RAGEngine

# Load local environment variables if available
load_dotenv()

# App Configuration & Setup
st.set_page_config(
    page_title="ISO 9001:2015 QMS AI Auditor",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Premium Styling Injection
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;500;600;700&display=swap');

/* Apply font to overall app */
html, body, [class*="css"], .stText, p, span, h1, h2, h3, h4, h5, h6 {
    font-family: 'Outfit', sans-serif !important;
}

/* Glassmorphism containers */
.qms-container {
    background: linear-gradient(135deg, rgba(21, 26, 38, 0.7) 0%, rgba(13, 17, 24, 0.9) 100%);
    border: 1px solid rgba(99, 102, 241, 0.2);
    border-radius: 16px;
    padding: 24px;
    margin-bottom: 24px;
    box-shadow: 0 8px 32px 0 rgba(0, 0, 0, 0.37);
    backdrop-filter: blur(12px);
    -webkit-backdrop-filter: blur(12px);
}

.qms-stat-card {
    background: rgba(30, 41, 59, 0.4);
    border-left: 4px solid #6366f1;
    border-radius: 8px;
    padding: 16px;
    margin: 8px 0;
    box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
}

.qms-stat-number {
    font-size: 2.2rem;
    font-weight: 700;
    color: #818cf8;
    line-height: 1;
}

.qms-stat-label {
    font-size: 0.85rem;
    color: #94a3b8;
    text-transform: uppercase;
    letter-spacing: 0.05em;
    margin-top: 4px;
}

/* Header gradient */
.qms-gradient-header {
    background: linear-gradient(90deg, #818cf8 0%, #c084fc 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    font-weight: 800;
    margin-bottom: 8px;
}

/* Badges for sources */
.qms-source-badge {
    background-color: rgba(99, 102, 241, 0.15);
    color: #a5b4fc;
    border: 1px solid rgba(99, 102, 241, 0.3);
    padding: 4px 10px;
    border-radius: 9999px;
    font-size: 0.8rem;
    font-weight: 500;
    display: inline-block;
    margin-right: 8px;
    margin-bottom: 8px;
}

.qms-score-badge {
    background-color: rgba(16, 185, 129, 0.15);
    color: #34d399;
    border: 1px solid rgba(16, 185, 129, 0.3);
    padding: 4px 10px;
    border-radius: 9999px;
    font-size: 0.8rem;
    font-weight: 500;
    display: inline-block;
}

/* Button micro-animations */
div.stButton > button:first-child {
    background: linear-gradient(90deg, #4f46e5 0%, #7c3aed 100%) !important;
    color: white !important;
    border: none !important;
    border-radius: 8px !important;
    padding: 10px 24px !important;
    font-weight: 600 !important;
    transition: all 0.3s ease !important;
    box-shadow: 0 4px 15px rgba(79, 70, 229, 0.3) !important;
}

div.stButton > button:first-child:hover {
    transform: translateY(-2px) !important;
    box-shadow: 0 6px 20px rgba(79, 70, 229, 0.5) !important;
}

/* Custom styles for sidebar */
[data-testid="stSidebar"] {
    background-color: #0b0d13 !important;
    border-right: 1px solid rgba(255, 255, 255, 0.05) !important;
}

/* Glowing divider line */
.glow-line {
    height: 2px;
    background: linear-gradient(90deg, transparent 0%, #6366f1 50%, transparent 100%);
    margin: 20px 0;
}
</style>
""", unsafe_allow_html=True)

# Initialize Session States
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "rag_engine" not in st.session_state:
    st.session_state.rag_engine = None
if "api_key" not in st.session_state:
    # Try reading from OS env or local .env
    st.session_state.api_key = os.environ.get("GEMINI_API_KEY", "")
if "active_clause_query" not in st.session_state:
    st.session_state.active_clause_query = None

# Sidebar Content
with st.sidebar:
    st.markdown("<h2 style='text-align: center; color: #818cf8;'>🛡️ System Config</h2>", unsafe_allow_html=True)
    st.markdown("<p style='text-align: center; color: #94a3b8; font-size: 0.9rem;'>ISO 9001:2015 QMS Auditor Settings</p>", unsafe_allow_html=True)
    st.markdown("<div class='glow-line'></div>", unsafe_allow_html=True)

    # 1. API Configuration
    st.markdown("### 🔑 API Authentication")
    api_key_input = st.text_input(
        "Enter Gemini API Key",
        value=st.session_state.api_key,
        type="password",
        help="Get an API key from Google AI Studio (ai.google.dev)"
    )
    
    if api_key_input:
        st.session_state.api_key = api_key_input
        # Initialize RAGEngine if not already done or if key changed
        if st.session_state.rag_engine is None or st.session_state.rag_engine.api_key != api_key_input:
            st.session_state.rag_engine = RAGEngine(api_key=api_key_input)
            
    if not st.session_state.api_key:
        st.warning("⚠️ Please provide a Gemini API Key to enable embeddings and generation.")
        
    st.markdown("<div class='glow-line'></div>", unsafe_allow_html=True)

    # 2. Knowledge Ingestion Manager
    st.markdown("### 📚 Knowledge Base Ingestion")
    pdf_path = "ISO-9001-2015-Fifth-Edition.pdf"
    
    # Check if index exists
    index_file = "vector_store.json"
    index_exists = os.path.exists(index_file)
    
    if index_exists:
        st.success("✅ Knowledge Index: Ready")
        # Load index stats to show
        try:
            with open(index_file, "r", encoding="utf-8") as f:
                index_data = json.load(f)
                num_chunks = len(index_data["chunks"])
        except:
            num_chunks = "Error loading stats"
            
        st.info(f"Loaded {num_chunks} chunks from disk.")
    else:
        st.warning("❌ Knowledge Index: Not Found")
        st.info("Run ingestion below to parse and embed the ISO PDF.")
        
    if st.session_state.rag_engine:
        if st.button("Ingest & Embed PDF" if not index_exists else "Rebuild QMS Index"):
            with st.spinner("Processing PDF, chunking and computing Gemini Embeddings... This takes about 15-30 seconds."):
                try:
                    num_chunks, cached = st.session_state.rag_engine.ingest_pdf(pdf_path, force_rebuild=True)
                    st.success(f"Successfully processed PDF! {num_chunks} chunks indexed.")
                    st.rerun()
                except Exception as e:
                    st.error(f"Ingestion failed: {e}")
    else:
        st.button("Ingest & Embed PDF", disabled=True, help="Provide a valid API Key first.")
        
    st.markdown("<div class='glow-line'></div>", unsafe_allow_html=True)

    # 3. Tuning Parameters
    st.markdown("### 🎛️ Auditor Tuning")
    model_name = st.selectbox(
        "Generative Model",
        options=["gemini-1.5-flash", "gemini-1.5-pro", "gemini-2.5-flash"],
        index=0,
        help="Pro is more analytical, Flash is faster."
    )
    
    top_k = st.slider(
        "Retrieved Source Count (Top-k)",
        min_value=3,
        max_value=8,
        value=5,
        help="Number of context passages sent to the auditor model."
    )
    
    temperature = st.slider(
        "Auditor Temperature",
        min_value=0.0,
        max_value=1.0,
        value=0.1,
        step=0.05,
        help="Lower values make the auditor more rigorous and strictly aligned with the context."
    )

# Main Application Layout
st.markdown("<h1 class='qms-gradient-header'>🛡️ ISO 9001:2015 QMS AI Auditor</h1>", unsafe_allow_html=True)
st.markdown("<p style='font-size: 1.15rem; color: #94a3b8; margin-bottom: 24px;'>A state-of-the-art Quality Management Systems auditor. Powered by semantic vector search and Google Gemini.</p>", unsafe_allow_html=True)

# Main Navigation Tabs
tab_chat, tab_navigator, tab_diagnostics = st.tabs([
    "💬 Q&A Auditor Chat", 
    "📑 Document Clause Navigator", 
    "📊 Ingestion Stats & Diagnostics"
])

# ==================== TAB 1: CHAT INTERFACE ====================
with tab_chat:
    # Quick Action prompts
    st.markdown("### ⚡ Quick Audit Queries")
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        if st.button("What is the scope of Clause 4.1?"):
            st.session_state.active_clause_query = "What is the scope of Clause 4.1 regarding understanding the organization and its context?"
    with col2:
        if st.button("Audit checklist for Clause 5.2?"):
            st.session_state.active_clause_query = "Generate an auditor checklist for Clause 5.2 (establishing and communicating the quality policy)."
    with col3:
        if st.button("Explain Clause 8.4 control?"):
            st.session_state.active_clause_query = "Explain Clause 8.4 requirements for control of externally provided processes, products and services."
    with col4:
        if st.button("Requirements for Clause 9.2?"):
            st.session_state.active_clause_query = "What are the requirements for internal audits according to Clause 9.2?"

    st.markdown("<br>", unsafe_allow_html=True)

    # Main Chat Area Container
    chat_container = st.container()

    # Display Chat History
    with chat_container:
        for message in st.session_state.chat_history:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])
                # If there are sources in the message, render them in an expander
                if "sources" in message and message["sources"]:
                    with st.expander("🔍 Verified Sources Used (Click to view)"):
                        for src in message["sources"]:
                            cols = st.columns([1, 1, 6])
                            with cols[0]:
                                st.markdown(f"<span class='qms-source-badge'>Page {src['page']}</span>", unsafe_allow_html=True)
                            with cols[1]:
                                st.markdown(f"<span class='qms-score-badge'>Sim: {src['score']:.3f}</span>", unsafe_allow_html=True)
                            with cols[2]:
                                st.markdown(f"**Clauses**: `{', '.join(src['clauses'])}`")
                            st.markdown(f"*{src['text']}*")
                            st.markdown("---")

    # Handle Input from Chat or Quick Actions
    user_query = st.chat_input("Ask about ISO 9001:2015 requirements, audits, or implementation...")
    
    # Pre-fill query if quick action was clicked
    if st.session_state.active_clause_query:
        user_query = st.session_state.active_clause_query
        st.session_state.active_clause_query = None

    if user_query:
        if not st.session_state.api_key:
            st.error("🔑 API Key Required! Please input your Gemini API Key in the sidebar.")
        elif not index_exists:
            st.error("📚 PDF Ingestion Required! Please click 'Ingest & Embed PDF' in the sidebar first.")
        else:
            # Display user message
            with chat_container:
                with st.chat_message("user"):
                    st.markdown(user_query)
            
            # Save user query to state
            st.session_state.chat_history.append({"role": "user", "content": user_query})
            
            # Generate Answer
            with chat_container:
                with st.chat_message("assistant"):
                    with st.spinner("Auditing standards knowledge base..."):
                        try:
                            # Apply configured parameters
                            st.session_state.rag_engine.vector_store.load(index_file) # Ensure current
                            
                            # Run the Q&A search
                            result = st.session_state.rag_engine.query(
                                user_query=user_query, 
                                top_k=top_k, 
                                temperature=temperature
                            )
                            
                            # Show response
                            st.markdown(result["answer"])
                            
                            # Show sources
                            if result["sources"]:
                                with st.expander("🔍 Verified Sources Used (Click to view)"):
                                    for src in result["sources"]:
                                        cols = st.columns([1, 1, 6])
                                        with cols[0]:
                                            st.markdown(f"<span class='qms-source-badge'>Page {src['page']}</span>", unsafe_allow_html=True)
                                        with cols[1]:
                                            st.markdown(f"<span class='qms-score-badge'>Sim: {src['score']:.3f}</span>", unsafe_allow_html=True)
                                        with cols[2]:
                                            st.markdown(f"**Clauses**: `{', '.join(src['clauses'])}`")
                                        st.markdown(f"*{src['text']}*")
                                        st.markdown("---")
                                        
                            # Save assistant response
                            st.session_state.chat_history.append({
                                "role": "assistant",
                                "content": result["answer"],
                                "sources": result["sources"]
                            })
                            
                        except Exception as e:
                            st.error(f"Error formulating response: {e}")

# ==================== TAB 2: CLAUSE NAVIGATOR ====================
with tab_navigator:
    st.markdown("### 📑 Interactive ISO 9001:2015 Clause Index")
    st.markdown("Select a clause to explore its core requirements. You can also directly query the AI Auditor on specific sub-clauses.")
    st.markdown("<br>", unsafe_allow_html=True)
    
    clauses_db = {
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

    # Render clauses cards with stream-filled buttons
    for clause_title, details in clauses_db.items():
        with st.container():
            st.markdown(f"""
            <div class='qms-container'>
                <h4 style='color: #a5b4fc; margin-top: 0;'>📘 {clause_title}</h4>
                <p style='color: #cbd5e1; font-size: 0.95rem; margin-bottom: 12px;'><strong>Core Focus:</strong> {details['focus']}</p>
                <div style='margin-left: 20px; border-left: 2px solid rgba(255,255,255,0.1); padding-left: 15px;'>
                    <ul style='color: #94a3b8; font-size: 0.9rem;'>
                        {"".join(f"<li>{sub}</li>" for sub in details['sub_clauses'])}
                    </ul>
                </div>
            </div>
            """, unsafe_allow_html=True)
            
            # Interaction buttons
            cols = st.columns([4, 4])
            with cols[0]:
                if st.button(f"Ask AI Auditor about {clause_title.split(':')[0]}", key=f"btn_ask_{clause_title}"):
                    st.session_state.active_clause_query = details["recommended_query"]
                    st.toast(f"Loaded recommended query for {clause_title.split(':')[0]}! Switch to Chat tab to see answer.")
            st.markdown("<br>", unsafe_allow_html=True)

# ==================== TAB 3: DIAGNOSTICS ====================
with tab_diagnostics:
    st.markdown("### 📊 Knowledge Base Diagnostics")
    st.markdown("Detailed breakdown of the processed document. This is highly useful for QMS managers to verify vector indexing coverage.")
    st.markdown("<br>", unsafe_allow_html=True)

    if index_exists:
        try:
            with open(index_file, "r", encoding="utf-8") as f:
                index_data = json.load(f)
                chunks = index_data["chunks"]
                
            # Compute stats
            total_chunks = len(chunks)
            total_chars = sum(len(c["text"]) for c in chunks)
            avg_chunk_len = total_chars / total_chunks
            
            pages_set = set(c["page"] for c in chunks)
            num_pages = len(pages_set)
            
            # Display metrics cards
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.markdown(f"""
                <div class='qms-stat-card'>
                    <div class='qms-stat-number'>{num_pages}</div>
                    <div class='qms-stat-label'>Total Pages Parsed</div>
                </div>
                """, unsafe_allow_html=True)
            with col2:
                st.markdown(f"""
                <div class='qms-stat-card'>
                    <div class='qms-stat-number'>{total_chunks}</div>
                    <div class='qms-stat-label'>Text Chunks Generated</div>
                </div>
                """, unsafe_allow_html=True)
            with col3:
                st.markdown(f"""
                <div class='qms-stat-card'>
                    <div class='qms-stat-number'>{avg_chunk_len:.0f}</div>
                    <div class='qms-stat-label'>Avg Characters / Chunk</div>
                </div>
                """, unsafe_allow_html=True)
            with col4:
                st.markdown(f"""
                <div class='qms-stat-card'>
                    <div class='qms-stat-number'>{total_chars:,}</div>
                    <div class='qms-stat-label'>Total Characters Index</div>
                </div>
                """, unsafe_allow_html=True)
                
            st.markdown("<br>### 📝 Vector Store Chunks Sample", unsafe_allow_html=True)
            # Display slider to select chunk sample
            chunk_idx = st.slider("Select chunk index to inspect details", 0, total_chunks - 1, 0)
            selected_chunk = chunks[chunk_idx]
            
            cols = st.columns([1, 1, 4])
            with cols[0]:
                st.markdown(f"**Chunk ID:** `{selected_chunk['id']}`")
            with cols[1]:
                st.markdown(f"**Source Page:** `{selected_chunk['page']}`")
            with cols[2]:
                st.markdown(f"**Clause Tags:** `{', '.join(selected_chunk['clauses'])}`")
                
            st.markdown(f"""
            <div style='background-color: rgba(30,41,59,0.3); border: 1px solid rgba(255,255,255,0.05); padding: 15px; border-radius: 8px; font-family: Courier, monospace; font-size: 0.9rem;'>
                {selected_chunk['text']}
            </div>
            """, unsafe_allow_html=True)
            
        except Exception as e:
            st.error(f"Could not load diagnostic details: {e}")
    else:
        st.info("No index statistics available. Please ingest and index your PDF standard first.")

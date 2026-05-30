// ISO 9001:2015 QMS AI Auditor Frontend Controller

// Dynamic API URL: resolves localhost locally and automatically falls back to your public cloud API URL when online
const API_BASE = window.location.hostname === "localhost" || window.location.hostname === "127.0.0.1"
    ? "http://127.0.0.1:8000"
    : "https://iso-9001-qms-rag.onrender.com";

// App Global State
let appState = {
    provider: "gemini",       // "gemini", "cohere", or "ollama"
    apiKeyConfigured: false,
    cohereKeyConfigured: false,
    ollamaActive: false,
    indexReady: false,
    indexedChunks: 0,
    diagnosticChunks: [],      // cached chunks for diagnostics tab
    selectedFile: null         // tracks currently selected dynamic PDF
};

// DOM ELEMENT CACHE
const elements = {
    // RAG Provider Selection
    providerSelect: document.getElementById("provider-select"),
    geminiConfigGroup: document.getElementById("gemini-config-group"),
    cohereConfigGroup: document.getElementById("cohere-config-group"),
    ollamaConfigGroup: document.getElementById("ollama-config-group"),
    ollamaStatusDot: document.getElementById("ollama-status-dot"),
    ollamaStatusText: document.getElementById("ollama-status-text"),
    ollamaPullInstructions: document.getElementById("ollama-pull-instructions"),
    
    // Sidebar config
    apiKeyInput: document.getElementById("api-key-input"),
    saveKeyBtn: document.getElementById("save-key-btn"),
    cohereApiKeyInput: document.getElementById("cohere-api-key-input"),
    saveCohereKeyBtn: document.getElementById("save-cohere-key-btn"),
    indexStatusDot: document.getElementById("index-status-dot"),
    indexStatusText: document.getElementById("index-status-text"),
    chunksStatText: document.getElementById("chunks-stat-text"),
    ingestBtn: document.getElementById("ingest-btn"),
    pdfFileInput: document.getElementById("pdf-file-input"),
    fileDropZone: document.getElementById("file-drop-zone"),
    fileInfoContainer: document.getElementById("file-info-container"),
    selectedFileName: document.getElementById("selected-file-name"),
    clearFileBtn: document.getElementById("clear-file-btn"),
    
    // Tuning Controls (Model Selection Groups)
    geminiModelsGroup: document.getElementById("gemini-models-group"),
    cohereModelsGroup: document.getElementById("cohere-models-group"),
    ollamaModelsGroup: document.getElementById("ollama-models-group"),
    modelSelect: document.getElementById("model-select"),
    cohereModelSelect: document.getElementById("cohere-model-select"),
    ollamaModelSelect: document.getElementById("ollama-model-select"),
    topKSlider: document.getElementById("top-k-slider"),
    topKVal: document.getElementById("top-k-val"),
    tempSlider: document.getElementById("temp-slider"),
    tempVal: document.getElementById("temp-val"),
    
    // Tab Navigation
    tabLinks: document.querySelectorAll(".tab-link"),
    tabPanels: document.querySelectorAll(".tab-panel"),
    
    // TAB 1: Chat Q&A
    chatMessages: document.getElementById("chat-messages"),
    chatForm: document.getElementById("chat-form"),
    chatInput: document.getElementById("chat-input"),
    sendBtn: document.getElementById("send-btn"),
    quickQueryBtns: document.querySelectorAll(".quick-query-btn"),
    
    // TAB 2: Clause Navigator
    clausesContainer: document.getElementById("clauses-accordion-container"),
    
    // TAB 3: Diagnostics
    diagnosticsReadyContent: document.getElementById("diagnostics-ready-content"),
    diagnosticsEmptyContent: document.getElementById("diagnostics-empty-content"),
    metricPages: document.getElementById("metric-pages"),
    metricChunks: document.getElementById("metric-chunks"),
    metricAvgLen: document.getElementById("metric-avg-len"),
    metricTotalChars: document.getElementById("metric-total-chars"),
    chunkInspectorSlider: document.getElementById("chunk-inspector-slider"),
    chunkIndexVal: document.getElementById("chunk-index-val"),
    inspectChunkId: document.getElementById("inspect-chunk-id"),
    inspectChunkPage: document.getElementById("inspect-chunk-page"),
    inspectChunkClauses: document.getElementById("inspect-chunk-clauses"),
    inspectChunkText: document.getElementById("inspect-chunk-text"),
    
    // TAB 4: Trace Timeline
    traceContainer: document.getElementById("trace-container"),
    traceEmptyContent: document.getElementById("trace-empty-content"),
    clearTraceBtn: document.getElementById("clear-trace-btn"),
    
    // TAB 5: Ragas Evaluator
    runEvalBtn: document.getElementById("run-eval-btn"),
    evalTableBody: document.getElementById("eval-table-body"),
    evalEmptyContent: document.getElementById("eval-empty-content"),
    evalResultsContent: document.getElementById("eval-results-content"),
    evalFaithfulnessVal: document.getElementById("eval-metric-faithfulness"),
    evalRelevanceVal: document.getElementById("eval-metric-relevance"),
    evalRecallVal: document.getElementById("eval-metric-recall"),
    evalPrecisionVal: document.getElementById("eval-metric-precision"),
    
    // Global Overlay
    loadingOverlay: document.getElementById("loading-overlay"),
    overlayTitle: document.getElementById("overlay-title"),
    overlayText: document.getElementById("overlay-text")
};

// INITIALIZATION
document.addEventListener("DOMContentLoaded", async () => {
    setupSliders();
    setupTabNavigation();
    setupEventListeners();
    setupQuickQueries();
    
    // Set initial provider view
    handleProviderSwitch();
    
    // Initial fetch of system status
    await checkSystemStatus();
    
    // Load navigator clause descriptions
    await loadClauseDatabase();
});

// SLIDER CONFIGURATIONS
function setupSliders() {
    elements.topKSlider.addEventListener("input", (e) => {
        elements.topKVal.textContent = e.target.value;
    });
    
    elements.tempSlider.addEventListener("input", (e) => {
        elements.tempVal.textContent = parseFloat(e.target.value).toFixed(2);
    });
    
    elements.chunkInspectorSlider.addEventListener("input", (e) => {
        const index = parseInt(e.target.value);
        elements.chunkIndexVal.textContent = index;
        renderSelectedChunk(index);
    });
}

// TAB NAVIGATION MANAGEMENT
function setupTabNavigation() {
    elements.tabLinks.forEach(link => {
        link.addEventListener("click", () => {
            const targetTab = link.getAttribute("data-tab");
            
            elements.tabLinks.forEach(l => l.classList.remove("active"));
            link.classList.add("active");
            
            elements.tabPanels.forEach(panel => {
                if (panel.id === targetTab) {
                    panel.classList.add("active");
                } else {
                    panel.classList.remove("active");
                }
            });
            
            // Reload diagnostics for the current active provider
            if (targetTab === "tab-diagnostics") {
                loadDiagnostics();
            } else if (targetTab === "tab-trace") {
                loadTraceLogs();
            } else if (targetTab === "tab-evaluate") {
                loadEvaluationReport();
            }
        });
    });
}

// PROVIDER SWITCH COORDINATION
function handleProviderSwitch() {
    const val = elements.providerSelect.value;
    appState.provider = val;
    
    // Hide all provider-specific blocks by default
    elements.geminiConfigGroup.classList.add("hidden");
    elements.geminiModelsGroup.classList.add("hidden");
    elements.cohereConfigGroup.classList.add("hidden");
    elements.cohereModelsGroup.classList.add("hidden");
    elements.ollamaConfigGroup.classList.add("hidden");
    elements.ollamaModelsGroup.classList.add("hidden");
    elements.ollamaPullInstructions.classList.add("hidden");
    
    if (val === "gemini") {
        elements.geminiConfigGroup.classList.remove("hidden");
        elements.geminiModelsGroup.classList.remove("hidden");
    } else if (val === "cohere") {
        elements.cohereConfigGroup.classList.remove("hidden");
        elements.cohereModelsGroup.classList.remove("hidden");
    } else {
        elements.ollamaConfigGroup.classList.remove("hidden");
        elements.ollamaModelsGroup.classList.remove("hidden");
        // Fetch local Ollama models list
        fetchOllamaModels();
    }
    
    // Clear chat messages to avoid mixing context between providers
    clearChat();
}

function clearChat() {
    elements.chatMessages.innerHTML = `
        <div class="chat-message assistant">
            <div class="message-header">
                <span class="avatar-icon">🛡️</span>
                <strong>Lead QMS Auditor</strong>
            </div>
            <div class="message-text">
                System mode switched to **${appState.provider.toUpperCase()}**.
                <br><br>
                Please verify the configuration state in the sidebar. Once the index status shows glowing green, I will be ready to perform Q&A with page-level citations!
            </div>
        </div>
    `;
}

// OLLAMA MODEL TAGS FETCHER
async function fetchOllamaModels() {
    try {
        const response = await fetch(`${API_BASE}/api/ollama/models`);
        if (!response.ok) throw new Error("Failed to contact Ollama models tag endpoint.");
        
        const result = await response.json();
        
        elements.ollamaModelSelect.innerHTML = "";
        
        if (result.status === "offline") {
            appState.ollamaActive = false;
            elements.ollamaStatusDot.className = "dot dot-red";
            elements.ollamaStatusText.textContent = "Ollama: Offline";
            elements.ollamaPullInstructions.classList.remove("hidden");
            elements.ollamaModelsGroup.classList.add("hidden");
            return;
        }
        
        appState.ollamaActive = true;
        elements.ollamaStatusDot.className = "dot dot-green";
        elements.ollamaStatusText.textContent = "Ollama: Connected";
        
        const models = result.models || [];
        
        // Filter out embedding models from LLM list (Ollama includes them in /api/tags)
        const chatModels = models.filter(m => !m.includes("embed") && !m.includes("colbert"));
        
        if (chatModels.length === 0) {
            elements.ollamaPullInstructions.classList.remove("hidden");
            elements.ollamaModelsGroup.classList.add("hidden");
            
            // Add a temporary fallback
            const opt = document.createElement("option");
            opt.value = "";
            opt.textContent = "No local models found";
            elements.ollamaModelSelect.appendChild(opt);
        } else {
            elements.ollamaPullInstructions.classList.add("hidden");
            elements.ollamaModelsGroup.classList.remove("hidden");
            
            chatModels.forEach(model => {
                const opt = document.createElement("option");
                opt.value = model;
                opt.textContent = model;
                elements.ollamaModelSelect.appendChild(opt);
            });
        }
    } catch (err) {
        console.error("Failed to query Ollama models:", err);
        elements.ollamaStatusDot.className = "dot dot-red";
        elements.ollamaStatusText.textContent = "Ollama: Service Error";
    }
}

// DYNAMIC FILE UPLOAD STATE SYNCHRONIZATION
function updateUploaderUI() {
    if (appState.selectedFile) {
        elements.fileInfoContainer.classList.remove("hidden");
        elements.selectedFileName.textContent = appState.selectedFile.name;
    } else {
        elements.fileInfoContainer.classList.add("hidden");
        elements.selectedFileName.textContent = "No file selected.";
        elements.pdfFileInput.value = "";
    }
    checkSystemStatus();
}

// GLOBAL EVENT HANDLERS
function setupEventListeners() {
    // Clear Trace button action
    elements.clearTraceBtn.addEventListener("click", clearTraceLog);

    // Run Compliance Sweep action
    elements.runEvalBtn.addEventListener("click", runEvaluationSweep);

    // 0. Provider Selector change
    elements.providerSelect.addEventListener("change", async () => {
        handleProviderSwitch();
        await checkSystemStatus();
        
        // Load tab statistics if active
        const activeTab = document.querySelector(".tab-link.active").getAttribute("data-tab");
        if (activeTab === "tab-diagnostics") {
            loadDiagnostics();
        }
    });

    // 1. Save Key Action (Gemini only)
    elements.saveKeyBtn.addEventListener("click", async () => {
        const key = elements.apiKeyInput.value.trim();
        if (!key) {
            alert("Please enter a valid API Key.");
            return;
        }
        
        showOverlay("Configuring System...", "Saving API Key and initializing Google Generative AI configurations.");
        try {
            const response = await fetch(`${API_BASE}/api/configure-key`, {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify({ api_key: key })
            });
            
            if (!response.ok) throw new Error("Failed to configure key.");
            
            alert("API Key configured successfully.");
            elements.apiKeyInput.value = "";
            await checkSystemStatus();
        } catch (err) {
            alert(`Error configuring API Key: ${err.message}`);
        } finally {
            hideOverlay();
        }
    });
    
    // 1b. Save Key Action (Cohere only)
    elements.saveCohereKeyBtn.addEventListener("click", async () => {
        const key = elements.cohereApiKeyInput.value.trim();
        if (!key) {
            alert("Please enter a valid Cohere API Key.");
            return;
        }
        
        showOverlay("Configuring System...", "Saving API Key and initializing Cohere configurations.");
        try {
            const response = await fetch(`${API_BASE}/api/configure-cohere-key`, {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify({ api_key: key })
            });
            
            if (!response.ok) throw new Error("Failed to configure key.");
            
            alert("Cohere API Key configured successfully.");
            elements.cohereApiKeyInput.value = "";
            await checkSystemStatus();
        } catch (err) {
            alert(`Error configuring API Key: ${err.message}`);
        } finally {
            hideOverlay();
        }
    });
    
    // 2. PDF Ingestion Action
    elements.ingestBtn.addEventListener("click", async () => {
        if (!appState.selectedFile) {
            alert("Please select a QMS or ISO 9001 PDF document first.");
            return;
        }
        
        const provUpper = appState.provider.toUpperCase();
        showOverlay(
            `Processing ${provUpper} Ingestion...`, 
            `The API is parsing your uploaded PDF standard, running sliding-window chunking, generating local or cloud embeddings using ${provUpper}, and serializing the vector database. This takes 15-30 seconds.`
        );
        
        try {
            const formData = new FormData();
            formData.append("file", appState.selectedFile);
            
            const response = await fetch(`${API_BASE}/api/ingest?force=true&provider=${appState.provider}`, {
                method: "POST",
                body: formData
            });
            
            if (!response.ok) {
                const data = await response.json();
                throw new Error(data.detail || "Ingestion failed.");
            }
            
            const result = await response.json();
            alert(`Ingestion Completed! ${result.chunks_count} vector chunks generated and indexed in ${provUpper} mode.`);
            await checkSystemStatus();
            
            // Reload diagnostics if tab active
            const activeTab = document.querySelector(".tab-link.active").getAttribute("data-tab");
            if (activeTab === "tab-diagnostics") {
                loadDiagnostics();
            }
        } catch (err) {
            alert(`Ingestion Error: ${err.message}`);
        } finally {
            hideOverlay();
        }
    });

    // 2a. Custom File Selector & Drag-Drop Zone Handlers
    elements.pdfFileInput.addEventListener("change", (e) => {
        const file = e.target.files[0];
        if (file) {
            appState.selectedFile = file;
            updateUploaderUI();
        }
    });
    
    elements.clearFileBtn.addEventListener("click", (e) => {
        e.preventDefault();
        e.stopPropagation();
        appState.selectedFile = null;
        updateUploaderUI();
    });
    
    // Setup drag-and-drop mechanics
    const dropZone = elements.fileDropZone;
    
    ['dragenter', 'dragover'].forEach(eventName => {
        dropZone.addEventListener(eventName, (e) => {
            e.preventDefault();
            e.stopPropagation();
            dropZone.classList.add('dragover');
        }, false);
    });
    
    ['dragleave', 'drop'].forEach(eventName => {
        dropZone.addEventListener(eventName, (e) => {
            e.preventDefault();
            e.stopPropagation();
            dropZone.classList.remove('dragover');
        }, false);
    });
    
    dropZone.addEventListener('drop', (e) => {
        const dt = e.dataTransfer;
        const file = dt.files[0];
        
        if (file && file.type === "application/pdf") {
            appState.selectedFile = file;
            updateUploaderUI();
        } else if (file) {
            alert("Only PDF files are supported!");
        }
    });
    
    // 3. Chat Q&A Submission Action
    elements.chatForm.addEventListener("submit", async (e) => {
        e.preventDefault();
        const query = elements.chatInput.value.trim();
        if (!query) return;
        
            elements.chatInput.value = "";
        await processUserQuery(query);
    });
}

// SYSTEM STATUS CHECK
async function checkSystemStatus() {
    try {
        const response = await fetch(`${API_BASE}/api/status?provider=${appState.provider}`);
        if (!response.ok) throw new Error("System status unreachable.");
        
        const status = await response.json();
        
        if (appState.provider === "gemini") {
            appState.apiKeyConfigured = status.api_key_configured;
        } else if (appState.provider === "cohere") {
            appState.cohereKeyConfigured = status.api_key_configured;
        }
        
        appState.ollamaActive = status.ollama_active;
        appState.indexReady = status.index_ready;
        appState.indexedChunks = status.indexed_chunks;
        
        // Update index indicators in UI
        if (status.index_ready) {
            elements.indexStatusDot.className = "dot dot-green";
            elements.indexStatusText.textContent = `Index: Ready (${appState.provider.toUpperCase()})`;
            elements.chunksStatText.textContent = `${status.indexed_chunks} QMS chunks indexed.`;
            elements.ingestBtn.textContent = "Ingest Selected Document";
        } else {
            elements.indexStatusDot.className = "dot dot-red";
            elements.indexStatusText.textContent = `Index: Not Found (${appState.provider.toUpperCase()})`;
            elements.chunksStatText.textContent = "Ingestion required.";
            elements.ingestBtn.textContent = "Ingest Selected Document";
        }
        
        // Manage active states based on Provider selection
        if (appState.provider === "gemini") {
            if (status.api_key_configured) {
                elements.ingestBtn.disabled = !appState.selectedFile;
                elements.apiKeyInput.placeholder = "•••••••••••••••••••••••• (API Key Active)";
                
                if (status.index_ready) {
                    elements.chatInput.disabled = false;
                    elements.sendBtn.disabled = false;
                    elements.chatInput.placeholder = "Ask about ISO 9001:2015 requirements, audits, or implementation...";
                } else {
                    elements.chatInput.disabled = true;
                    elements.sendBtn.disabled = true;
                    elements.chatInput.placeholder = "⚠️ Complete PDF Ingestion in the sidebar first...";
                }
            } else {
                elements.ingestBtn.disabled = true;
                elements.chatInput.disabled = true;
                elements.sendBtn.disabled = true;
                elements.chatInput.placeholder = "⚠️ Save a valid Gemini API Key in the sidebar first...";
            }
        } else if (appState.provider === "cohere") {
            if (status.api_key_configured) {
                elements.ingestBtn.disabled = !appState.selectedFile;
                elements.cohereApiKeyInput.placeholder = "•••••••••••••••••••••••• (API Key Active)";
                
                if (status.index_ready) {
                    elements.chatInput.disabled = false;
                    elements.sendBtn.disabled = false;
                    elements.chatInput.placeholder = "Cohere Cloud Auditor is active. Ask any question...";
                } else {
                    elements.chatInput.disabled = true;
                    elements.sendBtn.disabled = true;
                    elements.chatInput.placeholder = "⚠️ Complete PDF Ingestion in the sidebar first...";
                }
            } else {
                elements.ingestBtn.disabled = true;
                elements.chatInput.disabled = true;
                elements.sendBtn.disabled = true;
                elements.chatInput.placeholder = "⚠️ Save a valid Cohere API Key in the sidebar first...";
            }
        } else {
            // Ollama Mode
            elements.apiKeyInput.placeholder = "Ollama mode active";
            
            // Check if server is active
            if (status.ollama_active) {
                elements.ollamaStatusDot.className = "dot dot-green";
                elements.ollamaStatusText.textContent = "Ollama: Connected";
                elements.ingestBtn.disabled = !appState.selectedFile;
                
                // Verify if local models are loaded
                const hasLocalModels = elements.ollamaModelSelect.value !== "";
                
                if (status.index_ready && hasLocalModels) {
                    elements.chatInput.disabled = false;
                    elements.sendBtn.disabled = false;
                    elements.chatInput.placeholder = "Ollama Local Auditor is active. Ask any question...";
                } else if (!status.index_ready) {
                    elements.chatInput.disabled = true;
                    elements.sendBtn.disabled = true;
                    elements.chatInput.placeholder = "⚠️ Complete PDF Ingestion in the sidebar first...";
                } else {
                    elements.chatInput.disabled = true;
                    elements.sendBtn.disabled = true;
                    elements.chatInput.placeholder = "⚠️ Pull a local model first (see sidebar commands)...";
                }
            } else {
                elements.ollamaStatusDot.className = "dot dot-red";
                elements.ollamaStatusText.textContent = "Ollama: Offline";
                elements.ingestBtn.disabled = true;
                elements.chatInput.disabled = true;
                elements.sendBtn.disabled = true;
                elements.chatInput.placeholder = "⚠️ Start local Ollama daemon first...";
            }
        }
    } catch (err) {
        console.error("Status check failed:", err);
        elements.indexStatusText.textContent = "Server Status: Offline";
    }
}

// QUICK AUDIT QUERIES HANDLER
function setupQuickQueries() {
    elements.quickQueryBtns.forEach(btn => {
        btn.addEventListener("click", () => {
            const query = btn.getAttribute("data-query");
            if (elements.chatInput.disabled) {
                alert("Please complete the required sidebar settings for this provider first.");
                return;
            }
            processUserQuery(query);
        });
    });
}

// Q&A QUERY PROCESSOR
async function processUserQuery(queryText) {
    appendMessage("user", "User", queryText);
    const typingId = appendTypingIndicator();
    
    try {
        const top_k = parseInt(elements.topKSlider.value);
        const temperature = parseFloat(elements.tempSlider.value);
        
        const payload = {
            query: queryText,
            top_k: top_k,
            temperature: temperature,
            provider: appState.provider,
            ollama_model: elements.ollamaModelSelect.value || "llama3",
            gemini_model: elements.modelSelect.value || "gemini-2.5-flash",
            cohere_model: elements.cohereModelSelect.value || "command-r-08-2024"
        };
        
        // Call Backend API
        const response = await fetch(`${API_BASE}/api/query`, {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify(payload)
        });
        
        removeTypingIndicator(typingId);
        
        if (!response.ok) {
            const data = await response.json();
            throw new Error(data.detail || "Query failed.");
        }
        
        const result = await response.json();
        appendMessage("assistant", "Lead QMS Auditor", result.answer, result.sources);
        
    } catch (err) {
        removeTypingIndicator(typingId);
        appendMessage("assistant", "Lead QMS Auditor", `⚠️ Error executing audit search: ${err.message}`);
    }
}

// MESSAGE RENDERING CONTROLS
function appendMessage(role, senderName, text, sources = null) {
    const msgDiv = document.createElement("div");
    msgDiv.className = `chat-message ${role}`;
    
    const formattedText = formatMarkdown(text);
    
    let sourcesHTML = "";
    if (sources && sources.length > 0) {
        const uniqueId = `exp_${Date.now()}`;
        sourcesHTML = `
            <div class="sources-expander">
                <button class="sources-header-btn" onclick="toggleSources('${uniqueId}')">
                    <span>🔍 Verified Sources Used (Click to toggle)</span>
                    <span id="icon_${uniqueId}">▼</span>
                </button>
                <div id="${uniqueId}" class="sources-content hidden">
                    ${sources.map(src => `
                        <div class="source-item">
                            <div class="source-meta-row">
                                <span class="source-badge">Page ${src.page}</span>
                                <span class="score-badge">Sim: ${src.score.toFixed(3)}</span>
                                <span class="source-clauses"><strong>Clauses:</strong> ${src.clauses.join(", ")}</span>
                            </div>
                            <p class="source-text">"${src.text}"</p>
                        </div>
                    `).join('')}
                </div>
            </div>
        `;
    }
    
    msgDiv.innerHTML = `
        <div class="message-header">
            <span class="avatar-icon">${role === 'user' ? '👤' : '🛡️'}</span>
            <strong>${senderName}</strong>
        </div>
        <div class="message-text">${formattedText}</div>
        ${sourcesHTML}
    `;
    
    elements.chatMessages.appendChild(msgDiv);
    scrollToBottom();
}

function appendTypingIndicator() {
    const id = `typing_${Date.now()}`;
    const msgDiv = document.createElement("div");
    msgDiv.className = "chat-message assistant typing-indicator-msg";
    msgDiv.id = id;
    msgDiv.innerHTML = `
        <div class="message-header">
            <span class="avatar-icon">🛡️</span>
            <strong>Lead QMS Auditor</strong>
        </div>
        <div class="loading-spinner-container" style="padding: 10px; flex-direction: row;">
            <div class="spinner"></div>
            <span>Auditing QMS standards database...</span>
        </div>
    `;
    elements.chatMessages.appendChild(msgDiv);
    scrollToBottom();
    return id;
}

function removeTypingIndicator(id) {
    const indicator = document.getElementById(id);
    if (indicator) indicator.remove();
}

function scrollToBottom() {
    elements.chatMessages.scrollTop = elements.chatMessages.scrollHeight;
}

function formatMarkdown(text) {
    let html = text
        .replace(/&/g, "&amp;")
        .replace(/</g, "&lt;")
        .replace(/>/g, "&gt;");
        
    html = html.replace(/\*\*(.*?)\*\*/g, "<strong>$1</strong>");
    
    html = html.replace(/\[ \]/g, "☐");
    html = html.replace(/\[x\]/g, "☑");
    
    html = html.replace(/^\s*-\s+(.*)$/gm, "<li>$1</li>");
    html = html.replace(/(<li>.*<\/li>)/s, "<ul>$1</ul>");
    
    html = html.replace(/\n/g, "<br>");
    
    return html;
}

window.toggleSources = function(id) {
    const panel = document.getElementById(id);
    const icon = document.getElementById(`icon_${id}`);
    
    if (panel.classList.contains("hidden")) {
        panel.classList.remove("hidden");
        icon.textContent = "▲";
    } else {
        panel.classList.add("hidden");
        icon.textContent = "▼";
    }
};

// TAB 2: CLAUSE NAVIGATOR
async function loadClauseDatabase() {
    try {
        const response = await fetch(`${API_BASE}/api/clauses`);
        if (!response.ok) throw new Error("Failed to load clauses.");
        
        const database = await response.json();
        
        elements.clausesContainer.innerHTML = "";
        
        for (const [title, details] of Object.entries(database)) {
            const card = document.createElement("div");
            card.className = "qms-container";
            card.innerHTML = `
                <h4 class="clause-card-title">📘 ${title}</h4>
                <p style="color: var(--text-primary); font-size: 0.95rem; margin-bottom: 12px;">
                    <strong>Core Focus:</strong> ${details.focus}
                </p>
                <ul class="clause-subclauses-list">
                    ${details.sub_clauses.map(sub => `<li>${sub}</li>`).join('')}
                </ul>
                <div class="clause-action-row">
                    <button class="btn secondary-btn" onclick="queryNavigator('${details.recommended_query}')">
                        Ask Auditor about ${title.split(':')[0]}
                    </button>
                </div>
            `;
            elements.clausesContainer.appendChild(card);
        }
    } catch (err) {
        elements.clausesContainer.innerHTML = `
            <div class="loading-spinner-container">
                <p style="color: var(--color-red);">❌ Failed to load clause navigator: ${err.message}</p>
            </div>
        `;
    }
}

window.queryNavigator = function(query) {
    if (elements.chatInput.disabled) {
        alert("Please complete the required sidebar settings for this provider first.");
        return;
    }
    
    const chatTabLink = document.querySelector('[data-tab="tab-chat"]');
    chatTabLink.click();
    
    processUserQuery(query);
};

// TAB 3: DIAGNOSTICS LOAD
async function loadDiagnostics() {
    try {
        const response = await fetch(`${API_BASE}/api/stats?provider=${appState.provider}`);
        if (!response.ok) throw new Error("Failed to load diagnostics.");
        
        const stats = await response.json();
        
        if (stats.chunks_count === 0) {
            elements.diagnosticsReadyContent.classList.add("hidden");
            elements.diagnosticsEmptyContent.classList.remove("hidden");
            return;
        }
        
        elements.diagnosticsEmptyContent.classList.add("hidden");
        elements.diagnosticsReadyContent.classList.remove("hidden");
        
        appState.diagnosticChunks = stats.chunks;
        
        elements.metricPages.textContent = stats.total_pages;
        elements.metricChunks.textContent = stats.chunks_count;
        elements.metricAvgLen.textContent = stats.avg_chunk_length;
        elements.metricTotalChars.textContent = stats.total_characters.toLocaleString();
        
        elements.chunkInspectorSlider.max = stats.chunks_count - 1;
        elements.chunkInspectorSlider.value = 0;
        elements.chunkIndexVal.textContent = 0;
        
        renderSelectedChunk(0);
        
    } catch (err) {
        console.error("Diagnostics load failed:", err);
        elements.diagnosticsReadyContent.classList.add("hidden");
        elements.diagnosticsEmptyContent.classList.remove("hidden");
    }
}

function renderSelectedChunk(idx) {
    const chunk = appState.diagnosticChunks[idx];
    if (!chunk) return;
    
    elements.inspectChunkId.textContent = chunk.id;
    elements.inspectChunkPage.textContent = chunk.page;
    elements.inspectChunkClauses.textContent = chunk.clauses.join(", ");
    elements.inspectChunkText.textContent = chunk.text;
}

// OVERLAY INDICATORS
function showOverlay(title, text) {
    elements.overlayTitle.textContent = title;
    elements.overlayText.textContent = text;
    elements.loadingOverlay.classList.remove("hidden");
}

function hideOverlay() {
    elements.loadingOverlay.classList.add("hidden");
}

// DECISION TRACE LOG LOAD & RENDER
async function loadTraceLogs() {
    try {
        const response = await fetch(`${API_BASE}/api/trace`);
        if (!response.ok) throw new Error("Failed to load trace log");
        const traces = await response.json();
        
        if (!traces || traces.length === 0) {
            elements.traceContainer.classList.add("hidden");
            elements.traceEmptyContent.classList.remove("hidden");
            return;
        }
        
        elements.traceEmptyContent.classList.add("hidden");
        elements.traceContainer.classList.remove("hidden");
        
        renderTraceTimeline(traces);
    } catch (err) {
        console.error("Trace load failed:", err);
        elements.traceContainer.classList.add("hidden");
        elements.traceEmptyContent.classList.remove("hidden");
    }
}

function renderTraceTimeline(traces) {
    elements.traceContainer.innerHTML = "";
    
    const sortedTraces = [...traces].reverse();
    
    sortedTraces.forEach((trace) => {
        const traceDate = new Date(trace.timestamp).toLocaleString();
        
        let attemptsHTML = "";
        
        trace.query_attempts.forEach((attempt) => {
            const isRetry = attempt.attempt > 0;
            const statusClass = isRetry ? "triggered" : "passed";
            const statusLabel = isRetry ? `Rewrite Attempt #${attempt.attempt}` : "Original Retrieval";
            
            let candidatesHTML = "";
            if (attempt.candidates && attempt.candidates.length > 0) {
                candidatesHTML = `<div class="trace-scores-list">`;
                attempt.candidates.forEach((cand, idx) => {
                    let boostTagsHTML = "";
                    if (cand.boosts && Object.keys(cand.boosts).length > 0) {
                        boostTagsHTML = `<div class="trace-boosts-container">`;
                        for (const [key, value] of Object.entries(cand.boosts)) {
                            const formattedKey = key.replace(/_/g, " ");
                            boostTagsHTML += `<span class="trace-boost-tag">🔥 ${formattedKey}: +${value.toFixed(2)}</span>`;
                        }
                        boostTagsHTML += `</div>`;
                    }
                    
                    const origWidth = Math.max(cand.original_score * 100, 5);
                    const rerkWidth = Math.max(cand.reranked_score * 100, 5);
                    
                    candidatesHTML += `
                        <div class="trace-score-row">
                            <div class="trace-score-meta" style="margin-bottom: 8px;">
                                <span class="trace-chunk-info" style="font-weight:600; color:#818cf8;">Rank #${idx + 1} | Chunk #${cand.chunk_id} | Page ${cand.page}</span>
                                <span style="font-size: 0.76rem; color: #94a3b8; font-weight: 500;">Clauses: ${cand.clauses.join(", ")}</span>
                            </div>
                            <div style="font-size: 0.8rem; color: #cbd5e1; font-style: italic; margin-bottom: 10px; background: rgba(0,0,0,0.2); padding: 8px; border-radius: 4px; border-left: 2px solid #6366f1;">
                                "${cand.snippet}"
                            </div>
                            <div class="trace-scores-comparison">
                                <div class="trace-bar-wrapper">
                                    <span class="trace-bar-label">Original:</span>
                                    <div class="trace-bar-container">
                                        <div class="trace-bar-fill original" style="width: ${origWidth}%"></div>
                                    </div>
                                    <span class="trace-bar-value original">${cand.original_score.toFixed(4)}</span>
                                </div>
                                <div class="trace-bar-wrapper">
                                    <span class="trace-bar-label">Reranked:</span>
                                    <div class="trace-bar-container">
                                        <div class="trace-bar-fill reranked" style="width: ${rerkWidth}%"></div>
                                    </div>
                                    <span class="trace-bar-value reranked">${cand.reranked_score.toFixed(4)}</span>
                                </div>
                            </div>
                            ${boostTagsHTML}
                        </div>
                    `;
                });
                candidatesHTML += `</div>`;
            }
            
            let stepDesc = "";
            if (isRetry) {
                stepDesc = `Top similarity score was below 0.5 threshold. Programmatic search-rewriting was automatically triggered to optimize term coverage.`;
            } else {
                stepDesc = `Executed vector similarity scan on QMS standards index. Top raw semantic similarity score returned: <strong>${attempt.top_score.toFixed(4)}</strong>.`;
            }
            
            attemptsHTML += `
                <div class="trace-flow-step" style="margin-bottom: 24px;">
                    <div class="trace-step-title-row">
                        <span class="trace-step-icon">${isRetry ? "🔄" : "🔍"}</span>
                        <span class="trace-step-name">${statusLabel}</span>
                        <span class="trace-step-badge ${statusClass}">${attempt.top_score >= 0.5 ? "Similarity Passed" : "Low Similarity"}</span>
                    </div>
                    <div class="trace-step-detail">
                        <p>${stepDesc}</p>
                        <p style="margin-top: 6px; color: #818cf8; font-weight: 500;">Query used: "${attempt.query}"</p>
                        ${candidatesHTML}
                    </div>
                </div>
            `;
        });
        
        let rerankStepHTML = "";
        const finalAttempt = trace.query_attempts[trace.query_attempts.length - 1];
        if (finalAttempt && finalAttempt.candidates) {
            rerankStepHTML = `
                <div class="trace-flow-step" style="margin-bottom: 24px;">
                    <div class="trace-step-title-row">
                        <span class="trace-step-icon">⚖️</span>
                        <span class="trace-step-name">QMS Auditing Rerank Decision</span>
                        <span class="trace-step-badge info">Active</span>
                    </div>
                    <div class="trace-step-detail">
                        Candidate standard passages were prioritized and boosted in real-time by matching strict compliance obligation keywords (e.g. <code>shall</code>, <code>must</code>, <code>documented information</code>). Visualized comparison chart is rendered above for each candidate chunk.
                    </div>
                </div>
            `;
        }
        
        const traceHTML = `
            <div class="trace-timeline-item" style="margin-bottom: 20px;">
                <div class="trace-meta-header">
                    <span class="trace-time">🕒 Audit Timestamp: ${traceDate}</span>
                    <span class="trace-provider-tag">${trace.provider.toUpperCase()} RAG Path</span>
                </div>
                <div class="trace-query-box">
                    <h4 style="margin-top:0;">Original Question Submitted</h4>
                    <div class="trace-query-text">"${trace.query}"</div>
                </div>
                <div class="trace-flow">
                    ${attemptsHTML}
                    ${rerankStepHTML}
                    <div class="trace-flow-step">
                        <div class="trace-step-title-row">
                            <span class="trace-step-icon">🛡️</span>
                            <span class="trace-step-name">Generative Auditor Response</span>
                            <span class="trace-step-badge passed">Complete</span>
                        </div>
                        <div class="trace-step-detail">
                            Formulated QMS Auditor answer utilizing selected sources. Output summary: <em style="color:#e2e8f0;">"${trace.answer_summary}"</em>
                        </div>
                    </div>
                </div>
            </div>
        `;
        
        elements.traceContainer.innerHTML += traceHTML;
    });
}

async function clearTraceLog() {
    if (!confirm("Are you sure you want to clear your decision trace log history?")) return;
    
    try {
        const response = await fetch(`${API_BASE}/api/trace/clear`, { method: "POST" });
        if (!response.ok) throw new Error("Failed to clear trace");
        
        alert("Decision trace log history cleared successfully.");
        elements.traceContainer.classList.add("hidden");
        elements.traceEmptyContent.classList.remove("hidden");
    } catch (err) {
        console.error("Clear trace failed:", err);
        alert(`Error clearing trace: ${err.message}`);
    }
}

// TAB 5: RAGAS EVALUATOR FUNCTIONALITY
async function loadEvaluationReport() {
    try {
        const response = await fetch(`${API_BASE}/api/evaluate`);
        if (!response.ok) throw new Error("Failed to contact evaluation API");
        const data = await response.json();
        
        if (data.status === "empty" || !data.records || data.records.length === 0) {
            elements.evalResultsContent.classList.add("hidden");
            elements.evalEmptyContent.classList.remove("hidden");
            return;
        }
        
        elements.evalEmptyContent.classList.add("hidden");
        elements.evalResultsContent.classList.remove("hidden");
        
        // Render metrics cards
        elements.evalFaithfulnessVal.textContent = data.average_faithfulness.toFixed(4);
        elements.evalRelevanceVal.textContent = data.average_answer_relevance.toFixed(4);
        elements.evalRecallVal.textContent = data.average_context_recall.toFixed(4);
        elements.evalPrecisionVal.textContent = data.average_context_precision.toFixed(4);
        
        // Render detailed per-question rows
        renderEvaluationTable(data.records);
    } catch (err) {
        console.error("Failed to load evaluation report:", err);
        elements.evalResultsContent.classList.add("hidden");
        elements.evalEmptyContent.classList.remove("hidden");
    }
}

function renderEvaluationTable(records) {
    elements.evalTableBody.innerHTML = "";
    
    records.forEach(r => {
        // Color badge grading helper for premium aesthetics
        const getGradeBadge = (score) => {
            if (score >= 0.90) return `<span class="score-grade score-grade-excellent">${score.toFixed(4)}</span>`;
            if (score >= 0.70) return `<span class="score-grade score-grade-good">${score.toFixed(4)}</span>`;
            return `<span class="score-grade score-grade-warning">${score.toFixed(4)}</span>`;
        };
        
        const tr = document.createElement("tr");
        tr.style.borderBottom = "1px solid rgba(255,255,255,0.05)";
        tr.style.transition = "background-color 0.2s ease";
        tr.className = "eval-table-row";
        
        tr.innerHTML = `
            <td style="padding: 12px 8px; font-weight: 600; color: #818cf8; font-size: 0.85rem;">Clause ${r.id}</td>
            <td style="padding: 12px 8px; font-size: 0.88rem; color: #e2e8f0; line-height: 1.4;">
                <div style="font-weight: 600; color: #cbd5e1; margin-bottom: 2px;">${r.clause}</div>
                <div style="color: #94a3b8; font-style: italic;">"${r.question}"</div>
            </td>
            <td style="padding: 12px 8px; text-align: center;">${getGradeBadge(r.faithfulness)}</td>
            <td style="padding: 12px 8px; text-align: center;">${getGradeBadge(r.answer_relevance)}</td>
            <td style="padding: 12px 8px; text-align: center;">${getGradeBadge(r.context_recall)}</td>
            <td style="padding: 12px 8px; text-align: center;">${getGradeBadge(r.context_precision)}</td>
        `;
        
        elements.evalTableBody.appendChild(tr);
    });
}

async function runEvaluationSweep() {
    // Warn user that this takes time and requires API key if cloud provider is active
    if (appState.provider === "gemini" && !appState.apiKeyConfigured) {
        alert("Please configure a Gemini API key in the sidebar before running evaluations.");
        return;
    }
    if (appState.provider === "cohere" && !appState.cohereKeyConfigured) {
        alert("Please configure a Cohere API key in the sidebar before running evaluations.");
        return;
    }
    
    if (!confirm(`Are you sure you want to run a complete compliance evaluation sweep using Google Gemini as a judge? This will evaluate 10 compliance questions against the active agent and update all dashboard scores.`)) {
        return;
    }
    
    showOverlay(
        `Running Compliance Audit Sweep...`,
        `The AI Auditor is executing 10 compliance queries, fetching RAG documents, and grading responses across 4 standard metrics (Faithfulness, Relevance, Recall, Precision). Spaced safely to respect LLM quotas. Please wait (30-45 seconds).`
    );
    
    try {
        const response = await fetch(`${API_BASE}/api/evaluate/run`, {
            method: "POST"
        });
        
        if (!response.ok) {
            const data = await response.json();
            throw new Error(data.detail || "Evaluation failed.");
        }
        
        const result = await response.json();
        alert(`Compliance sweep completed successfully! Average Faithfulness: ${result.average_faithfulness.toFixed(4)} | Average Relevance: ${result.average_answer_relevance.toFixed(4)}`);
        
        await loadEvaluationReport();
    } catch (err) {
        console.error("Evaluation sweep execution error:", err);
        alert(`Evaluation Error: ${err.message}`);
    } finally {
        hideOverlay();
    }
}

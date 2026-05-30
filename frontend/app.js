// ISO 9001:2015 QMS AI Auditor Frontend Controller

// Dynamic API URL: resolves localhost locally and automatically falls back to your public cloud API URL when online
const API_BASE = window.location.hostname === "localhost" || window.location.hostname === "127.0.0.1"
    ? "http://127.0.0.1:8000"
    : "https://iso-9001-qms-rag-backend.onrender.com";

// App Global State
let appState = {
    provider: "gemini",       // "gemini" or "ollama"
    apiKeyConfigured: false,
    ollamaActive: false,
    indexReady: false,
    indexedChunks: 0,
    diagnosticChunks: []       // cached chunks for diagnostics tab
};

// DOM ELEMENT CACHE
const elements = {
    // RAG Provider Selection
    providerSelect: document.getElementById("provider-select"),
    geminiConfigGroup: document.getElementById("gemini-config-group"),
    ollamaConfigGroup: document.getElementById("ollama-config-group"),
    ollamaStatusDot: document.getElementById("ollama-status-dot"),
    ollamaStatusText: document.getElementById("ollama-status-text"),
    ollamaPullInstructions: document.getElementById("ollama-pull-instructions"),
    
    // Sidebar config
    apiKeyInput: document.getElementById("api-key-input"),
    saveKeyBtn: document.getElementById("save-key-btn"),
    indexStatusDot: document.getElementById("index-status-dot"),
    indexStatusText: document.getElementById("index-status-text"),
    chunksStatText: document.getElementById("chunks-stat-text"),
    ingestBtn: document.getElementById("ingest-btn"),
    
    // Tuning Controls (Model Selection Groups)
    geminiModelsGroup: document.getElementById("gemini-models-group"),
    ollamaModelsGroup: document.getElementById("ollama-models-group"),
    modelSelect: document.getElementById("model-select"),
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
            }
        });
    });
}

// PROVIDER SWITCH COORDINATION
function handleProviderSwitch() {
    const val = elements.providerSelect.value;
    appState.provider = val;
    
    if (val === "gemini") {
        elements.geminiConfigGroup.classList.remove("hidden");
        elements.geminiModelsGroup.classList.remove("hidden");
        
        elements.ollamaConfigGroup.classList.add("hidden");
        elements.ollamaModelsGroup.classList.add("hidden");
        elements.ollamaPullInstructions.classList.add("hidden");
    } else {
        elements.geminiConfigGroup.classList.add("hidden");
        elements.geminiModelsGroup.classList.add("hidden");
        
        elements.ollamaConfigGroup.classList.remove("hidden");
        elements.ollamaModelsGroup.classList.remove("hidden");
        
        // Fetch local Ollama models list
        fetchOllamaModels();
    }
    
    // Clear chat messages to avoid mixing context between Gemini and Ollama
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

// GLOBAL EVENT HANDLERS
function setupEventListeners() {
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
    
    // 2. PDF Ingestion Action
    elements.ingestBtn.addEventListener("click", async () => {
        const provUpper = appState.provider.toUpperCase();
        showOverlay(
            `Processing ${provUpper} Ingestion...`, 
            `The API is parsing your ISO PDF standard, running sliding-window chunking, generating local or cloud embeddings using ${provUpper}, and serializing the vector database. This takes 15-30 seconds.`
        );
        try {
            const response = await fetch(`${API_BASE}/api/ingest?force=true&provider=${appState.provider}`, {
                method: "POST"
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
        
        appState.apiKeyConfigured = status.api_key_configured;
        appState.ollamaActive = status.ollama_active;
        appState.indexReady = status.index_ready;
        appState.indexedChunks = status.indexed_chunks;
        
        // Update index indicators in UI
        if (status.index_ready) {
            elements.indexStatusDot.className = "dot dot-green";
            elements.indexStatusText.textContent = `Index: Ready (${appState.provider.toUpperCase()})`;
            elements.chunksStatText.textContent = `${status.indexed_chunks} QMS chunks indexed.`;
            elements.ingestBtn.textContent = "Rebuild QMS Index";
        } else {
            elements.indexStatusDot.className = "dot dot-red";
            elements.indexStatusText.textContent = `Index: Not Found (${appState.provider.toUpperCase()})`;
            elements.chunksStatText.textContent = "Ingestion required.";
            elements.ingestBtn.textContent = "Ingest & Embed PDF";
        }
        
        // Manage active states based on Provider selection
        if (appState.provider === "gemini") {
            if (status.api_key_configured) {
                elements.ingestBtn.disabled = false;
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
        } else {
            // Ollama Mode
            elements.apiKeyInput.placeholder = "Ollama mode active";
            
            // Check if server is active
            if (status.ollama_active) {
                elements.ollamaStatusDot.className = "dot dot-green";
                elements.ollamaStatusText.textContent = "Ollama: Connected";
                elements.ingestBtn.disabled = false;
                
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
            ollama_model: elements.ollamaModelSelect.value || "llama3"
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

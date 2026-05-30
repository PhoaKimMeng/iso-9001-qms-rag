// ISO 9001:2015 QMS AI Auditor Frontend Controller

const API_BASE = "http://127.0.0.1:8000";

// App Global State
let appState = {
    apiKeyConfigured: false,
    indexReady: false,
    indexedChunks: 0,
    diagnosticChunks: []  // cached chunks for diagnostics tab
};

// DOM ELEMENT CACHE
const elements = {
    // Sidebar config
    apiKeyInput: document.getElementById("api-key-input"),
    saveKeyBtn: document.getElementById("save-key-btn"),
    indexStatusDot: document.getElementById("index-status-dot"),
    indexStatusText: document.getElementById("index-status-text"),
    chunksStatText: document.getElementById("chunks-stat-text"),
    ingestBtn: document.getElementById("ingest-btn"),
    modelSelect: document.getElementById("model-select"),
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
    
    // Initial fetch of system status
    await checkSystemStatus();
    
    // Load navigator clause descriptions
    await loadClauseDatabase();
});

// SLIDER CONFIGURATIONS
function setupSliders() {
    // Top-K Slider
    elements.topKSlider.addEventListener("input", (e) => {
        elements.topKVal.textContent = e.target.value;
    });
    
    // Temperature Slider
    elements.tempSlider.addEventListener("input", (e) => {
        elements.tempVal.textContent = parseFloat(e.target.value).toFixed(2);
    });
    
    // Diagnostics inspector slider
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
            
            // Toggle active tab link classes
            elements.tabLinks.forEach(l => l.classList.remove("active"));
            link.classList.add("active");
            
            // Toggle active panel visibility
            elements.tabPanels.forEach(panel => {
                if (panel.id === targetTab) {
                    panel.classList.add("active");
                } else {
                    panel.classList.remove("active");
                }
            });
            
            // Trigger specific actions when tabs open
            if (targetTab === "tab-diagnostics" && appState.indexReady) {
                loadDiagnostics();
            }
        });
    });
}

// GLOBAL EVENT HANDLERS
function setupEventListeners() {
    // 1. Save Key Action
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
            elements.apiKeyInput.value = ""; // clear input visually
            await checkSystemStatus();
        } catch (err) {
            alert(`Error configuring API Key: ${err.message}`);
        } finally {
            hideOverlay();
        }
    });
    
    // 2. PDF Ingestion Action
    elements.ingestBtn.addEventListener("click", async () => {
        showOverlay(
            "Processing QMS Ingestion...", 
            "The API is parsing your ISO PDF standard, running sliding-window chunking, generating 1536-dimensional embeddings, and serializing your vector index. This takes 15-30 seconds."
        );
        try {
            const response = await fetch(`${API_BASE}/api/ingest?force=true`, {
                method: "POST"
            });
            
            if (!response.ok) {
                const data = await response.json();
                throw new Error(data.detail || "Ingestion failed.");
            }
            
            const result = await response.json();
            alert(`Ingestion Completed! ${result.chunks_count} vector chunks generated and indexed.`);
            await checkSystemStatus();
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
        const response = await fetch(`${API_BASE}/api/status`);
        if (!response.ok) throw new Error("System status unreachable.");
        
        const status = await response.json();
        
        appState.apiKeyConfigured = status.api_key_configured;
        appState.indexReady = status.index_ready;
        appState.indexedChunks = status.indexed_chunks;
        
        // Update index indicators in UI
        if (status.index_ready) {
            elements.indexStatusDot.className = "dot dot-green";
            elements.indexStatusText.textContent = "Index: Ready";
            elements.chunksStatText.textContent = `${status.indexed_chunks} QMS chunks indexed.`;
            elements.ingestBtn.textContent = "Rebuild QMS Index";
        } else {
            elements.indexStatusDot.className = "dot dot-red";
            elements.indexStatusText.textContent = "Index: Not Found";
            elements.chunksStatText.textContent = "Ingestion required.";
            elements.ingestBtn.textContent = "Ingest & Embed PDF";
        }
        
        // Manage active states based on API Key configuration
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
                alert("Please configure your API Key and ingest the PDF in the sidebar first.");
                return;
            }
            processUserQuery(query);
        });
    });
}

// Q&A QUERY PROCESSOR
async function processUserQuery(queryText) {
    // 1. Append User Message
    appendMessage("user", "User", queryText);
    
    // 2. Append Assistant Typing Indicator
    const typingId = appendTypingIndicator();
    
    try {
        const top_k = parseInt(elements.topKSlider.value);
        const temperature = parseFloat(elements.tempSlider.value);
        
        // Call Backend API
        const response = await fetch(`${API_BASE}/api/query`, {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({
                query: queryText,
                top_k: top_k,
                temperature: temperature
            })
        });
        
        // Remove typing indicator
        removeTypingIndicator(typingId);
        
        if (!response.ok) {
            const data = await response.json();
            throw new Error(data.detail || "Query failed.");
        }
        
        const result = await response.json();
        
        // 3. Append Assistant Answer with Sources
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
    
    // Formatting basic markdown syntax (like **bold** and newlines)
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

// Basic markdown to HTML renderer
function formatMarkdown(text) {
    // Escape standard tags
    let html = text
        .replace(/&/g, "&amp;")
        .replace(/</g, "&lt;")
        .replace(/>/g, "&gt;");
        
    // Bold matches (**text**)
    html = html.replace(/\*\*(.*?)\*\*/g, "<strong>$1</strong>");
    
    // Checklists [ ] and [x]
    html = html.replace(/\[ \]/g, "☐");
    html = html.replace(/\[x\]/g, "☑");
    
    // Process bullet points
    html = html.replace(/^\s*-\s+(.*)$/gm, "<li>$1</li>");
    // Wrap groups of <li> in <ul>
    // Note: This is a basic formatter, but works wonderfully for standard LLM streams
    html = html.replace(/(<li>.*<\/li>)/s, "<ul>$1</ul>");
    
    // Line breaks
    html = html.replace(/\n/g, "<br>");
    
    return html;
}

// Toggle Source expander collapse
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
        
        elements.clausesContainer.innerHTML = ""; // Clear loader
        
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

// Pre-fill chat from the Clause Navigator
window.queryNavigator = function(query) {
    if (elements.chatInput.disabled) {
        alert("Please configure your API Key and ingest the PDF in the sidebar first.");
        return;
    }
    
    // Switch to Tab 1 (Chat)
    const chatTabLink = document.querySelector('[data-tab="tab-chat"]');
    chatTabLink.click();
    
    // Start processing query
    processUserQuery(query);
};

// TAB 3: DIAGNOSTICS LOAD
async function loadDiagnostics() {
    try {
        const response = await fetch(`${API_BASE}/api/stats`);
        if (!response.ok) throw new Error("Failed to load diagnostics.");
        
        const stats = await response.json();
        
        if (stats.chunks_count === 0) {
            elements.diagnosticsReadyContent.classList.add("hidden");
            elements.diagnosticsEmptyContent.classList.remove("hidden");
            return;
        }
        
        elements.diagnosticsEmptyContent.classList.add("hidden");
        elements.diagnosticsReadyContent.classList.remove("hidden");
        
        // Cache diagnostics chunks globally
        appState.diagnosticChunks = stats.chunks;
        
        // Update stats metrics
        elements.metricPages.textContent = stats.total_pages;
        elements.metricChunks.textContent = stats.chunks_count;
        elements.metricAvgLen.textContent = stats.avg_chunk_length;
        elements.metricTotalChars.textContent = stats.total_characters.toLocaleString();
        
        // Update slider limit
        elements.chunkInspectorSlider.max = stats.chunks_count - 1;
        elements.chunkInspectorSlider.value = 0;
        elements.chunkIndexVal.textContent = 0;
        
        // Render index 0
        renderSelectedChunk(0);
        
    } catch (err) {
        console.error("Diagnostics load failed:", err);
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

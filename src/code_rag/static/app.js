/**
 * GitHub Code RAG — Interactive Frontend Application Logic (Persistent Chat Across Refreshes)
 */

document.addEventListener("DOMContentLoaded", () => {
    // DOM Elements
    const statusDot = document.getElementById("statusDot");
    const statusText = document.getElementById("statusText");
    const ingestForm = document.getElementById("ingestForm");
    const repoSourceInput = document.getElementById("repoSourceInput");
    const forceReindexCheckbox = document.getElementById("forceReindexCheckbox");
    const ingestBtn = document.getElementById("ingestBtn");
    const ingestBtnText = document.getElementById("ingestBtnText");
    const ingestSpinner = document.getElementById("ingestSpinner");
    const ingestAlert = document.getElementById("ingestAlert");
    const newChatBtn = document.getElementById("newChatBtn");
    const conversationsList = document.getElementById("conversationsList");
    const repoSelect = document.getElementById("repoSelect");
    const deleteRepoBtn = document.getElementById("deleteRepoBtn");
    const modelSelect = document.getElementById("modelSelect");
    const chatContainer = document.getElementById("chatContainer");
    const queryForm = document.getElementById("queryForm");
    const queryInput = document.getElementById("queryInput");
    const sendBtn = document.getElementById("sendBtn");
    const clearChatBtn = document.getElementById("clearChatBtn");
    const themeToggleBtn = document.getElementById("themeToggleBtn");
    const sidebarToggleBtn = document.getElementById("sidebarToggleBtn");
    const sidebar = document.getElementById("sidebar");
    const quickPasteLocalBtn = document.getElementById("quickPasteLocalBtn");
    const heroQuickIndexBtn = document.getElementById("heroQuickIndexBtn");
    const inputRepoLabel = document.getElementById("inputRepoLabel");
    const selectedRepoName = document.getElementById("selectedRepoName");
    const repoMeta = document.getElementById("repoMeta");
    const welcomeHero = document.getElementById("welcomeHero");
    const toastContainer = document.getElementById("toastContainer");

    const LOCAL_WORKSPACE_PATH = window.location.origin.includes("localhost") 
        ? "/Users/macbookpro/Downloads/Github-repositories-read-RAG" 
        : ".";

    let isSubmitting = false;
    let conversations = [];
    let activeConvId = null;
    let chatHistory = [];

    // Initialize application
    init();

    async function init() {
        setupTheme();
        setupEventListeners();
        loadConversationsFromMemory();
        await checkStatus();
        await loadModels();
        await loadRepos();
    }

    // =========================================================================
    // Long-Term Multi-Conversation Memory Management
    // =========================================================================
    function loadConversationsFromMemory() {
        try {
            const savedConvs = localStorage.getItem("coderag_conversations");
            if (savedConvs) {
                const parsed = JSON.parse(savedConvs);
                if (Array.isArray(parsed)) {
                    conversations = parsed;
                }
            }

            // Migration from legacy single-chat history if exists
            const legacyHistory = localStorage.getItem("coderag_chat_history");
            if (legacyHistory && conversations.length === 0) {
                try {
                    const parsedHist = JSON.parse(legacyHistory);
                    if (Array.isArray(parsedHist) && parsedHist.length > 0) {
                        const firstQ = parsedHist.find(m => m.type === "user");
                        const title = firstQ ? firstQ.text.slice(0, 32) : "Initial Conversation";
                        const legacyConv = {
                            id: "conv_" + Date.now(),
                            title: title,
                            timestamp: Date.now(),
                            repo: localStorage.getItem("coderag_selected_repo") || "",
                            model: localStorage.getItem("coderag_selected_model") || "",
                            messages: parsedHist
                        };
                        conversations = [legacyConv];
                        saveConversationsToMemory();
                    }
                } catch {
                    // Ignore legacy parse failure
                }
            }

            activeConvId = localStorage.getItem("coderag_active_conv_id");
            if (!activeConvId && conversations.length > 0) {
                activeConvId = conversations[0].id;
            }

            restoreActiveChatUI();
            renderConversationsList();
        } catch (err) {
            console.error("Error loading conversation memory:", err);
            startNewConversation(false);
        }
    }

    function saveConversationsToMemory() {
        try {
            localStorage.setItem("coderag_conversations", JSON.stringify(conversations));
            if (activeConvId) {
                localStorage.setItem("coderag_active_conv_id", activeConvId);
            } else {
                localStorage.removeItem("coderag_active_conv_id");
            }
        } catch (err) {
            console.warn("Unable to save conversations to memory:", err);
        }
    }

    function startNewConversation(shouldToast = true) {
        activeConvId = "conv_" + Date.now();
        chatHistory = [];
        saveConversationsToMemory();
        restoreActiveChatUI();
        renderConversationsList();
        if (shouldToast) {
            showToast("New conversation started", "info", 2000);
        }
    }

    function selectConversation(convId) {
        if (activeConvId === convId) return;
        activeConvId = convId;
        saveConversationsToMemory();
        restoreActiveChatUI();
        renderConversationsList();
    }

    function deleteConversation(convId, e) {
        if (e) e.stopPropagation();
        const idx = conversations.findIndex(c => c.id === convId);
        if (idx !== -1) {
            const deletedTitle = conversations[idx].title || "Conversation";
            conversations.splice(idx, 1);
            if (activeConvId === convId) {
                if (conversations.length > 0) {
                    activeConvId = conversations[0].id;
                } else {
                    activeConvId = "conv_" + Date.now();
                }
            }
            saveConversationsToMemory();
            restoreActiveChatUI();
            renderConversationsList();
            showToast(`Deleted "${deletedTitle}"`, "info", 2500);
        }
    }

    function restoreActiveChatUI() {
        chatContainer.innerHTML = "";
        const currentConv = conversations.find(c => c.id === activeConvId);
        chatHistory = currentConv ? (currentConv.messages || []) : [];

        if (chatHistory.length > 0) {
            chatHistory.forEach(item => {
                if (item.type === "user") {
                    appendMsg("user", escapeHtml(item.text));
                } else if (item.type === "bot") {
                    const row = appendMsg("bot", "");
                    const bubble = row.querySelector(".msg-bubble");
                    const textDiv = document.createElement("div");
                    textDiv.className = "markdown-body";
                    renderMarkdown(textDiv, item.text || "");
                    bubble.appendChild(textDiv);
                    renderExtras(bubble, item.sources || [], item.traceSteps || []);
                }
            });
            chatContainer.scrollTop = chatContainer.scrollHeight;
        } else {
            if (welcomeHero) {
                chatContainer.appendChild(welcomeHero);
            }
        }
    }

    function renderConversationsList() {
        if (!conversationsList) return;
        conversationsList.innerHTML = "";

        if (conversations.length === 0) {
            conversationsList.innerHTML = `<div class="conversations-empty">No previous chats yet</div>`;
            return;
        }

        // Sort latest first
        const sorted = [...conversations].sort((a, b) => (b.timestamp || 0) - (a.timestamp || 0));

        sorted.forEach(c => {
            const item = document.createElement("div");
            item.className = `conv-item ${c.id === activeConvId ? "active" : ""}`;
            const displayTitle = c.title || "New Chat";

            item.innerHTML = `
                <div class="conv-main">
                    <span class="conv-icon">
                        <svg width="13" height="13" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M21 15a2 2 0 0 1-2 2H7l-4 4V5a2 2 0 0 1 2-2h14a2 2 0 0 1 2 2z"></path></svg>
                    </span>
                    <span class="conv-title" title="${escapeHtml(displayTitle)}">${escapeHtml(displayTitle)}</span>
                </div>
                <button type="button" class="btn-delete-conv" title="Delete conversation">
                    <svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><polyline points="3 6 5 6 21 6"></polyline><path d="M19 6v14a2 2 0 0 1-2 2H7a2 2 0 0 1-2-2V6m3 0V4a2 2 0 0 1 2-2h4a2 2 0 0 1 2 2v2"></path></svg>
                </button>
            `;

            item.addEventListener("click", () => selectConversation(c.id));
            const delBtn = item.querySelector(".btn-delete-conv");
            delBtn.addEventListener("click", (e) => deleteConversation(c.id, e));

            conversationsList.appendChild(item);
        });
    }

    function recordUserMessageInConversation(text) {
        if (!activeConvId) {
            activeConvId = "conv_" + Date.now();
        }

        let conv = conversations.find(c => c.id === activeConvId);
        if (!conv) {
            conv = {
                id: activeConvId,
                title: text.slice(0, 36) + (text.length > 36 ? "..." : ""),
                timestamp: Date.now(),
                repo: repoSelect.value || "",
                model: modelSelect.value || "",
                messages: []
            };
            conversations.unshift(conv);
        } else {
            conv.timestamp = Date.now();
            conv.repo = repoSelect.value || conv.repo;
            conv.model = modelSelect.value || conv.model;
            if (!conv.title || conv.title === "New Chat" || conv.messages.length === 0) {
                conv.title = text.slice(0, 36) + (text.length > 36 ? "..." : "");
            }
        }

        conv.messages.push({ type: "user", text: text });
        chatHistory = conv.messages;
        saveConversationsToMemory();
        renderConversationsList();
    }

    function recordBotMessageInConversation(text, sources, traceSteps) {
        let conv = conversations.find(c => c.id === activeConvId);
        if (conv) {
            conv.timestamp = Date.now();
            conv.messages.push({
                type: "bot",
                text: text,
                sources: sources || [],
                traceSteps: traceSteps || []
            });
            chatHistory = conv.messages;
            saveConversationsToMemory();
            renderConversationsList();
        }
    }

    // =========================================================================
    // Event Listeners & UI Helpers
    // =========================================================================
    function setupEventListeners() {
        // New Conversation Button
        if (newChatBtn) {
            newChatBtn.addEventListener("click", () => {
                startNewConversation(true);
            });
        }
        // Quick suggestions chips
        document.querySelectorAll(".q-chip").forEach(chip => {
            chip.addEventListener("click", () => {
                const query = chip.dataset.query;
                if (!repoSelect.value) {
                    showToast("Please index and select a repository first.", "warning");
                    if (repoSourceInput) repoSourceInput.focus();
                    return;
                }
                queryInput.value = query;
                autoResizeTextarea();
                queryForm.dispatchEvent(new Event("submit"));
            });
        });

        // Quick paste local repo buttons
        if (quickPasteLocalBtn) {
            quickPasteLocalBtn.addEventListener("click", () => {
                repoSourceInput.value = LOCAL_WORKSPACE_PATH;
                showToast("Pre-filled current local repository path", "info");
            });
        }

        if (heroQuickIndexBtn) {
            heroQuickIndexBtn.addEventListener("click", () => {
                repoSourceInput.value = LOCAL_WORKSPACE_PATH;
                ingestForm.dispatchEvent(new Event("submit"));
            });
        }

        // Auto-expand textarea
        queryInput.addEventListener("input", autoResizeTextarea);
        queryInput.addEventListener("keydown", (e) => {
            if (e.key === "Enter" && !e.shiftKey) {
                e.preventDefault();
                queryForm.dispatchEvent(new Event("submit"));
            }
        });

        // Clear chat (resets active conversation)
        if (clearChatBtn) {
            clearChatBtn.addEventListener("click", () => {
                if (activeConvId) {
                    const conv = conversations.find(c => c.id === activeConvId);
                    if (conv) {
                        conv.messages = [];
                    }
                    chatHistory = [];
                    saveConversationsToMemory();
                }
                chatContainer.innerHTML = "";
                if (welcomeHero) {
                    chatContainer.appendChild(welcomeHero);
                }
                showToast("Active conversation cleared", "info", 2000);
            });
        }

        // Sidebar toggle (desktop & mobile)
        if (sidebarToggleBtn) {
            sidebarToggleBtn.addEventListener("click", () => {
                if (window.innerWidth <= 900) {
                    sidebar.classList.toggle("open");
                } else {
                    const isCollapsed = sidebar.classList.toggle("collapsed");
                    localStorage.setItem("coderag_sidebar_collapsed", isCollapsed ? "true" : "false");
                }
            });
        }

        // Global shortcut Ctrl+B / Cmd+B to toggle sidebar
        document.addEventListener("keydown", (e) => {
            if ((e.ctrlKey || e.metaKey) && e.key.toLowerCase() === "b") {
                e.preventDefault();
                if (sidebarToggleBtn) sidebarToggleBtn.click();
            }
        });

        // Restore saved sidebar collapsed state on desktop
        if (localStorage.getItem("coderag_sidebar_collapsed") === "true" && window.innerWidth > 900) {
            sidebar.classList.add("collapsed");
        }

        // Theme toggle
        if (themeToggleBtn) {
            themeToggleBtn.addEventListener("click", toggleTheme);
        }

        // Repository selection change
        repoSelect.addEventListener("change", () => {
            localStorage.setItem("coderag_selected_repo", repoSelect.value);
            updateSelectedRepoUI();
        });

        // Model selection change
        modelSelect.addEventListener("change", () => {
            localStorage.setItem("coderag_selected_model", modelSelect.value);
        });

        // Delete Repository Action
        if (deleteRepoBtn) {
            deleteRepoBtn.addEventListener("click", async () => {
                const targetRepo = repoSelect.value;
                if (!targetRepo) return;

                const confirmed = window.confirm(`Are you sure you want to delete indexed repository "${targetRepo}"?\n\nThis will remove its vector embeddings, BM25 search index, and cached files.`);
                if (!confirmed) return;

                deleteRepoBtn.disabled = true;
                const originalContent = deleteRepoBtn.innerHTML;
                deleteRepoBtn.innerHTML = `<span class="spinner" style="width: 12px; height: 12px; border-width: 2px;"></span>`;

                try {
                    const res = await fetch(`/api/repos/${encodeURIComponent(targetRepo)}`, {
                        method: "DELETE"
                    });
                    const data = await res.json();
                    if (res.ok) {
                        showToast(`Repository "${targetRepo}" deleted successfully`, "success");
                        if (localStorage.getItem("coderag_selected_repo") === targetRepo) {
                            localStorage.removeItem("coderag_selected_repo");
                        }
                        await loadRepos();
                    } else {
                        showToast(data.detail || `Failed to delete "${targetRepo}"`, "error");
                    }
                } catch (err) {
                    showToast(`Error deleting repository: ${err.message}`, "error");
                } finally {
                    deleteRepoBtn.innerHTML = originalContent;
                    updateSelectedRepoUI();
                }
            });
        }
    }

    function autoResizeTextarea() {
        queryInput.style.height = "auto";
        queryInput.style.height = Math.min(queryInput.scrollHeight, 120) + "px";
    }

    function updateSelectedRepoUI() {
        const val = repoSelect.value;
        if (val) {
            inputRepoLabel.textContent = val;
            if (selectedRepoName) selectedRepoName.textContent = val;
            if (repoMeta) repoMeta.style.display = "block";
            if (deleteRepoBtn) {
                deleteRepoBtn.disabled = false;
                deleteRepoBtn.title = `Delete repository: ${val}`;
            }
        } else {
            inputRepoLabel.textContent = "No Repo Selected";
            if (repoMeta) repoMeta.style.display = "none";
            if (deleteRepoBtn) {
                deleteRepoBtn.disabled = true;
                deleteRepoBtn.title = "No repository selected";
            }
        }
    }

    // =========================================================================
    // Toast Notification System
    // =========================================================================
    function showToast(message, type = "info", duration = 4000) {
        const toast = document.createElement("div");
        toast.className = `toast toast-${type}`;

        const icons = {
            success: `<svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.5"><path d="M20 6L9 17l-5-5"/></svg>`,
            error: `<svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.5"><circle cx="12" cy="12" r="10"/><line x1="15" y1="9" x2="9" y2="15"/><line x1="9" y1="9" x2="15" y2="15"/></svg>`,
            warning: `<svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.5"><path d="M10.29 3.86L1.82 18a2 2 0 0 0 1.71 3h16.94a2 2 0 0 0 1.71-3L13.71 3.86a2 2 0 0 0-3.42 0z"/><line x1="12" y1="9" x2="12" y2="13"/><line x1="12" y1="17" x2="12.01" y2="17"/></svg>`,
            info: `<svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.5"><circle cx="12" cy="12" r="10"/><line x1="12" y1="16" x2="12" y2="12"/><line x1="12" y1="8" x2="12.01" y2="8"/></svg>`
        };

        toast.innerHTML = `
            <div class="toast-icon">${icons[type] || icons.info}</div>
            <div class="toast-message">${escapeHtml(message)}</div>
            <button class="toast-close" title="Dismiss">&times;</button>
        `;

        toast.querySelector(".toast-close").addEventListener("click", () => {
            dismissToast(toast);
        });

        toastContainer.appendChild(toast);

        setTimeout(() => {
            dismissToast(toast);
        }, duration);
    }

    function dismissToast(toast) {
        toast.classList.add("toast-hiding");
        setTimeout(() => {
            if (toast.parentNode) toast.parentNode.removeChild(toast);
        }, 250);
    }

    // =========================================================================
    // Theme Management
    // =========================================================================
    function setupTheme() {
        const savedTheme = localStorage.getItem("coderag_theme") || "dark";
        document.documentElement.setAttribute("data-theme", savedTheme);
        updateThemeIcons(savedTheme);
    }

    function toggleTheme() {
        const currentTheme = document.documentElement.getAttribute("data-theme") || "dark";
        const newTheme = currentTheme === "dark" ? "light" : "dark";
        document.documentElement.setAttribute("data-theme", newTheme);
        localStorage.setItem("coderag_theme", newTheme);
        updateThemeIcons(newTheme);
    }

    function updateThemeIcons(theme) {
        const sun = themeToggleBtn.querySelector(".sun-icon");
        const moon = themeToggleBtn.querySelector(".moon-icon");
        if (theme === "light") {
            sun.style.display = "none";
            moon.style.display = "block";
        } else {
            sun.style.display = "block";
            moon.style.display = "none";
        }
    }

    // =========================================================================
    // Backend API Calls
    // =========================================================================
    async function checkStatus() {
        try {
            const res = await fetch("/api/health");
            if (res.ok) {
                const data = await res.json();
                if (statusDot) statusDot.className = "status-indicator online";
                if (statusText) statusText.textContent = `Online (${data.active_llm || "Ready"})`;
            } else {
                throw new Error();
            }
        } catch {
            if (statusDot) statusDot.className = "status-indicator offline";
            if (statusText) statusText.textContent = "Offline (Check Ollama)";
        }
    }

    async function loadModels() {
        try {
            const res = await fetch("/api/models");
            const savedModel = localStorage.getItem("coderag_selected_model");
            if (res.ok) {
                const data = await res.json();
                modelSelect.innerHTML = "";
                if (data.models && data.models.length > 0) {
                    data.models.forEach(m => {
                        const opt = document.createElement("option");
                        opt.value = m;
                        opt.textContent = m;
                        if (savedModel ? m === savedModel : m === data.active_model) {
                            opt.selected = true;
                        }
                        modelSelect.appendChild(opt);
                    });
                } else {
                    modelSelect.innerHTML = '<option value="llama3.2:latest">llama3.2:latest</option><option value="olmo:latest">olmo:latest</option>';
                }
            }
        } catch {
            modelSelect.innerHTML = '<option value="llama3.2:latest">llama3.2:latest</option>';
        }
    }

    async function loadRepos() {
        try {
            const res = await fetch("/api/repos");
            const savedRepo = localStorage.getItem("coderag_selected_repo");
            if (res.ok) {
                const data = await res.json();
                repoSelect.innerHTML = "";
                if (data.repos && data.repos.length > 0) {
                    let hasSelected = false;
                    data.repos.forEach((r, idx) => {
                        const opt = document.createElement("option");
                        opt.value = r;
                        opt.textContent = r;
                        if (savedRepo && r === savedRepo) {
                            opt.selected = true;
                            hasSelected = true;
                        } else if (!savedRepo && idx === 0) {
                            opt.selected = true;
                            hasSelected = true;
                        }
                        repoSelect.appendChild(opt);
                    });
                    if (!hasSelected && data.repos.length > 0) {
                        repoSelect.options[0].selected = true;
                    }
                } else {
                    repoSelect.innerHTML = '<option value="">No repositories indexed</option>';
                }
                updateSelectedRepoUI();
            }
        } catch (err) {
            console.error("Failed to load repositories:", err);
        }
    }

    // Indexing Form Submit
    ingestForm.addEventListener("submit", async (e) => {
        e.preventDefault();
        const source = repoSourceInput.value.trim();
        if (!source) return;

        ingestBtnText.textContent = "Indexing Code...";
        ingestSpinner.style.display = "inline-block";
        ingestBtn.disabled = true;
        ingestAlert.style.display = "none";

        try {
            const res = await fetch("/api/index", {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify({
                    repo_source: source,
                    force_reindex: forceReindexCheckbox.checked
                })
            });
            const data = await res.json();
            if (res.ok) {
                ingestAlert.className = "alert-box success";
                ingestAlert.textContent = `Successfully indexed ${data.total_chunks} chunks from ${data.total_files} files.`;
                ingestAlert.style.display = "block";
                repoSourceInput.value = "";
                localStorage.setItem("coderag_selected_repo", data.repo_id);
                showToast(`Repository ${data.repo_name} indexed (${data.total_chunks} chunks)`, "success");
                await loadRepos();
            } else {
                throw new Error(data.detail || "Indexing failed");
            }
        } catch (err) {
            ingestAlert.className = "alert-box error";
            ingestAlert.textContent = err.message;
            ingestAlert.style.display = "block";
            showToast(err.message, "error");
        } finally {
            ingestBtnText.textContent = "Index Repository";
            ingestSpinner.style.display = "none";
            ingestBtn.disabled = false;
        }
    });

    // Query Submission & Streaming
    queryForm.addEventListener("submit", async (e) => {
        e.preventDefault();
        if (isSubmitting) return;

        const q = queryInput.value.trim();
        const repo = repoSelect.value;
        const model = modelSelect.value;

        if (!q) return;
        if (!repo) {
            showToast("Please index and select a repository first.", "warning");
            if (repoSourceInput) repoSourceInput.focus();
            return;
        }

        // Hide welcome hero when conversation begins
        if (welcomeHero && welcomeHero.parentNode === chatContainer) {
            welcomeHero.remove();
        }

        // Append User Message and persist to conversation memory
        appendMsg("user", escapeHtml(q));
        recordUserMessageInConversation(q);

        queryInput.value = "";
        queryInput.style.height = "auto";

        // Append Loading Bot Message
        const botMsg = appendMsg("bot", `
            <div class="loading-box">
                <span class="spinner" style="border-color: rgba(99, 102, 241, 0.2); border-top-color: var(--primary-color);"></span>
                <span class="stream-status">Analyzing intent & searching AST symbols...</span>
            </div>
        `);
        const contentEl = botMsg.querySelector(".msg-bubble");

        isSubmitting = true;
        sendBtn.disabled = true;

        let accumulatedAnswer = "";
        let sources = [];
        let traceSteps = [];

        try {
            const res = await fetch("/api/query/stream", {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify({
                    repo_id: repo,
                    question: q,
                    model_name: model
                })
            });

            if (!res.ok) {
                const errJson = await res.json().catch(() => ({}));
                throw new Error(errJson.detail || "Query execution failed");
            }

            const reader = res.body.getReader();
            const decoder = new TextDecoder();
            let buffer = "";
            let textDiv = null;

            while (true) {
                const { value, done } = await reader.read();
                if (done) break;

                buffer += decoder.decode(value, { stream: true });
                const lines = buffer.split("\n\n");
                buffer = lines.pop() || "";

                for (const line of lines) {
                    const trimmed = line.trim();
                    if (trimmed.startsWith("data: ")) {
                        try {
                            const data = JSON.parse(trimmed.slice(6));
                            if (data.type === "trace") {
                                const statusMsg = contentEl.querySelector(".stream-status");
                                if (statusMsg) statusMsg.textContent = data.step;
                                traceSteps.push(data.step);
                            } else if (data.type === "token") {
                                if (!textDiv) {
                                    contentEl.innerHTML = "";
                                    textDiv = document.createElement("div");
                                    textDiv.className = "markdown-body";
                                    contentEl.appendChild(textDiv);
                                }
                                accumulatedAnswer += data.content;
                                renderMarkdown(textDiv, accumulatedAnswer);
                                chatContainer.scrollTop = chatContainer.scrollHeight;
                            } else if (data.type === "done") {
                                sources = data.sources || [];
                                if (data.trace_steps) traceSteps = data.trace_steps;
                            } else if (data.type === "error") {
                                throw new Error(data.error);
                            }
                        } catch (parseErr) {
                            console.debug("Parse stream chunk error:", parseErr);
                        }
                    }
                }
            }

            if (!textDiv && accumulatedAnswer) {
                contentEl.innerHTML = "";
                textDiv = document.createElement("div");
                textDiv.className = "markdown-body";
                renderMarkdown(textDiv, accumulatedAnswer);
                contentEl.appendChild(textDiv);
            }

            renderExtras(contentEl, sources, traceSteps);

            // Persist bot message to active conversation
            if (accumulatedAnswer) {
                recordBotMessageInConversation(accumulatedAnswer, sources, traceSteps);
            }

        } catch (err) {
            contentEl.innerHTML = `
                <div style="color: var(--accent-rose); display: flex; align-items: center; gap: 8px;">
                    <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><circle cx="12" cy="12" r="10"/><line x1="12" y1="8" x2="12" y2="12"/><line x1="12" y1="16" x2="12.01" y2="16"/></svg>
                    <span><strong>Error:</strong> ${escapeHtml(err.message)}</span>
                </div>
            `;
            showToast(err.message, "error");
        } finally {
            isSubmitting = false;
            sendBtn.disabled = false;
            chatContainer.scrollTop = chatContainer.scrollHeight;
        }
    });

    // =========================================================================
    // Rendering & Helpers
    // =========================================================================
    function appendMsg(type, html) {
        const row = document.createElement("div");
        row.className = `msg-row ${type}`;
        row.innerHTML = `<div class="msg-bubble">${html}</div>`;
        chatContainer.appendChild(row);
        chatContainer.scrollTop = chatContainer.scrollHeight;
        return row;
    }

    function renderMarkdown(element, markdownText) {
        if (window.marked) {
            element.innerHTML = marked.parse(markdownText, { breaks: true, gfm: true });
        } else {
            element.innerHTML = formatMarkdownFallback(markdownText);
        }
        if (window.Prism) {
            Prism.highlightAllUnder(element);
        }
    }

    function renderExtras(contentEl, sources, traceSteps) {
        // Citations List
        if (sources && sources.length > 0) {
            const citeSec = document.createElement("div");
            citeSec.className = "citations-section";
            citeSec.innerHTML = `
                <div class="citations-header">
                    <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8z"/><polyline points="14 2 14 8 20 8"/></svg>
                    <span>Verified Source Citations (${sources.length})</span>
                </div>
            `;

            sources.forEach(s => {
                const card = document.createElement("div");
                card.className = "citation-card";
                card.innerHTML = `
                    <div class="citation-meta">
                        <span class="citation-file">
                            <svg width="13" height="13" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M13 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V9z"></path><polyline points="13 2 13 9 20 9"></polyline></svg>
                            ${escapeHtml(s.file_path)}
                        </span>
                        <span class="citation-badge">L${s.start_line}–${s.end_line}</span>
                    </div>
                    <pre><code class="language-${s.language || 'python'}">${escapeHtml(s.snippet || '')}</code></pre>
                `;
                citeSec.appendChild(card);
            });
            contentEl.appendChild(citeSec);
        }

        // Trace Section
        if (traceSteps && traceSteps.length > 0) {
            const traceSec = document.createElement("div");
            traceSec.className = "trace-wrapper";
            
            const btn = document.createElement("button");
            btn.className = "trace-toggle-btn";
            btn.innerHTML = `
                <svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.5"><polyline points="9 18 15 12 9 6"></polyline></svg>
                <span>Execution Trace (${traceSteps.length} steps)</span>
            `;

            const panel = document.createElement("div");
            panel.className = "trace-panel";
            panel.style.display = "none";

            traceSteps.forEach(st => {
                const item = document.createElement("div");
                item.className = "trace-step-item";
                item.innerHTML = `<span class="trace-step-dot"></span><span>${escapeHtml(st)}</span>`;
                panel.appendChild(item);
            });

            btn.addEventListener("click", () => {
                const isHidden = panel.style.display === "none";
                panel.style.display = isHidden ? "flex" : "none";
                btn.innerHTML = isHidden 
                    ? `<svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.5"><polyline points="6 9 12 15 18 9"></polyline></svg><span>Execution Trace (${traceSteps.length} steps)</span>`
                    : `<svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.5"><polyline points="9 18 15 12 9 6"></polyline></svg><span>Execution Trace (${traceSteps.length} steps)</span>`;
            });

            traceSec.appendChild(btn);
            traceSec.appendChild(panel);
            contentEl.appendChild(traceSec);
        }

        if (window.Prism) {
            Prism.highlightAllUnder(contentEl);
        }
    }

    function formatMarkdownFallback(str) {
        if (!str) return "";
        let h = escapeHtml(str);
        h = h.replace(/\*\*(.*?)\*\*/g, '<strong>$1</strong>');
        h = h.replace(/`([^`]+)`/g, '<code>$1</code>');
        h = h.replace(/\n\n/g, '</p><p>');
        h = h.replace(/\n/g, '<br/>');
        return `<p>${h}</p>`;
    }

    function escapeHtml(s) {
        if (!s) return "";
        return String(s)
            .replace(/&/g, "&amp;")
            .replace(/</g, "&lt;")
            .replace(/>/g, "&gt;")
            .replace(/"/g, "&quot;")
            .replace(/'/g, "&#039;");
    }
});

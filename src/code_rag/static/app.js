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
    const navNewChatBtn = document.getElementById("navNewChatBtn");
    const conversationsList = document.getElementById("conversationsList");
    const repoSelect = document.getElementById("repoSelect");
    const deleteRepoBtn = document.getElementById("deleteRepoBtn");
    const indexedReposList = document.getElementById("indexedReposList");
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
    const scrollBottomBtn = document.getElementById("scrollBottomBtn");

    const LOCAL_WORKSPACE_PATH = window.location.origin.includes("localhost") 
        ? "/Users/macbookpro/Downloads/Github-repositories-read-RAG" 
        : ".";

    let isSubmitting = false;
    let conversations = [];
    let activeConvId = null;
    let chatHistory = [];
    let autoScrollEnabled = true;

    // Helper to test if user is near bottom of chat
    function isNearBottom(threshold = 90) {
        if (!chatContainer) return true;
        return (chatContainer.scrollHeight - chatContainer.scrollTop - chatContainer.clientHeight) <= threshold;
    }

    function updateScrollState() {
        if (!chatContainer) return;
        const nearBottom = isNearBottom(100);
        autoScrollEnabled = nearBottom;
        if (scrollBottomBtn) {
            if (!nearBottom && chatContainer.scrollHeight > chatContainer.clientHeight + 100) {
                scrollBottomBtn.classList.add("visible");
            } else {
                scrollBottomBtn.classList.remove("visible");
            }
        }
    }

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
            chatHistory.forEach((item, index) => {
                if (item.type === "user") {
                    appendMsg("user", item.text, index);
                } else if (item.type === "bot") {
                    const row = appendMsg("bot", "", index);
                    const bubble = row.querySelector(".msg-bubble");
                    const textDiv = document.createElement("div");
                    textDiv.className = "markdown-body";
                    renderMarkdown(textDiv, item.text || "");
                    bubble.appendChild(textDiv);
                    renderExtras(bubble, item.sources || [], item.traceSteps || [], item.text || "");
                }
            });
            chatContainer.scrollTop = chatContainer.scrollHeight;
        } else {
            if (welcomeHero) {
                chatContainer.appendChild(welcomeHero);
            }
        }
        updateScrollState();
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

    let currentModel = "llama3.2:latest";

    function recordUserMessageInConversation(text) {
        if (!activeConvId) {
            activeConvId = "conv_" + Date.now();
        }

        let conv = conversations.find(c => c.id === activeConvId);
        const activeModelName = (modelSelect && modelSelect.value) ? modelSelect.value : currentModel;
        if (!conv) {
            conv = {
                id: activeConvId,
                title: text.slice(0, 36) + (text.length > 36 ? "..." : ""),
                timestamp: Date.now(),
                repo: (repoSelect && repoSelect.value) ? repoSelect.value : "",
                model: activeModelName,
                messages: []
            };
            conversations.unshift(conv);
        } else {
            conv.timestamp = Date.now();
            if (repoSelect && repoSelect.value) conv.repo = repoSelect.value;
            conv.model = activeModelName || conv.model;
            if (!conv.title || conv.title === "New Chat" || conv.messages.length === 0) {
                conv.title = text.slice(0, 36) + (text.length > 36 ? "..." : "");
            }
        }

        conv.messages.push({ type: "user", text: text });
        chatHistory = conv.messages;
        saveConversationsToMemory();
        renderConversationsList();
        return conv.messages.length - 1;
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
        // New Conversation Buttons (Sidebar & Navbar)
        if (newChatBtn) {
            newChatBtn.addEventListener("click", () => {
                startNewConversation(true);
            });
        }
        if (navNewChatBtn) {
            navNewChatBtn.addEventListener("click", () => {
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

        // Auto-expand textarea and Enter key submission
        if (queryInput) {
            queryInput.addEventListener("input", autoResizeTextarea);
            queryInput.addEventListener("keydown", (e) => {
                if (e.key === "Enter" && !e.shiftKey) {
                    e.preventDefault();
                    submitUserQuery();
                }
            });
        }

        // Send Button Click
        if (sendBtn) {
            sendBtn.addEventListener("click", (e) => {
                e.preventDefault();
                submitUserQuery();
            });
        }

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

        // Global shortcut Ctrl+B / Cmd+B to toggle sidebar, Ctrl+N / Alt+N to new chat
        document.addEventListener("keydown", (e) => {
            if ((e.ctrlKey || e.metaKey) && e.key.toLowerCase() === "b") {
                e.preventDefault();
                if (sidebarToggleBtn) sidebarToggleBtn.click();
            } else if (((e.ctrlKey || e.metaKey) && e.key.toLowerCase() === "n") || (e.altKey && e.key.toLowerCase() === "n")) {
                e.preventDefault();
                startNewConversation(true);
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

        // Model selection change (if element exists)
        if (modelSelect) {
            modelSelect.addEventListener("change", () => {
                localStorage.setItem("coderag_selected_model", modelSelect.value);
                currentModel = modelSelect.value;
            });
        }

        async function executeRepoDelete(targetRepo) {
            if (!targetRepo) return;

            const confirmed = window.confirm(`Are you sure you want to delete indexed repository "${targetRepo}"?\n\nThis will permanently remove its vector embeddings, BM25 search index, and cached files.`);
            if (!confirmed) return;

            if (deleteRepoBtn) {
                deleteRepoBtn.disabled = true;
                deleteRepoBtn.innerHTML = `<span class="spinner" style="width: 12px; height: 12px; border-width: 2px;"></span>`;
            }

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
                if (deleteRepoBtn) {
                    deleteRepoBtn.innerHTML = `
                        <svg width="15" height="15" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
                            <polyline points="3 6 5 6 21 6"></polyline>
                            <path d="M19 6v14a2 2 0 0 1-2 2H7a2 2 0 0 1-2-2V6m3 0V4a2 2 0 0 1 2-2h4a2 2 0 0 1 2 2v2"></path>
                            <line x1="10" y1="11" x2="10" y2="17"></line>
                            <line x1="14" y1="11" x2="14" y2="17"></line>
                        </svg>
                    `;
                }
                updateSelectedRepoUI();
            }
        }

        // Delete Repository Action (Toolbar Button)
        if (deleteRepoBtn) {
            deleteRepoBtn.addEventListener("click", async () => {
                const targetRepo = repoSelect.value;
                if (!targetRepo) return;
                await executeRepoDelete(targetRepo);
            });
        }

        // Indexed Repositories List Click Delegation
        if (indexedReposList) {
            indexedReposList.addEventListener("click", async (e) => {
                const delBtn = e.target.closest(".btn-delete-repo-item");
                if (delBtn) {
                    const targetRepo = delBtn.getAttribute("data-repo");
                    if (targetRepo) {
                        await executeRepoDelete(targetRepo);
                    }
                    return;
                }

                const nameBtn = e.target.closest(".indexed-repo-name-btn");
                if (nameBtn) {
                    const targetRepo = nameBtn.getAttribute("data-repo");
                    if (targetRepo && repoSelect) {
                        repoSelect.value = targetRepo;
                        localStorage.setItem("coderag_selected_repo", targetRepo);
                        updateSelectedRepoUI();
                        showToast(`Switched active repository to "${targetRepo}"`, "info", 1800);
                    }
                }
            });
        }

        // Chat Container Scroll Detection for Non-Intrusive Streaming
        if (chatContainer) {
            chatContainer.addEventListener("scroll", updateScrollState, { passive: true });
            chatContainer.addEventListener("wheel", () => {
                setTimeout(updateScrollState, 40);
            }, { passive: true });
            chatContainer.addEventListener("touchmove", () => {
                setTimeout(updateScrollState, 40);
            }, { passive: true });
        }

        // Scroll to Bottom Button Click
        if (scrollBottomBtn) {
            scrollBottomBtn.addEventListener("click", () => {
                autoScrollEnabled = true;
                if (chatContainer) {
                    chatContainer.scrollTo({
                        top: chatContainer.scrollHeight,
                        behavior: "smooth"
                    });
                }
                scrollBottomBtn.classList.remove("visible");
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

        // Sync active state in indexedReposList
        if (indexedReposList) {
            indexedReposList.querySelectorAll(".indexed-repo-item").forEach(item => {
                const btn = item.querySelector(".indexed-repo-name-btn");
                const repo = btn ? btn.getAttribute("data-repo") : null;
                if (repo === val) {
                    item.classList.add("active");
                    if (!item.querySelector(".indexed-repo-active-dot")) {
                        const dot = document.createElement("span");
                        dot.className = "indexed-repo-active-dot";
                        btn.insertBefore(dot, btn.firstChild);
                    }
                } else {
                    item.classList.remove("active");
                    const dot = item.querySelector(".indexed-repo-active-dot");
                    if (dot) dot.remove();
                }
            });
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
                currentModel = savedModel || data.active_model || (data.models && data.models[0]) || "llama3.2:latest";
                if (modelSelect) {
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
            }
        } catch {
            if (modelSelect) {
                modelSelect.innerHTML = '<option value="llama3.2:latest">llama3.2:latest</option>';
            }
        }
    }

    async function loadRepos() {
        try {
            const res = await fetch("/api/repos");
            const savedRepo = localStorage.getItem("coderag_selected_repo");
            if (res.ok) {
                const data = await res.json();
                repoSelect.innerHTML = "";
                if (indexedReposList) indexedReposList.innerHTML = "";

                if (data.repos && data.repos.length > 0) {
                    let hasSelected = false;
                    data.repos.forEach((r, idx) => {
                        const opt = document.createElement("option");
                        opt.value = r;
                        opt.textContent = r;
                        const isSelected = (savedRepo && r === savedRepo) || (!savedRepo && idx === 0);
                        if (savedRepo && r === savedRepo) {
                            opt.selected = true;
                            hasSelected = true;
                        } else if (!savedRepo && idx === 0) {
                            opt.selected = true;
                            hasSelected = true;
                        }
                        repoSelect.appendChild(opt);

                        // Populate indexed repository list item with instant delete button
                        if (indexedReposList) {
                            const item = document.createElement("div");
                            item.className = `indexed-repo-item ${isSelected ? "active" : ""}`;
                            item.innerHTML = `
                                <button type="button" class="indexed-repo-name-btn" data-repo="${escapeHtml(r)}" title="Switch to ${escapeHtml(r)}">
                                    ${isSelected ? '<span class="indexed-repo-active-dot"></span>' : ''}
                                    <span>${escapeHtml(r)}</span>
                                </button>
                                <button type="button" class="btn-delete-repo-item" data-repo="${escapeHtml(r)}" title="Delete repository ${escapeHtml(r)}">
                                    <svg width="13" height="13" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
                                        <polyline points="3 6 5 6 21 6"></polyline>
                                        <path d="M19 6v14a2 2 0 0 1-2 2H7a2 2 0 0 1-2-2V6m3 0V4a2 2 0 0 1 2-2h4a2 2 0 0 1 2 2v2"></path>
                                    </svg>
                                </button>
                            `;
                            indexedReposList.appendChild(item);
                        }
                    });
                    if (!hasSelected && data.repos.length > 0) {
                        repoSelect.options[0].selected = true;
                    }
                } else {
                    repoSelect.innerHTML = '<option value="">No repositories indexed</option>';
                    if (indexedReposList) {
                        indexedReposList.innerHTML = '<div style="font-size:0.75rem; color:var(--text-muted); padding:4px 0;">No indexed repositories</div>';
                    }
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
    if (queryForm) {
        queryForm.addEventListener("submit", (e) => {
            e.preventDefault();
            submitUserQuery();
        });
    }

    async function submitUserQuery() {
        if (isSubmitting) return;

        const q = queryInput.value.trim();
        if (!q) {
            queryInput.focus();
            return;
        }

        let repo = repoSelect ? repoSelect.value : "";
        if (!repo && repoSelect && repoSelect.options && repoSelect.options.length > 0) {
            for (let i = 0; i < repoSelect.options.length; i++) {
                if (repoSelect.options[i].value) {
                    repoSelect.selectedIndex = i;
                    repo = repoSelect.options[i].value;
                    updateSelectedRepoUI();
                    break;
                }
            }
        }

        if (!repo) {
            showToast("Please index or select a target repository first from the sidebar.", "warning", 3500);
            if (repoSourceInput) repoSourceInput.focus();
            return;
        }

        const model = (modelSelect && modelSelect.value) ? modelSelect.value : (currentModel || "llama3.2:latest");

        // Hide welcome hero when conversation begins
        if (welcomeHero && welcomeHero.parentNode === chatContainer) {
            welcomeHero.remove();
        }

        // Reset autoscroll on new user message
        autoScrollEnabled = true;

        // Persist User Message to conversation memory and append to UI
        const userMsgIndex = recordUserMessageInConversation(q);
        appendMsg("user", q, userMsgIndex);

        queryInput.value = "";
        queryInput.style.height = "auto";        const startTime = performance.now();
        let timerSeconds = 0;

        // Append Loading Bot Message with live counting timer
        const botMsg = appendMsg("bot", `
            <div class="loading-box">
                <span class="spinner" style="border-color: rgba(99, 102, 241, 0.2); border-top-color: var(--primary-color);"></span>
                <div class="stream-status-wrapper">
                    <span class="stream-status">Analyzing intent & searching AST symbols...</span>
                    <span class="stream-timer-badge">⏱️ <span class="stream-timer-val">0</span>s</span>
                </div>
            </div>
        `, userMsgIndex + 1);
        const contentEl = botMsg.querySelector(".msg-bubble");

        const timerInterval = setInterval(() => {
            timerSeconds++;
            const timerValEl = contentEl.querySelector(".stream-timer-val");
            if (timerValEl) {
                timerValEl.textContent = timerSeconds;
            }
        }, 1000);

        isSubmitting = true;
        if (sendBtn) sendBtn.disabled = true;

        let accumulatedAnswer = "";
        let sources = [];
        let traceSteps = [];
        let finalElapsedSec = null;

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
            let renderPending = false;
            let lastRenderTime = 0;

            const scheduleRender = (force = false) => {
                const now = performance.now();
                if (force || (now - lastRenderTime >= 30 && !renderPending)) {
                    renderPending = true;
                    requestAnimationFrame(() => {
                        if (textDiv) {
                            renderFastMarkdown(textDiv, accumulatedAnswer);
                            // Only auto-scroll if user has not scrolled up to read previous context
                            if (autoScrollEnabled && chatContainer) {
                                chatContainer.scrollTop = chatContainer.scrollHeight;
                            }
                        }
                        lastRenderTime = performance.now();
                        renderPending = false;
                    });
                }
            };

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
                                scheduleRender();
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

            finalElapsedSec = ((performance.now() - startTime) / 1000).toFixed(1);

            if (!textDiv && accumulatedAnswer) {
                contentEl.innerHTML = "";
                textDiv = document.createElement("div");
                textDiv.className = "markdown-body";
                contentEl.appendChild(textDiv);
            }

            if (textDiv) {
                renderMarkdown(textDiv, accumulatedAnswer);
            }

            renderExtras(contentEl, sources, traceSteps, accumulatedAnswer, finalElapsedSec);

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
            clearInterval(timerInterval);
            isSubmitting = false;
            if (sendBtn) sendBtn.disabled = false;
            if (autoScrollEnabled && chatContainer) {
                chatContainer.scrollTop = chatContainer.scrollHeight;
            }
        }      updateScrollState();
        }
    }

    // =========================================================================
    // Rendering & Helpers
    // =========================================================================
    function appendMsg(type, textOrHtml, msgIndex = null) {
        const row = document.createElement("div");
        row.className = `msg-row ${type}`;
        if (msgIndex !== null && msgIndex !== undefined) {
            row.dataset.msgIndex = msgIndex;
        }

        if (type === "user") {
            const rawText = textOrHtml;
            row.innerHTML = `
                <div class="msg-bubble">${escapeHtml(rawText)}</div>
                <div class="user-msg-actions">
                    <button type="button" class="btn-user-action btn-edit-user-msg" title="Edit question">
                        <svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
                            <path d="M11 4H4a2 2 0 0 0-2 2v14a2 2 0 0 0 2 2h14a2 2 0 0 0 2-2v-7"></path>
                            <path d="M18.5 2.5a2.121 2.121 0 0 1 3 3L12 15l-4 1 1-4 9.5-9.5z"></path>
                        </svg>
                        <span>Edit</span>
                    </button>
                    <button type="button" class="btn-user-action btn-delete-user-msg" title="Delete question & answer">
                        <svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
                            <polyline points="3 6 5 6 21 6"></polyline>
                            <path d="M19 6v14a2 2 0 0 1-2 2H7a2 2 0 0 1-2-2V6m3 0V4a2 2 0 0 1 2-2h4a2 2 0 0 1 2 2v2"></path>
                        </svg>
                        <span>Delete</span>
                    </button>
                    <button type="button" class="btn-user-action btn-copy-user-msg" title="Copy question">
                        <svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
                            <rect x="9" y="9" width="13" height="13" rx="2" ry="2"></rect>
                            <path d="M5 15H4a2 2 0 0 1-2-2V4a2 2 0 0 1 2-2h9a2 2 0 0 1 2 2v1"></path>
                        </svg>
                        <span>Copy</span>
                    </button>
                </div>
            `;

            const editBtn = row.querySelector(".btn-edit-user-msg");
            if (editBtn) {
                editBtn.addEventListener("click", () => {
                    const currentIdx = row.dataset.msgIndex !== undefined ? parseInt(row.dataset.msgIndex, 10) : msgIndex;
                    startInlineEdit(row, rawText, currentIdx);
                });
            }

            const deleteBtn = row.querySelector(".btn-delete-user-msg");
            if (deleteBtn) {
                deleteBtn.addEventListener("click", () => {
                    const currentIdx = row.dataset.msgIndex !== undefined ? parseInt(row.dataset.msgIndex, 10) : msgIndex;
                    deleteQuestionAt(currentIdx);
                });
            }

            const copyBtn = row.querySelector(".btn-copy-user-msg");
            if (copyBtn) {
                copyBtn.addEventListener("click", async () => {
                    await navigator.clipboard.writeText(rawText);
                    copyBtn.innerHTML = `
                        <svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.5" stroke-linecap="round" stroke-linejoin="round" style="color: var(--accent-emerald);">
                            <polyline points="20 6 9 17 4 12"></polyline>
                        </svg>
                        <span style="color: var(--accent-emerald);">Copied!</span>
                    `;
                    setTimeout(() => {
                        copyBtn.innerHTML = `
                            <svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
                                <rect x="9" y="9" width="13" height="13" rx="2" ry="2"></rect>
                                <path d="M5 15H4a2 2 0 0 1-2-2V4a2 2 0 0 1 2-2h9a2 2 0 0 1 2 2v1"></path>
                            </svg>
                            <span>Copy</span>
                        `;
                    }, 1800);
                });
            }
        } else {
            row.innerHTML = `<div class="msg-bubble">${textOrHtml}</div>`;
        }

        chatContainer.appendChild(row);
        if (type === "user" || autoScrollEnabled) {
            chatContainer.scrollTop = chatContainer.scrollHeight;
        }
        return row;
    }

    function deleteQuestionAt(index) {
        if (isSubmitting) {
            showToast("Please wait for the current query to finish", "warning");
            return;
        }

        let conv = conversations.find(c => c.id === activeConvId);
        if (!conv || !conv.messages || index === null || index === undefined || index >= conv.messages.length) return;

        // If next message is bot answer, delete both question and answer
        const isNextBot = (index + 1 < conv.messages.length) && conv.messages[index + 1].type === "bot";
        const deleteCount = isNextBot ? 2 : 1;

        conv.messages.splice(index, deleteCount);
        chatHistory = conv.messages;

        // If conversation is now empty, update title
        if (conv.messages.length === 0) {
            conv.title = "New Chat";
        } else {
            const firstUser = conv.messages.find(m => m.type === "user");
            if (firstUser) {
                conv.title = firstUser.text.slice(0, 36) + (firstUser.text.length > 36 ? "..." : "");
            }
        }

        saveConversationsToMemory();
        restoreActiveChatUI();
        renderConversationsList();
        showToast("Question deleted", "info", 2000);
    }

    function startInlineEdit(row, originalText, index) {
        if (isSubmitting) {
            showToast("Please wait for current response to finish before editing", "warning");
            return;
        }

        const bubble = row.querySelector(".msg-bubble");
        const actions = row.querySelector(".user-msg-actions");
        if (bubble) bubble.style.display = "none";
        if (actions) actions.style.display = "none";

        const editContainer = document.createElement("div");
        editContainer.className = "user-edit-container";
        editContainer.innerHTML = `
            <textarea class="user-edit-textarea" rows="2">${escapeHtml(originalText)}</textarea>
            <div class="user-edit-buttons">
                <button type="button" class="btn-edit-cancel">Cancel</button>
                <button type="button" class="btn-edit-save">Save & Resubmit</button>
            </div>
        `;

        row.appendChild(editContainer);
        const textarea = editContainer.querySelector(".user-edit-textarea");
        const cancelBtn = editContainer.querySelector(".btn-edit-cancel");
        const saveBtn = editContainer.querySelector(".btn-edit-save");

        textarea.focus();
        textarea.setSelectionRange(textarea.value.length, textarea.value.length);

        const cancel = () => {
            editContainer.remove();
            if (bubble) bubble.style.display = "";
            if (actions) actions.style.display = "";
        };

        cancelBtn.addEventListener("click", cancel);

        const submitEdit = () => {
            const newText = textarea.value.trim();
            if (!newText) {
                showToast("Question cannot be empty", "warning");
                textarea.focus();
                return;
            }

            editContainer.remove();
            resubmitEditedQuestion(index, newText);
        };

        saveBtn.addEventListener("click", submitEdit);
        textarea.addEventListener("keydown", (e) => {
            if (e.key === "Enter" && !e.shiftKey) {
                e.preventDefault();
                submitEdit();
            } else if (e.key === "Escape") {
                cancel();
            }
        });
    }

    async function resubmitEditedQuestion(index, newText) {
        let conv = conversations.find(c => c.id === activeConvId);
        if (!conv || !conv.messages) return;

        // Truncate from this user message index onward
        conv.messages = conv.messages.slice(0, index);
        chatHistory = conv.messages;
        saveConversationsToMemory();
        restoreActiveChatUI();

        // Submit the new text
        if (queryInput) {
            queryInput.value = newText;
        }
        await submitUserQuery();
    }

    const LANG_EXTENSION_MAP = {
        python: "py",
        py: "py",
        javascript: "js",
        js: "js",
        typescript: "ts",
        ts: "ts",
        tsx: "tsx",
        jsx: "jsx",
        html: "html",
        htm: "html",
        css: "css",
        scss: "scss",
        sass: "sass",
        less: "less",
        cpp: "cpp",
        "c++": "cpp",
        c: "c",
        h: "h",
        hpp: "hpp",
        cs: "cs",
        csharp: "cs",
        java: "java",
        kotlin: "kt",
        kt: "kt",
        swift: "swift",
        rust: "rs",
        rs: "rs",
        go: "go",
        golang: "go",
        ruby: "rb",
        rb: "rb",
        php: "php",
        sql: "sql",
        json: "json",
        yaml: "yaml",
        yml: "yaml",
        toml: "toml",
        xml: "xml",
        svg: "svg",
        bash: "sh",
        sh: "sh",
        shell: "sh",
        zsh: "sh",
        markdown: "md",
        md: "md",
        text: "txt",
        txt: "txt",
        dockerfile: "dockerfile",
        makefile: "makefile"
    };

    function getCodeBlockFilename(wrapper, lang) {
        const rawLang = (lang || "").toLowerCase().trim();
        const ext = LANG_EXTENSION_MAP[rawLang] || (rawLang.length <= 4 && rawLang.match(/^[a-z0-9]+$/) ? rawLang : "txt");

        // Look for preceding filename mentions in headers or text
        let prev = wrapper.previousElementSibling;
        let count = 0;
        while (prev && count < 5) {
            const text = prev.textContent || "";
            // Find patterns like `filename.ext` or path/to/file.ext
            const match = text.match(/`?([a-zA-Z0-9_\-\.\/]+\.([a-zA-Z0-9_]+))`?/);
            if (match && match[1]) {
                const rawName = match[1].trim().replace(/^`|`$/g, "");
                const parts = rawName.split("/").filter(Boolean);
                const baseName = parts[parts.length - 1];
                if (baseName && baseName.includes(".") && !baseName.endsWith(".")) {
                    return baseName;
                }
            }
            if (prev.tagName && prev.tagName.match(/^H[1-6]$/)) break;
            prev = prev.previousElementSibling;
            count++;
        }

        return `code.${ext}`;
    }

    function enhanceCodeBlocks(container) {
        if (!container) return;
        const pres = container.querySelectorAll("pre");
        pres.forEach((pre) => {
            if (pre.parentElement && pre.parentElement.classList.contains("code-block-wrapper")) {
                return;
            }

            const codeEl = pre.querySelector("code");
            let lang = "code";
            if (codeEl) {
                const match = codeEl.className.match(/language-([a-zA-Z0-9_-]+)/);
                if (match && match[1]) {
                    lang = match[1];
                }
            }

            const wrapper = document.createElement("div");
            wrapper.className = "code-block-wrapper";

            const header = document.createElement("div");
            header.className = "code-block-header";
            header.innerHTML = `
                <span class="code-lang-label">${escapeHtml(lang)}</span>
                <div class="code-block-actions">
                    <button type="button" class="btn-code-action btn-copy-code" title="Copy code">
                        <svg class="action-icon" width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
                            <rect x="9" y="9" width="13" height="13" rx="2" ry="2"></rect>
                            <path d="M5 15H4a2 2 0 0 1-2-2V4a2 2 0 0 1 2-2h9a2 2 0 0 1 2 2v1"></path>
                        </svg>
                        <span class="action-text">Copy</span>
                    </button>
                    <button type="button" class="btn-code-action btn-download-code" title="Download file">
                        <svg class="action-icon" width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
                            <path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4"></path>
                            <polyline points="7 10 12 15 17 10"></polyline>
                            <line x1="12" y1="15" x2="12" y2="3"></line>
                        </svg>
                        <span class="action-text">Download</span>
                    </button>
                </div>
            `;

            pre.parentNode.insertBefore(wrapper, pre);
            wrapper.appendChild(header);
            wrapper.appendChild(pre);
        });
    }

    // Delegated click handler for code block Copy and Download buttons
    document.addEventListener("click", async (e) => {
        // Copy Code Button
        const copyBtn = e.target.closest(".btn-copy-code");
        if (copyBtn) {
            const wrapper = copyBtn.closest(".code-block-wrapper");
            if (!wrapper) return;

            const codeEl = wrapper.querySelector("pre code") || wrapper.querySelector("pre");
            if (!codeEl) return;

            const codeText = codeEl.innerText || codeEl.textContent || "";
            
            try {
                await navigator.clipboard.writeText(codeText);
                copyBtn.classList.add("copied");
                copyBtn.innerHTML = `
                    <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.5" stroke-linecap="round" stroke-linejoin="round" style="color: var(--accent-emerald);">
                        <polyline points="20 6 9 17 4 12"></polyline>
                    </svg>
                    <span class="action-text" style="color: var(--accent-emerald);">Copied!</span>
                `;
                showToast("Code copied to clipboard", "success", 2000);
                setTimeout(() => {
                    copyBtn.classList.remove("copied");
                    copyBtn.innerHTML = `
                        <svg class="action-icon" width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
                            <rect x="9" y="9" width="13" height="13" rx="2" ry="2"></rect>
                            <path d="M5 15H4a2 2 0 0 1-2-2V4a2 2 0 0 1 2-2h9a2 2 0 0 1 2 2v1"></path>
                        </svg>
                        <span class="action-text">Copy</span>
                    `;
                }, 2000);
            } catch (err) {
                showToast("Failed to copy code", "error");
            }
            return;
        }

        // Download Code Button
        const downloadBtn = e.target.closest(".btn-download-code");
        if (downloadBtn) {
            const wrapper = downloadBtn.closest(".code-block-wrapper");
            if (!wrapper) return;

            const codeEl = wrapper.querySelector("pre code") || wrapper.querySelector("pre");
            if (!codeEl) return;

            const langEl = wrapper.querySelector(".code-lang-label");
            const lang = langEl ? langEl.textContent.trim() : "text";
            const filename = getCodeBlockFilename(wrapper, lang);

            const codeText = codeEl.innerText || codeEl.textContent || "";

            try {
                const blob = new Blob([codeText], { type: "text/plain;charset=utf-8" });
                const url = URL.createObjectURL(blob);
                const a = document.createElement("a");
                a.href = url;
                a.download = filename;
                document.body.appendChild(a);
                a.click();
                document.body.removeChild(a);
                URL.revokeObjectURL(url);

                downloadBtn.classList.add("success");
                downloadBtn.innerHTML = `
                    <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.5" stroke-linecap="round" stroke-linejoin="round" style="color: var(--accent-emerald);">
                        <polyline points="20 6 9 17 4 12"></polyline>
                    </svg>
                    <span class="action-text" style="color: var(--accent-emerald);">Downloaded!</span>
                `;
                showToast(`Downloaded ${filename}`, "success", 2500);

                setTimeout(() => {
                    downloadBtn.classList.remove("success");
                    downloadBtn.innerHTML = `
                        <svg class="action-icon" width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
                            <path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4"></path>
                            <polyline points="7 10 12 15 17 10"></polyline>
                            <line x1="12" y1="15" x2="12" y2="3"></line>
                        </svg>
                        <span class="action-text">Download</span>
                    `;
                }, 2200);
            } catch (err) {
                showToast("Failed to download file", "error");
            }
            return;
        }
    });

    function renderFastMarkdown(element, markdownText) {
        if (window.marked) {
            element.innerHTML = marked.parse(markdownText, { breaks: true, gfm: true });
        } else {
            element.innerHTML = formatMarkdownFallback(markdownText);
        }
        enhanceCodeBlocks(element);
    }

    function renderMarkdown(element, markdownText) {
        if (window.marked) {
            element.innerHTML = marked.parse(markdownText, { breaks: true, gfm: true });
        } else {
            element.innerHTML = formatMarkdownFallback(markdownText);
        }
        enhanceCodeBlocks(element);
        if (window.Prism) {
            Prism.highlightAllUnder(element);
        }
    }

    function renderExtras(contentEl, sources, traceSteps, text, elapsedTime) {
        // Actions Bar (Copy Answer & Time Badge)
        if (text && text.trim()) {
            const actionsDiv = document.createElement("div");
            actionsDiv.className = "msg-actions-bar";
            
            const copyBtn = document.createElement("button");
            copyBtn.type = "button";
            copyBtn.className = "btn-copy-msg";
            copyBtn.title = "Copy answer to clipboard";
            copyBtn.innerHTML = `
                <svg width="13" height="13" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
                    <rect x="9" y="9" width="13" height="13" rx="2" ry="2"></rect>
                    <path d="M5 15H4a2 2 0 0 1-2-2V4a2 2 0 0 1 2-2h9a2 2 0 0 1 2 2v1"></path>
                </svg>
                <span>Copy</span>
            `;

            copyBtn.addEventListener("click", async () => {
                try {
                    await navigator.clipboard.writeText(text);
                    copyBtn.innerHTML = `
                        <svg width="13" height="13" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.5" stroke-linecap="round" stroke-linejoin="round" style="color: var(--accent-emerald);">
                            <polyline points="20 6 9 17 4 12"></polyline>
                        </svg>
                        <span style="color: var(--accent-emerald);">Copied!</span>
                    `;
                    showToast("Answer copied to clipboard", "success", 2000);
                    setTimeout(() => {
                        copyBtn.innerHTML = `
                            <svg width="13" height="13" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
                                <rect x="9" y="9" width="13" height="13" rx="2" ry="2"></rect>
                                <path d="M5 15H4a2 2 0 0 1-2-2V4a2 2 0 0 1 2-2h9a2 2 0 0 1 2 2v1"></path>
                            </svg>
                            <span>Copy</span>
                        `;
                    }, 2200);
                } catch (e) {
                    showToast("Failed to copy", "error");
                }
            });

            actionsDiv.appendChild(copyBtn);

            if (elapsedTime) {
                const timeBadge = document.createElement("span");
                timeBadge.className = "time-elapsed-meta";
                timeBadge.title = `Completed in ${elapsedTime}s`;
                timeBadge.innerHTML = `
                    <svg width="11" height="11" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
                        <circle cx="12" cy="12" r="10"></circle>
                        <polyline points="12 6 12 12 16 14"></polyline>
                    </svg>
                    <span>${elapsedTime}s</span>
                `;
                actionsDiv.appendChild(timeBadge);
            }

            contentEl.appendChild(actionsDiv);
        }

        // Trace Section (Collapsible)
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

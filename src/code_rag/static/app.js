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
    const welcomeHero = document.getElementById("welcomeHero");
    const toastContainer = document.getElementById("toastContainer");
    const scrollBottomBtn = document.getElementById("scrollBottomBtn");

    const LOCAL_WORKSPACE_PATH = window.location.origin.includes("localhost")
        ? "/Users/macbookpro/Downloads/Github-repositories-read-RAG"
        : ".";

    let isSubmitting = false;
    let currentAbortController = null;
    let currentBotRow = null;
    let currentAccumulatedAnswer = "";
    let queryQueue = [];
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
        await loadConversationsFromBackend();
        await checkStatus();
        await loadModels();
        await loadRepos();
    }

    // =========================================================================
    // Custom Modal System (Confirm & Prompt Dialogs) & Project Guide Modal
    // =========================================================================
    const customModalOverlay = document.getElementById("customModalOverlay");
    const customModalCloseBtn = document.getElementById("customModalCloseBtn");
    const customModalIconWrap = document.getElementById("customModalIconWrap");
    const customModalTitle = document.getElementById("customModalTitle");
    const customModalDesc = document.getElementById("customModalDesc");
    const customModalInputWrap = document.getElementById("customModalInputWrap");
    const customModalInput = document.getElementById("customModalInput");
    const customModalCancelBtn = document.getElementById("customModalCancelBtn");
    const customModalConfirmBtn = document.getElementById("customModalConfirmBtn");

    const guideModalOverlay = document.getElementById("guideModalOverlay");
    const guideModalCloseBtn = document.getElementById("guideModalCloseBtn");
    const guideModalGotItBtn = document.getElementById("guideModalGotItBtn");
    const projectGuideBtn = document.getElementById("projectGuideBtn");

    function openGuideModal() {
        if (!guideModalOverlay) return;
        guideModalOverlay.style.display = "flex";
        requestAnimationFrame(() => {
            guideModalOverlay.classList.add("active");
        });
    }

    function closeGuideModal() {
        if (!guideModalOverlay) return;
        guideModalOverlay.classList.remove("active");
        setTimeout(() => {
            guideModalOverlay.style.display = "none";
        }, 180);
    }

    if (projectGuideBtn) {
        projectGuideBtn.addEventListener("click", openGuideModal);
    }
    if (guideModalCloseBtn) {
        guideModalCloseBtn.addEventListener("click", closeGuideModal);
    }
    if (guideModalGotItBtn) {
        guideModalGotItBtn.addEventListener("click", closeGuideModal);
    }
    if (guideModalOverlay) {
        guideModalOverlay.addEventListener("click", (e) => {
            if (e.target === guideModalOverlay) closeGuideModal();
        });
    }

    let activeModalResolver = null;

    function closeModal(result) {
        if (!customModalOverlay) return;
        customModalOverlay.classList.remove("active");
        setTimeout(() => {
            customModalOverlay.style.display = "none";
            if (activeModalResolver) {
                const resolver = activeModalResolver;
                activeModalResolver = null;
                resolver(result);
            }
        }, 180);
    }

    if (customModalCloseBtn) {
        customModalCloseBtn.addEventListener("click", () => closeModal(null));
    }
    if (customModalCancelBtn) {
        customModalCancelBtn.addEventListener("click", () => closeModal(null));
    }
    if (customModalOverlay) {
        customModalOverlay.addEventListener("click", (e) => {
            if (e.target === customModalOverlay) {
                closeModal(null);
            }
        });
    }

    document.addEventListener("keydown", (e) => {
        if (customModalOverlay && customModalOverlay.classList.contains("active")) {
            if (e.key === "Escape") {
                e.preventDefault();
                closeModal(null);
            } else if (e.key === "Enter" && !e.shiftKey) {
                if (document.activeElement === customModalCancelBtn) {
                    closeModal(null);
                } else {
                    e.preventDefault();
                    if (customModalConfirmBtn) customModalConfirmBtn.click();
                }
            }
        } else if (guideModalOverlay && guideModalOverlay.classList.contains("active")) {
            if (e.key === "Escape") {
                e.preventDefault();
                closeGuideModal();
            }
        }
    });

    function showConfirmModal({
        title = "Are you sure?",
        message = "",
        subtext = "",
        confirmText = "Confirm",
        cancelText = "Cancel",
        isDanger = false
    }) {
        return new Promise((resolve) => {
            activeModalResolver = resolve;

            customModalTitle.textContent = title;
            customModalDesc.textContent = subtext ? `${message}\n\n${subtext}` : message;

            customModalInputWrap.style.display = "none";

            if (isDanger) {
                customModalIconWrap.className = "custom-modal-icon-wrap danger";
                customModalIconWrap.innerHTML = `
                    <svg width="22" height="22" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
                        <polyline points="3 6 5 6 21 6"></polyline>
                        <path d="M19 6v14a2 2 0 0 1-2 2H7a2 2 0 0 1-2-2V6m3 0V4a2 2 0 0 1 2-2h4a2 2 0 0 1 2 2v2"></path>
                        <line x1="10" y1="11" x2="10" y2="17"></line>
                        <line x1="14" y1="11" x2="14" y2="17"></line>
                    </svg>
                `;
                customModalConfirmBtn.className = "btn-modal btn-modal-confirm danger";
            } else {
                customModalIconWrap.className = "custom-modal-icon-wrap primary";
                customModalIconWrap.innerHTML = `
                    <svg width="22" height="22" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
                        <circle cx="12" cy="12" r="10"></circle>
                        <line x1="12" y1="16" x2="12" y2="12"></line>
                        <line x1="12" y1="8" x2="12.01" y2="8"></line>
                    </svg>
                `;
                customModalConfirmBtn.className = "btn-modal btn-modal-confirm";
            }

            customModalConfirmBtn.textContent = confirmText;
            customModalCancelBtn.textContent = cancelText;

            customModalConfirmBtn.onclick = () => closeModal(true);
            customModalCancelBtn.onclick = () => closeModal(false);

            customModalOverlay.style.display = "flex";
            requestAnimationFrame(() => {
                customModalOverlay.classList.add("active");
                customModalConfirmBtn.focus();
            });
        });
    }

    function showPromptModal({
        title = "Prompt",
        message = "",
        defaultValue = "",
        placeholder = "",
        confirmText = "Save",
        cancelText = "Cancel"
    }) {
        return new Promise((resolve) => {
            activeModalResolver = resolve;

            customModalTitle.textContent = title;
            customModalDesc.textContent = message;

            customModalInputWrap.style.display = "block";
            customModalInput.value = defaultValue || "";
            customModalInput.placeholder = placeholder || "";

            customModalIconWrap.className = "custom-modal-icon-wrap primary";
            customModalIconWrap.innerHTML = `
                <svg width="22" height="22" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
                    <path d="M11 4H4a2 2 0 0 0-2 2v14a2 2 0 0 0 2 2h14a2 2 0 0 0 2-2v-7"></path>
                    <path d="M18.5 2.5a2.121 2.121 0 0 1 3 3L12 15l-4 1 1-4 9.5-9.5z"></path>
                </svg>
            `;
            customModalConfirmBtn.className = "btn-modal btn-modal-confirm";
            customModalConfirmBtn.textContent = confirmText;
            customModalCancelBtn.textContent = cancelText;

            customModalConfirmBtn.onclick = () => closeModal(customModalInput.value);
            customModalCancelBtn.onclick = () => closeModal(null);

            customModalOverlay.style.display = "flex";
            requestAnimationFrame(() => {
                customModalOverlay.classList.add("active");
                customModalInput.focus();
                customModalInput.select();
            });
        });
    }

    // =========================================================================
    // Permanent SQLite3 Conversation & Turn History Management
    // =========================================================================

    async function fetchConversationsList() {
        try {
            const res = await fetch("/api/conversations");
            if (res.ok) {
                const data = await res.json();
                conversations = data.conversations || [];
                renderConversationsList();
            }
        } catch (err) {
            console.warn("Unable to fetch conversations list from backend:", err);
        }
    }

    async function loadConversationsFromBackend() {
        try {
            await fetchConversationsList();
            const savedActiveId = localStorage.getItem("coderag_active_conv_id");
            if (savedActiveId && conversations.some(c => (c.conversation_id || c.id) === savedActiveId)) {
                activeConvId = savedActiveId;
                await loadConversationMessages(activeConvId);
            } else if (conversations.length > 0) {
                activeConvId = conversations[0].conversation_id || conversations[0].id;
                localStorage.setItem("coderag_active_conv_id", activeConvId);
                await loadConversationMessages(activeConvId);
            } else {
                startNewConversation(false);
            }
            renderConversationsList();
        } catch (err) {
            console.error("Error loading conversations from SQLite3 backend:", err);
            startNewConversation(false);
        }
    }

    async function loadConversationMessages(convId) {
        if (!convId) return;
        try {
            const res = await fetch(`/api/conversations/${encodeURIComponent(convId)}`);
            if (!res.ok) {
                if (res.status === 404) {
                    startNewConversation(false);
                }
                return;
            }
            const conv = await res.json();
            renderConversationMessages(conv.messages || []);
        } catch (err) {
            console.error("Error loading conversation messages from SQLite3 backend:", err);
        }
    }

    function renderConversationMessages(messages) {
        chatContainer.innerHTML = "";
        if (!messages || messages.length === 0) {
            if (welcomeHero) {
                chatContainer.appendChild(welcomeHero);
            }
            updateScrollState();
            return;
        }

        messages.forEach((m, idx) => {
            if (m.role === "user") {
                appendMsg("user", m.content, idx, m.turn_id);
            } else if (m.role === "assistant") {
                const row = appendMsg("bot", "", idx, m.turn_id);
                const bubble = row.querySelector(".msg-bubble");
                const textDiv = document.createElement("div");
                textDiv.className = "markdown-body";
                renderMarkdown(textDiv, m.content || "");
                bubble.appendChild(textDiv);
                renderExtras(bubble, m.sources || [], [], m.content || "", null);
            }
        });

        chatContainer.scrollTop = chatContainer.scrollHeight;
        updateScrollState();
    }

    function startNewConversation(shouldToast = true) {
        if (isSubmitting && currentAbortController) {
            currentAbortController.abort();
            currentAbortController = null;
            isSubmitting = false;
        }
        queryQueue = [];
        activeConvId = "conv_" + Date.now();
        localStorage.setItem("coderag_active_conv_id", activeConvId);
        chatContainer.innerHTML = "";
        if (welcomeHero) {
            chatContainer.appendChild(welcomeHero);
        }
        renderConversationsList();
        if (queryInput) {
            queryInput.value = "";
            queryInput.style.height = "auto";
            queryInput.focus();
        }
        if (chatContainer) {
            chatContainer.scrollTop = 0;
        }
        if (shouldToast) {
            showToast("New conversation started", "info", 2000);
        }
    }

    async function selectConversation(convId) {
        if (activeConvId === convId) return;
        if (isSubmitting && currentAbortController) {
            currentAbortController.abort();
            currentAbortController = null;
            isSubmitting = false;
        }
        queryQueue = [];
        activeConvId = convId;
        localStorage.setItem("coderag_active_conv_id", activeConvId);
        renderConversationsList();
        await loadConversationMessages(activeConvId);
    }

    async function deleteConversation(convId, e) {
        if (e) e.stopPropagation();
        const confirmed = await showConfirmModal({
            title: "Delete Conversation",
            message: "Are you sure you want to delete this conversation?",
            subtext: "This will permanently remove the conversation and all its Q&A messages from your database.",
            confirmText: "Delete",
            cancelText: "Cancel",
            isDanger: true
        });
        if (!confirmed) return;

        try {
            const res = await fetch(`/api/conversations/${encodeURIComponent(convId)}`, {
                method: "DELETE"
            });
            if (res.ok) {
                showToast("Conversation deleted", "info", 2000);
                if (activeConvId === convId) {
                    activeConvId = null;
                    localStorage.removeItem("coderag_active_conv_id");
                }
                await fetchConversationsList();
                if (!activeConvId) {
                    if (conversations.length > 0) {
                        selectConversation(conversations[0].conversation_id || conversations[0].id);
                    } else {
                        startNewConversation(false);
                    }
                }
            } else {
                showToast("Failed to delete conversation", "error");
            }
        } catch (err) {
            console.error("Error deleting conversation:", err);
            showToast("Error deleting conversation", "error");
        }
    }

    async function renameConversation(convId, currentTitle, e) {
        if (e) e.stopPropagation();
        const newTitle = await showPromptModal({
            title: "Rename Conversation",
            message: "Enter a new title for this conversation:",
            defaultValue: currentTitle,
            placeholder: "Conversation title",
            confirmText: "Save",
            cancelText: "Cancel"
        });
        if (!newTitle || !newTitle.trim() || newTitle.trim() === currentTitle) return;

        try {
            const res = await fetch(`/api/conversations/${encodeURIComponent(convId)}`, {
                method: "PATCH",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify({ title: newTitle.trim() })
            });
            if (res.ok) {
                showToast("Conversation renamed", "success", 2000);
                await fetchConversationsList();
            } else {
                showToast("Failed to rename conversation", "error");
            }
        } catch (err) {
            console.error("Error renaming conversation:", err);
            showToast("Error renaming conversation", "error");
        }
    }

    function renderConversationsList() {
        if (!conversationsList) return;
        conversationsList.innerHTML = "";

        if (!conversations || conversations.length === 0) {
            conversationsList.innerHTML = `<div class="conversations-empty">No previous chats yet</div>`;
            return;
        }

        conversations.forEach(c => {
            const cid = c.conversation_id || c.id;
            const item = document.createElement("div");
            item.className = `conv-item ${cid === activeConvId ? "active" : ""}`;
            const displayTitle = c.title || "New Chat";

            item.innerHTML = `
                <div class="conv-main">
                    <span class="conv-icon">
                        <svg width="13" height="13" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M21 15a2 2 0 0 1-2 2H7l-4 4V5a2 2 0 0 1 2-2h14a2 2 0 0 1 2 2z"></path></svg>
                    </span>
                    <span class="conv-title" title="${escapeHtml(displayTitle)}">${escapeHtml(displayTitle)}</span>
                </div>
                <div class="conv-actions">
                    <button type="button" class="btn-rename-conv" title="Rename conversation">
                        <svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
                            <path d="M11 4H4a2 2 0 0 0-2 2v14a2 2 0 0 0 2 2h14a2 2 0 0 0 2-2v-7"></path>
                            <path d="M18.5 2.5a2.121 2.121 0 0 1 3 3L12 15l-4 1 1-4 9.5-9.5z"></path>
                        </svg>
                    </button>
                    <button type="button" class="btn-delete-conv" title="Delete conversation">
                        <svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><polyline points="3 6 5 6 21 6"></polyline><path d="M19 6v14a2 2 0 0 1-2 2H7a2 2 0 0 1-2-2V6m3 0V4a2 2 0 0 1 2-2h4a2 2 0 0 1 2 2v2"></path></svg>
                    </button>
                </div>
            `;

            item.addEventListener("click", () => selectConversation(cid));
            const renBtn = item.querySelector(".btn-rename-conv");
            if (renBtn) {
                renBtn.addEventListener("click", (e) => renameConversation(cid, displayTitle, e));
            }
            const delBtn = item.querySelector(".btn-delete-conv");
            if (delBtn) {
                delBtn.addEventListener("click", (e) => deleteConversation(cid, e));
            }

            conversationsList.appendChild(item);
        });
    }

    let currentModel = "llama3.2:latest";

    // =========================================================================
    // Event Listeners & UI Helpers
    // =========================================================================
    function setupEventListeners() {
        // New Conversation Buttons (Sidebar & Navbar)
        if (newChatBtn) {
            newChatBtn.addEventListener("click", (e) => {
                e.preventDefault();
                startNewConversation(true);
            });
        }
        if (navNewChatBtn) {
            navNewChatBtn.addEventListener("click", (e) => {
                e.preventDefault();
                startNewConversation(true);
            });
        }

        // Delegated click listener for any new chat button
        document.addEventListener("click", (e) => {
            const btn = e.target.closest("#navNewChatBtn, #newChatBtn, .nav-new-chat-btn, .btn-new-chat");
            if (btn) {
                e.preventDefault();
                startNewConversation(true);
            }
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
                queryQueue = [];
                startNewConversation(true);
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

            const confirmed = await showConfirmModal({
                title: "Delete Indexed Repository",
                message: `Are you sure you want to delete indexed repository "${targetRepo}"?`,
                subtext: "This will permanently remove its vector embeddings, BM25 search index, and cached files.",
                confirmText: "Delete",
                cancelText: "Cancel",
                isDanger: true
            });
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
            if (deleteRepoBtn) {
                deleteRepoBtn.disabled = false;
                deleteRepoBtn.title = `Delete repository: ${val}`;
            }
        } else {
            inputRepoLabel.textContent = "No Repo Selected";
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

        // --- Progress animation ---
        const steps = [
            "Connecting to repository…",
            "Cloning repository…",
            "Parsing source files…",
            "Building vector index…",
            "Finalizing embeddings…"
        ];
        let stepIdx = 0;
        ingestAlert.className = "alert-box info indexing-progress";
        ingestAlert.innerHTML = `<span class="spinner" style="width:12px;height:12px;border-width:2px;display:inline-block;vertical-align:middle;margin-right:6px;"></span><span class="progress-step-text">${steps[0]}</span>`;
        ingestAlert.style.display = "block";

        const progressInterval = setInterval(() => {
            stepIdx = (stepIdx + 1) % steps.length;
            ingestAlert.innerHTML = `<span class="spinner" style="width:12px;height:12px;border-width:2px;display:inline-block;vertical-align:middle;margin-right:6px;"></span><span class="progress-step-text">${steps[stepIdx]}</span>`;
        }, 2200);

        ingestBtnText.textContent = "Indexing…";
        ingestSpinner.style.display = "inline-block";
        ingestBtn.disabled = true;

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
            clearInterval(progressInterval);

            if (res.ok) {
                // Show success in-place with stats
                ingestAlert.className = "alert-box success";
                ingestAlert.innerHTML = `
                    <strong>✅ Indexed successfully!</strong><br>
                    <span style="font-size:0.82rem;opacity:0.9;">
                        📁 ${data.total_files} files &nbsp;·&nbsp; 🧩 ${data.total_chunks} chunks &nbsp;·&nbsp; 📦 ${data.repo_name}
                    </span>`;
                ingestAlert.style.display = "block";

                // Save and update dropdown
                localStorage.setItem("coderag_selected_repo", data.repo_id);
                showToast(`✅ ${data.repo_name} indexed — ${data.total_chunks} chunks`, "success", 5000);

                // Clear input after a short delay so user sees what was indexed
                setTimeout(() => { repoSourceInput.value = ""; }, 1200);

                await loadRepos();

                // Auto-hide alert after 6 s
                setTimeout(() => {
                    ingestAlert.style.display = "none";
                }, 6000);
            } else {
                throw new Error(data.detail || "Indexing failed");
            }
        } catch (err) {
            clearInterval(progressInterval);
            ingestAlert.className = "alert-box error";
            ingestAlert.innerHTML = `<strong>❌ Error:</strong> ${escapeHtml(err.message)}`;
            ingestAlert.style.display = "block";
            showToast(err.message, "error");
        } finally {
            ingestBtnText.textContent = "Index Repository";
            ingestSpinner.style.display = "none";
            ingestBtn.disabled = false;
        }
    });

    // Query Submission & Streaming with Question Queueing
    if (queryForm) {
        queryForm.addEventListener("submit", (e) => {
            e.preventDefault();
            submitUserQuery();
        });
    }

    async function submitUserQuery(customQuestion = null, existingRowEl = null, existingMsgIndex = null) {
        // Terminate previous running question immediately — do not add to queue
        if (currentAbortController) {
            currentAbortController.abort();
            currentAbortController = null;
        }

        if (currentBotRow) {
            const loadingBox = currentBotRow.querySelector(".loading-box");
            if (loadingBox) loadingBox.remove();
            const bubble = currentBotRow.querySelector(".msg-bubble");
            if (bubble && !bubble.textContent.trim()) {
                bubble.innerHTML = `<span style="color: var(--text-muted); font-style: italic;">(Generation stopped)</span>`;
            }
            currentBotRow = null;
        }

        isSubmitting = false;
        queryQueue = [];

        const q = customQuestion !== null ? customQuestion : (queryInput ? queryInput.value.trim() : "");
        if (!q) {
            if (queryInput) queryInput.focus();
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

        let currentTurnId = "turn_" + Date.now() + "_" + Math.random().toString(36).substr(2, 6);
        let userMsgIndex;
        if (existingRowEl) {
            // Unmark queued state
            existingRowEl.classList.remove("msg-queued");
            const qHeader = existingRowEl.querySelector(".msg-queued-header");
            if (qHeader) qHeader.remove();
            userMsgIndex = existingMsgIndex !== null ? existingMsgIndex : parseInt(existingRowEl.dataset.msgIndex || "0", 10);
            if (existingRowEl.dataset.turnId) {
                currentTurnId = existingRowEl.dataset.turnId;
            } else {
                existingRowEl.dataset.turnId = currentTurnId;
            }
        } else {
            userMsgIndex = document.querySelectorAll(".msg-row.user").length;
            appendMsg("user", q, userMsgIndex, currentTurnId);
            if (queryInput) {
                queryInput.value = "";
                queryInput.style.height = "auto";
            }
        }

        const startTime = performance.now();

        // Append Loading Bot Message
        const botMsg = appendMsg("bot", `
            <div class="loading-box">
                <span class="spinner" style="border-color: rgba(99, 102, 241, 0.2); border-top-color: var(--primary-color);"></span>
                <span class="stream-status">Analyzing intent & searching AST symbols...</span>
            </div>
        `, userMsgIndex + 1, currentTurnId);
        const contentEl = botMsg.querySelector(".msg-bubble");

        isSubmitting = true;
        currentBotRow = botMsg;
        currentAccumulatedAnswer = "";

        const controller = new AbortController();
        currentAbortController = controller;

        let sources = [];
        let traceSteps = [];
        let finalElapsedSec = null;

        try {
            const res = await fetch("/api/query/stream", {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                signal: controller.signal,
                body: JSON.stringify({
                    repo_id: repo,
                    question: q,
                    model_name: model,
                    conversation_id: activeConvId || null,
                    turn_id: currentTurnId
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
                            renderFastMarkdown(textDiv, currentAccumulatedAnswer);
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
                            if (data.type === "meta") {
                                if (data.conversation_id) {
                                    activeConvId = data.conversation_id;
                                    localStorage.setItem("coderag_active_conv_id", activeConvId);
                                }
                                if (data.turn_id) {
                                    currentTurnId = data.turn_id;
                                    if (botMsg) botMsg.dataset.turnId = currentTurnId;
                                    const userRow = chatContainer.querySelector(`.msg-row.user:last-of-type`);
                                    if (userRow && !userRow.dataset.turnId) userRow.dataset.turnId = currentTurnId;
                                }
                            } else if (data.type === "trace") {
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
                                currentAccumulatedAnswer += data.content;
                                scheduleRender();
                            } else if (data.type === "done") {
                                sources = data.sources || [];
                                if (data.trace_steps) traceSteps = data.trace_steps;
                                fetchConversationsList();
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

            if (!textDiv && currentAccumulatedAnswer) {
                contentEl.innerHTML = "";
                textDiv = document.createElement("div");
                textDiv.className = "markdown-body";
                contentEl.appendChild(textDiv);
            }

            if (textDiv) {
                renderMarkdown(textDiv, currentAccumulatedAnswer);
            }

            renderExtras(contentEl, sources, traceSteps, currentAccumulatedAnswer, finalElapsedSec);
            fetchConversationsList();

        } catch (err) {
            if (err.name === "AbortError" || controller.signal.aborted) {
                console.info("Previous query stream terminated by user");
                return;
            }
            contentEl.innerHTML = `
                <div style="color: var(--accent-rose); display: flex; align-items: center; gap: 8px;">
                    <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><circle cx="12" cy="12" r="10"/><line x1="12" y1="8" x2="12" y2="12"/><line x1="12" y1="16" x2="12.01" y2="16"/></svg>
                    <span><strong>Error:</strong> ${escapeHtml(err.message)}</span>
                </div>
            `;
            showToast(err.message, "error");
        } finally {
            if (currentAbortController === controller) {
                currentAbortController = null;
                isSubmitting = false;
                currentBotRow = null;
            }
            if (autoScrollEnabled && chatContainer) {
                chatContainer.scrollTop = chatContainer.scrollHeight;
            }
            updateScrollState();
        }
    }

    // =========================================================================
    // Rendering & Helpers
    // =========================================================================
    function appendMsg(type, textOrHtml, msgIndex = null, turnId = null) {
        const row = document.createElement("div");
        row.className = `msg-row ${type}`;
        if (msgIndex !== null && msgIndex !== undefined) {
            row.dataset.msgIndex = msgIndex;
        }
        if (turnId) {
            row.dataset.turnId = turnId;
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
                    deleteTurnOrQuestion(row.dataset.turnId, currentIdx, row);
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

    async function deleteTurnOrQuestion(turnId, index, rowEl = null) {
        if (isSubmitting && currentAbortController) {
            if (currentBotRow && currentBotRow.dataset.turnId === turnId) {
                currentAbortController.abort();
                currentAbortController = null;
                isSubmitting = false;
            }
        }

        if (activeConvId && turnId) {
            try {
                const res = await fetch(`/api/conversations/${encodeURIComponent(activeConvId)}/turns/${encodeURIComponent(turnId)}`, {
                    method: "DELETE"
                });
                if (!res.ok) {
                    const err = await res.json().catch(() => ({}));
                    console.warn("Delete turn API returned error:", err);
                }
            } catch (apiErr) {
                console.error("Failed to delete turn via API:", apiErr);
            }
        }

        // Remove the user question element
        if (rowEl) {
            rowEl.remove();
        }

        // Also remove the corresponding bot answer element with matching turnId
        if (turnId) {
            const botEl = chatContainer.querySelector(`.msg-row.bot[data-turn-id="${turnId}"]`);
            if (botEl) {
                botEl.remove();
            }
        } else if (rowEl && rowEl.nextElementSibling && rowEl.nextElementSibling.classList.contains("bot")) {
            rowEl.nextElementSibling.remove();
        }

        // If no messages left in chatContainer, show welcomeHero
        const remainingMsgs = chatContainer.querySelectorAll(".msg-row");
        if (remainingMsgs.length === 0 && welcomeHero) {
            chatContainer.appendChild(welcomeHero);
        }

        updateScrollState();
        await fetchConversationsList();
        showToast("Question and corresponding AI answer deleted", "info", 2000);
    }

    function startInlineEdit(row, originalText, index) {
        if (isSubmitting && currentAbortController) {
            currentAbortController.abort();
            currentAbortController = null;
            isSubmitting = false;
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
            resubmitEditedQuestion(row, index, newText);
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

    async function resubmitEditedQuestion(row, index, newText) {
        if (row && row.dataset.turnId) {
            await deleteTurnOrQuestion(row.dataset.turnId, index, row);
        } else if (row) {
            row.remove();
        }

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

    function detectCodeLanguage(code) {
        if (!code || !code.trim()) return "text";
        const sample = code.slice(0, 1000);
        // HTML / XML
        if (/^\s*<(!DOCTYPE|html|div|span|p|head|body|script|style|svg|table|form|nav|header|button|input|h[1-6]|ul|li|section|article|main|aside)/i.test(sample) ||
            /<\/[a-zA-Z0-9_-]+>/i.test(sample) ||
            /<[a-zA-Z0-9_-]+[^>]*class=["'][^"']*["']/i.test(sample)) {
            return "markup";
        }
        // JSON
        if (/^\s*[\{\[][\s\S]*["'][\s\S]*:[\s\S]*[\}\]]\s*$/.test(sample.trim())) {
            return "json";
        }
        // Python
        if (/\b(def\s+[a-zA-Z_]\w*\s*\(|import\s+[a-zA-Z_]|from\s+[a-zA-Z_].*import|class\s+[a-zA-Z_]\w*(\s*\(|:)|elif\s+|if\s+__name__\s*==|print\s*\(|self\.[a-zA-Z_])/.test(sample)) {
            return "python";
        }
        // JavaScript / TypeScript
        if (/\b(const|let|var)\s+[a-zA-Z_$]\w*\s*=|function\s+[a-zA-Z_$]\w*\(|=>\s*\{|\bconsole\.(log|error|warn)|\bdocument\.query|async\s+function|export\s+(default|const)|\baddEventListener\b/.test(sample)) {
            return "javascript";
        }
        // CSS
        if (/^[.#]?[a-zA-Z0-9_-]+\s*\{[\s\S]*:[^;]+;[\s\S]*\}/m.test(sample) || /@(media|keyframes|import)\b/.test(sample)) {
            return "css";
        }
        // Bash / Shell
        if (/^\s*(#!\/bin\/bash|npm\s+(run|install)|pip\s+install|git\s+(commit|push|pull|clone)|cd\s+|sudo\s+|echo\s+)/m.test(sample)) {
            return "bash";
        }
        // SQL
        if (/\b(SELECT|INSERT\s+INTO|UPDATE|DELETE\s+FROM|CREATE\s+TABLE|ALTER\s+TABLE)\b/i.test(sample)) {
            return "sql";
        }
        return "javascript";
    }

    // Configure marked options
    if (window.marked && typeof marked.setOptions === "function") {
        try {
            marked.setOptions({ breaks: true, gfm: true });
        } catch (e) { }
    }

    function enhanceCodeBlocks(container) {
        if (!container) return;
        const pres = container.querySelectorAll("pre");
        pres.forEach((pre) => {
            if (pre.parentElement && pre.parentElement.classList.contains("code-block-wrapper")) {
                return;
            }

            const codeEl = pre.querySelector("code");
            let lang = "";
            if (codeEl) {
                const match = codeEl.className.match(/language-([a-zA-Z0-9_-]+)/);
                if (match && match[1]) {
                    lang = match[1];
                } else {
                    const detected = detectCodeLanguage(codeEl.textContent || "");
                    if (detected) {
                        lang = detected;
                        codeEl.classList.add(`language-${detected}`);
                    } else {
                        lang = "code";
                    }
                }
            } else {
                lang = detectCodeLanguage(pre.textContent || "") || "code";
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
        if (window.Prism) {
            try {
                Prism.highlightAllUnder(element);
            } catch (e) { }
        }
    }

    function renderMarkdown(element, markdownText) {
        if (window.marked) {
            element.innerHTML = marked.parse(markdownText, { breaks: true, gfm: true });
        } else {
            element.innerHTML = formatMarkdownFallback(markdownText);
        }
        enhanceCodeBlocks(element);
        if (window.Prism) {
            try {
                Prism.highlightAllUnder(element);
            } catch (e) { }
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

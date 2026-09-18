document.addEventListener("DOMContentLoaded", () => {
    const statusDot = document.getElementById("statusDot");
    const statusText = document.getElementById("statusText");
    const ingestForm = document.getElementById("ingestForm");
    const repoSourceInput = document.getElementById("repoSourceInput");
    const forceReindexCheckbox = document.getElementById("forceReindexCheckbox");
    const ingestBtn = document.getElementById("ingestBtn");
    const ingestBtnText = document.getElementById("ingestBtnText");
    const ingestSpinner = document.getElementById("ingestSpinner");
    const ingestAlert = document.getElementById("ingestAlert");
    const repoSelect = document.getElementById("repoSelect");
    const modelSelect = document.getElementById("modelSelect");
    const chatContainer = document.getElementById("chatContainer");
    const queryForm = document.getElementById("queryForm");
    const queryInput = document.getElementById("queryInput");
    const qButtons = document.querySelectorAll(".q-btn");

    init();

    async function init() {
        await checkStatus();
        await loadModels();
        await loadRepos();
    }

    qButtons.forEach(btn => {
        btn.addEventListener("click", () => {
            queryInput.value = btn.dataset.query;
            queryForm.dispatchEvent(new Event("submit"));
        });
    });

    async function checkStatus() {
        try {
            const res = await fetch("/api/health");
            if (res.ok) {
                const data = await res.json();
                statusDot.className = "status-dot online";
                statusText.textContent = `Online (${data.active_llm || "Ready"})`;
            } else {
                throw new Error();
            }
        } catch {
            statusDot.className = "status-dot offline";
            statusText.textContent = "Offline";
        }
    }

    async function loadModels() {
        try {
            const res = await fetch("/api/models");
            if (res.ok) {
                const data = await res.json();
                modelSelect.innerHTML = "";
                if (data.models && data.models.length > 0) {
                    data.models.forEach(m => {
                        const opt = document.createElement("option");
                        opt.value = m;
                        opt.textContent = m;
                        if (m === data.active_model) opt.selected = true;
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
            if (res.ok) {
                const data = await res.json();
                repoSelect.innerHTML = "";
                if (data.repos && data.repos.length > 0) {
                    data.repos.forEach((r, idx) => {
                        const opt = document.createElement("option");
                        opt.value = r;
                        opt.textContent = r;
                        if (idx === 0) opt.selected = true;
                        repoSelect.appendChild(opt);
                    });
                } else {
                    repoSelect.innerHTML = '<option value="">No repositories indexed</option>';
                }
            }
        } catch {}
    }

    ingestForm.addEventListener("submit", async (e) => {
        e.preventDefault();
        const source = repoSourceInput.value.trim();
        if (!source) return;

        ingestBtnText.textContent = "Indexing...";
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
                ingestAlert.className = "alert alert-success";
                ingestAlert.textContent = `Indexed ${data.total_chunks} chunks from ${data.total_files} files.`;
                ingestAlert.style.display = "block";
                repoSourceInput.value = "";
                await loadRepos();
            } else {
                throw new Error(data.detail || "Indexing failed");
            }
        } catch (err) {
            ingestAlert.className = "alert alert-error";
            ingestAlert.textContent = err.message;
            ingestAlert.style.display = "block";
        } finally {
            ingestBtnText.textContent = "Index Repository";
            ingestSpinner.style.display = "none";
            ingestBtn.disabled = false;
        }
    });

    queryForm.addEventListener("submit", async (e) => {
        e.preventDefault();
        const q = queryInput.value.trim();
        const repo = repoSelect.value;
        const model = modelSelect.value;

        if (!q) return;
        if (!repo) {
            alert("Please index and select a repository first.");
            return;
        }

        appendMsg("user", q);
        queryInput.value = "";

        const botMsg = appendMsg("bot", '<div class="loading-box" style="display:flex;align-items:center;gap:8px;color:#666055;"><span class="spinner"></span> <span class="stream-status">Analyzing query...</span></div>');
        const contentEl = botMsg.querySelector(".msg-content");

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
                throw new Error(errJson.detail || "Query failed");
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
                                    contentEl.appendChild(textDiv);
                                }
                                accumulatedAnswer += data.content;
                                textDiv.innerHTML = formatMarkdown(accumulatedAnswer);
                                chatContainer.scrollTop = chatContainer.scrollHeight;
                            } else if (data.type === "done") {
                                sources = data.sources || [];
                                if (data.trace_steps) traceSteps = data.trace_steps;
                            } else if (data.type === "error") {
                                throw new Error(data.error);
                            }
                        } catch (parseErr) {}
                    }
                }
            }

            if (!textDiv && accumulatedAnswer) {
                contentEl.innerHTML = "";
                textDiv = document.createElement("div");
                textDiv.innerHTML = formatMarkdown(accumulatedAnswer);
                contentEl.appendChild(textDiv);
            }

            renderExtras(contentEl, sources, traceSteps);

        } catch (err) {
            contentEl.innerHTML = `<p style="color:#B91C1C;">Error: ${escape(err.message)}</p>`;
        } finally {
            chatContainer.scrollTop = chatContainer.scrollHeight;
        }
    });

    function appendMsg(type, html) {
        const div = document.createElement("div");
        div.className = `msg ${type}`;
        div.innerHTML = `<div class="msg-content">${html}</div>`;
        chatContainer.appendChild(div);
        chatContainer.scrollTop = chatContainer.scrollHeight;
        return div;
    }

    function renderExtras(contentEl, sources, traceSteps) {
        if (sources && sources.length > 0) {
            const citeSec = document.createElement("div");
            citeSec.className = "citations";
            citeSec.innerHTML = `<div class="citations-title">Sources (${sources.length})</div>`;

            sources.forEach(s => {
                const card = document.createElement("div");
                card.className = "cite-card";
                card.innerHTML = `
                    <div class="cite-header">
                        <span class="cite-file">${escape(s.file_path)}</span>
                        <span class="cite-lines">Lines ${s.start_line}-${s.end_line}</span>
                    </div>
                    <pre><code class="language-${s.language || 'python'}">${escape(s.snippet || '')}</code></pre>
                `;
                citeSec.appendChild(card);
            });
            contentEl.appendChild(citeSec);
        }

        if (traceSteps && traceSteps.length > 0) {
            const traceSec = document.createElement("div");
            traceSec.className = "trace-sec";
            
            const btn = document.createElement("button");
            btn.className = "trace-toggle";
            btn.textContent = `▶ Execution trace (${traceSteps.length} steps)`;

            const list = document.createElement("div");
            list.className = "trace-content";
            list.style.display = "none";

            traceSteps.forEach(st => {
                const item = document.createElement("div");
                item.textContent = st;
                list.appendChild(item);
            });

            btn.addEventListener("click", () => {
                const isHidden = list.style.display === "none";
                list.style.display = isHidden ? "flex" : "none";
                btn.textContent = isHidden ? `▼ Execution trace (${traceSteps.length} steps)` : `▶ Execution trace (${traceSteps.length} steps)`;
            });

            traceSec.appendChild(btn);
            traceSec.appendChild(list);
            contentEl.appendChild(traceSec);
        }

        if (window.Prism) {
            Prism.highlightAllUnder(contentEl);
        }
    }

    function formatMarkdown(str) {
        if (!str) return "";
        let h = escape(str);
        h = h.replace(/\*\*(.*?)\*\*/g, '<strong>$1</strong>');
        h = h.replace(/`([^`]+)`/g, '<code style="background:#F1EEE8;color:#92400E;padding:2px 6px;border-radius:4px;font-family:monospace;border:1px solid #E2DDD2;">$1</code>');
        h = h.replace(/\n\n/g, '</p><p>');
        h = h.replace(/\n/g, '<br/>');
        return `<p>${h}</p>`;
    }

    function escape(s) {
        if (!s) return "";
        return s.replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;");
    }
});

const DEFAULT_TOP_K = 4;
const MOBILE_BREAKPOINT = 1180;
const UUID_PREFIX_PATTERN = /^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}[-_]/i;

let state = {
  activeSessionId: null,
  activeDraftTitle: "新对话",
  sessions: [],
  messages: [],
  citations: [],
  isStreaming: false,
  isUploading: false,
  leftOpen: window.innerWidth > MOBILE_BREAKPOINT,
  rightOpen: false,
  serviceStatusText: "服务状态检测中",
  streamStatusText: "准备就绪",
  documentCount: 0,
};

const elements = {
  appRoot: document.getElementById("app-root"),
  sessionList: document.getElementById("session-list"),
  messageThread: document.getElementById("message-thread"),
  sourcesList: document.getElementById("sources-list"),
  chatTitle: document.getElementById("chat-title"),
  streamStatus: document.getElementById("stream-status"),
  ingestFeedback: document.getElementById("ingest-feedback"),
  askForm: document.getElementById("ask-form"),
  askButton: document.getElementById("ask-button"),
  questionInput: document.getElementById("question-input"),
  uploadTrigger: document.getElementById("upload-trigger"),
  fileInput: document.getElementById("file-input"),
  newChatButton: document.getElementById("new-chat-button"),
  closeLeftButton: document.getElementById("close-left-button"),
  openLeftButton: document.getElementById("open-left-button"),
  closeRightButton: document.getElementById("close-right-button"),
  openRightButton: document.getElementById("open-right-button"),
};

function setState(patch) {
  state = {
    ...state,
    ...patch,
  };
}

function createSessionId() {
  if (window.crypto && typeof window.crypto.randomUUID === "function") {
    return window.crypto.randomUUID().replaceAll("-", "");
  }

  return `session-${Date.now()}-${Math.random().toString(16).slice(2)}`;
}

function formatTime(value) {
  if (!value) {
    return "刚刚";
  }

  return new Date(value).toLocaleString("zh-CN", {
    month: "2-digit",
    day: "2-digit",
    hour: "2-digit",
    minute: "2-digit",
  });
}

function escapeHtml(value) {
  return String(value)
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;")
    .replaceAll('"', "&quot;")
    .replaceAll("'", "&#39;");
}

function stripUuidPrefix(value) {
  return String(value || "").replace(UUID_PREFIX_PATTERN, "");
}

function formatSourceName(sourceName) {
  const normalized = stripUuidPrefix(sourceName).trim();
  return normalized || "Unknown Source";
}

function formatSourcePath(sourceUriOrPath, fallbackName) {
  const rawValue = String(sourceUriOrPath || "").trim();
  if (!rawValue) {
    return formatSourceName(fallbackName);
  }

  if (/^https?:\/\//i.test(rawValue)) {
    return rawValue;
  }

  const normalizedPath = rawValue.replaceAll("\\", "/");
  const fileName = normalizedPath.split("/").filter(Boolean).pop() || normalizedPath;
  return stripUuidPrefix(fileName).trim() || formatSourceName(fallbackName);
}

function buildCitationKey(citation) {
  return formatSourcePath(citation.source_uri_or_path, citation.source_name).toLowerCase();
}

function normalizeCitations(citations) {
  const uniqueCitations = [];
  const seenKeys = new Set();

  citations.forEach((citation) => {
    const key = buildCitationKey(citation);
    if (seenKeys.has(key)) {
      return;
    }

    seenKeys.add(key);
    uniqueCitations.push({
      ...citation,
      index: uniqueCitations.length + 1,
      source_name: formatSourceName(citation.source_name),
      source_uri_or_path: formatSourcePath(citation.source_uri_or_path, citation.source_name),
    });
  });

  return uniqueCitations;
}

function getActiveTitle() {
  if (!state.activeSessionId) {
    return "新对话";
  }

  const session = state.sessions.find((item) => item.session_id === state.activeSessionId);
  if (session) {
    return session.title;
  }

  const firstUserMessage = state.messages.find((message) => message.role === "user");
  if (firstUserMessage?.content?.trim()) {
    return firstUserMessage.content.trim().slice(0, 32);
  }

  return state.activeDraftTitle || "新对话";
}

function getDisplayedSessions() {
  const hasCurrentSession = state.sessions.some((session) => session.session_id === state.activeSessionId);
  if (!state.activeSessionId || hasCurrentSession) {
    return [...state.sessions];
  }

  return [
    {
      session_id: state.activeSessionId,
      title: getActiveTitle(),
      message_count: state.messages.length,
      updated_at: new Date().toISOString(),
    },
    ...state.sessions,
  ];
}

function extractLatestCitations(messages) {
  const assistantMessage = [...messages]
    .reverse()
    .find((message) => message.role === "assistant" && Array.isArray(message.citations) && message.citations.length);

  return assistantMessage ? normalizeCitations(assistantMessage.citations) : [];
}

function renderStatus() {
  elements.streamStatus.textContent = `${state.streamStatusText} · ${state.serviceStatusText} · 文档 ${state.documentCount}`;
}

function renderPanelState() {
  elements.appRoot.classList.toggle("left-collapsed", !state.leftOpen);
  elements.appRoot.classList.toggle("right-collapsed", !state.rightOpen);
  elements.openLeftButton.hidden = state.leftOpen;
  elements.openRightButton.hidden = state.rightOpen;
  elements.closeLeftButton.setAttribute("aria-expanded", String(state.leftOpen));
  elements.openLeftButton.setAttribute("aria-expanded", String(state.leftOpen));
  elements.closeRightButton.setAttribute("aria-expanded", String(state.rightOpen));
  elements.openRightButton.setAttribute("aria-expanded", String(state.rightOpen));
}

function renderSessions() {
  const sessions = getDisplayedSessions();
  elements.chatTitle.textContent = getActiveTitle();

  if (!sessions.length) {
    elements.sessionList.innerHTML = '<p class="panel-empty">暂无历史对话，点击“新建对话”开始。</p>';
    return;
  }

  elements.sessionList.innerHTML = sessions
    .map((session) => {
      const activeClass = session.session_id === state.activeSessionId ? "active" : "";
      return `
        <article class="session-card ${activeClass}">
          <button
            type="button"
            class="session-main"
            data-session-action="open"
            data-session-id="${escapeHtml(session.session_id)}"
            ${state.isStreaming ? "disabled" : ""}
          >
            <p class="session-card-title">${escapeHtml(session.title || "新对话")}</p>
            <p class="session-card-meta">
              <span>${session.message_count || 0} 条消息</span>
              <span>${escapeHtml(formatTime(session.updated_at))}</span>
            </p>
          </button>
          <div class="session-tools">
            <button
              type="button"
              class="session-tool"
              data-session-action="rename"
              data-session-id="${escapeHtml(session.session_id)}"
              data-session-title="${escapeHtml(session.title || "新对话")}" 
              aria-label="重命名对话"
              ${state.isStreaming ? "disabled" : ""}
            >
              <svg viewBox="0 0 24 24" aria-hidden="true"><path d="m12 20h9"/><path d="M16.5 3.5a2.1 2.1 0 0 1 3 3L7 19l-4 1 1-4Z"/></svg>
            </button>
            <button
              type="button"
              class="session-tool danger"
              data-session-action="delete"
              data-session-id="${escapeHtml(session.session_id)}"
              data-session-title="${escapeHtml(session.title || "新对话")}" 
              aria-label="删除对话"
              ${state.isStreaming ? "disabled" : ""}
            >
              <svg viewBox="0 0 24 24" aria-hidden="true"><path d="M3 6h18"/><path d="M8 6V4h8v2"/><path d="M6 6l1 14h10l1-14"/><path d="M10 11v6M14 11v6"/></svg>
            </button>
          </div>
        </article>
      `;
    })
    .join("");
}

function renderThread() {
  if (!state.messages.length) {
    elements.messageThread.innerHTML = `
      <div class="message-row assistant empty-thread">
        <article class="message-bubble">
          <div class="message-meta"><span>模型回答</span><span>新对话</span></div>
          上传工程文档后直接提问，我会基于检索到的内容作答，并在右侧按引用顺序展示来源。
        </article>
      </div>
    `;
    scrollThreadToBottom();
    return;
  }

  elements.messageThread.innerHTML = state.messages
    .map((message) => {
      const roleLabel = message.role === "user" ? "用户" : "模型回答";
      const groundedText = message.role === "assistant" && message.grounded === false ? "证据不足" : "";
      return `
        <div class="message-row ${escapeHtml(message.role)}">
          <article class="message-bubble">
            <div class="message-meta">
              <span>${roleLabel}</span>
              <span>${escapeHtml(groundedText || formatTime(message.created_at || message.createdAt))}</span>
            </div>
            <div>${escapeHtml(message.content || "")}</div>
          </article>
        </div>
      `;
    })
    .join("");

  scrollThreadToBottom();
}

function renderSources() {
  if (!state.citations.length) {
    elements.sourcesList.innerHTML = '<p class="panel-empty">当前回答暂无引用。生成带引用回答后，右侧栏会自动展开。</p>';
    return;
  }

  elements.sourcesList.innerHTML = state.citations
    .map((citation) => {
      const sectionText = citation.page_or_section
        ? `<span class="badge">${escapeHtml(citation.page_or_section)}</span>`
        : "";
      return `
        <article class="source-card">
          <div class="source-head">
            <h3 class="source-title">${escapeHtml(formatSourceName(citation.source_name))}</h3>
            <span class="source-index">[${escapeHtml(citation.index || "-")}]</span>
          </div>
          ${sectionText ? `<div class="badge-row">${sectionText}</div>` : ""}
          <p class="source-path">${escapeHtml(formatSourcePath(citation.source_uri_or_path, citation.source_name))}</p>
          <p class="source-snippet">${escapeHtml(citation.snippet || "")}</p>
        </article>
      `;
    })
    .join("");
}

function scrollThreadToBottom() {
  requestAnimationFrame(() => {
    elements.messageThread.scrollTop = elements.messageThread.scrollHeight;
  });
}

function resizeQuestionInput() {
  elements.questionInput.style.height = "auto";
  elements.questionInput.style.height = `${Math.min(elements.questionInput.scrollHeight, 168)}px`;
}

function setFeedback(message, isError = false) {
  elements.ingestFeedback.textContent = message;
  elements.ingestFeedback.style.color = isError ? "#b91c1c" : "#64748b";
}

function updateComposerDisabled() {
  const disabled = state.isStreaming || state.isUploading;
  elements.askButton.disabled = disabled;
  elements.uploadTrigger.disabled = disabled;
  elements.questionInput.disabled = disabled;
  elements.newChatButton.disabled = disabled;
}

function startNewConversation() {
  if (state.isStreaming) {
    return;
  }

  setState({
    activeSessionId: createSessionId(),
    activeDraftTitle: "新对话",
    messages: [],
    citations: [],
    rightOpen: false,
    streamStatusText: "已创建新对话",
  });
  renderPanelState();
  renderSessions();
  renderThread();
  renderSources();
  renderStatus();
  elements.questionInput.focus();
}

async function syncDraftTitleToServer() {
  if (state.activeDraftTitle === "新对话" || !state.activeSessionId) {
    return;
  }

  const response = await fetch(`/sessions/${encodeURIComponent(state.activeSessionId)}`, {
    method: "PATCH",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ title: state.activeDraftTitle }),
  });

  if (!response.ok) {
    throw new Error(`HTTP ${response.status}`);
  }

  setState({ activeDraftTitle: "新对话" });
}

async function renameSession(sessionId, currentTitle) {
  if (state.isStreaming) {
    return;
  }

  const nextTitle = window.prompt("请输入新的对话名称", currentTitle || "新对话");
  if (nextTitle === null) {
    return;
  }

  const normalizedTitle = nextTitle.trim().slice(0, 32);
  if (!normalizedTitle) {
    setFeedback("对话名称不能为空。", true);
    return;
  }

  const isDraftSession = !state.sessions.some((session) => session.session_id === sessionId);
  if (isDraftSession && sessionId === state.activeSessionId) {
    setState({ activeDraftTitle: normalizedTitle, streamStatusText: "对话已重命名" });
    renderSessions();
    renderStatus();
    return;
  }

  try {
    const response = await fetch(`/sessions/${encodeURIComponent(sessionId)}`, {
      method: "PATCH",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ title: normalizedTitle }),
    });
    if (!response.ok) {
      throw new Error(`HTTP ${response.status}`);
    }
    setFeedback("对话名称已更新。");
    setState({
      sessions: state.sessions.map((session) => (
        session.session_id === sessionId ? { ...session, title: normalizedTitle } : session
      )),
      streamStatusText: "对话已重命名",
    });
    renderSessions();
    renderStatus();
  } catch (error) {
    console.error("Failed to rename session:", error);
    setFeedback("重命名失败，请稍后重试。", true);
  }
}

async function deleteSession(sessionId, sessionTitle) {
  if (state.isStreaming) {
    return;
  }

  const confirmed = window.confirm(`确定删除对话“${sessionTitle || "新对话"}”吗？`);
  if (!confirmed) {
    return;
  }

  const isDraftSession = !state.sessions.some((session) => session.session_id === sessionId);
  if (isDraftSession) {
    startNewConversation();
    return;
  }

  try {
    const response = await fetch(`/sessions/${encodeURIComponent(sessionId)}`, {
      method: "DELETE",
    });
    if (!response.ok) {
      throw new Error(`HTTP ${response.status}`);
    }

    const remainingSessions = state.sessions.filter((session) => session.session_id !== sessionId);
    setState({ sessions: remainingSessions, streamStatusText: "对话已删除" });

    if (sessionId === state.activeSessionId) {
      if (remainingSessions.length) {
        await loadSessionMessages(remainingSessions[0].session_id);
      } else {
        startNewConversation();
      }
    } else {
      renderSessions();
      renderStatus();
    }
  } catch (error) {
    console.error("Failed to delete session:", error);
    setFeedback("删除失败，请稍后重试。", true);
  }
}

async function loadSessionMessages(sessionId) {
  if (state.isStreaming) {
    return;
  }

  try {
    setState({
      activeSessionId: sessionId,
      activeDraftTitle: "新对话",
      streamStatusText: "正在加载历史记录",
    });
    renderSessions();
    renderStatus();

    const response = await fetch(`/sessions/${encodeURIComponent(sessionId)}/messages`);
    if (!response.ok) {
      throw new Error(`HTTP ${response.status}`);
    }

    const payload = await response.json();
    const messages = Array.isArray(payload.messages) ? payload.messages : [];
    const citations = extractLatestCitations(messages);
    setState({
      messages,
      citations,
      rightOpen: citations.length ? true : state.rightOpen,
      streamStatusText: "历史记录已加载",
      leftOpen: window.innerWidth > MOBILE_BREAKPOINT ? state.leftOpen : false,
    });
    renderPanelState();
    renderSessions();
    renderThread();
    renderSources();
    renderStatus();
  } catch (error) {
    console.error("Failed to load session messages:", error);
    setState({ streamStatusText: "历史记录加载失败" });
    renderStatus();
  }
}

async function fetchHealth() {
  try {
    const response = await fetch("/health");
    if (!response.ok) {
      throw new Error(`HTTP ${response.status}`);
    }
    const payload = await response.json();
    setState({ serviceStatusText: payload.status === "ok" ? "服务在线" : "服务异常" });
  } catch (error) {
    console.error("Failed to fetch health:", error);
    setState({ serviceStatusText: "服务不可用" });
  }
  renderStatus();
}

async function fetchDocuments() {
  try {
    const response = await fetch("/documents");
    if (!response.ok) {
      throw new Error(`HTTP ${response.status}`);
    }
    const payload = await response.json();
    const documents = Array.isArray(payload.documents) ? payload.documents : [];
    setState({ documentCount: documents.length });
  } catch (error) {
    console.error("Failed to fetch documents:", error);
    setState({ documentCount: 0 });
  }
  renderStatus();
}

async function refreshSessions() {
  try {
    const response = await fetch("/sessions");
    if (!response.ok) {
      throw new Error(`HTTP ${response.status}`);
    }
    const payload = await response.json();
    setState({ sessions: Array.isArray(payload.sessions) ? payload.sessions : [] });
  } catch (error) {
    console.error("Failed to fetch sessions:", error);
    setState({ sessions: [] });
  }
  renderSessions();
}

async function handleFileUpload() {
  const files = Array.from(elements.fileInput.files || []);
  if (!files.length) {
    return;
  }

  const formData = new FormData();
  files.forEach((file) => formData.append("files", file));

  try {
    setState({ isUploading: true, streamStatusText: "正在导入文件" });
    setFeedback(`正在上传 ${files.length} 个文件...`);
    updateComposerDisabled();
    renderStatus();

    const response = await fetch("/ingest/files", {
      method: "POST",
      body: formData,
    });
    if (!response.ok) {
      throw new Error(`HTTP ${response.status}`);
    }

    const payload = await response.json();
    const uploadedNames = (payload.results || [])
      .map((item) => formatSourceName(item.source_name))
      .filter(Boolean)
      .join("、");
    setFeedback(uploadedNames ? `导入成功：${uploadedNames}` : "文件导入成功");
    setState({ streamStatusText: "文件导入完成" });
    await fetchDocuments();
  } catch (error) {
    console.error("Failed to upload files:", error);
    setFeedback("文件导入失败，请检查格式后重试。", true);
    setState({ streamStatusText: "文件导入失败" });
    renderStatus();
  } finally {
    elements.fileInput.value = "";
    setState({ isUploading: false });
    updateComposerDisabled();
  }
}

function appendUserMessage(content) {
  const nextMessages = [
    ...state.messages,
    {
      session_id: state.activeSessionId,
      role: "user",
      content,
      citations: [],
      createdAt: new Date().toISOString(),
    },
    {
      session_id: state.activeSessionId,
      role: "assistant",
      content: "",
      citations: [],
      grounded: null,
      createdAt: new Date().toISOString(),
    },
  ];

  setState({
    messages: nextMessages,
    citations: [],
    streamStatusText: "正在检索并生成回答",
    rightOpen: false,
  });
  renderPanelState();
  renderSessions();
  renderThread();
  renderSources();
  renderStatus();
}

function updateStreamingAnswer(token) {
  if (!state.messages.length) {
    return;
  }

  const lastIndex = state.messages.length - 1;
  const lastMessage = state.messages[lastIndex];
  if (lastMessage.role !== "assistant") {
    return;
  }

  const nextMessages = state.messages.map((message, index) => (
    index === lastIndex
      ? { ...message, content: `${message.content || ""}${token || ""}` }
      : message
  ));

  setState({ messages: nextMessages });
  renderThread();
}

function finalizeAssistantMessage(payload) {
  if (!state.messages.length) {
    return;
  }

  const lastIndex = state.messages.length - 1;
  const nextMessages = state.messages.map((message, index) => (
    index === lastIndex && message.role === "assistant"
      ? { ...message, grounded: Boolean(payload.grounded), citations: [...state.citations] }
      : message
  ));

  setState({
    messages: nextMessages,
    activeSessionId: payload.session_id || state.activeSessionId,
    streamStatusText: payload.grounded ? "回答已完成，引用已更新" : "回答完成，但当前证据不足",
  });
  renderSessions();
  renderThread();
  renderStatus();
}

function processSseMessage(rawMessage) {
  const lines = rawMessage.split("\n");
  let eventName = "message";
  let dataText = "";

  lines.forEach((line) => {
    if (line.startsWith("event:")) {
      eventName = line.slice(6).trim();
    }
    if (line.startsWith("data:")) {
      dataText += line.slice(5).trimStart();
    }
  });

  if (!dataText) {
    return;
  }

  const payload = JSON.parse(dataText);
  if (eventName === "token") {
    updateStreamingAnswer(payload.text || "");
    return;
  }

  if (eventName === "sources") {
    const citations = normalizeCitations(Array.isArray(payload.citations) ? payload.citations : []);
    setState({
      citations,
      rightOpen: citations.length ? true : state.rightOpen,
    });
    renderPanelState();
    renderSources();
    return;
  }

  if (eventName === "done") {
    finalizeAssistantMessage(payload);
  }
}

async function streamAnswer(question) {
  const sessionId = state.activeSessionId || createSessionId();
  setState({ activeSessionId: sessionId, isStreaming: true });
  updateComposerDisabled();
  appendUserMessage(question);

  try {
    const response = await fetch("/ask/stream", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        question,
        session_id: sessionId,
        top_k: DEFAULT_TOP_K,
      }),
    });

    if (!response.ok || !response.body) {
      throw new Error(`HTTP ${response.status}`);
    }

    const reader = response.body.getReader();
    const decoder = new TextDecoder("utf-8");
    let buffer = "";

    while (true) {
      const { value, done } = await reader.read();
      if (done) {
        break;
      }

      buffer += decoder.decode(value, { stream: true });
      const chunks = buffer.split("\n\n");
      buffer = chunks.pop() || "";
      chunks.forEach((chunk) => processSseMessage(chunk));
    }

    buffer += decoder.decode();
    if (buffer.trim()) {
      processSseMessage(buffer);
    }

    await syncDraftTitleToServer();
    await refreshSessions();
  } catch (error) {
    console.error("Failed to stream answer:", error);
    const lastIndex = state.messages.length - 1;
    const nextMessages = state.messages.map((message, index) => (
      index === lastIndex && message.role === "assistant"
        ? { ...message, content: "请求失败，请稍后重试。", grounded: false, citations: [] }
        : message
    ));
    setState({
      messages: nextMessages,
      citations: [],
      streamStatusText: "请求失败",
    });
    renderThread();
    renderSources();
    renderStatus();
  } finally {
    setState({ isStreaming: false });
    updateComposerDisabled();
  }
}

function toggleLeftPanel(isOpen) {
  setState({ leftOpen: isOpen });
  renderPanelState();
}

function toggleRightPanel(isOpen) {
  setState({ rightOpen: isOpen });
  renderPanelState();
}

async function bootstrap() {
  renderPanelState();
  renderThread();
  renderSources();
  renderSessions();
  renderStatus();
  setFeedback("可通过左下角“+”上传文件，再在输入框提问。");

  await Promise.all([fetchHealth(), fetchDocuments(), refreshSessions()]);

  if (state.sessions.length) {
    await loadSessionMessages(state.sessions[0].session_id);
  } else {
    startNewConversation();
  }
}

elements.newChatButton.addEventListener("click", startNewConversation);
elements.closeLeftButton.addEventListener("click", () => toggleLeftPanel(false));
elements.openLeftButton.addEventListener("click", () => toggleLeftPanel(true));
elements.closeRightButton.addEventListener("click", () => toggleRightPanel(false));
elements.openRightButton.addEventListener("click", () => toggleRightPanel(true));

elements.sessionList.addEventListener("click", (event) => {
  const actionButton = event.target.closest("[data-session-action]");
  if (!actionButton) {
    return;
  }

  const action = actionButton.getAttribute("data-session-action");
  const sessionId = actionButton.getAttribute("data-session-id");
  const sessionTitle = actionButton.getAttribute("data-session-title") || "新对话";
  if (!action || !sessionId) {
    return;
  }

  if (action === "open") {
    void loadSessionMessages(sessionId);
    return;
  }

  if (action === "rename") {
    void renameSession(sessionId, sessionTitle);
    return;
  }

  if (action === "delete") {
    void deleteSession(sessionId, sessionTitle);
  }
});

elements.uploadTrigger.addEventListener("click", () => elements.fileInput.click());
elements.fileInput.addEventListener("change", () => {
  void handleFileUpload();
});

elements.questionInput.addEventListener("input", resizeQuestionInput);
elements.questionInput.addEventListener("keydown", (event) => {
  if (event.key !== "Enter" || event.shiftKey) {
    return;
  }

  event.preventDefault();
  if (!state.isStreaming && !state.isUploading) {
    elements.askForm.requestSubmit();
  }
});

elements.askForm.addEventListener("submit", (event) => {
  event.preventDefault();
  const question = elements.questionInput.value.trim();
  if (!question) {
    setState({ streamStatusText: "请输入问题后再发送" });
    renderStatus();
    return;
  }

  elements.questionInput.value = "";
  resizeQuestionInput();
  void streamAnswer(question);
});

window.addEventListener("resize", () => {
  if (window.innerWidth > MOBILE_BREAKPOINT) {
    return;
  }

  renderPanelState();
});

void bootstrap();

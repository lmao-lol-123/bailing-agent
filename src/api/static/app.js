const state = {
  answer: "",
  citations: [],
};

const elements = {
  healthStatus: document.getElementById("health-status"),
  documentCount: document.getElementById("document-count"),
  documentsList: document.getElementById("documents-list"),
  ingestFeedback: document.getElementById("ingest-feedback"),
  uploadForm: document.getElementById("upload-form"),
  fileInput: document.getElementById("file-input"),
  urlForm: document.getElementById("url-form"),
  urlInput: document.getElementById("url-input"),
  refreshDocuments: document.getElementById("refresh-documents"),
  askForm: document.getElementById("ask-form"),
  askButton: document.getElementById("ask-button"),
  questionInput: document.getElementById("question-input"),
  topKInput: document.getElementById("top-k-input"),
  streamStatus: document.getElementById("stream-status"),
  answerOutput: document.getElementById("answer-output"),
  sourcesList: document.getElementById("sources-list"),
};

function setFeedback(message, isError = false) {
  elements.ingestFeedback.textContent = message;
  elements.ingestFeedback.style.background = isError ? "rgba(239, 131, 84, 0.18)" : "rgba(0, 109, 119, 0.12)";
}

function renderDocuments(documents) {
  elements.documentCount.textContent = String(documents.length);

  if (!documents.length) {
    elements.documentsList.innerHTML = '<p class="placeholder">暂无文档</p>';
    return;
  }

  elements.documentsList.innerHTML = documents.map((document) => `
    <article class="document-item">
      <h3>${escapeHtml(document.source_name)}</h3>
      <div class="doc-meta">
        <span class="badge">${escapeHtml(document.source_type)}</span>
        <span class="badge">${document.chunk_count} chunks</span>
      </div>
      <p class="muted">${escapeHtml(document.source_uri_or_path)}</p>
      ${document.page_or_sections.length ? `<p class="muted">Sections: ${escapeHtml(document.page_or_sections.join(", "))}</p>` : ""}
    </article>
  `).join("");
}

function renderSources(citations) {
  if (!citations.length) {
    elements.sourcesList.innerHTML = '<p class="placeholder">回答完成后显示引用片段。</p>';
    return;
  }

  elements.sourcesList.innerHTML = citations.map((citation) => `
    <article class="source-item">
      <h3>[${citation.index}] ${escapeHtml(citation.source_name)}</h3>
      <div class="source-meta">
        <span class="badge">${escapeHtml(citation.doc_id)}</span>
        ${citation.page_or_section ? `<span class="badge">${escapeHtml(citation.page_or_section)}</span>` : ""}
      </div>
      <p class="muted">${escapeHtml(citation.source_uri_or_path)}</p>
      <p class="snippet">${escapeHtml(citation.snippet)}</p>
    </article>
  `).join("");
}

function resetAnswer() {
  state.answer = "";
  state.citations = [];
  elements.answerOutput.innerHTML = '<p class="placeholder">模型正在准备回答。</p>';
  renderSources([]);
}

function renderAnswer() {
  if (!state.answer) {
    elements.answerOutput.innerHTML = '<p class="placeholder">模型正在准备回答。</p>';
    return;
  }
  elements.answerOutput.textContent = state.answer;
}

async function fetchHealth() {
  try {
    const response = await fetch("/health");
    const data = await response.json();
    elements.healthStatus.textContent = data.status === "ok" ? "在线" : "异常";
  } catch {
    elements.healthStatus.textContent = "不可达";
  }
}

async function fetchDocuments() {
  try {
    const response = await fetch("/documents");
    const data = await response.json();
    renderDocuments(data.documents || []);
  } catch {
    renderDocuments([]);
  }
}

async function handleUpload(event) {
  event.preventDefault();
  const files = elements.fileInput.files;
  if (!files.length) {
    setFeedback("请先选择至少一个文件。", true);
    return;
  }

  const formData = new FormData();
  Array.from(files).forEach((file) => formData.append("files", file));

  try {
    setFeedback("正在上传并导入文件...");
    toggleIngestForms(true);
    const response = await fetch("/ingest/files", { method: "POST", body: formData });
    if (!response.ok) {
      throw new Error("upload failed");
    }
    const data = await response.json();
    const names = data.results.map((item) => item.source_name).join("，");
    setFeedback(`导入完成：${names}`);
    elements.uploadForm.reset();
    await fetchDocuments();
  } catch {
    setFeedback("文件导入失败，请检查服务日志。", true);
  } finally {
    toggleIngestForms(false);
  }
}

async function handleUrlIngest(event) {
  event.preventDefault();
  const url = elements.urlInput.value.trim();
  if (!url) {
    setFeedback("请先输入 URL。", true);
    return;
  }

  try {
    setFeedback("正在抓取网页并导入...");
    toggleIngestForms(true);
    const response = await fetch("/ingest/url", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ url }),
    });
    if (!response.ok) {
      throw new Error("url ingest failed");
    }
    const data = await response.json();
    setFeedback(`导入完成：${data.results[0].source_name}`);
    elements.urlForm.reset();
    await fetchDocuments();
  } catch {
    setFeedback("网页导入失败，请确认 URL 可访问。", true);
  } finally {
    toggleIngestForms(false);
  }
}

async function handleAsk(event) {
  event.preventDefault();
  const question = elements.questionInput.value.trim();
  const topK = Number(elements.topKInput.value);

  if (!question) {
    elements.streamStatus.textContent = "请输入问题";
    return;
  }

  resetAnswer();
  elements.streamStatus.textContent = "检索与生成中";
  elements.askButton.disabled = true;

  try {
    const response = await fetch("/ask/stream", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ question, top_k: topK || 4 }),
    });
    if (!response.ok || !response.body) {
      throw new Error("stream failed");
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
      const messages = buffer.split("\n\n");
      buffer = messages.pop() || "";

      messages.forEach((message) => handleSseMessage(message));
    }
  } catch {
    elements.streamStatus.textContent = "问答失败";
  } finally {
    elements.askButton.disabled = false;
  }
}

function handleSseMessage(message) {
  const lines = message.split("\n");
  let eventName = "message";
  let data = "";

  lines.forEach((line) => {
    if (line.startsWith("event:")) {
      eventName = line.slice(6).trim();
    }
    if (line.startsWith("data:")) {
      data += line.slice(5).trim();
    }
  });

  if (!data) {
    return;
  }

  const payload = JSON.parse(data);

  if (eventName === "token") {
    state.answer += payload.text || "";
    renderAnswer();
    return;
  }

  if (eventName === "sources") {
    state.citations = payload.citations || [];
    renderSources(state.citations);
    return;
  }

  if (eventName === "done") {
    elements.streamStatus.textContent = payload.grounded ? "回答完成" : "未找到可靠依据";
  }
}

function toggleIngestForms(disabled) {
  [...elements.uploadForm.elements, ...elements.urlForm.elements].forEach((control) => {
    control.disabled = disabled;
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

elements.uploadForm.addEventListener("submit", handleUpload);
elements.urlForm.addEventListener("submit", handleUrlIngest);
elements.askForm.addEventListener("submit", handleAsk);
elements.refreshDocuments.addEventListener("click", fetchDocuments);

fetchHealth();
fetchDocuments();
setFeedback("前端已就绪，可以开始导入资料。");


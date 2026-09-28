"use strict";

/* =========================================================
   DocLens AI — Frontend
   Backend: Flask API with dynamic endpoint resolution
   ========================================================= */

/* -------------------------
   Configuration & Endpoints
------------------------- */

function resolveInitialApiBaseUrl() {
  if (window.location.protocol.startsWith("http")) {
    // If served from Flask backend on port 5000, use same-origin relative paths
    if (window.location.port === "5000") {
      return window.location.origin;
    }
  }
  // Default to IPv4 loopback
  return "http://127.0.0.1:5000";
}

let API_BASE_URL = resolveInitialApiBaseUrl();

function getEndpoint(path) {
  return `${API_BASE_URL}${path}`;
}

/* -------------------------
   DOM elements
------------------------- */

const uploadZone = document.getElementById("uploadZone");
const pdfInput = document.getElementById("pdfInput");
const chooseFileButton = document.getElementById("chooseFileButton");

const uploadStatus = document.getElementById("uploadStatus");
const documentInfo = document.getElementById("documentInfo");
const documentFilename = document.getElementById("documentFilename");
const documentStats = document.getElementById("documentStats");
const documentIdElement = document.getElementById("documentId");

const backendStatus = document.getElementById("backendStatus");
const backendStatusText = document.getElementById("backendStatusText");

const chatMessages = document.getElementById("chatMessages");
const emptyState = document.getElementById("emptyState");

const questionForm = document.getElementById("questionForm");
const questionInput = document.getElementById("questionInput");
const askButton = document.getElementById("askButton");

/* -------------------------
   Application state
------------------------- */

const appState = {
  selectedDocument: null,
  isUploading: false,
  isAsking: false,
  isBackendConnected: false
};

/* -------------------------
   Initial DOM validation
------------------------- */

const requiredElements = {
  pdfInput,
  chooseFileButton,
  uploadStatus,
  documentInfo,
  documentFilename,
  documentStats,
  documentIdElement,
  backendStatus,
  backendStatusText,
  chatMessages,
  questionForm,
  questionInput,
  askButton
};

const missingElements = Object.entries(requiredElements)
  .filter(([, element]) => !element)
  .map(([id]) => id);

if (missingElements.length > 0) {
  console.error(
    "DocLens AI: Required HTML elements are missing:",
    missingElements
  );
} else {
  console.log("DocLens AI frontend initialized.");
  initializeApp();
}

/* =========================================================
   Initialization
========================================================= */

function initializeApp() {
  // 1. Setup browse button click
  chooseFileButton.addEventListener("click", (event) => {
    event.stopPropagation();
    handleBrowseClick();
  });

  // 2. Setup whole upload zone interaction
  if (uploadZone) {
    uploadZone.addEventListener("click", () => {
      handleBrowseClick();
    });

    uploadZone.addEventListener("keydown", (event) => {
      if (event.key === "Enter" || event.key === " ") {
        event.preventDefault();
        handleBrowseClick();
      }
    });

    setupDragAndDrop(uploadZone);
  }

  // 3. Setup file input change
  pdfInput.addEventListener("change", handleFileInputChange);

  // 4. Setup question submission
  questionForm.addEventListener("submit", handleQuestionSubmit);

  // 5. Restore saved document from sessionStorage if available
  restorePersistedDocument();

  // 6. Check health immediately, then poll every 8 seconds
  checkBackendHealth();
  setInterval(checkBackendHealth, 8000);

  // 7. Prevent default browser drop behavior on window to avoid navigating away
  window.addEventListener("dragover", (event) => event.preventDefault());
  window.addEventListener("drop", (event) => event.preventDefault());

  console.log("DocLens AI frontend ready.");
}

/* =========================================================
   Drag and Drop Support
========================================================= */

function setupDragAndDrop(zone) {
  ["dragenter", "dragover"].forEach((eventName) => {
    zone.addEventListener(eventName, (event) => {
      event.preventDefault();
      event.stopPropagation();
      if (!appState.isUploading) {
        zone.classList.add("dragover");
      }
    });
  });

  ["dragleave", "dragend"].forEach((eventName) => {
    zone.addEventListener(eventName, (event) => {
      event.preventDefault();
      event.stopPropagation();
      zone.classList.remove("dragover");
    });
  });

  zone.addEventListener("drop", async (event) => {
    event.preventDefault();
    event.stopPropagation();
    zone.classList.remove("dragover");

    if (appState.isUploading) {
      showUploadStatus("An upload is already in progress.", "info");
      return;
    }

    const files = event.dataTransfer?.files;
    if (!files || files.length === 0) {
      return;
    }

    const file = files[0];
    await processAndUploadFile(file);
  });
}

/* =========================================================
   Backend Health & Auto-Discovery
========================================================= */

async function checkBackendHealth() {
  const candidateUrls = [
    API_BASE_URL,
    "http://127.0.0.1:5000",
    "http://localhost:5000"
  ];

  const uniqueCandidates = [...new Set(candidateUrls.filter(Boolean))];

  for (const candidate of uniqueCandidates) {
    try {
      const response = await fetch(`${candidate}/health`, {
        method: "GET"
      });

      if (response.ok) {
        const result = await response.json();
        if (result.status === "ok") {
          API_BASE_URL = candidate;
          appState.isBackendConnected = true;
          setBackendStatus(true, "Backend connected");
          return true;
        }
      }
    } catch {
      // Continue to next candidate endpoint
    }
  }

  appState.isBackendConnected = false;
  setBackendStatus(false, "Backend disconnected (start server: python app.py)");
  return false;
}

function setBackendStatus(isConnected, message) {
  backendStatusText.textContent = message;
  backendStatus.classList.toggle("connected", isConnected);
  backendStatus.classList.toggle("disconnected", !isConnected);
}

/* =========================================================
   PDF Selection & Upload Flow
========================================================= */

function handleBrowseClick() {
  if (appState.isUploading) {
    console.log("Upload already in progress.");
    return;
  }

  // Clear value so re-selecting the exact same PDF triggers the change event
  pdfInput.value = "";
  pdfInput.click();
}

async function handleFileInputChange(event) {
  const file = event.target.files?.[0];
  if (!file) {
    return;
  }
  await processAndUploadFile(file);
}

async function processAndUploadFile(file) {
  console.log("Selected file:", file.name, file.size, "bytes");

  // Extension check
  if (!file.name.toLowerCase().endsWith(".pdf")) {
    showUploadStatus("Only PDF documents are allowed. Please select a .pdf file.", "error");
    return;
  }

  // Empty file check
  if (file.size === 0) {
    showUploadStatus("The selected file is empty (0 bytes). Please choose a valid PDF.", "error");
    return;
  }

  if (appState.isUploading) {
    showUploadStatus("An upload is already in progress.", "info");
    return;
  }

  await uploadPDF(file);
}

async function uploadPDF(file) {
  appState.isUploading = true;

  chooseFileButton.disabled = true;
  chooseFileButton.textContent = "Processing PDF...";

  showUploadStatus(`Uploading and indexing "${file.name}"... This may take a few moments.`, "info");

  const formData = new FormData();
  formData.append("file", file);

  try {
    // Check/refresh backend connection discovery
    if (!appState.isBackendConnected) {
      await checkBackendHealth();
    }

    const uploadUrl = getEndpoint("/upload");
    console.log("Sending PDF to backend:", uploadUrl);

    const response = await fetch(uploadUrl, {
      method: "POST",
      body: formData
    });

    console.log("Upload HTTP status:", response.status);

    const result = await readJsonResponse(response);
    console.log("Upload response:", result);

    if (!response.ok) {
      throw new Error(
        result.error || `Upload failed with HTTP ${response.status}`
      );
    }

    if (result.status !== "success" || !result.document_id) {
      throw new Error(
        result.error || "The backend did not return a valid document ID."
      );
    }

    const selectedDocument = {
      document_id: result.document_id,
      filename: result.filename || file.name,
      pages: Number(result.pages) || 0,
      chunks: Number(result.chunks) || 0,
      duplicate: result.duplicate === true
    };

    appState.selectedDocument = selectedDocument;

    // Save to sessionStorage to survive accidental reloads
    persistDocument(selectedDocument);

    renderSelectedDocument(selectedDocument);
    enableQuestionForm();

    if (selectedDocument.duplicate) {
      showUploadStatus(
        `"${selectedDocument.filename}" already exists in the vector store and is ready to query!`,
        "success"
      );
    } else {
      showUploadStatus(
        `Successfully uploaded and indexed "${selectedDocument.filename}" (${selectedDocument.pages} pages, ${selectedDocument.chunks} chunks).`,
        "success"
      );
    }

  } catch (error) {
    console.error("PDF upload failed:", error);

    let displayMessage = error.message;

    if (error.name === "TypeError" && error.message.toLowerCase().includes("fetch")) {
      displayMessage = `Cannot connect to DocLens backend at ${API_BASE_URL}. Please ensure the backend server is running with 'python app.py'.`;
    }

    showUploadStatus(displayMessage || "Unable to upload the PDF. Please try again.", "error");

  } finally {
    appState.isUploading = false;
    chooseFileButton.disabled = false;
    chooseFileButton.textContent = "Browse PDF";
    pdfInput.value = "";
  }
}

/* =========================================================
   Selected Document UI & Persistence
========================================================= */

function renderSelectedDocument(document) {
  documentFilename.textContent = document.filename;

  if (document.duplicate) {
    documentStats.textContent =
      "Status: Ready (Cached in document store)";
  } else {
    documentStats.textContent =
      `Pages: ${document.pages} | Chunks: ${document.chunks}`;
  }

  documentIdElement.textContent =
    `Document ID: ${document.document_id}`;

  documentInfo.classList.add("visible");
}

function persistDocument(doc) {
  try {
    sessionStorage.setItem("doclens_active_doc", JSON.stringify(doc));
  } catch (err) {
    console.warn("Unable to persist document to sessionStorage:", err);
  }
}

function restorePersistedDocument() {
  try {
    const saved = sessionStorage.getItem("doclens_active_doc");
    if (saved) {
      const doc = JSON.parse(saved);
      if (doc && doc.document_id) {
        appState.selectedDocument = doc;
        renderSelectedDocument(doc);
        enableQuestionForm();
        showUploadStatus(`Restored session for "${doc.filename}". Ready for questions.`, "success");
      }
    }
  } catch (err) {
    console.warn("Unable to restore document from sessionStorage:", err);
  }
}

function enableQuestionForm() {
  questionInput.disabled = false;
  askButton.disabled = false;
  questionInput.placeholder = "Ask a question about your PDF...";
  questionInput.focus();
}

function showUploadStatus(message, type = "info") {
  uploadStatus.textContent = message;
  uploadStatus.className = `upload-status visible ${type}`;
}

/* =========================================================
   Question Submission Flow
========================================================= */

async function handleQuestionSubmit(event) {
  event.preventDefault();

  if (appState.isAsking) {
    console.log("A question is already being processed.");
    return;
  }

  const question = questionInput.value.trim();

  if (!question) {
    addMessage("Please enter a question.", "assistant");
    return;
  }

  const selectedDocument = appState.selectedDocument;

  if (!selectedDocument?.document_id) {
    addMessage("Please upload a PDF before asking a question.", "assistant");
    return;
  }

  appState.isAsking = true;
  askButton.disabled = true;
  questionInput.disabled = true;

  addMessage(question, "user");
  const loadingMessage = addMessage("Thinking and verifying evidence...", "assistant");

  try {
    const askUrl = getEndpoint("/ask");
    console.log("Submitting question to:", askUrl);

    const response = await fetch(askUrl, {
      method: "POST",
      headers: {
        "Content-Type": "application/json"
      },
      body: JSON.stringify({
        document_id: selectedDocument.document_id,
        question: question
      })
    });

    console.log("Question HTTP status:", response.status);

    const result = await readJsonResponse(response);
    console.log("Question response:", result);

    if (!response.ok) {
      // Show a friendly message for rate-limit errors
      if (response.status === 429) {
        throw new Error(
          "DocLens AI has temporarily reached its AI usage limit. Please try again later."
        );
      }
      throw new Error(
        result.error || `Question request failed with HTTP ${response.status}`
      );
    }

    loadingMessage.remove();
    renderAnswer(result);
    questionInput.value = "";

  } catch (error) {
    console.error("Question request failed:", error);

    let displayMessage = error.message;
    if (error.name === "TypeError" && error.message.toLowerCase().includes("fetch")) {
      displayMessage = `Failed to connect to backend at ${API_BASE_URL}. Ensure 'python app.py' is running.`;
    }

    loadingMessage.textContent = `Unable to get an answer: ${displayMessage}`;

  } finally {
    appState.isAsking = false;
    questionInput.disabled = !appState.selectedDocument?.document_id;
    askButton.disabled = !appState.selectedDocument?.document_id;
    questionInput.focus();
  }
}

/* =========================================================
   Chat Message Rendering
========================================================= */

function addMessage(text, role = "assistant") {
  const currentEmptyState = document.getElementById("emptyState");
  if (currentEmptyState) {
    currentEmptyState.remove();
  }

  const message = document.createElement("div");
  message.className = `message ${role}`;

  const label = document.createElement("span");
  label.className = "message-label";
  label.textContent = role === "user" ? "You" : "DocLens AI";

  const content = document.createElement("div");
  content.textContent = text;

  message.appendChild(label);
  message.appendChild(content);

  chatMessages.appendChild(message);
  chatMessages.scrollTop = chatMessages.scrollHeight;

  return content;
}

/* =========================================================
   Answer & Evidence Rendering
========================================================= */

function renderAnswer(result) {
  const answer = result.answer || "No answer was returned.";
  const answerMessage = addMessage(answer, "assistant");

  if (result.status || result.gate) {
    const statusLine = document.createElement("p");
    statusLine.style.margin = "10px 0 0";
    statusLine.style.fontSize = "12px";
    statusLine.style.color = "#718096";

    const gate = result.gate || {};
    const status = result.status || "UNKNOWN";
    const supported = gate.supported_requirements;
    const total = gate.total_requirements;
    const coverage = gate.coverage;

    let gateText = `Evidence status: ${status}`;

    if (
      Number.isFinite(Number(supported)) &&
      Number.isFinite(Number(total))
    ) {
      gateText += ` · Supported requirements: ${supported}/${total}`;
    }

    if (Number.isFinite(Number(coverage))) {
      gateText += ` · Coverage: ${(Number(coverage) * 100).toFixed(0)}%`;
    }

    const unsupported = Number(total) - Number(supported);
    if (Number.isFinite(unsupported) && unsupported > 0) {
      gateText += ` · Unsupported: ${unsupported}`;
    }

    statusLine.textContent = gateText;
    answerMessage.appendChild(statusLine);
  }

  renderEvidence(result.evidence, answerMessage);
}

function renderEvidence(evidence, parentElement) {
  if (!evidence || isEmptyEvidence(evidence)) {
    return;
  }

  const evidenceContainer = document.createElement("div");
  evidenceContainer.style.marginTop = "12px";
  evidenceContainer.style.paddingTop = "10px";
  evidenceContainer.style.borderTop = "1px solid #e2e8f0";

  const heading = document.createElement("strong");
  heading.textContent = "Supporting evidence";
  heading.style.display = "block";
  heading.style.marginBottom = "8px";

  evidenceContainer.appendChild(heading);

  const evidenceItems = normalizeEvidence(evidence);

  evidenceItems.forEach((item, index) => {
    const card = document.createElement("div");
    card.style.marginTop = "8px";
    card.style.padding = "10px";
    card.style.border = "1px solid #e2e8f0";
    card.style.borderRadius = "8px";
    card.style.background = "#ffffff";
    card.style.fontSize = "12px";
    card.style.lineHeight = "1.6";
    card.style.overflowWrap = "anywhere";

    const title = document.createElement("strong");
    title.textContent = `Evidence ${index + 1}`;
    title.style.display = "block";
    title.style.marginBottom = "5px";

    card.appendChild(title);

    appendEvidenceField(card, "Requirement", item.requirement);
    appendEvidenceField(card, "Source", item.source);
    appendEvidenceField(card, "Page", item.page);
    appendEvidenceField(card, "Quote", item.evidence || item.quote);

    evidenceContainer.appendChild(card);
  });

  parentElement.appendChild(evidenceContainer);
  chatMessages.scrollTop = chatMessages.scrollHeight;
}

function appendEvidenceField(container, label, value) {
  if (value === undefined || value === null || value === "") {
    return;
  }

  const line = document.createElement("div");

  const labelElement = document.createElement("span");
  labelElement.textContent = `${label}: `;
  labelElement.style.fontWeight = "650";

  const valueElement = document.createElement("span");
  valueElement.textContent = String(value);

  line.appendChild(labelElement);
  line.appendChild(valueElement);

  container.appendChild(line);
}

function normalizeEvidence(evidence) {
  if (Array.isArray(evidence)) {
    return evidence;
  }

  if (typeof evidence === "object" && evidence !== null) {
    if (
      evidence.evidence ||
      evidence.quote ||
      evidence.source ||
      evidence.page ||
      evidence.requirement
    ) {
      return [evidence];
    }

    return Object.values(evidence).flatMap((value) => {
      if (Array.isArray(value)) {
        return value;
      }
      if (value && typeof value === "object") {
        return [value];
      }
      return [];
    });
  }

  return [];
}

function isEmptyEvidence(evidence) {
  if (!evidence) {
    return true;
  }
  if (Array.isArray(evidence)) {
    return evidence.length === 0;
  }
  if (typeof evidence === "object") {
    return Object.keys(evidence).length === 0;
  }
  return false;
}

/* =========================================================
   Response Parsing Helper
========================================================= */

async function readJsonResponse(response) {
  const responseText = await response.text();

  if (!responseText) {
    return {};
  }

  try {
    return JSON.parse(responseText);
  } catch (error) {
    console.error("Backend response text:", responseText);
    if (!response.ok) {
      throw new Error(`Server returned HTTP ${response.status}: ${responseText.slice(0, 150)}`);
    }
    throw new Error("The backend returned an unexpected response format.");
  }
}
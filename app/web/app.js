const form = document.querySelector("#chat-form");
const input = document.querySelector("#message-input");
const messages = document.querySelector("#messages");
const welcome = document.querySelector("#welcome");
const sendButton = document.querySelector("#send-button");
const errorMessage = document.querySelector("#error-message");
const conversationList = document.querySelector("#conversation-list");
const sidebar = document.querySelector("#sidebar");
const mobileMenu = document.querySelector("#mobile-menu");
const mobileScrim = document.querySelector("#mobile-scrim");

let conversations = [];
let activeId = null;
let isSending = false;

function createConversation() {
  const conversation = { id: crypto.randomUUID(), title: "Neues Gespräch", messages: [], updatedAt: Date.now() };
  conversations.unshift(conversation);
  activeId = conversation.id;
  render();
  input.focus();
  closeMobileMenu();
}

function currentConversation() {
  return conversations.find((conversation) => conversation.id === activeId);
}

function appendMessage(role, text, pending = false) {
  const row = document.createElement("div");
  row.className = `message ${role}${pending ? " pending" : ""}`;
  if (role === "assistant") {
    const avatar = document.createElement("span");
    avatar.className = "message-avatar";
    avatar.textContent = "CA";
    row.append(avatar);
  }
  const body = document.createElement("div");
  body.className = "message-body";
  body.textContent = text;
  row.append(body);
  messages.append(row);
  return body;
}

function render() {
  conversationList.replaceChildren();
  const recent = [...conversations].sort((a, b) => b.updatedAt - a.updatedAt);
  if (recent.length === 0) {
    const empty = document.createElement("div");
    empty.className = "empty-history";
    empty.textContent = "Deine Unterhaltungen erscheinen hier.";
    conversationList.append(empty);
  }
  for (const conversation of recent) {
    const row = document.createElement("div");
    row.className = "history-row";
    const button = document.createElement("button");
    button.type = "button";
    button.className = `history-item${conversation.id === activeId ? " active" : ""}`;
    button.setAttribute("aria-current", conversation.id === activeId ? "page" : "false");
    button.innerHTML = '<svg viewBox="0 0 24 24" aria-hidden="true"><path d="M20 11.5a7.5 7.5 0 0 1-8 7.5 8.5 8.5 0 0 1-3.5-.8L4 20l1.4-3.7A7.4 7.4 0 0 1 4 11.5 7.5 7.5 0 0 1 12 4a7.5 7.5 0 0 1 8 7.5Z"/></svg>';
    const label = document.createElement("span");
    label.textContent = conversation.title;
    button.append(label);
    const deleteButton = document.createElement("button");
    deleteButton.type = "button";
    deleteButton.className = "history-delete";
    deleteButton.setAttribute("aria-label", `Gespräch „${conversation.title}“ löschen`);
    deleteButton.title = "Gespräch löschen";
    deleteButton.innerHTML = '<svg viewBox="0 0 24 24" aria-hidden="true"><path d="M4 7h16M10 11v6m4-6v6M6 7l1 14h10l1-14M9 7V4h6v3"/></svg>';
    deleteButton.addEventListener("click", (event) => {
      event.stopPropagation();
      conversations = conversations.filter((item) => item.id !== conversation.id);
      if (activeId === conversation.id) {
        activeId = conversations[0]?.id || null;
      }
      render();
    });
    row.append(button, deleteButton);
    button.addEventListener("click", () => {
      activeId = conversation.id;
      render();
      closeMobileMenu();
    });
    conversationList.append(row);
  }

  const conversation = currentConversation();
  const hasMessages = Boolean(conversation?.messages.length);
  welcome.hidden = hasMessages;
  messages.hidden = !hasMessages;
  messages.replaceChildren();
  if (conversation) {
    for (const message of conversation.messages) appendMessage(message.role, message.text);
  }
  messages.scrollTop = messages.scrollHeight;
  errorMessage.hidden = true;
}

async function sendMessage(text) {
  const question = text.trim();
  if (!question || isSending) return;
  let conversation = currentConversation();
  if (!conversation) {
    conversation = { id: crypto.randomUUID(), title: question.slice(0, 42), messages: [], updatedAt: Date.now() };
    conversations.unshift(conversation);
    activeId = conversation.id;
  }

  conversation.title = conversation.messages.length ? conversation.title : question.slice(0, 42);
  conversation.messages.push({ role: "user", text: question });
  conversation.updatedAt = Date.now();
  render();
  appendMessage("assistant", "Ich suche eine Antwort …", true);
  isSending = true;
  sendButton.disabled = true;
  input.disabled = true;
  errorMessage.hidden = true;

  try {
    const response = await fetch("/chat/quick", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ conversation_id: conversation.id, message: question }),
    });
    if (!response.ok) throw new Error(`Die Anfrage konnte nicht verarbeitet werden (${response.status}).`);
    const data = await response.json();
    if (typeof data.answer !== "string") throw new Error("Die Antwort des Servers hatte ein unerwartetes Format.");
    conversation.messages.push({ role: "assistant", text: data.answer });
    conversation.updatedAt = Date.now();
    render();
  } catch (error) {
    messages.lastElementChild?.remove();
    errorMessage.textContent = `${error.message || "Der Chat ist gerade nicht erreichbar."} Bitte versuche es erneut.`;
    errorMessage.hidden = false;
  } finally {
    isSending = false;
    sendButton.disabled = false;
    input.disabled = false;
    input.focus();
  }
}

function closeMobileMenu() {
  sidebar.classList.remove("open");
  mobileScrim.classList.remove("visible");
  mobileMenu.setAttribute("aria-expanded", "false");
}

form.addEventListener("submit", (event) => {
  event.preventDefault();
  const text = input.value;
  input.value = "";
  input.style.height = "auto";
  sendMessage(text);
});

input.addEventListener("input", () => {
  input.style.height = "auto";
  input.style.height = `${Math.min(input.scrollHeight, 140)}px`;
});

input.addEventListener("keydown", (event) => {
  if (event.key === "Enter" && !event.shiftKey) {
    event.preventDefault();
    form.requestSubmit();
  }
});

document.querySelectorAll("[data-prompt]").forEach((button) => {
  button.addEventListener("click", () => sendMessage(button.dataset.prompt));
});

document.querySelector("#new-chat").addEventListener("click", createConversation);
mobileMenu.addEventListener("click", () => {
  const open = sidebar.classList.toggle("open");
  mobileScrim.classList.toggle("visible", open);
  mobileMenu.setAttribute("aria-expanded", String(open));
});
mobileScrim.addEventListener("click", closeMobileMenu);
document.addEventListener("keydown", (event) => {
  if ((event.metaKey || event.ctrlKey) && event.key.toLowerCase() === "k") {
    event.preventDefault();
    createConversation();
  }
});

render();
// All backend calls for the Ava chat live here.
// Override with VITE_API_BASE at build/dev time; defaults to the local backend.
const BASE = import.meta.env.VITE_API_BASE || "http://localhost:8000/api";

// Optional client key (VITE_AVA_CLIENT_KEY): ties chats to a client account for
// internal usage metering. It ships in the browser bundle, so it identifies the
// tenant — it does not authenticate end users.
const CLIENT_KEY = import.meta.env.VITE_AVA_CLIENT_KEY;

export async function startSession() {
  const headers = CLIENT_KEY ? { "X-Client-Key": CLIENT_KEY } : {};
  const res = await fetch(BASE + "/session", { method: "POST", headers });
  if (!res.ok) throw new Error("Could not start a session");
  return res.json(); // { session_id, greeting }
}

export async function uploadCv(sessionId, file) {
  const form = new FormData();
  form.append("session_id", sessionId);
  form.append("file", file);
  const res = await fetch(BASE + "/upload", { method: "POST", body: form });
  if (!res.ok) {
    const detail = await res.json().catch(() => ({}));
    throw new Error(detail.detail || "Upload failed");
  }
  return res.json(); // { ok, filename, chars }
}

// Streams Ava's reply. Calls onToken(text) per delta, onDocument(info) when a
// file is saved, onAction(info) when Ava drafts an email awaiting approval,
// onInterview(question) when a mock-interview question is asked, and
// onReport(report) when the interview ends. Resolves when the stream closes.
export async function streamMessage(
  sessionId,
  message,
  { onToken, onDocument = () => {}, onAction = () => {}, onInterview = () => {}, onReport = () => {} }
) {
  const res = await fetch(BASE + "/message", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ session_id: sessionId, message }),
  });
  if (!res.ok || !res.body) throw new Error("Message failed");

  const reader = res.body.getReader();
  const decoder = new TextDecoder();
  let buffer = "";

  while (true) {
    const { done, value } = await reader.read();
    if (done) break;
    buffer += decoder.decode(value, { stream: true });

    let sep;
    while ((sep = buffer.indexOf("\n\n")) !== -1) {
      const frame = buffer.slice(0, sep).trim();
      buffer = buffer.slice(sep + 2);
      if (!frame.startsWith("data:")) continue;
      let evt;
      try {
        evt = JSON.parse(frame.slice(5).trim());
      } catch {
        continue;
      }
      if (evt.type === "token") onToken(evt.text);
      else if (evt.type === "document") onDocument({ id: evt.id, kind: evt.kind, title: evt.title });
      else if (evt.type === "action")
        onAction({
          id: evt.id,
          kind: evt.kind,
          to: evt.to,
          subject: evt.subject,
          body: evt.body,
          attachments: evt.attachments || [],
        });
      else if (evt.type === "interview")
        onInterview({
          id: evt.id,
          role: evt.role,
          index: evt.index,
          total: evt.total,
          question: evt.question,
          kind: evt.kind,
        });
      else if (evt.type === "report")
        onReport({
          role: evt.role,
          readiness: evt.readiness,
          band: evt.band,
          answered: evt.answered,
          skipped: evt.skipped,
          total: evt.total,
          strengths: evt.strengths || [],
          improvements: evt.improvements || [],
          per_question: evt.per_question || [],
        });
    }
  }
}

export function documentUrl(id, fmt = "docx") {
  return `${BASE}/document/${id}?fmt=${fmt}`;
}

export async function getConfig() {
  const res = await fetch(BASE + "/config");
  return res.ok ? res.json() : {};
}

async function actionPost(path, sessionId, actionId) {
  const res = await fetch(BASE + path, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ session_id: sessionId, action_id: actionId }),
  });
  const data = await res.json().catch(() => ({}));
  if (!res.ok) throw new Error(data.detail || "Action failed");
  return data; // { status, error }
}

export const confirmAction = (sessionId, actionId) =>
  actionPost("/action/confirm", sessionId, actionId);
export const cancelAction = (sessionId, actionId) =>
  actionPost("/action/cancel", sessionId, actionId);

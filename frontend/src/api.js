// All backend calls for the Ava chat live here.
// Override with VITE_API_BASE at build/dev time; defaults to the local backend.
const BASE = import.meta.env.VITE_API_BASE || "http://localhost:8000/api";

export async function startSession() {
  const res = await fetch(BASE + "/session", { method: "POST" });
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

// Streams Ava's reply. Calls onToken(text) per delta and onDocument(info) when a
// file is saved. Resolves when the stream closes.
export async function streamMessage(sessionId, message, { onToken, onDocument }) {
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
    }
  }
}

export function documentUrl(id, fmt = "docx") {
  return `${BASE}/document/${id}?fmt=${fmt}`;
}

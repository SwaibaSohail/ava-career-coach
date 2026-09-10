import { useEffect, useRef, useState } from "react";
import Message from "./components/Message.jsx";
import Composer from "./components/Composer.jsx";
import { startSession, uploadCv, streamMessage, getConfig } from "./api.js";

export default function App() {
  const [sessionId, setSessionId] = useState(null);
  const [messages, setMessages] = useState([]);
  const [busy, setBusy] = useState(false); // streaming or uploading
  const [error, setError] = useState("");
  const [smtpConfigured, setSmtpConfigured] = useState(false);
  const endRef = useRef(null);

  // Open a session and show Ava's greeting.
  useEffect(() => {
    startSession()
      .then((d) => {
        setSessionId(d.session_id);
        setMessages([{ role: "assistant", content: d.greeting }]);
      })
      .catch(() => setError("Couldn't reach the backend. Is it running on http://localhost:8000?"));
    // Whether the server can actually send email (controls the Approve button).
    getConfig().then((c) => setSmtpConfigured(!!c.smtp)).catch(() => {});
  }, []);

  useEffect(() => {
    endRef.current?.scrollIntoView({ block: "end" });
  }, [messages]);

  // Append a delta / document to the last (streaming) assistant message.
  const patchLast = (fn) =>
    setMessages((prev) => {
      const copy = prev.slice();
      copy[copy.length - 1] = fn(copy[copy.length - 1]);
      return copy;
    });

  async function runAva(backendMessage, showUser) {
    setBusy(true);
    setMessages((prev) => [
      ...prev,
      ...(showUser ? [{ role: "user", content: backendMessage }] : []),
      { role: "assistant", content: "" },
    ]);
    try {
      await streamMessage(sessionId, backendMessage, {
        onToken: (t) => patchLast((m) => ({ ...m, content: m.content + t })),
        onDocument: (doc) => patchLast((m) => ({ ...m, documents: [...(m.documents || []), doc] })),
        onAction: (action) => patchLast((m) => ({ ...m, actions: [...(m.actions || []), action] })),
      });
    } catch {
      patchLast((m) => ({
        ...m,
        content: m.content || "Sorry — something went wrong. Please try again.",
      }));
    } finally {
      setBusy(false);
    }
  }

  function handleSend(text) {
    if (sessionId && !busy) runAva(text, true);
  }

  async function handleAttach(file) {
    if (!sessionId || busy) return;
    setBusy(true);
    try {
      const res = await uploadCv(sessionId, file);
      setMessages((prev) => [...prev, { role: "system", content: `CV attached · ${res.filename}` }]);
    } catch (e) {
      setMessages((prev) => [...prev, { role: "system", content: `Upload failed: ${e.message}` }]);
      setBusy(false);
      return;
    }
    // Let Ava react — she now has the CV in context.
    runAva("I've just uploaded my CV.", false);
  }

  return (
    <div className="chat">
      <header className="chat-header">
        <h1>Ava · Career Coach</h1>
        <p className="muted">Tailor your CV, write a cover letter, or build one from scratch — just chat.</p>
      </header>

      <div className="messages">
        {error && <div className="msg system">{error}</div>}
        {messages.map((m, i) => (
          <Message
            key={i}
            message={m}
            streaming={busy && i === messages.length - 1 && m.role === "assistant"}
            sessionId={sessionId}
            smtpConfigured={smtpConfigured}
          />
        ))}
        <div ref={endRef} />
      </div>

      <Composer onSend={handleSend} onAttach={handleAttach} disabled={busy || !sessionId} />
    </div>
  );
}

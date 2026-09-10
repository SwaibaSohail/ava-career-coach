import { useState } from "react";
import { confirmAction, cancelAction } from "../api.js";

const PREVIEW = 220;

export default function ActionCard({ action, sessionId, smtpConfigured }) {
  const [status, setStatus] = useState(action.status || "pending");
  const [error, setError] = useState(action.error || "");
  const [busy, setBusy] = useState(false);
  const [showFull, setShowFull] = useState(false);

  const body = action.body || "";
  const long = body.length > PREVIEW;
  const shown = showFull || !long ? body : body.slice(0, PREVIEW) + "…";

  async function run(fn) {
    setBusy(true);
    try {
      const r = await fn(sessionId, action.id);
      setStatus(r.status);
      setError(r.error || "");
    } catch (e) {
      setStatus("failed");
      setError(e.message || "Action failed");
    } finally {
      setBusy(false);
    }
  }

  const open = status === "pending" || status === "failed";

  return (
    <div className="action-card">
      <div className="action-head">
        <span className="action-ic" aria-hidden="true">✉️</span>
        <span className="action-title">Email — review before sending</span>
      </div>
      <div className="action-field"><span>To</span> {action.to}</div>
      <div className="action-field"><span>Subject</span> {action.subject}</div>
      <pre className="action-body">{shown}</pre>
      {long && (
        <button className="action-link" onClick={() => setShowFull((v) => !v)}>
          {showFull ? "Show less" : "Show full"}
        </button>
      )}

      {open ? (
        <div className="action-btns">
          <button
            className="action-approve"
            disabled={busy || !smtpConfigured}
            title={smtpConfigured ? "" : "Email isn't configured on the server"}
            onClick={() => run(confirmAction)}
          >
            {status === "failed" ? "Retry" : "Approve & send"}
          </button>
          <button className="action-cancel" disabled={busy} onClick={() => run(cancelAction)}>
            Cancel
          </button>
          {!smtpConfigured && <span className="action-hint">email not configured</span>}
          {status === "failed" && error && <span className="action-hint">Failed: {error}</span>}
        </div>
      ) : (
        <div className={`action-status action-${status}`}>
          {status === "sent" ? "Sent ✓" : status === "cancelled" ? "Cancelled" : status}
        </div>
      )}
    </div>
  );
}

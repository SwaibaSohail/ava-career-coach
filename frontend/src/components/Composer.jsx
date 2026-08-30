import { useRef, useState } from "react";

export default function Composer({ onSend, onAttach, disabled }) {
  const [text, setText] = useState("");
  const fileRef = useRef(null);

  function send() {
    const trimmed = text.trim();
    if (!trimmed || disabled) return;
    setText("");
    onSend(trimmed);
  }

  function onKeyDown(e) {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      send();
    }
  }

  function grow(e) {
    e.target.style.height = "auto";
    e.target.style.height = Math.min(e.target.scrollHeight, 160) + "px";
  }

  function pickFile(e) {
    const file = e.target.files?.[0];
    e.target.value = ""; // let the same file be re-selected later
    if (file) onAttach(file);
  }

  return (
    <div className="composer">
      <input ref={fileRef} type="file" accept="application/pdf" onChange={pickFile} hidden />
      <button
        className="icon-btn"
        title="Attach CV (PDF)"
        aria-label="Attach CV"
        onClick={() => fileRef.current?.click()}
        disabled={disabled}
      >
        <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
          <path d="M12 15V4" />
          <path d="m7 9 5-5 5 5" />
          <path d="M5 20h14" />
        </svg>
      </button>
      <textarea
        value={text}
        onChange={(e) => setText(e.target.value)}
        onInput={grow}
        onKeyDown={onKeyDown}
        placeholder="Message Ava…"
        rows={1}
        disabled={disabled}
      />
      <button className="send-btn" onClick={send} disabled={disabled || !text.trim()}>
        Send
      </button>
    </div>
  );
}

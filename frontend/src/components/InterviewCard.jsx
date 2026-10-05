const KIND_LABEL = {
  technical: "Technical",
  behavioral: "Behavioral",
  situational: "Situational",
  cv_gap: "CV gap",
};

export default function InterviewCard({ q, onSend, active }) {
  const pct = Math.round((q.index / q.total) * 100);
  const send = (text) => {
    if (active && onSend) onSend(text);
  };
  return (
    <div className="iv-card">
      <div className="iv-head">
        <span className="iv-title">Mock interview · {q.role}</span>
        <span className="iv-step">
          Question {q.index} of {q.total}
        </span>
      </div>
      <div className="iv-bar" aria-hidden="true">
        <span style={{ width: `${pct}%` }} />
      </div>
      <span className="iv-kind">{KIND_LABEL[q.kind] || q.kind}</span>
      <p className="iv-q">{q.question}</p>
      <div className="iv-foot">
        <span className="iv-hint">Type your answer below, or</span>
        <button type="button" disabled={!active} onClick={() => send("skip")}>
          Skip
        </button>
        <button type="button" disabled={!active} onClick={() => send("repeat")}>
          Repeat
        </button>
        <button
          type="button"
          className="iv-end"
          disabled={!active}
          onClick={() => send("end interview")}
        >
          End interview
        </button>
      </div>
    </div>
  );
}

const BAND_CLASS = { Strong: "strong", Promising: "promising", "Needs work": "needs" };

export default function ReportCard({ report }) {
  const r = report;
  return (
    <div className="rp-card">
      <div className="rp-head">
        <span className="rp-title">Interview report · {r.role}</span>
      </div>
      <div className="rp-score">
        <span className="rp-num">
          {r.readiness}
          <small>/100</small>
        </span>
        <span className={`rp-band ${BAND_CLASS[r.band] || "needs"}`}>{r.band}</span>
      </div>
      <p className="rp-meta">
        Estimated readiness · answered {r.answered} of {r.total}
        {r.skipped ? ` · ${r.skipped} skipped` : ""}
      </p>
      {r.strengths.length > 0 && (
        <div className="rp-list">
          <h4>Strengths</h4>
          <ul>
            {r.strengths.map((s, i) => (
              <li key={i}>{s}</li>
            ))}
          </ul>
        </div>
      )}
      {r.improvements.length > 0 && (
        <div className="rp-list">
          <h4>To improve</h4>
          <ul>
            {r.improvements.map((s, i) => (
              <li key={i}>{s}</li>
            ))}
          </ul>
        </div>
      )}
      {r.per_question.length > 0 && (
        <table className="rp-table">
          <thead>
            <tr>
              <th>#</th>
              <th>Question</th>
              <th>Score</th>
            </tr>
          </thead>
          <tbody>
            {r.per_question.map((p) => (
              <tr key={p.index}>
                <td>{p.index}</td>
                <td>{p.question}</td>
                <td>{p.skipped ? "Skipped" : `${p.score}/10`}</td>
              </tr>
            ))}
          </tbody>
        </table>
      )}
      <p className="rp-note">
        Based on your answers in this practice session — not a prediction of the hiring outcome.
      </p>
    </div>
  );
}

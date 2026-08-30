import { documentUrl } from "../api.js";

export default function DocumentCard({ doc }) {
  const label = doc.kind === "cover_letter" ? "Cover letter" : "CV";
  return (
    <div className="doc-card">
      <span className="doc-ic" aria-hidden="true">
        <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.7" strokeLinecap="round" strokeLinejoin="round">
          <path d="M14 3H7a2 2 0 0 0-2 2v14a2 2 0 0 0 2 2h10a2 2 0 0 0 2-2V8z" />
          <path d="M14 3v5h5" />
        </svg>
      </span>
      <span className="doc-meta">
        <span className="doc-title">{doc.title}</span>
        <span className="doc-sub">{label}</span>
      </span>
      <span className="doc-actions">
        <a href={documentUrl(doc.id, "docx")}>DOCX</a>
        <a href={documentUrl(doc.id, "pdf")}>PDF</a>
      </span>
    </div>
  );
}

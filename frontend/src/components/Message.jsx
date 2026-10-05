import ReactMarkdown from "react-markdown";
import remarkGfm from "remark-gfm";
import DocumentCard from "./DocumentCard.jsx";
import ActionCard from "./ActionCard.jsx";
import InterviewCard from "./InterviewCard.jsx";
import ReportCard from "./ReportCard.jsx";

const clean = (t) =>
  (t || "")
    // Hide any save_document payload the model prints instead of calling the tool.
    .replace(/```(?:json)?\s*\{[\s\S]*?\}\s*```/g, "")
    .replace(/<br\s*\/?>/gi, "\n");

export default function Message({ message, streaming, sessionId, smtpConfigured, onSend, active }) {
  const { role, content, documents, actions, interview, report } = message;

  if (role === "system") return <div className="msg system">{content}</div>;
  if (role === "user") return <div className="msg user">{content}</div>;

  return (
    <div className="msg assistant">
      {content ? (
        <ReactMarkdown
          remarkPlugins={[remarkGfm]}
          components={{
            a: ({ node, ...props }) => (
              <a {...props} target="_blank" rel="noopener noreferrer" />
            ),
          }}
        >
          {clean(content)}
        </ReactMarkdown>
      ) : streaming ? (
        <span className="typing">Ava is typing…</span>
      ) : null}
      {documents?.map((doc) => (
        <DocumentCard key={doc.id} doc={doc} />
      ))}
      {actions?.map((action) => (
        <ActionCard
          key={action.id}
          action={action}
          sessionId={sessionId}
          smtpConfigured={smtpConfigured}
        />
      ))}
      {interview && <InterviewCard q={interview} onSend={onSend} active={active} />}
      {report && <ReportCard report={report} />}
    </div>
  );
}

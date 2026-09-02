import ReactMarkdown from "react-markdown";
import remarkGfm from "remark-gfm";
import DocumentCard from "./DocumentCard.jsx";

const clean = (t) =>
  (t || "")
    // Hide any save_document payload the model prints instead of calling the tool.
    .replace(/```(?:json)?\s*\{[\s\S]*?\}\s*```/g, "")
    .replace(/<br\s*\/?>/gi, "\n");

export default function Message({ message, streaming }) {
  const { role, content, documents } = message;

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
    </div>
  );
}

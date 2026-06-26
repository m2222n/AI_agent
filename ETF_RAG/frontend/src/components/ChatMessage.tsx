import ReactMarkdown from "react-markdown";
import remarkGfm from "remark-gfm";
import type { UiMessage } from "@/lib/types";
import { questionTypeLabel } from "@/lib/labels";
import StructuredDataView from "./StructuredData";
import { Spinner } from "./Feedback";

export default function ChatMessage({ message }: { message: UiMessage }) {
  const isUser = message.role === "user";

  return (
    <div className={`flex ${isUser ? "justify-end" : "justify-start"}`}>
      <div
        className={[
          "max-w-[85%] rounded-2xl px-4 py-3 text-sm leading-relaxed",
          isUser
            ? "bg-blue-600 text-white"
            : message.isError
              ? "bg-red-50 text-red-700 border border-red-200 dark:bg-red-950 dark:text-red-300 dark:border-red-900"
              : "bg-gray-100 text-gray-900 dark:bg-gray-800 dark:text-gray-100",
        ].join(" ")}
      >
        {!isUser && (message.questionType || message.model) && (
          <div className="mb-1 text-xs text-gray-500">
            {questionTypeLabel(message.questionType)}
            {message.model ? ` · ${message.model}` : ""}
          </div>
        )}

        {/* 스트리밍 중 상태줄 (도구 호출 등). 본문이 아직 없을 때만. */}
        {!isUser && message.status && !message.content && (
          <div role="status" aria-live="polite" className="flex items-center gap-2 text-xs text-gray-500">
            <Spinner className="h-3 w-3" />
            <span>{message.status}</span>
          </div>
        )}

        {isUser ? (
          <div className="whitespace-pre-wrap break-words">
            {message.content}
          </div>
        ) : (
          message.content && (
            <div className="markdown-body break-words">
              <ReactMarkdown remarkPlugins={[remarkGfm]}>
                {message.content}
              </ReactMarkdown>
            </div>
          )
        )}

        {/* 차트/비교표 (structured_data) */}
        {!isUser &&
          message.structured?.map((d, i) => (
            <StructuredDataView key={i} data={d} />
          ))}
      </div>
    </div>
  );
}

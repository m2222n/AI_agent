import type { UiMessage } from "@/lib/types";
import { questionTypeLabel } from "@/lib/labels";

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
              ? "bg-red-50 text-red-700 border border-red-200"
              : "bg-gray-100 text-gray-900",
        ].join(" ")}
      >
        {!isUser && (message.questionType || message.model) && (
          <div className="mb-1 text-xs text-gray-500">
            {questionTypeLabel(message.questionType)}
            {message.model ? ` · ${message.model}` : ""}
          </div>
        )}
        {/* 4a: 본문은 plain text (markdown은 4b). 줄바꿈 보존. */}
        <div className="whitespace-pre-wrap break-words">{message.content}</div>
      </div>
    </div>
  );
}

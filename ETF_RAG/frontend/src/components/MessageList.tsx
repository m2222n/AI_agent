import type { UiMessage } from "@/lib/types";
import ChatMessage from "./ChatMessage";

export default function MessageList({
  messages,
  isLoading,
}: {
  messages: UiMessage[];
  isLoading: boolean;
}) {
  return (
    <div className="flex flex-col gap-3">
      {messages.map((m, i) => (
        <ChatMessage key={i} message={m} />
      ))}
      {isLoading && (
        <div className="flex justify-start">
          <div className="rounded-2xl bg-gray-100 px-4 py-3 text-sm text-gray-500">
            <span className="inline-flex gap-1">
              답변 생성 중
              <span className="animate-pulse">...</span>
            </span>
          </div>
        </div>
      )}
    </div>
  );
}

import type { UiMessage } from "@/lib/types";
import ChatMessage from "./ChatMessage";

export default function MessageList({ messages }: { messages: UiMessage[] }) {
  return (
    <div className="flex flex-col gap-3">
      {messages.map((m, i) => (
        <ChatMessage key={i} message={m} />
      ))}
    </div>
  );
}

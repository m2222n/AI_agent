"use client";

import { useState } from "react";

export default function ChatInput({
  disabled,
  onSend,
  placeholder,
}: {
  disabled: boolean;
  onSend: (text: string) => void;
  placeholder: string;
}) {
  const [text, setText] = useState("");

  const submit = () => {
    const q = text.trim();
    if (!q || disabled) return;
    onSend(q);
    setText("");
  };

  const onKeyDown = (e: React.KeyboardEvent<HTMLTextAreaElement>) => {
    // Enter 전송, Shift+Enter 줄바꿈
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      submit();
    }
  };

  return (
    <div className="flex items-end gap-2">
      <textarea
        value={text}
        onChange={(e) => setText(e.target.value)}
        onKeyDown={onKeyDown}
        disabled={disabled}
        rows={1}
        placeholder={placeholder}
        className="flex-1 resize-none rounded-xl border border-gray-300 px-4 py-3 text-sm focus:border-blue-500 focus:outline-none disabled:bg-gray-100 disabled:text-gray-400"
      />
      <button
        type="button"
        onClick={submit}
        disabled={disabled || !text.trim()}
        className="rounded-xl bg-blue-600 px-5 py-3 text-sm font-medium text-white hover:bg-blue-700 disabled:cursor-not-allowed disabled:bg-gray-300"
      >
        전송
      </button>
    </div>
  );
}

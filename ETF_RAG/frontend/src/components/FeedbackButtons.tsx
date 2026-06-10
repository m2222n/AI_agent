"use client";

import { useState } from "react";
import { sendFeedback } from "@/lib/api";

const REASONS = [
  "정보가 부정확해요",
  "원하는 답변이 아니에요",
  "더 자세한 정보가 필요해요",
  "기타",
];

export default function FeedbackButtons({
  question,
  answer,
}: {
  question: string;
  answer: string;
}) {
  const [done, setDone] = useState(false);
  const [showReason, setShowReason] = useState(false);

  if (done) {
    return <p className="mt-1 text-xs text-gray-400">피드백 감사합니다 🙏</p>;
  }

  const positive = () => {
    sendFeedback(question, answer, "positive");
    setDone(true);
  };

  const negative = (reason: string) => {
    sendFeedback(question, answer, "negative", reason);
    setDone(true);
  };

  return (
    <div className="mt-1">
      {!showReason ? (
        <div className="flex gap-1">
          <button
            type="button"
            onClick={positive}
            className="rounded px-2 py-0.5 text-xs text-gray-400 hover:bg-gray-100"
            aria-label="도움됨"
          >
            👍
          </button>
          <button
            type="button"
            onClick={() => setShowReason(true)}
            className="rounded px-2 py-0.5 text-xs text-gray-400 hover:bg-gray-100"
            aria-label="아쉬워요"
          >
            👎
          </button>
        </div>
      ) : (
        <div className="flex flex-wrap gap-1">
          {REASONS.map((r) => (
            <button
              key={r}
              type="button"
              onClick={() => negative(r)}
              className="rounded-full border border-gray-200 px-2 py-0.5 text-xs text-gray-500 hover:bg-gray-100"
            >
              {r}
            </button>
          ))}
        </div>
      )}
    </div>
  );
}

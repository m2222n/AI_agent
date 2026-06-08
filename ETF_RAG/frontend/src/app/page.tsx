"use client";

import { useEffect, useRef, useState } from "react";
import { chatOnce, getHealth } from "@/lib/api";
import type { Health, UiMessage } from "@/lib/types";
import MessageList from "@/components/MessageList";
import ChatInput from "@/components/ChatInput";

export default function Home() {
  const [health, setHealth] = useState<Health | null>(null);
  const [messages, setMessages] = useState<UiMessage[]>([]);
  const [isLoading, setIsLoading] = useState(false);
  const scrollRef = useRef<HTMLDivElement>(null);

  // mount 시 /health 폴링 — ready가 될 때까지 3초 간격 재시도
  useEffect(() => {
    let cancelled = false;
    let timer: ReturnType<typeof setTimeout>;

    const poll = async () => {
      try {
        const h = await getHealth();
        if (cancelled) return;
        setHealth(h);
        if (!h.ready) timer = setTimeout(poll, 3000);
      } catch {
        if (cancelled) return;
        setHealth({ ready: false, error: "백엔드에 연결할 수 없어요." });
        timer = setTimeout(poll, 3000);
      }
    };
    poll();

    return () => {
      cancelled = true;
      clearTimeout(timer);
    };
  }, []);

  // 새 메시지마다 하단으로 스크롤
  useEffect(() => {
    scrollRef.current?.scrollTo({
      top: scrollRef.current.scrollHeight,
      behavior: "smooth",
    });
  }, [messages, isLoading]);

  const inputDisabled = !health?.ready || isLoading;

  const handleSend = async (question: string) => {
    setMessages((prev) => [...prev, { role: "user", content: question }]);
    setIsLoading(true);
    try {
      // 4a: 멀티턴 생략 — 빈 히스토리 전송 (백엔드는 null로 처리)
      const res = await chatOnce(question, []);
      setMessages((prev) => [
        ...prev,
        {
          role: "assistant",
          content: res.answer,
          questionType: res.question_type,
          model: res.model,
        },
      ]);
    } catch {
      setMessages((prev) => [
        ...prev,
        {
          role: "assistant",
          content:
            "답변을 가져오지 못했어요. 백엔드가 실행 중인지 확인하고 다시 시도해주세요.",
          isError: true,
        },
      ]);
    } finally {
      setIsLoading(false);
    }
  };

  const statusText = !health
    ? "백엔드 상태 확인 중…"
    : health.ready
      ? "준비 완료"
      : `백엔드 준비 중… ${health.error ? `(${health.error})` : ""}`;

  return (
    <main className="mx-auto flex h-full w-full max-w-3xl flex-col px-4">
      <header className="border-b border-gray-200 py-4">
        <h1 className="text-lg font-bold text-gray-900">
          📈 투자 AI 어시스턴트
        </h1>
        <p className="text-xs text-gray-500">
          ETF · 주식 · 기술적 분석 · 재무제표 · 가격 전망
        </p>
        <p
          className={`mt-1 text-xs ${
            health?.ready ? "text-green-600" : "text-amber-600"
          }`}
        >
          {statusText}
        </p>
      </header>

      <div ref={scrollRef} className="flex-1 overflow-y-auto py-4">
        {messages.length === 0 ? (
          <div className="mt-12 text-center text-sm text-gray-400">
            궁금한 ETF나 주식을 물어보세요.
            <br />예: &ldquo;KODEX 200 수익률 알려줘&rdquo;
          </div>
        ) : (
          <MessageList messages={messages} isLoading={isLoading} />
        )}
      </div>

      <div className="border-t border-gray-200 py-4">
        <ChatInput
          disabled={inputDisabled}
          onSend={handleSend}
          placeholder={
            health?.ready ? "메시지를 입력하세요…" : "백엔드 준비 중…"
          }
        />
      </div>
    </main>
  );
}

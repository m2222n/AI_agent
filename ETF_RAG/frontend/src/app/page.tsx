"use client";

import { useEffect, useRef, useState } from "react";
import { getHealth, streamChat } from "@/lib/api";
import type { Health, UiMessage } from "@/lib/types";
import { toolLabel } from "@/lib/labels";
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

  // 마지막(진행 중) assistant 메시지를 불변 업데이트
  const patchLastAssistant = (patch: Partial<UiMessage>) => {
    setMessages((prev) => {
      const next = [...prev];
      const last = next[next.length - 1];
      if (last && last.role === "assistant") {
        next[next.length - 1] = { ...last, ...patch };
      }
      return next;
    });
  };

  const handleSend = (question: string) => {
    // user 메시지 + 빈 assistant placeholder push
    setMessages((prev) => [
      ...prev,
      { role: "user", content: question },
      { role: "assistant", content: "", status: "분석 중…" },
    ]);
    setIsLoading(true);

    // 4b: 멀티턴 생략 — 빈 히스토리 전송. (4d에서 실제 chat_history)
    streamChat(question, [], {
      onQuestionType: (t) => patchLastAssistant({ questionType: t }),
      onToolCall: (c) =>
        patchLastAssistant({ status: `${toolLabel(c.name)} 중…` }),
      onStructuredData: (d) =>
        setMessages((prev) => {
          const next = [...prev];
          const last = next[next.length - 1];
          if (last && last.role === "assistant") {
            next[next.length - 1] = {
              ...last,
              structured: [...(last.structured ?? []), d],
            };
          }
          return next;
        }),
      onToken: (cumulative) =>
        // 누적 텍스트 → replace (델타 아님)
        patchLastAssistant({ content: cumulative, status: undefined }),
      onDone: (d) => {
        patchLastAssistant({
          // done.answer가 마지막 토큰보다 길면 우선 (CoV 보정 등)
          model: d.model,
          questionType: d.question_type,
          status: undefined,
        });
        setMessages((prev) => {
          const next = [...prev];
          const last = next[next.length - 1];
          if (
            last &&
            last.role === "assistant" &&
            d.answer.length > last.content.length
          ) {
            next[next.length - 1] = { ...last, content: d.answer };
          }
          return next;
        });
        setIsLoading(false);
      },
      onError: (msg) => {
        patchLastAssistant({
          content: msg,
          isError: true,
          status: undefined,
        });
        setIsLoading(false);
      },
    });
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
          <MessageList messages={messages} />
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

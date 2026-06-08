"use client";

import { useEffect, useRef, useState } from "react";
import { getHealth, streamChat } from "@/lib/api";
import type { ChatHistoryItem, Health, UiMessage } from "@/lib/types";
import { toolLabel } from "@/lib/labels";
import { getFollowupSuggestions } from "@/lib/followup";
import MessageList from "@/components/MessageList";
import ChatInput from "@/components/ChatInput";

const STORAGE_KEY = "etfrag.messages.v1";

export default function Home() {
  const [health, setHealth] = useState<Health | null>(null);
  const [messages, setMessages] = useState<UiMessage[]>([]);
  const [isLoading, setIsLoading] = useState(false);
  const [hydrated, setHydrated] = useState(false);
  const scrollRef = useRef<HTMLDivElement>(null);

  // mount 시 localStorage 복원 (1회)
  useEffect(() => {
    try {
      const raw = localStorage.getItem(STORAGE_KEY);
      if (raw) setMessages(JSON.parse(raw) as UiMessage[]);
    } catch {
      /* 손상 시 무시 */
    }
    setHydrated(true);
  }, []);

  // 대화 변경 시 영속. base64 차트(structured)·전이 상태(status)는 제외 —
  // 이미지가 200KB+라 localStorage(~5MB) 초과 위험. 텍스트 대화만 보존.
  useEffect(() => {
    if (!hydrated) return;
    try {
      const slim = messages
        .filter((m) => m.content)
        .map((m) => ({
          role: m.role,
          content: m.content,
          questionType: m.questionType,
          model: m.model,
          isError: m.isError,
          followups: m.followups,
        }));
      localStorage.setItem(STORAGE_KEY, JSON.stringify(slim));
    } catch {
      /* 용량 초과 등 무시 */
    }
  }, [messages, hydrated]);

  // mount 시 /health 폴링 — ready까지 3초 간격 재시도
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
    // 멀티턴: 직전까지 완료된 대화를 chat_history로 (현재 user/placeholder 추가 전 기준)
    const history: ChatHistoryItem[] = messages
      .filter((m) => m.content && !m.isError)
      .map((m) => ({ role: m.role, content: m.content }));

    setMessages((prev) => [
      ...prev,
      { role: "user", content: question },
      { role: "assistant", content: "", status: "분석 중…" },
    ]);
    setIsLoading(true);

    const toolsUsed: string[] = [];
    let qType = "";

    streamChat(question, history, {
      onQuestionType: (t) => {
        qType = t;
        patchLastAssistant({ questionType: t });
      },
      onToolCall: (c) => {
        toolsUsed.push(c.name);
        patchLastAssistant({ status: `${toolLabel(c.name)} 중…` });
      },
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
        patchLastAssistant({ content: cumulative, status: undefined }),
      onDone: (d) => {
        const followups = getFollowupSuggestions(
          question,
          toolsUsed,
          d.question_type || qType,
        );
        patchLastAssistant({
          model: d.model,
          questionType: d.question_type,
          status: undefined,
          followups,
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
        patchLastAssistant({ content: msg, isError: true, status: undefined });
        setIsLoading(false);
      },
    });
  };

  const handleReset = () => {
    if (isLoading) return;
    setMessages([]);
    try {
      localStorage.removeItem(STORAGE_KEY);
    } catch {
      /* 무시 */
    }
  };

  const statusText = !health
    ? "백엔드 상태 확인 중…"
    : health.ready
      ? "준비 완료"
      : `백엔드 준비 중… ${health.error ? `(${health.error})` : ""}`;

  // 마지막 assistant 메시지의 후속질문 (스트리밍 끝났을 때만)
  const lastMsg = messages[messages.length - 1];
  const followups =
    !isLoading && lastMsg?.role === "assistant" && !lastMsg.isError
      ? (lastMsg.followups ?? [])
      : [];

  return (
    <main className="mx-auto flex h-full w-full max-w-3xl flex-col px-3 sm:px-4">
      <header className="flex items-start justify-between border-b border-gray-200 py-3 sm:py-4">
        <div>
          <h1 className="text-base font-bold text-gray-900 sm:text-lg">
            📈 투자 AI 어시스턴트
          </h1>
          <p className="hidden text-xs text-gray-500 sm:block">
            ETF · 주식 · 기술적 분석 · 재무제표 · 가격 전망
          </p>
          <p
            className={`mt-0.5 text-xs ${
              health?.ready ? "text-green-600" : "text-amber-600"
            }`}
          >
            {statusText}
          </p>
        </div>
        {messages.length > 0 && (
          <button
            type="button"
            onClick={handleReset}
            disabled={isLoading}
            className="shrink-0 rounded-lg border border-gray-300 px-3 py-1.5 text-xs text-gray-600 hover:bg-gray-100 disabled:opacity-40"
          >
            🗑️ 대화 초기화
          </button>
        )}
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

      {followups.length > 0 && (
        <div className="flex flex-wrap gap-2 pb-2">
          {followups.map((f, i) => (
            <button
              key={i}
              type="button"
              onClick={() => handleSend(f)}
              disabled={inputDisabled}
              className="rounded-full border border-blue-200 bg-blue-50 px-3 py-1.5 text-xs text-blue-700 hover:bg-blue-100 disabled:opacity-40"
            >
              {f}
            </button>
          ))}
        </div>
      )}

      <div className="sticky bottom-0 border-t border-gray-200 bg-white py-3 sm:py-4">
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

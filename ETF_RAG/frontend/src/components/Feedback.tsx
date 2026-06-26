/**
 * 공통 UX 피드백 컴포넌트 — 로딩/에러/성공/빈 상태를 한 곳에서 통일.
 * 페이지마다 인라인으로 흩어져 있던 "...중…"/text-red-600/text-gray-400을 대체한다.
 */

/** 작은 회전 스피너 (Tailwind animate-spin, 추가 CSS 불필요). */
export function Spinner({ className = "" }: { className?: string }) {
  return (
    <span
      aria-hidden="true"
      className={`inline-block animate-spin rounded-full border-2 border-gray-300 border-t-blue-500 ${className}`}
    />
  );
}

/** 데이터 로딩 중 — 스피너 + 문구. role=status/aria-busy로 스크린리더 안내. */
export function Loading({ text = "불러오는 중…" }: { text?: string }) {
  return (
    <div
      role="status"
      aria-busy="true"
      className="mt-6 flex items-center justify-center gap-2 text-sm text-gray-400"
    >
      <Spinner className="h-4 w-4" />
      <span>{text}</span>
    </div>
  );
}

/** API 실패 등 에러 메시지. aria-live로 알림. */
export function ErrorText({ message }: { message: string }) {
  return (
    <p role="alert" className="mt-6 text-center text-sm text-red-600">
      {message}
    </p>
  );
}

/** 액션 결과 알림(성공/실패). 폼·거래 등 인라인 피드백 통일. */
export function Notice({ message, kind = "ok" }: { message: string; kind?: "ok" | "err" }) {
  return (
    <p
      role={kind === "err" ? "alert" : "status"}
      className={`mt-2 text-xs ${kind === "ok" ? "text-green-600" : "text-red-600"}`}
    >
      {message}
    </p>
  );
}

/** 빈 상태 — 아이콘 + 메시지 + (선택) 행동 유도. */
export function EmptyState({
  icon = "📭",
  message,
  action,
}: {
  icon?: string;
  message: string;
  action?: React.ReactNode;
}) {
  return (
    <div className="mt-4 flex flex-col items-center gap-2 py-6 text-center">
      <span className="text-2xl" aria-hidden="true">{icon}</span>
      <p className="text-xs text-gray-400">{message}</p>
      {action}
    </div>
  );
}

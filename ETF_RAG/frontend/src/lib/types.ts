// 백엔드(api/) 계약과 1:1 대응하는 타입.

export type Role = "user" | "assistant";

export type QuestionType =
  | "simple"
  | "compare"
  | "recommend"
  | "risk"
  | "general"
  | "technical"
  | "correlation"
  | "portfolio";

/** /chat·/stream 요청의 chat_history 항목 */
export interface ChatHistoryItem {
  role: Role;
  content: string;
}

/** POST /chat 응답 */
export interface ChatResponse {
  answer: string;
  question_type: QuestionType;
  model: string;
}

/** GET /health 응답 */
export interface Health {
  ready: boolean;
  error: string | null;
}

// ── SSE structured_data 변형 (data를 JSON.parse한 결과) ──
export interface ComparisonItem {
  name: string;
  ticker?: string;
  close?: number;
  change_pct?: number;
  per?: number;
  pbr?: number;
  [k: string]: unknown;
}
export interface ComparisonTable {
  __type__: "comparison_table";
  items: ComparisonItem[];
  comparison_chart_b64?: string;
}
export interface TechnicalChart {
  __type__: "technical_chart";
  image_b64: string;
  name: string;
}
export interface PortfolioChart {
  __type__: "portfolio_chart";
  image_b64: string;
  names: string[];
}
export type StructuredData = ComparisonTable | TechnicalChart | PortfolioChart;

/** done 이벤트 payload */
export interface DonePayload {
  answer: string;
  model: string;
  question_type: QuestionType;
  cov_applied: boolean;
}

/** streamChat 콜백 인터페이스 */
export interface StreamCallbacks {
  onQuestionType?: (t: QuestionType) => void;
  onToolCall?: (c: { name: string; args: Record<string, unknown> }) => void;
  onToolResult?: (s: string) => void;
  onCovRevision?: (s: string) => void;
  onToken?: (cumulativeText: string) => void; // 누적 텍스트 → replace
  onStructuredData?: (d: StructuredData) => void;
  onError?: (msg: string) => void;
  onDone?: (d: DonePayload) => void;
}

/** UI가 다루는 메시지 모델 */
export interface UiMessage {
  role: Role;
  content: string;
  questionType?: QuestionType;
  model?: string;
  isError?: boolean;
  structured?: StructuredData[]; // 4c에서 렌더 (4b는 수집만)
  status?: string; // 스트리밍 중 상태줄 텍스트
}

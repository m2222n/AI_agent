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

/** UI가 다루는 메시지 모델 (4a는 plain text 본문) */
export interface UiMessage {
  role: Role;
  content: string;
  questionType?: QuestionType;
  model?: string;
  isError?: boolean;
}

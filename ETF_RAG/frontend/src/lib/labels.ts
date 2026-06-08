import type { QuestionType } from "./types";

const QUESTION_TYPE_LABELS: Record<QuestionType, string> = {
  simple: "📝 단순 정보",
  compare: "⚖️ 비교 분석",
  recommend: "💡 종목 추천",
  risk: "⚠️ 위험 분석",
  general: "💬 일반 질문",
  technical: "📊 기술적 분석",
  correlation: "🔗 상관관계",
  portfolio: "📈 포트폴리오",
};

export function questionTypeLabel(t?: QuestionType): string {
  if (!t) return "";
  return QUESTION_TYPE_LABELS[t] ?? t;
}

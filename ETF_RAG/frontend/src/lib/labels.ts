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

const TOOL_LABELS: Record<string, string> = {
  search_etf: "🔍 ETF 정보 검색",
  compare_etfs: "⚖️ ETF 비교",
  get_etf_list: "📋 ETF 목록 조회",
  search_stock: "🔍 주식 정보 검색",
  compare_stocks: "⚖️ 주식 비교",
  get_stock_list: "📋 주식 목록 조회",
  get_realtime_price: "📈 실시간 시세 조회",
  analyze_sector: "🏭 섹터 분석",
  get_technical_indicators: "📊 기술적 지표 분석",
  get_stock_correlation: "🔗 상관관계 분석",
  simulate_portfolio: "📈 포트폴리오 시뮬레이션",
  get_financial_statements: "📑 재무제표 조회",
  predict_price_outlook: "🔮 가격 전망 분석",
  get_stock_news: "📰 뉴스 수집·감성 분석",
};

export function toolLabel(name: string): string {
  return TOOL_LABELS[name] ?? `🔍 ${name}`;
}

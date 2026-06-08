// 후속 질문 제안 — 백엔드 src/ui/chat.py:_get_followup_suggestions 규칙을 클라이언트 복제.
// tool_call 이벤트로 수집한 도구 이름 + 질문에서 추출한 종목명 기반, 최대 3개.

const COMMON_STOCKS = [
  "삼성전자",
  "SK하이닉스",
  "현대차",
  "LG에너지솔루션",
  "카카오",
  "네이버",
  "셀트리온",
  "기아",
  "포스코홀딩스",
  "삼성SDI",
];
const COMMON_ETFS = ["KODEX 200", "TIGER 200", "KODEX 레버리지", "TIGER 미국S&P500"];

export function getFollowupSuggestions(
  question: string,
  toolsUsed: string[],
  questionType: string,
): string[] {
  const stockNames = COMMON_STOCKS.filter((n) => question.includes(n));
  const etfNames = COMMON_ETFS.filter((n) => question.includes(n));
  const target = stockNames[0] ?? etfNames[0] ?? "";

  const out: string[] = [];
  const has = (t: string) => toolsUsed.includes(t);

  if (has("search_stock") || has("search_etf")) {
    if (target) {
      out.push(`${target} 기술적 분석해줘`);
      out.push(`${target} 앞으로 전망은?`);
    }
  } else if (has("get_technical_indicators")) {
    if (target) {
      out.push(`${target} 재무제표 보여줘`);
      out.push(`${target} 최근 실적은 어때?`);
    }
  } else if (has("predict_price_outlook")) {
    if (target) out.push(`${target} 기술적 분석해줘`);
  } else if (has("compare_etfs") || has("compare_stocks")) {
    if (stockNames[0]) out.push(`${stockNames[0]} 기술적 분석해줘`);
  } else if (has("get_financial_statements")) {
    if (target) {
      out.push(`${target} 기술적 분석해줘`);
      out.push(`${target} 앞으로 전망은?`);
    }
  }

  if (questionType === "simple" && target) {
    const s = `${target} 기술적 분석해줘`;
    if (!out.includes(s)) out.push(s);
  }
  if (out.length === 0 && target) out.push(`${target} 기술적 분석해줘`);

  return out.slice(0, 3);
}

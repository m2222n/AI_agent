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

// ── 동적 추천질문 (movers) ──────────────────────────────
export interface MoverItem {
  name: string;
  ticker: string;
  change_pct: number;
}
export interface MoversResponse {
  gainers: MoverItem[];
  losers: MoverItem[];
  most_traded: MoverItem[];
}

// ── 사이드바 개요 ────────────────────────────────────────
export interface InstrumentItem {
  name: string;
  ticker: string;
  close: number;
  change_pct: number;
  trade_value: number;
  sector?: string | null;
  per?: number | null;
  market_cap?: number | null;
}
export interface OverviewResponse {
  etf_count: number;
  stock_count: number;
  as_of: string | null;
  top_etfs: InstrumentItem[];
  top_stocks: InstrumentItem[];
  sectors: string[];
}

export interface VisitorResponse {
  daily: number;
  total: number;
}

// ── 탭 API ──────────────────────────────────────────────
/** /tabs/technical 응답 (summary는 복잡 중첩 dict — 느슨한 타입) */
export interface TechnicalResponse {
  ticker: string;
  name: string;
  summary: Record<string, unknown>;
  chart_b64: string | null;
}

/** 종목 검색 옵션 ("이름 (티커)") */
export interface TickerSearchResponse {
  options: string[];
}

/** 실시간 시세 (/tabs/price). source: kis(실시간) | yfinance(지연) | close(종가) */
export interface PriceData {
  name: string;
  ticker: string;
  price: number;
  prev_close: number | null;
  change: number | null;
  change_pct: number | null;
  volume: number | null;
  source: "kis" | "yfinance" | "close";
  is_live: boolean;
  timestamp: string | null;
  market_open: boolean;
}

/** 호가 10단계 (/tabs/orderbook, KIS 전용) */
export interface OrderbookLevel {
  price: number;
  qty: number;
}
export interface OrderbookData {
  name: string;
  ticker: string;
  asks: OrderbookLevel[]; // 매도호가 1~10
  bids: OrderbookLevel[]; // 매수호가 1~10
  total_ask_qty: number;
  total_bid_qty: number;
  timestamp: string | null;
  source: string;
}

/** /tabs/financial 응답 */
export interface FinancialRow {
  fiscal_year: number;
  fiscal_quarter: number;
  revenue?: number | null;
  operating_profit?: number | null;
  net_income?: number | null;
  operating_margin?: number | null;
  net_margin?: number | null;
  revenue_growth_yoy?: number | null;
  op_growth_yoy?: number | null;
}
export interface FinancialResponse {
  ticker: string;
  name: string;
  rows: FinancialRow[];
  chart_b64: string | null;
}

/** /tabs/comparison 응답 (items = 구조화 데이터 dict) */
export interface ComparisonResponse {
  items: Record<string, unknown>[];
  comparison_chart_b64: string | null;
  valuation_chart_b64: string | null;
}

/** /tabs/outlook 응답 (복잡 중첩 — 느슨한 타입) */
export interface OutlookResponse {
  ticker?: string;
  name?: string;
  horizon?: string;
  current_price?: number;
  composite_score?: number;
  confidence_grade?: string;
  technical?: Record<string, unknown>;
  fundamental?: Record<string, unknown>;
  statistical?: Record<string, unknown>;
  prophet?: Record<string, unknown>;
  scenarios?: Record<string, { probability?: number; target_return?: number; description?: string }>;
  risk_factors?: string[];
  [k: string]: unknown;
}

/** /tabs/sector 응답 */
export interface SectorStat {
  sector: string;
  count: number;
  market_cap: number;
  change_pct: number;
  median_per: number;
  up_count: number;
  down_count: number;
}
export interface SectorResponse {
  stats: SectorStat[];
  overview_chart_b64: string | null;
  period?: string;
  sector?: string;
  detail_chart_b64?: string | null;
  stocks?: Record<string, unknown>[];
  // 기간 추이 (period !== "1d" + 섹터 선택 시)
  trend_chart_b64?: string | null;
  trend_return_pct?: number;
  trend_constituents?: number;
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
  structured?: StructuredData[]; // 4c 렌더
  status?: string; // 스트리밍 중 상태줄 텍스트
  followups?: string[]; // 4d: 후속 질문 제안 (assistant 완료 시)
}

// ── 가상투자(모의투자) ─────────────────────────────────
export interface PaperHolding {
  ticker: string;
  name: string;
  qty: number;
  avg_price: number;
  current_price: number;
  eval_value: number;
  cost_value: number;
  pnl: number;
  pnl_pct: number;
  price_source: string;
}
export interface PaperPortfolio {
  cash: number;
  holdings: PaperHolding[];
  holdings_value: number;
  total_value: number;
  initial_cash: number;
  total_pnl: number;
  total_pnl_pct: number;
}
export interface PaperTradeResult {
  ok: boolean;
  side: "buy" | "sell";
  ticker: string;
  name: string;
  qty: number;
  price: number;
  amount: number;
  cash: number;
  realized_pnl?: number | null;
  price_source: string;
}
export interface PaperTradeHistoryItem {
  ticker: string;
  name: string | null;
  side: "buy" | "sell";
  qty: number;
  price: number;
  amount: number;
  realized_pnl: number | null;
  created_at: string;
}
export interface PaperRankingItem {
  rank: number;
  nickname: string;
  total_value: number;
  total_pnl_pct: number;
  is_me: boolean;
}
export interface PaperRanking {
  rankings: PaperRankingItem[];
  my_rank: number | null;
  total_players: number;
}

export interface PaperHistoryPoint {
  date: string;
  total_value: number;
  pnl_pct: number;
}
export interface PaperHistory {
  points: PaperHistoryPoint[];
  chart_b64: string | null;
}

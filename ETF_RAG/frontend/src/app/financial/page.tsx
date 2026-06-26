"use client";

import { useState } from "react";
import { getFinancial } from "@/lib/api";
import type { FinancialResponse } from "@/lib/types";
import TickerSearch from "@/components/TickerSearch";
import ChartImage from "@/components/ChartImage";
import DataRangeNote from "@/components/DataRangeNote";
import { Loading, ErrorText } from "@/components/Feedback";
import WatchlistStar from "@/components/WatchlistStar";

// 억원 단위
function eok(v: number | null | undefined): string {
  if (typeof v !== "number") return "-";
  return `${Math.round(v / 1_0000_0000).toLocaleString("ko-KR")}억`;
}
function pct(v: number | null | undefined): string {
  return typeof v === "number" ? `${v.toFixed(1)}%` : "-";
}

// 조회 기간 옵션 (분기 수 ↔ 라벨). 1년=4분기.
const RANGES = [
  { quarters: 4, label: "1년" },
  { quarters: 8, label: "2년" },
  { quarters: 12, label: "3년" },
  { quarters: 20, label: "5년" },
  { quarters: 40, label: "10년" },
];
const MAX_YEARS = 12; // 데이터 보존 한계(2015~)에 맞춘 상한

export default function FinancialPage() {
  const [data, setData] = useState<FinancialResponse | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [ticker, setTicker] = useState<string | null>(null);
  const [quarters, setQuarters] = useState(12);
  const [customYears, setCustomYears] = useState(""); // 직접설정(년) 입력값

  const fetchFin = async (tk: string, q: number) => {
    setLoading(true);
    setError(null);
    try {
      const res = await getFinancial(tk, q);
      if (!res) setError("재무 데이터를 찾을 수 없어요. (재무제표는 상장 주식만 제공)");
      setData(res);
    } catch {
      setError("데이터를 가져오지 못했어요.");
    } finally {
      setLoading(false);
    }
  };

  const onSelect = (sel: { ticker: string }) => {
    setTicker(sel.ticker);
    fetchFin(sel.ticker, quarters);
  };

  const onRange = (q: number) => {
    setQuarters(q);
    setCustomYears("");
    if (ticker) fetchFin(ticker, q);
  };

  // 직접설정: 연 수 입력 → 분기(년×4)로 환산. 1~MAX_YEARS로 clamp.
  const applyCustom = () => {
    const years = Math.round(Number(customYears));
    if (!years || years < 1) return;
    const clamped = Math.min(years, MAX_YEARS);
    if (clamped !== years) setCustomYears(String(clamped));
    const q = clamped * 4;
    setQuarters(q);
    if (ticker) fetchFin(ticker, q);
  };

  return (
    <main className="mx-auto w-full max-w-3xl px-3 py-5 sm:px-4">
      <h1 className="mb-1 text-lg font-bold text-gray-900">📑 재무제표</h1>
      <p className="mb-4 text-xs text-gray-500">분기별 매출·영업이익·순이익·마진</p>

      <TickerSearch
        onSelect={onSelect}
        assetType="stock"
        placeholder="종목명 또는 종목코드 (주식만)"
      />

      {/* 조회 기간 선택 */}
      <div className="mt-3 flex flex-wrap items-center gap-1.5">
        {RANGES.map((r) => {
          const active = quarters === r.quarters && customYears === "";
          return (
            <button
              key={r.quarters}
              onClick={() => onRange(r.quarters)}
              className={`rounded-full px-3 py-1 text-xs font-medium transition ${
                active
                  ? "bg-gray-900 text-white"
                  : "bg-gray-100 text-gray-600 hover:bg-gray-200"
              }`}
            >
              {r.label}
            </button>
          );
        })}
        {/* 직접 설정 (년) */}
        <span className="ml-1 flex items-center gap-1">
          <input
            type="number"
            min={1}
            max={MAX_YEARS}
            value={customYears}
            onChange={(e) => setCustomYears(e.target.value)}
            onKeyDown={(e) => {
              if (e.key === "Enter") applyCustom();
            }}
            placeholder="직접"
            className="w-16 rounded-full border border-gray-300 px-2.5 py-1 text-xs focus:border-blue-500 focus:outline-none"
          />
          <span className="text-xs text-gray-500">년</span>
          <button
            onClick={applyCustom}
            disabled={!customYears}
            className="rounded-full bg-blue-600 px-2.5 py-1 text-xs font-medium text-white disabled:opacity-40"
          >
            적용
          </button>
        </span>
      </div>

      {loading && <Loading text="조회 중…" />}
      {error && <ErrorText message={error} />}

      {data && !loading && (
        <div className="mt-5">
          <div className="mb-3 flex items-center gap-2">
            <h2 className="text-base font-semibold">{data.name}</h2>
            <span className="text-xs text-gray-400">{data.ticker}</span>
            <WatchlistStar ticker={data.ticker} />
          </div>

          <div className="overflow-x-auto">
            <table className="comparison-table text-xs">
              <thead>
                <tr>
                  <th className="text-left">분기</th>
                  <th className="text-right">매출액</th>
                  <th className="text-right">영업이익</th>
                  <th className="text-right">순이익</th>
                  <th className="text-right">영업이익률</th>
                  <th className="text-right">매출 YoY</th>
                </tr>
              </thead>
              <tbody>
                {data.rows.map((r, i) => (
                  <tr key={i}>
                    <td className="text-left">
                      {r.fiscal_year} Q{r.fiscal_quarter}
                    </td>
                    <td className="text-right tabular-nums">{eok(r.revenue)}</td>
                    <td className="text-right tabular-nums">{eok(r.operating_profit)}</td>
                    <td className="text-right tabular-nums">{eok(r.net_income)}</td>
                    <td className="text-right tabular-nums">{pct(r.operating_margin)}</td>
                    <td className="text-right tabular-nums">{pct(r.revenue_growth_yoy)}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>

          <ChartImage b64={data.chart_b64} alt={`${data.name} 실적 추이`} />
        </div>
      )}

      <DataRangeNote />
    </main>
  );
}

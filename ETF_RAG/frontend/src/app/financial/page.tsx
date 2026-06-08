"use client";

import { useState } from "react";
import { getFinancial } from "@/lib/api";
import type { FinancialResponse } from "@/lib/types";
import TickerSearch from "@/components/TickerSearch";
import ChartImage from "@/components/ChartImage";

// 억원 단위
function eok(v: number | null | undefined): string {
  if (typeof v !== "number") return "-";
  return `${Math.round(v / 1_0000_0000).toLocaleString("ko-KR")}억`;
}
function pct(v: number | null | undefined): string {
  return typeof v === "number" ? `${v.toFixed(1)}%` : "-";
}

export default function FinancialPage() {
  const [data, setData] = useState<FinancialResponse | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const onSelect = async (sel: { ticker: string }) => {
    setLoading(true);
    setError(null);
    try {
      const res = await getFinancial(sel.ticker, 12);
      if (!res) setError("재무 데이터를 찾을 수 없어요. (재무제표는 상장 주식만 제공)");
      setData(res);
    } catch {
      setError("데이터를 가져오지 못했어요.");
    } finally {
      setLoading(false);
    }
  };

  return (
    <main className="mx-auto w-full max-w-3xl px-3 py-5 sm:px-4">
      <h1 className="mb-1 text-lg font-bold text-gray-900">📑 재무제표</h1>
      <p className="mb-4 text-xs text-gray-500">분기별 매출·영업이익·순이익·마진</p>

      <TickerSearch onSelect={onSelect} placeholder="종목명 또는 티커 (주식만)" />

      {loading && <p className="mt-6 text-center text-sm text-gray-400">조회 중…</p>}
      {error && <p className="mt-6 text-center text-sm text-red-600">{error}</p>}

      {data && !loading && (
        <div className="mt-5">
          <h2 className="mb-3 text-base font-semibold">
            {data.name} <span className="text-xs text-gray-400">{data.ticker}</span>
          </h2>

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
    </main>
  );
}

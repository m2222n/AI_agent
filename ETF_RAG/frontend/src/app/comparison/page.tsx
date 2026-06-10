"use client";

import { useState } from "react";
import { postComparison, getFinancial } from "@/lib/api";
import type {
  ComparisonResponse,
  ComparisonItem,
  FinancialResponse,
  FinancialRow,
} from "@/lib/types";
import TickerSearch from "@/components/TickerSearch";
import ChartImage from "@/components/ChartImage";
import ComparisonTable from "@/components/ComparisonTable";
import DataRangeNote from "@/components/DataRangeNote";

// 상대 수익률 차트 기간 (기술 분석 탭과 동일 옵션)
const PERIODS: { label: string; days: number }[] = [
  { label: "6개월", days: 120 },
  { label: "1년", days: 250 },
  { label: "3년", days: 750 },
  { label: "5년", days: 1250 },
  { label: "10년", days: 2500 },
];

export default function ComparisonPage() {
  const [t1, setT1] = useState<{ name: string; ticker: string } | null>(null);
  const [t2, setT2] = useState<{ name: string; ticker: string } | null>(null);
  const [days, setDays] = useState(120);
  const [data, setData] = useState<ComparisonResponse | null>(null);
  const [fin, setFin] = useState<(FinancialResponse | null)[] | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const run = async (
    a: { ticker: string },
    b: { ticker: string },
    d: number,
  ) => {
    setLoading(true);
    setError(null);
    setFin(null);
    try {
      const res = await postComparison([a.ticker, b.ticker], d);
      if (!res) setError("비교할 종목을 찾을 수 없어요.");
      setData(res);
      // 최근 분기 실적 비교 (각 종목 재무제표 병렬 — 없으면 null, 표 자체 생략)
      try {
        const [f1, f2] = await Promise.all([
          getFinancial(a.ticker, 8),
          getFinancial(b.ticker, 8),
        ]);
        if (f1 || f2) setFin([f1, f2]);
      } catch {
        /* 재무제표 실패는 비교 자체를 막지 않음 */
      }
    } catch {
      setError("데이터를 가져오지 못했어요.");
    } finally {
      setLoading(false);
    }
  };

  const pick1 = (sel: { name: string; ticker: string }) => {
    setT1(sel);
    if (t2) run(sel, t2, days);
  };
  const pick2 = (sel: { name: string; ticker: string }) => {
    setT2(sel);
    if (t1) run(t1, sel, days);
  };
  const onPeriod = (d: number) => {
    setDays(d);
    if (t1 && t2) run(t1, t2, d);
  };

  return (
    <main className="mx-auto w-full max-w-3xl px-3 py-5 sm:px-4">
      <h1 className="mb-1 text-lg font-bold text-gray-900">⚖️ 비교 분석</h1>
      <p className="mb-4 text-xs text-gray-500">두 종목의 시세·밸류에이션·상대 수익률</p>

      <div className="grid gap-3 sm:grid-cols-2">
        <div>
          <div className="mb-1 text-xs text-gray-500">종목 1</div>
          <TickerSearch onSelect={pick1} />
        </div>
        <div>
          <div className="mb-1 text-xs text-gray-500">종목 2</div>
          <TickerSearch onSelect={pick2} />
        </div>
      </div>

      {/* 상대 수익률 차트 기간 선택 */}
      <div className="mt-3 flex flex-wrap gap-1.5">
        {PERIODS.map((p) => (
          <button
            key={p.days}
            type="button"
            onClick={() => onPeriod(p.days)}
            className={`rounded-full px-3 py-1 text-xs font-medium transition ${
              days === p.days
                ? "bg-blue-600 text-white"
                : "bg-gray-100 text-gray-600 hover:bg-gray-200"
            }`}
          >
            {p.label}
          </button>
        ))}
      </div>

      {loading && <p className="mt-6 text-center text-sm text-gray-400">비교 중…</p>}
      {error && <p className="mt-6 text-center text-sm text-red-600">{error}</p>}

      {data && !loading && (
        <div className="mt-5">
          <ComparisonTable items={data.items as ComparisonItem[]} />
          <ChartImage b64={data.comparison_chart_b64} alt="상대 수익률 추이" />
          <ChartImage b64={data.valuation_chart_b64} alt="밸류에이션 비교" />
          {fin && <FinancialCompare fin={fin} />}
        </div>
      )}

      <DataRangeNote />
    </main>
  );
}

// 두 종목의 최근 분기 실적 나란히 비교 (둘 중 하나라도 재무제표 있으면 표시)
function FinancialCompare({ fin }: { fin: (FinancialResponse | null)[] }) {
  const cols = fin.map((f) => {
    const last = f?.rows?.[f.rows.length - 1];
    return f && last ? { name: f.name, row: last } : null;
  });
  if (!cols.some(Boolean)) return null;

  const 억 = (v: number | null | undefined) =>
    typeof v === "number" ? `${Math.round(v / 1_0000_0000).toLocaleString("ko-KR")}억원` : "-";
  const pct = (v: number | null | undefined) =>
    typeof v === "number" ? `${v > 0 ? "+" : ""}${v.toFixed(2)}%` : "-";

  const metrics: { label: string; get: (r: FinancialRow) => number | null | undefined; fmt: (v: number | null | undefined) => string }[] = [
    { label: "매출액", get: (r) => r.revenue, fmt: 억 },
    { label: "영업이익", get: (r) => r.operating_profit, fmt: 억 },
    { label: "순이익", get: (r) => r.net_income, fmt: 억 },
    { label: "영업이익률", get: (r) => r.operating_margin, fmt: pct },
    { label: "순이익률", get: (r) => r.net_margin, fmt: pct },
    { label: "매출 YoY", get: (r) => r.revenue_growth_yoy, fmt: pct },
    { label: "영업이익 YoY", get: (r) => r.op_growth_yoy, fmt: pct },
  ];

  return (
    <div className="mt-6">
      <h2 className="mb-2 text-sm font-semibold text-gray-900">최근 분기 실적 비교</h2>
      <div className="overflow-x-auto">
        <table className="comparison-table text-xs">
          <thead>
            <tr>
              <th className="text-left">항목</th>
              {cols.map((c, i) => (
                <th key={i} className="text-right">
                  {c?.name ?? "-"}
                  {c?.row ? (
                    <span className="block font-normal text-gray-400">
                      {c.row.fiscal_year} {c.row.fiscal_quarter}Q
                    </span>
                  ) : null}
                </th>
              ))}
            </tr>
          </thead>
          <tbody>
            {metrics.map((m) => (
              <tr key={m.label}>
                <td className="text-left text-gray-500">{m.label}</td>
                {cols.map((c, i) => (
                  <td key={i} className="text-right tabular-nums">
                    {c?.row ? m.fmt(m.get(c.row)) : "-"}
                  </td>
                ))}
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </div>
  );
}

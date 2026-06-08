"use client";

import { useState } from "react";
import { getTechnical } from "@/lib/api";
import type { TechnicalResponse } from "@/lib/types";
import TickerSearch from "@/components/TickerSearch";

const PERIODS: { label: string; days: number }[] = [
  { label: "6개월", days: 120 },
  { label: "1년", days: 250 },
  { label: "3년", days: 750 },
  { label: "5년", days: 1250 },
];

// summary에서 안전하게 값 추출
function n(v: unknown): string {
  if (typeof v === "number") return v.toLocaleString("ko-KR");
  return "-";
}
function obj(v: unknown): Record<string, unknown> {
  return v && typeof v === "object" ? (v as Record<string, unknown>) : {};
}

export default function TechnicalPage() {
  const [days, setDays] = useState(120);
  const [data, setData] = useState<TechnicalResponse | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [selected, setSelected] = useState<{ name: string; ticker: string } | null>(null);

  const run = async (ticker: string, d: number) => {
    setLoading(true);
    setError(null);
    try {
      const res = await getTechnical(ticker, d);
      if (!res) setError("해당 종목의 기술적 데이터를 찾을 수 없어요.");
      setData(res);
    } catch {
      setError("데이터를 가져오지 못했어요. 백엔드 상태를 확인해주세요.");
    } finally {
      setLoading(false);
    }
  };

  const onSelect = (sel: { name: string; ticker: string }) => {
    setSelected(sel);
    run(sel.ticker, days);
  };

  const onPeriod = (d: number) => {
    setDays(d);
    if (selected) run(selected.ticker, d);
  };

  const s = data?.summary ?? {};
  const ma = obj(s.ma);
  const macd = obj(s.macd);
  const boll = obj(s.bollinger);

  return (
    <main className="mx-auto w-full max-w-3xl px-3 py-5 sm:px-4">
      <h1 className="mb-1 text-lg font-bold text-gray-900">📊 기술적 분석</h1>
      <p className="mb-4 text-xs text-gray-500">
        이동평균·RSI·MACD·볼린저밴드 등 지표와 차트
      </p>

      <TickerSearch onSelect={onSelect} />

      {selected && (
        <div className="mt-3 flex flex-wrap gap-2">
          {PERIODS.map((p) => (
            <button
              key={p.days}
              type="button"
              onClick={() => onPeriod(p.days)}
              className={[
                "rounded-full px-3 py-1 text-xs",
                days === p.days
                  ? "bg-blue-600 text-white"
                  : "border border-gray-300 text-gray-600 hover:bg-gray-100",
              ].join(" ")}
            >
              {p.label}
            </button>
          ))}
        </div>
      )}

      {loading && (
        <p className="mt-6 text-center text-sm text-gray-400">분석 중…</p>
      )}
      {error && <p className="mt-6 text-center text-sm text-red-600">{error}</p>}

      {data && !loading && (
        <div className="mt-5">
          <div className="mb-3 flex items-baseline gap-2">
            <h2 className="text-base font-semibold">{data.name}</h2>
            <span className="text-xs text-gray-400">{data.ticker}</span>
          </div>

          {/* 핵심 지표 */}
          <div className="grid grid-cols-2 gap-2 sm:grid-cols-4">
            <Metric label="종가" value={`${n(s.close)}원`} />
            <Metric label="추세" value={String(s.trend ?? "-")} />
            <Metric label="RSI(14)" value={n(s.rsi)} />
            <Metric label="MACD" value={n(macd.macd)} />
            <Metric label="MA5" value={n(ma.ma5)} />
            <Metric label="MA20" value={n(ma.ma20)} />
            <Metric label="MA60" value={n(ma.ma60)} />
            <Metric label="볼린저 %B" value={n(boll.pct_b)} />
          </div>

          {/* 차트 */}
          {data.chart_b64 && (
            <img
              // eslint-disable-next-line @next/next/no-img-element
              src={`data:image/png;base64,${data.chart_b64}`}
              alt={`${data.name} 기술적 분석 차트`}
              className="mt-4 w-full rounded-lg border border-gray-200"
            />
          )}
        </div>
      )}
    </main>
  );
}

function Metric({ label, value }: { label: string; value: string }) {
  return (
    <div className="rounded-lg border border-gray-200 px-3 py-2">
      <div className="text-xs text-gray-500">{label}</div>
      <div className="text-sm font-semibold tabular-nums">{value}</div>
    </div>
  );
}

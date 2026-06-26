"use client";

import { useState } from "react";
import { getOutlook } from "@/lib/api";
import type { OutlookResponse } from "@/lib/types";
import TickerSearch from "@/components/TickerSearch";
import DataRangeNote from "@/components/DataRangeNote";
import { Loading, ErrorText } from "@/components/Feedback";
import WatchlistStar from "@/components/WatchlistStar";

const HORIZONS = ["1m", "3m", "6m", "1y"];

function str(v: unknown): string {
  return v == null ? "-" : String(v);
}
function axisFactors(axis: Record<string, unknown> | undefined): string[] {
  const f = axis?.key_factors;
  return Array.isArray(f) ? (f as string[]) : [];
}
function num(v: unknown): number | null {
  return typeof v === "number" ? v : null;
}
function signedPct(v: unknown): string {
  const n = num(v);
  return n == null ? "-" : `${n > 0 ? "+" : ""}${n.toFixed(2)}%`;
}
// statistical/prophet 축은 예측 수익률·신뢰구간·추세 형태(key_factors 없음) → 별도 렌더
function forecastView(axis: Record<string, unknown> | undefined) {
  if (!axis) return null;
  const ci = axis.confidence_interval;
  const hasForecast =
    "predicted_return" in axis || (Array.isArray(ci) && ci.length === 2);
  if (!hasForecast) return null;
  const [lo, hi] = Array.isArray(ci) ? ci : [undefined, undefined];
  return {
    predicted: signedPct(axis.predicted_return),
    ci:
      num(lo) != null && num(hi) != null
        ? `${signedPct(lo)} ~ ${signedPct(hi)}`
        : null,
    trend: typeof axis.trend === "string" ? axis.trend : null,
    available: axis.available !== false,
  };
}

export default function OutlookPage() {
  const [horizon, setHorizon] = useState("1m");
  const [selected, setSelected] = useState<{ ticker: string; name?: string } | null>(null);
  const [data, setData] = useState<OutlookResponse | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const run = async (ticker: string, h: string) => {
    setLoading(true);
    setError(null);
    try {
      const res = await getOutlook(ticker, h);
      if (!res) setError("전망을 생성할 수 없어요.");
      setData(res);
    } catch {
      setError("데이터를 가져오지 못했어요.");
    } finally {
      setLoading(false);
    }
  };

  const onSelect = (sel: { ticker: string; name: string }) => {
    setSelected(sel);
    run(sel.ticker, horizon);
  };
  const onHorizon = (h: string) => {
    setHorizon(h);
    if (selected) run(selected.ticker, h);
  };

  const scenarios = data?.scenarios ?? {};

  return (
    <main className="mx-auto w-full max-w-3xl px-3 py-5 sm:px-4">
      <h1 className="mb-1 text-lg font-bold text-gray-900">🔮 가격 전망</h1>
      <p className="mb-4 text-xs text-gray-500">
        기술적·펀더멘털·통계·Prophet 4축 종합 + 시나리오
      </p>

      <TickerSearch onSelect={onSelect} minDays={60} />

      {selected && (
        <div className="mt-3 flex flex-wrap gap-2">
          {HORIZONS.map((h) => (
            <button
              key={h}
              type="button"
              onClick={() => onHorizon(h)}
              className={[
                "rounded-full px-3 py-1 text-xs",
                horizon === h
                  ? "bg-blue-600 text-white"
                  : "border border-gray-300 text-gray-600 hover:bg-gray-100",
              ].join(" ")}
            >
              {h}
            </button>
          ))}
        </div>
      )}

      {loading && <Loading text="분석 중…" />}
      {error && <ErrorText message={error} />}

      {data && !loading && (
        <div className="mt-5 space-y-4">
          {selected && (
            <div className="flex items-center gap-2">
              <h2 className="text-base font-semibold">{selected.name ?? selected.ticker}</h2>
              <span className="text-xs text-gray-400">{selected.ticker}</span>
              <WatchlistStar ticker={selected.ticker} />
            </div>
          )}
          <div className="flex flex-wrap gap-2">
            <Metric label="종합 점수" value={str(data.composite_score)} />
            <Metric label="신뢰 등급" value={str(data.confidence_grade)} />
            <Metric label="현재가" value={`${str(data.current_price)}원`} />
          </div>

          {/* 4축 */}
          <div className="grid gap-2 sm:grid-cols-2">
            <Axis title="기술적" axis={data.technical} />
            <Axis title="펀더멘털" axis={data.fundamental} />
            <Axis title="통계(회귀)" axis={data.statistical} />
            <Axis title="Prophet 시계열" axis={data.prophet} accent />
          </div>

          {/* 시나리오 */}
          <div className="grid grid-cols-1 gap-2 sm:grid-cols-3">
            {(["bullish", "neutral", "bearish"] as const).map((k) => {
              const sc = scenarios[k];
              const label = { bullish: "🔼 상승", neutral: "➖ 중립", bearish: "🔽 하락" }[k];
              return (
                <div key={k} className="rounded-lg border border-gray-200 p-3 text-xs">
                  <div className="font-semibold">{label}</div>
                  <div className="mt-1 text-gray-500">
                    확률 {sc?.probability != null ? `${Math.round(sc.probability * 100)}%` : "-"}
                  </div>
                  <div className="text-gray-500">
                    목표 {sc?.target_return != null ? `${sc.target_return}%` : "-"}
                  </div>
                </div>
              );
            })}
          </div>

          {/* 리스크 */}
          {Array.isArray(data.risk_factors) && data.risk_factors.length > 0 && (
            <div className="rounded-lg border border-amber-200 bg-amber-50 p-3 text-xs">
              <div className="mb-1 font-semibold text-amber-800">⚠️ 리스크 요인</div>
              <ul className="list-disc pl-4 text-amber-700">
                {data.risk_factors.map((r, i) => (
                  <li key={i}>{r}</li>
                ))}
              </ul>
            </div>
          )}

        </div>
      )}

      <p className="mt-4 text-[11px] leading-relaxed text-gray-400">
        ℹ️ 가격 전망은 회귀·Prophet 등 모델 학습에 충분한 과거 시세가 필요해, 상장
        직후라 시세가 60거래일 미만인 신규 종목은 검색에서 자동 제외됩니다.
      </p>
      <DataRangeNote />
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

function Axis({
  title,
  axis,
  accent = false,
}: {
  title: string;
  axis: Record<string, unknown> | undefined;
  accent?: boolean;
}) {
  const factors = axisFactors(axis);
  const fc = forecastView(axis);
  return (
    <div
      className={[
        "rounded-lg border p-3",
        accent ? "border-violet-300 bg-violet-50" : "border-gray-200",
      ].join(" ")}
    >
      <div className="flex items-baseline justify-between">
        <span className="text-sm font-semibold">
          {accent && <span className="mr-1">📈</span>}
          {title}
        </span>
        {/* signal 형태 축(기술/펀더멘털) → signal, 예측 형태 축(통계/Prophet) → 추세 */}
        <span className="text-xs text-gray-500">
          {fc ? fc.trend ?? "-" : str(axis?.signal)}
        </span>
      </div>

      {/* 예측 수익률 + 신뢰구간 (통계/Prophet) */}
      {fc &&
        (fc.available ? (
          <div className="mt-1.5 text-xs">
            <div className="font-semibold tabular-nums text-gray-800">
              예측 수익률 {fc.predicted}
            </div>
            {fc.ci && (
              <div className="text-gray-500 tabular-nums">
                신뢰구간 {fc.ci}
              </div>
            )}
          </div>
        ) : (
          <div className="mt-1.5 text-xs text-gray-400">
            데이터 부족으로 예측 불가
          </div>
        ))}

      {/* key_factors (기술/펀더멘털) */}
      {factors.length > 0 && (
        <ul className="mt-1 list-disc pl-4 text-xs text-gray-600">
          {factors.slice(0, 4).map((f, i) => (
            <li key={i}>{f}</li>
          ))}
        </ul>
      )}
    </div>
  );
}

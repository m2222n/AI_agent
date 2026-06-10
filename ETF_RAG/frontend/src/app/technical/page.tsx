"use client";

import { useEffect, useState } from "react";
import { getTechnical, getIntraday } from "@/lib/api";
import type { TechnicalResponse } from "@/lib/types";
import TickerSearch from "@/components/TickerSearch";
import WatchlistStar from "@/components/WatchlistStar";

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
function str(v: unknown): string {
  return v == null || v === "" ? "-" : String(v);
}

export default function TechnicalPage() {
  const [days, setDays] = useState(120);
  const [data, setData] = useState<TechnicalResponse | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [selected, setSelected] = useState<{ name: string; ticker: string } | null>(null);
  const [intraday, setIntraday] = useState<string | null>(null);
  const [intradayMsg, setIntradayMsg] = useState<string | null>(null);

  const run = async (ticker: string, d: number) => {
    setLoading(true);
    setError(null);
    setIntraday(null);
    setIntradayMsg(null);
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

  const loadIntraday = async () => {
    if (!data) return;
    setIntradayMsg("불러오는 중…");
    try {
      const res = await getIntraday(data.ticker);
      if (res) {
        setIntraday(res.chart_b64);
        setIntradayMsg(null);
      } else {
        setIntradayMsg("장중 시세를 불러올 수 없어요. (장 외 시간이거나 데이터 없음)");
      }
    } catch {
      setIntradayMsg("장중 시세 조회 실패.");
    }
  };

  // 관심종목 칩 등에서 /technical?ticker=005930 로 진입 시 자동 조회 (mount 1회)
  useEffect(() => {
    const t = new URLSearchParams(window.location.search).get("ticker");
    if (t) {
      setSelected({ name: t, ticker: t });
      run(t, 120);
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  const onPeriod = (d: number) => {
    setDays(d);
    if (selected) run(selected.ticker, d);
  };

  const s = data?.summary ?? {};
  const ma = obj(s.ma);
  const macd = obj(s.macd);
  const boll = obj(s.bollinger);
  const stoch = obj(s.stochastic);
  const cci = obj(s.cci);
  const adx = obj(s.adx);
  const ichi = obj(s.ichimoku);
  const obv = obj(s.obv);
  const atr = obj(s.atr);
  // 골든/데드 크로스 — cross: {"5_20": {type,label}|null, ...}
  const cross = obj(s.cross);
  const crossLabels = Object.values(cross)
    .filter((c): c is Record<string, unknown> => !!c && typeof c === "object")
    .map((c) => {
      const t = c.type;
      const label = String(c.label ?? "");
      const icon = t === "golden" ? "🟢" : t === "dead" ? "🔴" : "";
      return `${icon} ${label}`.trim();
    })
    .filter((x) => x.length > 1);

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
          <div className="mb-3 flex items-center gap-2">
            <h2 className="text-base font-semibold">{data.name}</h2>
            <span className="text-xs text-gray-400">{data.ticker}</span>
            <WatchlistStar ticker={data.ticker} />
          </div>

          {/* 핵심 지표 */}
          <div className="grid grid-cols-2 gap-2 sm:grid-cols-4">
            <Metric label="종가" value={`${n(s.close)}원`} />
            <Metric label="추세" value={String(s.trend ?? "-")} />
            <Metric label="MA5" value={n(ma.ma5)} />
            <Metric label="MA20" value={n(ma.ma20)} />
            <Metric label="MA60" value={n(ma.ma60)} />
            <Metric label="MA120" value={n(ma.ma120)} />
            <Metric label="RSI(14)" value={n(s.rsi)} />
            <Metric label="볼린저 %B" value={n(boll.pct_b)} />
          </div>

          {/* 골든/데드 크로스 */}
          {crossLabels.length > 0 && (
            <div className="mt-2 rounded-lg border border-gray-200 bg-gray-50 px-3 py-2 text-xs">
              {crossLabels.map((c, i) => (
                <span key={i} className="mr-3">
                  {c}
                </span>
              ))}
            </div>
          )}

          {/* 모멘텀 / 추세·변동성 2단 */}
          <div className="mt-3 grid gap-3 sm:grid-cols-2">
            <IndicatorCard title="📈 모멘텀">
              <Row label="MACD" value={`${n(macd.macd)} (시그널 ${n(macd.signal)})`} />
              <Row label="스토캐스틱" value={stoch.k != null ? `%K ${n(stoch.k)} / %D ${n(stoch.d)} ${str(stoch.signal)}` : "-"} />
              <Row label="CCI(20)" value={cci.cci != null ? `${n(cci.cci)} ${str(cci.signal)}` : "-"} />
            </IndicatorCard>
            <IndicatorCard title="📉 추세 / 변동성">
              <Row label="볼린저" value={boll.lower != null ? `${n(boll.lower)}~${n(boll.upper)}` : "-"} />
              <Row label="ADX(14)" value={adx.adx != null ? `${n(adx.adx)} ${str(adx.trend_strength)}` : "-"} />
              <Row label="일목균형표" value={str(ichi.cloud_status)} />
              <Row label="OBV" value={str(obv.trend)} />
              <Row label="ATR(14)" value={atr.atr_pct != null ? `${n(atr.atr_pct)}% ${str(atr.volatility)}` : "-"} />
            </IndicatorCard>
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

          {/* 장중 시세 차트 (yfinance 15분봉) */}
          <div className="mt-4">
            {!intraday && (
              <button
                type="button"
                onClick={loadIntraday}
                className="rounded-lg border border-gray-300 px-3 py-1.5 text-xs text-gray-600 hover:bg-gray-100"
              >
                📈 장중 시세 보기
              </button>
            )}
            {intradayMsg && (
              <p className="mt-1 text-xs text-gray-400">{intradayMsg}</p>
            )}
            {intraday && (
              <>
                <img
                  // eslint-disable-next-line @next/next/no-img-element
                  src={`data:image/png;base64,${intraday}`}
                  alt={`${data.name} 장중 시세`}
                  className="w-full rounded-lg border border-gray-200"
                />
                <p className="mt-1 text-xs text-gray-400">
                  ⏱ yfinance 15분봉 (약 15분 지연)
                </p>
              </>
            )}
          </div>
        </div>
      )}
    </main>
  );
}

function IndicatorCard({
  title,
  children,
}: {
  title: string;
  children: React.ReactNode;
}) {
  return (
    <div className="rounded-lg border border-gray-200 p-3">
      <div className="mb-1 text-xs font-semibold text-gray-700">{title}</div>
      <div className="space-y-1">{children}</div>
    </div>
  );
}

function Row({ label, value }: { label: string; value: string }) {
  return (
    <div className="flex justify-between text-xs">
      <span className="text-gray-500">{label}</span>
      <span className="tabular-nums text-gray-800">{value}</span>
    </div>
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

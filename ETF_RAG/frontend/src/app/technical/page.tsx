"use client";

import { Suspense, useEffect, useState } from "react";
import { useSearchParams } from "next/navigation";
import { getTechnical, getIntraday } from "@/lib/api";
import type { TechnicalResponse } from "@/lib/types";
import TickerSearch from "@/components/TickerSearch";
import WatchlistStar from "@/components/WatchlistStar";
import DataRangeNote from "@/components/DataRangeNote";
import { Loading, ErrorText } from "@/components/Feedback";
import PriceCard from "@/components/PriceCard";
import OrderbookCard from "@/components/OrderbookCard";

const PERIODS: { label: string; days: number }[] = [
  { label: "6개월", days: 120 },
  { label: "1년", days: 250 },
  { label: "3년", days: 750 },
  { label: "5년", days: 1250 },
  { label: "10년", days: 2500 },
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

function TechnicalInner() {
  const searchParams = useSearchParams();
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

  // 사이드바/관심종목에서 /technical?ticker=X 진입 시 자동 조회.
  // useSearchParams로 쿼리 변화를 감지 — 이미 이 페이지에 있을 때 다른 종목을
  // 클릭하면 URL만 바뀌고 리마운트가 안 돼(mount 1회 useEffect는 안 돎) 종목이
  // 안 바뀌던 버그 수정.
  const queryTicker = searchParams.get("ticker");
  useEffect(() => {
    if (queryTicker) {
      setSelected({ name: queryTicker, ticker: queryTicker });
      run(queryTicker, 120);
      setDays(120);
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [queryTicker]);

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

      <TickerSearch onSelect={onSelect} minDays={20} />

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

      {loading && <Loading text="분석 중…" />}
      {error && <ErrorText message={error} />}

      {data && !loading && (
        <div className="mt-5">
          <div className="mb-3 flex items-center gap-2">
            <h2 className="text-base font-semibold">{data.name}</h2>
            <span className="text-xs text-gray-400">{data.ticker}</span>
            <WatchlistStar ticker={data.ticker} />
          </div>

          {/* 실시간 시세 (KIS 우선 → yfinance, 장 외엔 종가) */}
          <PriceCard ticker={data.ticker} />

          {/* 선택 기간 대비 등락률 (종가 기준) */}
          {(() => {
            const cur = Number(s.close);
            const base = Number(s.first_close);
            const fd = String(s.first_date ?? "");
            if (!cur || !base || base <= 0) return null;
            const pct = ((cur - base) / base) * 100;
            const label = PERIODS.find((p) => p.days === days)?.label ?? "기간";
            const color =
              pct > 0 ? "text-red-600" : pct < 0 ? "text-blue-600" : "text-gray-500";
            const fdFmt =
              fd.length === 8 ? `${fd.slice(0, 4)}.${fd.slice(4, 6)}.${fd.slice(6, 8)}` : fd;
            return (
              <p className="mt-2 text-xs text-gray-500">
                {label} 전({fdFmt}) 대비{" "}
                <span className={`font-semibold ${color}`}>
                  {pct > 0 ? "+" : ""}
                  {pct.toFixed(2)}%
                </span>{" "}
                <span className="text-gray-400">
                  ({base.toLocaleString("ko-KR")} → {cur.toLocaleString("ko-KR")}원)
                </span>
              </p>
            );
          })()}

          {/* 호가 10단계 (KIS 전용, 장중에만 표시) */}
          <div className="mt-3">
            <OrderbookCard ticker={data.ticker} />
          </div>

          {/* 핵심 지표 */}
          <div className="grid grid-cols-2 gap-2 sm:grid-cols-4 max-[380px]:grid-cols-1">
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

      <p className="mt-4 text-[11px] leading-relaxed text-gray-400">
        ℹ️ 기술적 분석은 이동평균·RSI 등 계산에 충분한 과거 시세가 필요해, 상장 직후라
        시세가 20거래일 미만인 신규 종목은 검색에서 자동 제외됩니다.
      </p>
      <DataRangeNote />
    </main>
  );
}

// useSearchParams는 Suspense 경계 필요(Next.js CSR bailout) → 래핑.
export default function TechnicalPage() {
  return (
    <Suspense fallback={<main className="mx-auto w-full max-w-3xl px-3 py-5" />}>
      <TechnicalInner />
    </Suspense>
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

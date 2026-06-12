"use client";

import { useEffect, useRef, useState } from "react";
import { getPrice } from "@/lib/api";
import type { PriceData } from "@/lib/types";

/**
 * 종목 실시간 시세 카드.
 * - 장중: KIS 실시간(또는 yfinance 지연)을 30초마다 자동 갱신
 * - 장 외: 수집 종가 1회 표시(자동 갱신 없음)
 * source 배지로 출처 구분(🔴 실시간 / 🟡 지연 / 종가).
 */
export default function PriceCard({ ticker }: { ticker: string }) {
  const [data, setData] = useState<PriceData | null>(null);
  const [loading, setLoading] = useState(true);
  const timer = useRef<ReturnType<typeof setInterval> | null>(null);

  useEffect(() => {
    let alive = true;
    setLoading(true);
    setData(null);

    const load = async () => {
      try {
        const p = await getPrice(ticker);
        if (alive) setData(p);
      } catch {
        if (alive) setData(null);
      } finally {
        if (alive) setLoading(false);
      }
    };

    load().then(() => {
      // 장중일 때만 30초 폴링 (load 결과는 다음 tick에서 확인 — is_live 기준)
      if (timer.current) clearInterval(timer.current);
      timer.current = setInterval(async () => {
        try {
          const p = await getPrice(ticker);
          if (alive && p && p.is_live) setData(p);
          else if (alive && p && !p.market_open && timer.current) {
            clearInterval(timer.current); // 장 마감 감지 시 폴링 중단
          }
        } catch {
          /* 일시 오류 무시 */
        }
      }, 30_000);
    });

    return () => {
      alive = false;
      if (timer.current) clearInterval(timer.current);
    };
  }, [ticker]);

  if (loading) {
    return (
      <div className="rounded-xl border border-gray-200 bg-gray-50 px-4 py-3 text-xs text-gray-400">
        시세 불러오는 중…
      </div>
    );
  }
  if (!data) return null;

  const up = (data.change_pct ?? 0) > 0;
  const down = (data.change_pct ?? 0) < 0;
  const color = up ? "text-red-600" : down ? "text-blue-600" : "text-gray-500";

  const badge =
    data.source === "kis"
      ? { label: "🔴 실시간 (KIS)", cls: "bg-red-50 text-red-600" }
      : data.source === "yfinance"
        ? { label: "🟡 15분 지연 (yfinance)", cls: "bg-yellow-50 text-yellow-700" }
        : { label: "종가", cls: "bg-gray-100 text-gray-500" };

  return (
    <div className="rounded-xl border border-gray-200 bg-white px-4 py-3">
      <div className="flex items-center justify-between gap-2">
        <div className="flex items-baseline gap-2">
          <span className="text-xl font-bold tabular-nums text-gray-900">
            {data.price.toLocaleString("ko-KR")}원
          </span>
          {data.change_pct != null && (
            <span className={`text-sm font-medium tabular-nums ${color}`}>
              {data.change != null
                ? `${up ? "+" : ""}${data.change.toLocaleString("ko-KR")}원 `
                : ""}
              ({up ? "+" : ""}
              {data.change_pct}%)
            </span>
          )}
        </div>
        <span className={`shrink-0 rounded-full px-2 py-0.5 text-[11px] ${badge.cls}`}>
          {badge.label}
        </span>
      </div>
      <div className="mt-1 flex items-center gap-3 text-[11px] text-gray-400">
        {data.volume != null && (
          <span>거래량 {data.volume.toLocaleString("ko-KR")}주</span>
        )}
        {data.timestamp && <span>{data.timestamp}</span>}
        {data.source === "close" && !data.market_open && <span>장 마감</span>}
      </div>
    </div>
  );
}

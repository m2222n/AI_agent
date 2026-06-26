"use client";

import { useEffect, useRef, useState } from "react";
import { getPrice, streamPrice } from "@/lib/api";
import type { PriceData } from "@/lib/types";

/**
 * 종목 실시간 시세 카드.
 * - 1순위: KIS WebSocket SSE(체결 틱 실시간) — source 배지 "실시간 (KIS)"
 * - fallback: SSE 미연동/장외/오류 시 REST 폴링(장중 30초; KIS우선→yfinance→종가)
 * 먼저 REST 1회로 기준값(prev_close 등)을 채운 뒤 SSE 틱으로 price/change를 갱신한다.
 */
export default function PriceCard({ ticker }: { ticker: string }) {
  const [data, setData] = useState<PriceData | null>(null);
  const [loading, setLoading] = useState(true);
  const [live, setLive] = useState(false); // SSE 틱 수신 중인지
  const pollTimer = useRef<ReturnType<typeof setInterval> | null>(null);
  const sseCtrl = useRef<AbortController | null>(null);

  useEffect(() => {
    let alive = true;
    setLoading(true);
    setData(null);
    setLive(false);

    const clearPoll = () => {
      if (pollTimer.current) clearInterval(pollTimer.current);
      pollTimer.current = null;
    };

    const startPolling = () => {
      clearPoll();
      pollTimer.current = setInterval(async () => {
        try {
          const p = await getPrice(ticker);
          if (alive && p && p.is_live) setData(p);
          else if (alive && p && !p.market_open) clearPoll(); // 장 마감
        } catch {
          /* ignore */
        }
      }, 30_000);
    };

    // 1) REST 1회로 기준값 확보
    getPrice(ticker)
      .then((p) => {
        if (!alive) return;
        if (p) setData(p);
        setLoading(false);

        // 2) KIS WebSocket SSE 시도 (틱 수신 시 live, 아니면 폴링 fallback)
        sseCtrl.current = streamPrice(ticker, {
          onTick: (t) => {
            if (!alive) return;
            setLive(true);
            clearPoll(); // 실시간 받으면 폴링 불필요
            setData((prev) => ({
              ...(prev ?? ({} as PriceData)),
              name: prev?.name ?? "",
              ticker: t.ticker || ticker,
              price: t.price,
              change: t.change,
              change_pct: t.change_pct,
              volume: t.volume ?? prev?.volume ?? null,
              prev_close: prev?.prev_close ?? null,
              source: "kis",
              is_live: true,
              market_open: true,
              timestamp: t.timestamp ?? prev?.timestamp ?? null,
            }));
          },
          onUnavailable: () => {
            if (alive) startPolling(); // KIS 미연동/장외 → 폴링
          },
          onError: () => {
            if (alive && !live) startPolling();
          },
        });
      })
      .catch(() => {
        if (alive) setLoading(false);
      });

    return () => {
      alive = false;
      clearPoll();
      if (sseCtrl.current) sseCtrl.current.abort();
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [ticker]);

  if (loading) {
    return (
      <div className="rounded-xl border border-gray-200 dark:border-gray-800 bg-gray-50 px-4 py-3 text-xs text-gray-400">
        시세 불러오는 중…
      </div>
    );
  }
  if (!data) return null;

  const up = (data.change_pct ?? 0) > 0;
  const down = (data.change_pct ?? 0) < 0;
  const color = up ? "text-red-600" : down ? "text-blue-600" : "text-gray-500";

  const badge =
    live || data.source === "kis"
      ? {
          label: live ? "🔴 실시간 (KIS)" : "🔴 KIS",
          cls: "bg-red-50 text-red-600",
        }
      : data.source === "yfinance"
        ? { label: "🟡 15분 지연 (yfinance)", cls: "bg-yellow-50 text-yellow-700" }
        : { label: "종가", cls: "bg-gray-100 text-gray-500" };

  // 현재가(장중)인지 종가(장외)인지에 따라 라벨 구분
  const isLivePrice = data.source !== "close";
  const headLabel = isLivePrice ? "현재 시세" : "최근 종가";

  return (
    <div className="rounded-xl border border-gray-200 dark:border-gray-800 bg-white dark:bg-gray-900 px-4 py-3">
      <div className="flex items-start justify-between gap-2">
        <span className="text-[11px] font-medium text-gray-500">{headLabel}</span>
        <span
          className={`shrink-0 rounded-full px-2 py-0.5 text-[11px] ${badge.cls} ${live ? "animate-pulse" : ""}`}
        >
          {badge.label}
        </span>
      </div>
      <div className="mt-0.5 flex items-baseline gap-2">
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
      {/* 등락 기준 명시 */}
      {data.change_pct != null && (
        <div className="mt-0.5 text-[11px] text-gray-400">
          전일 종가 대비
          {data.prev_close != null &&
            ` (전일 ${data.prev_close.toLocaleString("ko-KR")}원)`}
        </div>
      )}
      <div className="mt-1 flex items-center gap-3 text-[11px] text-gray-400">
        {data.volume != null && data.volume > 0 && (
          <span>거래량 {data.volume.toLocaleString("ko-KR")}주</span>
        )}
        {data.timestamp && <span>{data.timestamp}</span>}
        {data.source === "close" && !data.market_open && <span>장 마감</span>}
      </div>
    </div>
  );
}

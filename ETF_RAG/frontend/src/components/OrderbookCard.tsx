"use client";

import { useEffect, useRef, useState } from "react";
import { getOrderbook } from "@/lib/api";
import type { OrderbookData } from "@/lib/types";

/**
 * 호가 10단계 (KIS 전용). 장중에만 데이터가 있고, 5초마다 갱신.
 * KIS 미연동/장 외/조회 실패(null) 시 카드 자체를 숨김.
 * KRX 관례: 매도호가(위) 파랑, 매수호가(아래) 빨강. 잔량은 막대로 비례 표시.
 */
export default function OrderbookCard({ ticker }: { ticker: string }) {
  const [data, setData] = useState<OrderbookData | null>(null);
  const timer = useRef<ReturnType<typeof setInterval> | null>(null);

  useEffect(() => {
    let alive = true;
    setData(null);

    const load = async () => {
      try {
        const ob = await getOrderbook(ticker);
        if (alive) setData(ob);
        return ob;
      } catch {
        if (alive) setData(null);
        return null;
      }
    };

    load().then((ob) => {
      if (timer.current) clearInterval(timer.current);
      // 데이터가 있을 때(장중)만 폴링
      if (ob) {
        timer.current = setInterval(async () => {
          try {
            const next = await getOrderbook(ticker);
            if (alive) {
              setData(next);
              if (!next && timer.current) clearInterval(timer.current); // 장 마감
            }
          } catch {
            /* 일시 오류 무시 */
          }
        }, 5_000);
      }
    });

    return () => {
      alive = false;
      if (timer.current) clearInterval(timer.current);
    };
  }, [ticker]);

  if (!data) return null; // KIS 미연동/장 외 → 숨김

  const maxQty = Math.max(
    1,
    ...data.asks.map((a) => a.qty),
    ...data.bids.map((b) => b.qty),
  );
  const fmt = (n: number) => n.toLocaleString("ko-KR");

  // 매도호가: 고가 → 저가 순으로 위에서 아래 (API는 1단계=최저 매도가라 역순)
  const asksTopDown = [...data.asks].reverse();

  return (
    <div className="rounded-xl border border-gray-200 bg-white p-3">
      <div className="mb-2 flex items-center justify-between">
        <span className="text-xs font-semibold text-gray-700">📋 호가 10단계</span>
        <span className="rounded-full bg-red-50 px-2 py-0.5 text-[11px] text-red-600">
          🔴 실시간 (KIS){data.timestamp ? ` · ${data.timestamp}` : ""}
        </span>
      </div>

      {/* 매도호가 (파랑, 위) */}
      <div className="space-y-0.5">
        {asksTopDown.map((a, i) => (
          <Row key={`a${i}`} price={a.price} qty={a.qty} maxQty={maxQty} side="ask" fmt={fmt} />
        ))}
      </div>
      <div className="my-1 border-t border-dashed border-gray-200" />
      {/* 매수호가 (빨강, 아래) */}
      <div className="space-y-0.5">
        {data.bids.map((b, i) => (
          <Row key={`b${i}`} price={b.price} qty={b.qty} maxQty={maxQty} side="bid" fmt={fmt} />
        ))}
      </div>

      <div className="mt-2 flex justify-between text-[11px] text-gray-400">
        <span>총 매도잔량 {fmt(data.total_ask_qty)}</span>
        <span>총 매수잔량 {fmt(data.total_bid_qty)}</span>
      </div>
    </div>
  );
}

function Row({
  price,
  qty,
  maxQty,
  side,
  fmt,
}: {
  price: number;
  qty: number;
  maxQty: number;
  side: "ask" | "bid";
  fmt: (n: number) => string;
}) {
  const pct = Math.round((qty / maxQty) * 100);
  const barColor = side === "ask" ? "bg-blue-100" : "bg-red-100";
  const priceColor = side === "ask" ? "text-blue-600" : "text-red-600";
  return (
    <div className="grid grid-cols-2 items-center gap-1 text-xs tabular-nums">
      <span className={`text-right ${priceColor}`}>{price ? fmt(price) : "-"}</span>
      <div className="relative h-5 overflow-hidden rounded">
        <div
          className={`absolute inset-y-0 left-0 ${barColor}`}
          style={{ width: `${pct}%` }}
        />
        <span className="relative z-10 ml-1 leading-5 text-gray-600">
          {qty ? fmt(qty) : ""}
        </span>
      </div>
    </div>
  );
}

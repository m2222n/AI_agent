"use client";

import Link from "next/link";
import { useEffect, useState } from "react";
import { useWatchlist } from "@/lib/useWatchlist";
import { getWatchlistDetail } from "@/lib/auth";
import type { WatchlistDetailItem } from "@/lib/types";

/** 로그인 사용자의 관심종목 칩. 종목명 표시(티커 작게). 클릭 시 기술적 분석 탭으로.
 * 비어있거나 비로그인이면 숨김. */
export default function WatchlistBar() {
  const { enabled, tickers } = useWatchlist();
  const [detail, setDetail] = useState<WatchlistDetailItem[]>([]);

  // 관심종목 목록 변화 시 종목명 포함 상세 재로드
  useEffect(() => {
    if (!enabled || tickers.length === 0) {
      setDetail([]);
      return;
    }
    let cancelled = false;
    getWatchlistDetail().then((d) => {
      if (!cancelled) setDetail(d);
    });
    return () => {
      cancelled = true;
    };
    // tickers 배열 내용 변화를 키로 감지
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [enabled, tickers.join(",")]);

  if (!enabled || tickers.length === 0) return null;

  // detail 로드 전/실패 시엔 티커로 fallback
  const items: WatchlistDetailItem[] =
    detail.length > 0 ? detail : tickers.map((t) => ({ ticker: t, name: t }));

  return (
    <div className="flex flex-wrap items-center gap-2 border-b border-gray-100 pb-3">
      <span className="text-xs text-gray-500">⭐ 관심종목</span>
      {items.map((it) => (
        <Link
          key={it.ticker}
          href={`/technical?ticker=${encodeURIComponent(it.ticker)}`}
          title={it.ticker}
          className="rounded-full border border-gray-200 dark:border-gray-800 bg-gray-50 px-3 py-1 text-xs text-gray-700 hover:bg-gray-100"
        >
          {it.name}
          {it.name !== it.ticker && (
            <span className="ml-1 text-[10px] text-gray-400">{it.ticker}</span>
          )}
        </Link>
      ))}
    </div>
  );
}

"use client";

import Link from "next/link";
import { useWatchlist } from "@/lib/useWatchlist";

/** 로그인 사용자의 관심종목 칩 목록. 클릭 시 기술적 분석 탭으로. 비어있거나 비로그인이면 숨김. */
export default function WatchlistBar() {
  const { enabled, tickers } = useWatchlist();
  if (!enabled || tickers.length === 0) return null;

  return (
    <div className="flex flex-wrap items-center gap-2 border-b border-gray-100 pb-3">
      <span className="text-xs text-gray-500">⭐ 관심종목</span>
      {tickers.map((t) => (
        <Link
          key={t}
          href={`/technical?ticker=${encodeURIComponent(t)}`}
          className="rounded-full border border-gray-200 bg-gray-50 px-3 py-1 text-xs text-gray-700 hover:bg-gray-100"
        >
          {t}
        </Link>
      ))}
    </div>
  );
}

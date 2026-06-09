"use client";

import { useWatchlist } from "@/lib/useWatchlist";

/** 종목 관심 토글 버튼(⭐). 로그인 시에만 표시. */
export default function WatchlistStar({
  ticker,
  name,
}: {
  ticker: string;
  name?: string;
}) {
  const { enabled, has, toggle } = useWatchlist();
  if (!enabled) return null;

  const active = has(ticker);
  return (
    <button
      type="button"
      onClick={() => toggle(ticker)}
      title={active ? "관심종목에서 제거" : "관심종목에 추가"}
      aria-label={active ? "관심종목 제거" : "관심종목 추가"}
      className="shrink-0 rounded-lg px-2 py-1 text-base hover:bg-gray-100"
    >
      <span className={active ? "" : "opacity-30"}>
        {active ? "⭐" : "☆"}
      </span>
      {name ? <span className="ml-1 text-xs text-gray-400">{name}</span> : null}
    </button>
  );
}

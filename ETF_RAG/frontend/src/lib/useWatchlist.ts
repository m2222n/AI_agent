"use client";

import { useCallback, useEffect, useState } from "react";
import { useAuth } from "./AuthContext";
import { addWatchlist, getWatchlist, removeWatchlist } from "./auth";

/** 관심종목 상태 + 토글. 로그인 시에만 동작(비로그인은 빈 set + enabled=false). */
export function useWatchlist() {
  const { user } = useAuth();
  const [tickers, setTickers] = useState<Set<string>>(new Set());

  // 로그인 변화 시 서버에서 로드
  useEffect(() => {
    if (!user) {
      setTickers(new Set());
      return;
    }
    let cancelled = false;
    getWatchlist().then((list) => {
      if (!cancelled) setTickers(new Set(list));
    });
    return () => {
      cancelled = true;
    };
  }, [user]);

  const has = useCallback((t: string) => tickers.has(t), [tickers]);

  const toggle = useCallback(
    async (ticker: string) => {
      if (!user) return;
      const adding = !tickers.has(ticker);
      // 낙관적 업데이트
      setTickers((prev) => {
        const next = new Set(prev);
        if (adding) next.add(ticker);
        else next.delete(ticker);
        return next;
      });
      // 서버 반영 → 성공 시 정본으로 동기화. 실패(null)면 낙관적 상태를 유지한다.
      // (과거엔 실패 시 []로 덮어써 관심종목이 화면에서 전부 사라지는 버그가 있었다.)
      const server = adding ? await addWatchlist(ticker) : await removeWatchlist(ticker);
      if (server !== null) setTickers(new Set(server));
    },
    [user, tickers],
  );

  return {
    enabled: !!user,
    tickers: Array.from(tickers),
    has,
    toggle,
  };
}

"use client";

import { useCallback, useEffect, useState } from "react";
import Link from "next/link";
import { useAuth } from "@/lib/AuthContext";
import {
  getPortfolio,
  getTradeHistory,
  getRanking,
  getPaperHistory,
  buyStock,
  sellStock,
  resetPaper,
} from "@/lib/auth";
import type {
  PaperPortfolio,
  PaperTradeHistoryItem,
  PaperRanking,
  PaperHistory,
} from "@/lib/types";
import TickerSearch from "@/components/TickerSearch";
import ChartImage from "@/components/ChartImage";

const won = (v: number) => `${Math.round(v).toLocaleString("ko-KR")}원`;
function signColor(v: number): string {
  return v > 0 ? "text-red-600" : v < 0 ? "text-blue-600" : "text-gray-500";
}
function pctStr(v: number): string {
  return `${v > 0 ? "+" : ""}${v.toFixed(2)}%`;
}

export default function InvestPage() {
  const { user, loading } = useAuth();
  const [pf, setPf] = useState<PaperPortfolio | null>(null);
  const [trades, setTrades] = useState<PaperTradeHistoryItem[]>([]);
  const [ranking, setRanking] = useState<PaperRanking | null>(null);
  const [hist, setHist] = useState<PaperHistory | null>(null);
  const [busy, setBusy] = useState(false);
  const [msg, setMsg] = useState<{ k: "ok" | "err"; t: string } | null>(null);

  // 매수/매도 입력
  const [ticker, setTicker] = useState("");
  const [tickerLabel, setTickerLabel] = useState("");
  const [qty, setQty] = useState("");

  const refresh = useCallback(async () => {
    const [p, t, r, hh] = await Promise.all([
      getPortfolio(),
      getTradeHistory(),
      getRanking(),
      getPaperHistory(),
    ]);
    setPf(p);
    setTrades(t);
    setRanking(r);
    setHist(hh);
  }, []);

  useEffect(() => {
    if (user) refresh();
  }, [user, refresh]);

  if (!loading && !user) {
    return (
      <main className="mx-auto w-full max-w-sm px-4 py-10 text-center">
        <p className="mb-4 text-sm text-gray-600">
          가상투자는 로그인 후 이용할 수 있어요. (1억 원 가상 자금 지급)
        </p>
        <Link
          href="/login"
          className="rounded-lg bg-blue-600 px-4 py-2 text-sm font-medium text-white"
        >
          로그인하러 가기
        </Link>
      </main>
    );
  }
  if (loading || !user) return null;

  const trade = async (side: "buy" | "sell") => {
    if (!ticker || !qty) return;
    const n = parseInt(qty, 10);
    if (!n || n < 1) return;
    setBusy(true);
    setMsg(null);
    try {
      const fn = side === "buy" ? buyStock : sellStock;
      const r = await fn(ticker, n);
      const label = side === "buy" ? "매수" : "매도";
      const extra =
        side === "sell" && typeof r.realized_pnl === "number"
          ? ` (실현손익 ${r.realized_pnl > 0 ? "+" : ""}${r.realized_pnl.toLocaleString("ko-KR")}원)`
          : "";
      setMsg({
        k: "ok",
        t: `${r.name} ${r.qty}주 ${label} 완료 — ${won(r.amount)}${extra}`,
      });
      setQty("");
      await refresh();
    } catch (e) {
      setMsg({ k: "err", t: e instanceof Error ? e.message : "거래 실패" });
    } finally {
      setBusy(false);
    }
  };

  const onReset = async () => {
    if (!confirm("계좌를 초기화할까요? 보유 종목과 거래 내역이 모두 사라지고 현금 1억 원으로 돌아갑니다."))
      return;
    setBusy(true);
    const p = await resetPaper();
    if (p) {
      setPf(p);
      setMsg({ k: "ok", t: "계좌를 초기화했어요." });
    }
    await refresh();
    setBusy(false);
  };

  return (
    <main className="mx-auto w-full max-w-3xl px-3 py-5 sm:px-4">
      <h1 className="mb-1 text-lg font-bold text-gray-900">💰 가상투자</h1>
      <p className="mb-4 text-xs text-gray-500">
        가상 자금 1억 원으로 실제 시세 기반 모의투자 (수수료·세금 미반영)
      </p>

      {/* 자산 요약 */}
      {pf && (
        <div className="mb-4 grid grid-cols-2 gap-2 sm:grid-cols-4">
          <Stat label="총 자산" value={won(pf.total_value)} />
          <Stat
            label="총 수익률"
            value={pctStr(pf.total_pnl_pct)}
            color={signColor(pf.total_pnl)}
          />
          <Stat
            label="평가손익"
            value={`${pf.total_pnl > 0 ? "+" : ""}${pf.total_pnl.toLocaleString("ko-KR")}원`}
            color={signColor(pf.total_pnl)}
          />
          <Stat label="현금" value={won(pf.cash)} />
        </div>
      )}

      {/* 수익률 추이 차트 (스냅샷 2일+ 누적 시) */}
      {hist?.chart_b64 && (
        <div className="mb-4">
          <ChartImage b64={hist.chart_b64} alt="가상투자 수익률 추이" />
        </div>
      )}

      {/* 매수/매도 */}
      <section className="mb-5 rounded-xl border border-gray-200 p-4">
        <h2 className="mb-2 text-sm font-semibold text-gray-800">주문</h2>
        <TickerSearch
          onSelect={(sel) => {
            setTicker(sel.ticker);
            setTickerLabel(sel.name);
          }}
          placeholder="종목명 또는 종목코드 검색"
        />
        {tickerLabel && (
          <p className="mt-1 text-xs text-gray-500">선택: {tickerLabel}</p>
        )}
        <div className="mt-2 flex gap-2">
          <input
            type="number"
            min={1}
            value={qty}
            onChange={(e) => setQty(e.target.value)}
            placeholder="수량"
            className="w-28 rounded-lg border border-gray-300 px-3 py-2 text-sm focus:border-blue-500 focus:outline-none"
          />
          <button
            type="button"
            onClick={() => trade("buy")}
            disabled={busy || !ticker || !qty}
            className="flex-1 rounded-lg bg-red-600 px-4 py-2 text-sm font-medium text-white disabled:opacity-40"
          >
            매수
          </button>
          <button
            type="button"
            onClick={() => trade("sell")}
            disabled={busy || !ticker || !qty}
            className="flex-1 rounded-lg bg-blue-600 px-4 py-2 text-sm font-medium text-white disabled:opacity-40"
          >
            매도
          </button>
        </div>
        {msg && (
          <p className={`mt-2 text-xs ${msg.k === "ok" ? "text-green-600" : "text-red-600"}`}>
            {msg.t}
          </p>
        )}
      </section>

      {/* 보유 현황 */}
      <section className="mb-5">
        <h2 className="mb-2 text-sm font-semibold text-gray-800">보유 종목</h2>
        {pf && pf.holdings.length > 0 ? (
          <div className="overflow-x-auto">
            <table className="comparison-table text-xs">
              <thead>
                <tr>
                  <th className="text-left">종목</th>
                  <th className="text-right">수량</th>
                  <th className="text-right">평단가</th>
                  <th className="text-right">현재가</th>
                  <th className="text-right">평가금액</th>
                  <th className="text-right">평가손익</th>
                </tr>
              </thead>
              <tbody>
                {pf.holdings.map((h) => (
                  <tr key={h.ticker}>
                    <td className="text-left">{h.name}</td>
                    <td className="text-right tabular-nums">{h.qty.toLocaleString("ko-KR")}</td>
                    <td className="text-right tabular-nums">{Math.round(h.avg_price).toLocaleString("ko-KR")}</td>
                    <td className="text-right tabular-nums">{h.current_price.toLocaleString("ko-KR")}</td>
                    <td className="text-right tabular-nums">{h.eval_value.toLocaleString("ko-KR")}</td>
                    <td className={`text-right tabular-nums ${signColor(h.pnl)}`}>
                      {h.pnl > 0 ? "+" : ""}
                      {h.pnl.toLocaleString("ko-KR")}
                      <span className="block text-[10px]">{pctStr(h.pnl_pct)}</span>
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        ) : (
          <p className="text-xs text-gray-400">보유 종목이 없어요. 위에서 매수해 보세요.</p>
        )}
      </section>

      {/* 랭킹 */}
      {ranking && ranking.total_players > 0 && (
        <section className="mb-5">
          <h2 className="mb-2 text-sm font-semibold text-gray-800">
            🏆 수익률 랭킹{" "}
            <span className="text-xs font-normal text-gray-400">
              (참가 {ranking.total_players}명{ranking.my_rank ? ` · 내 순위 ${ranking.my_rank}위` : ""})
            </span>
          </h2>
          <div className="overflow-x-auto">
            <table className="comparison-table text-xs">
              <thead>
                <tr>
                  <th className="text-left">순위</th>
                  <th className="text-left">닉네임</th>
                  <th className="text-right">총 자산</th>
                  <th className="text-right">수익률</th>
                </tr>
              </thead>
              <tbody>
                {ranking.rankings.map((r) => (
                  <tr key={r.rank} className={r.is_me ? "bg-blue-50" : ""}>
                    <td className="text-left tabular-nums">{r.rank}</td>
                    <td className="text-left">{r.nickname}{r.is_me ? " (나)" : ""}</td>
                    <td className="text-right tabular-nums">{won(r.total_value)}</td>
                    <td className={`text-right tabular-nums ${signColor(r.total_pnl_pct)}`}>
                      {pctStr(r.total_pnl_pct)}
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </section>
      )}

      {/* 거래 내역 */}
      {trades.length > 0 && (
        <section className="mb-5">
          <h2 className="mb-2 text-sm font-semibold text-gray-800">거래 내역</h2>
          <div className="overflow-x-auto">
            <table className="comparison-table text-xs">
              <thead>
                <tr>
                  <th className="text-left">일시</th>
                  <th className="text-left">종목</th>
                  <th className="text-center">구분</th>
                  <th className="text-right">수량</th>
                  <th className="text-right">단가</th>
                  <th className="text-right">금액</th>
                </tr>
              </thead>
              <tbody>
                {trades.map((t, i) => (
                  <tr key={i}>
                    <td className="text-left text-gray-500">
                      {t.created_at ? t.created_at.slice(0, 16).replace("T", " ") : "-"}
                    </td>
                    <td className="text-left">{t.name ?? t.ticker}</td>
                    <td className={`text-center ${t.side === "buy" ? "text-red-600" : "text-blue-600"}`}>
                      {t.side === "buy" ? "매수" : "매도"}
                    </td>
                    <td className="text-right tabular-nums">{t.qty.toLocaleString("ko-KR")}</td>
                    <td className="text-right tabular-nums">{Math.round(t.price).toLocaleString("ko-KR")}</td>
                    <td className="text-right tabular-nums">{t.amount.toLocaleString("ko-KR")}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </section>
      )}

      {/* 리셋 */}
      <button
        type="button"
        onClick={onReset}
        disabled={busy}
        className="rounded-lg border border-gray-300 px-4 py-2 text-xs text-gray-600 hover:bg-gray-100 disabled:opacity-40"
      >
        🔄 계좌 초기화
      </button>
    </main>
  );
}

function Stat({
  label,
  value,
  color = "text-gray-900",
}: {
  label: string;
  value: string;
  color?: string;
}) {
  return (
    <div className="rounded-xl border border-gray-200 px-3 py-2">
      <div className="text-[11px] text-gray-500">{label}</div>
      <div className={`text-sm font-semibold tabular-nums ${color}`}>{value}</div>
    </div>
  );
}

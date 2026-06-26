"use client";

import { useCallback, useEffect, useState } from "react";
import Link from "next/link";
import { useAuth } from "@/lib/AuthContext";
import {
  getPortfolio,
  getTradeHistory,
  getRanking,
  getPaperHistory,
  getPaperStats,
  getPaperRounds,
  buyStock,
  sellStock,
  resetPaper,
  collectDividend,
} from "@/lib/auth";
import { getPrice } from "@/lib/api";
import type {
  PaperPortfolio,
  PaperTradeHistoryItem,
  PaperRanking,
  PaperHistory,
  PaperTradeStats,
  PaperRound,
  PriceData,
} from "@/lib/types";
import TickerSearch from "@/components/TickerSearch";
import ChartImage from "@/components/ChartImage";
import PortfolioPie from "@/components/PortfolioPie";
import { Notice, EmptyState } from "@/components/Feedback";

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
  const [stats, setStats] = useState<PaperTradeStats | null>(null);
  const [pastRounds, setPastRounds] = useState<PaperRound[]>([]);
  const [busy, setBusy] = useState(false);
  const [msg, setMsg] = useState<{ k: "ok" | "err"; t: string } | null>(null);
  const [resetOpen, setResetOpen] = useState(false);
  const [resetInput, setResetInput] = useState("");

  // 매수/매도 입력
  const [ticker, setTicker] = useState("");
  const [tickerLabel, setTickerLabel] = useState("");
  const [qty, setQty] = useState("");
  const [price, setPrice] = useState<PriceData | null>(null); // 선택 종목 현재가

  const refresh = useCallback(async () => {
    const [p, t, r, hh, st, rd] = await Promise.all([
      getPortfolio(),
      getTradeHistory(),
      getRanking(),
      getPaperHistory(),
      getPaperStats(),
      getPaperRounds(),
    ]);
    setPf(p);
    setTrades(t);
    setRanking(r);
    setHist(hh);
    setStats(st);
    setPastRounds(rd);
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

  // 종목 선택 → 현재가 조회(등락률 표시 + 비율 버튼 수량 계산용)
  const onSelectTicker = async (sel: { ticker: string; name: string }) => {
    setTicker(sel.ticker);
    setTickerLabel(sel.name);
    setQty("");
    setPrice(null);
    const p = await getPrice(sel.ticker);
    if (p) setPrice(p);
  };

  // 현금의 일정 비율로 매수 가능한 수량 자동 입력 (현재가 기준 내림)
  const setQtyByRatio = (ratio: number) => {
    if (!pf || !price || price.price <= 0) return;
    const budget = pf.cash * ratio;
    const n = Math.floor(budget / price.price);
    setQty(n > 0 ? String(n) : "");
  };

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

  const doReset = async () => {
    if (resetInput.trim() !== "초기화") return;
    setBusy(true);
    setMsg(null);
    try {
      const p = await resetPaper(resetInput.trim());
      setPf(p);
      setMsg({ k: "ok", t: "새 라운드를 시작했어요. 지난 성적은 아래에 기록됐어요." });
      setResetOpen(false);
      setResetInput("");
      await refresh();
    } catch (e) {
      setMsg({ k: "err", t: e instanceof Error ? e.message : "초기화 실패" });
    } finally {
      setBusy(false);
    }
  };

  // 거래내역 CSV 다운로드 (클라이언트 측, 백엔드 불필요). Excel 한글 위해 BOM.
  const exportCsv = () => {
    if (!trades.length) return;
    const esc = (v: string | number | null) => {
      const s = v == null ? "" : String(v);
      return /[",\n]/.test(s) ? `"${s.replace(/"/g, '""')}"` : s;
    };
    const header = ["일시", "종목", "종목코드", "구분", "수량", "단가", "금액", "실현손익"];
    const rows = trades.map((t) => [
      t.created_at?.replace("T", " ").slice(0, 19) ?? "",
      t.name ?? t.ticker,
      t.ticker,
      t.side === "buy" ? "매수" : "매도",
      t.qty,
      Math.round(t.price),
      t.amount,
      t.realized_pnl ?? "",
    ]);
    const csv = [header, ...rows].map((r) => r.map(esc).join(",")).join("\r\n");
    const blob = new Blob(["﻿" + csv], { type: "text/csv;charset=utf-8;" });
    const url = URL.createObjectURL(blob);
    const a = document.createElement("a");
    a.href = url;
    a.download = `가상투자_거래내역_${new Date().toISOString().slice(0, 10)}.csv`;
    a.click();
    URL.revokeObjectURL(url);
  };

  const doDividend = async () => {
    setBusy(true);
    setMsg(null);
    try {
      const d = await collectDividend();
      setMsg({ k: d.paid ? "ok" : "err", t: d.message });
      await refresh();
    } catch (e) {
      setMsg({ k: "err", t: e instanceof Error ? e.message : "배당 정산 실패" });
    } finally {
      setBusy(false);
    }
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
      <section className="mb-5 rounded-xl border border-gray-200 dark:border-gray-800 p-4">
        <h2 className="mb-2 text-sm font-semibold text-gray-800">주문</h2>
        <TickerSearch
          onSelect={onSelectTicker}
          placeholder="종목명 또는 종목코드 검색"
        />
        {tickerLabel && (
          <div className="mt-2 flex items-baseline justify-between">
            <span className="text-xs text-gray-500">{tickerLabel}</span>
            {price && (
              <span className="text-sm font-semibold tabular-nums text-gray-900">
                {price.price.toLocaleString("ko-KR")}원
                {typeof price.change_pct === "number" && (
                  <span className={`ml-1 text-xs font-medium ${signColor(price.change_pct)}`}>
                    ({price.change_pct > 0 ? "+" : ""}{price.change_pct.toFixed(2)}%)
                  </span>
                )}
                {price.source === "close" && (
                  <span className="ml-1 text-[10px] text-gray-400">종가</span>
                )}
              </span>
            )}
          </div>
        )}

        {/* 금액 비율 버튼 — 현금의 N%로 매수 가능 수량 자동 입력 */}
        <div className="mt-2 flex gap-1.5">
          {[0.1, 0.25, 0.5, 1].map((r) => (
            <button
              key={r}
              type="button"
              onClick={() => setQtyByRatio(r)}
              disabled={!price || price.price <= 0}
              className="flex-1 rounded-lg border border-gray-300 dark:border-gray-700 px-2 py-1 text-xs text-gray-600 hover:bg-gray-100 disabled:opacity-40"
            >
              {r === 1 ? "100%" : `${r * 100}%`}
            </button>
          ))}
        </div>

        <div className="mt-2 flex gap-2">
          <input
            type="number"
            min={1}
            value={qty}
            onChange={(e) => setQty(e.target.value)}
            placeholder="수량"
            className="w-28 rounded-lg border border-gray-300 dark:border-gray-700 px-3 py-2 text-sm focus:border-blue-500 focus:outline-none"
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
        {/* 예상 체결 금액 (수량 × 현재가) */}
        {price && qty && parseInt(qty, 10) > 0 && (
          <p className="mt-2 text-xs text-gray-500">
            예상 금액 약 {won(price.price * parseInt(qty, 10))}
          </p>
        )}
        {msg && <Notice message={msg.t} kind={msg.k} />}
      </section>

      {/* 보유 현황 */}
      <section className="mb-5">
        <div className="mb-2 flex items-center justify-between">
          <h2 className="text-sm font-semibold text-gray-800">보유 종목</h2>
          {pf && pf.holdings.length > 0 && (
            <button
              type="button"
              onClick={doDividend}
              disabled={busy}
              title="보유 종목의 예상 연간 배당금을 현금으로 1회 지급(라운드당 1회)"
              className="rounded-lg border border-emerald-300 px-2.5 py-1 text-xs font-medium text-emerald-700 hover:bg-emerald-50 disabled:opacity-40"
            >
              💰 배당 받기
            </button>
          )}
        </div>
        {pf && pf.holdings.length > 0 ? (
          <>
          <div className="mb-3">
            <PortfolioPie holdings={pf.holdings} cash={pf.cash} />
          </div>
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
                    <td className="text-left">
                      {h.name}
                      {h.holding_days != null && (
                        <span className="block text-[10px] text-gray-400">
                          {h.since} · {h.holding_days}일째
                        </span>
                      )}
                    </td>
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
          </>
        ) : (
          <EmptyState icon="📦" message="보유 종목이 없어요. 위에서 매수해 보세요." />
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

      {/* 거래 통계 (청산 1건 이상일 때) */}
      {stats && stats.sell_count > 0 && (
        <section className="mb-5">
          <h2 className="mb-2 text-sm font-semibold text-gray-800">거래 통계</h2>
          <div className="grid grid-cols-2 gap-2 sm:grid-cols-4">
            <Stat label="승률" value={`${stats.win_rate}%`} sub={`${stats.win_count}승 ${stats.loss_count}패`} />
            <Stat
              label="실현손익"
              value={`${stats.realized_pnl > 0 ? "+" : ""}${stats.realized_pnl.toLocaleString("ko-KR")}`}
            />
            <Stat
              label="손익비"
              value={stats.profit_factor == null ? "-" : `${stats.profit_factor}`}
              sub="총이익/총손실"
            />
            <Stat label="청산 횟수" value={`${stats.sell_count}회`} sub={`매수 ${stats.buy_count}회`} />
            <Stat
              label="평균 이익"
              value={`+${stats.avg_win.toLocaleString("ko-KR")}`}
            />
            <Stat
              label="평균 손실"
              value={`${stats.avg_loss.toLocaleString("ko-KR")}`}
            />
            <Stat
              label="최고 거래"
              value={stats.best_trade == null ? "-" : `+${stats.best_trade.toLocaleString("ko-KR")}`}
            />
            <Stat
              label="최악 거래"
              value={stats.worst_trade == null ? "-" : `${stats.worst_trade.toLocaleString("ko-KR")}`}
            />
          </div>
        </section>
      )}

      {/* 거래 내역 */}
      {trades.length > 0 && (
        <section className="mb-5">
          <div className="mb-2 flex items-center justify-between">
            <h2 className="text-sm font-semibold text-gray-800">거래 내역</h2>
            <button
              type="button"
              onClick={exportCsv}
              title="거래 내역을 CSV 파일로 내려받기"
              className="rounded-lg border border-gray-300 dark:border-gray-700 px-2.5 py-1 text-xs font-medium text-gray-600 hover:bg-gray-100"
            >
              ⬇ CSV 내보내기
            </button>
          </div>
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
                  <th className="text-right">실현손익</th>
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
                    <td className={`text-right tabular-nums ${t.realized_pnl == null ? "text-gray-300" : signColor(t.realized_pnl)}`}>
                      {t.realized_pnl == null
                        ? "-"
                        : `${t.realized_pnl > 0 ? "+" : ""}${t.realized_pnl.toLocaleString("ko-KR")}`}
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </section>
      )}

      {/* 지난 성적 (라운드 결산) */}
      {pastRounds.length > 0 && (
        <section className="mb-5">
          <h2 className="mb-2 text-sm font-semibold text-gray-800">📚 지난 성적</h2>
          <div className="space-y-2">
            {pastRounds.map((rd) => (
              <details key={rd.round_no} className="rounded-xl border border-gray-200 dark:border-gray-800 p-3">
                <summary className="cursor-pointer text-xs">
                  <b>R{rd.round_no}</b>{" "}
                  <span className="text-gray-400">
                    {rd.started_at.slice(0, 10)} ~ {rd.ended_at.slice(0, 10)} · 거래 {rd.trade_count}회
                  </span>{" "}
                  <span className={`font-semibold ${signColor(rd.return_pct)}`}>
                    {pctStr(rd.return_pct)}
                  </span>{" "}
                  <span className="text-gray-500">({won(rd.final_value)})</span>
                </summary>
                {rd.symbols.length > 0 && (
                  <div className="mt-2 overflow-x-auto">
                    <table className="comparison-table text-xs">
                      <thead>
                        <tr>
                          <th className="text-left">종목</th>
                          <th className="text-right">실현</th>
                          <th className="text-right">미실현</th>
                          <th className="text-right">합계</th>
                        </tr>
                      </thead>
                      <tbody>
                        {rd.symbols.map((s) => (
                          <tr key={s.ticker}>
                            <td className="text-left">{s.name}</td>
                            <td className={`text-right tabular-nums ${signColor(s.realized)}`}>
                              {s.realized > 0 ? "+" : ""}{s.realized.toLocaleString("ko-KR")}
                            </td>
                            <td className={`text-right tabular-nums ${signColor(s.unrealized)}`}>
                              {s.unrealized > 0 ? "+" : ""}{s.unrealized.toLocaleString("ko-KR")}
                            </td>
                            <td className={`text-right tabular-nums font-semibold ${signColor(s.total)}`}>
                              {s.total > 0 ? "+" : ""}{s.total.toLocaleString("ko-KR")}
                            </td>
                          </tr>
                        ))}
                      </tbody>
                    </table>
                  </div>
                )}
              </details>
            ))}
          </div>
        </section>
      )}

      {/* 계좌 초기화 (새 라운드) */}
      <button
        type="button"
        onClick={() => { setResetOpen(true); setResetInput(""); }}
        disabled={busy}
        className="rounded-lg border border-gray-300 dark:border-gray-700 px-4 py-2 text-xs text-gray-600 hover:bg-gray-100 disabled:opacity-40"
      >
        🔄 계좌 초기화 (새 라운드 시작)
      </button>

      {/* 초기화 확인 모달 */}
      {resetOpen && (
        <div className="fixed inset-0 z-50 flex items-center justify-center p-4">
          <button
            type="button"
            aria-label="닫기"
            onClick={() => setResetOpen(false)}
            className="absolute inset-0 bg-black/40"
          />
          <div className="relative w-full max-w-sm rounded-2xl bg-white dark:bg-gray-900 p-5 shadow-xl">
            <h3 className="text-sm font-bold text-gray-900">계좌 초기화</h3>
            <p className="mt-2 text-xs leading-relaxed text-gray-600">
              현재 라운드를 <b>결산해 &lsquo;지난 성적&rsquo;에 기록</b>하고, 보유 종목·거래
              내역을 비운 뒤 현금 <b>1억 원</b>으로 새 라운드를 시작해요.
              <br />
              계속하려면 아래에 <b>초기화</b>를 입력하세요.
            </p>
            <input
              value={resetInput}
              onChange={(e) => setResetInput(e.target.value)}
              onKeyDown={(e) => { if (e.key === "Enter") doReset(); }}
              placeholder="초기화"
              autoFocus
              className="mt-3 w-full rounded-lg border border-gray-300 dark:border-gray-700 px-3 py-2 text-sm focus:border-blue-500 focus:outline-none"
            />
            <div className="mt-3 flex gap-2">
              <button
                type="button"
                onClick={doReset}
                disabled={busy || resetInput.trim() !== "초기화"}
                className="flex-1 rounded-lg bg-red-600 px-4 py-2 text-sm font-medium text-white disabled:opacity-40"
              >
                초기화
              </button>
              <button
                type="button"
                onClick={() => setResetOpen(false)}
                className="rounded-lg px-4 py-2 text-sm text-gray-600 hover:bg-gray-100"
              >
                취소
              </button>
            </div>
          </div>
        </div>
      )}
    </main>
  );
}

function Stat({
  label,
  value,
  color = "text-gray-900",
  sub,
}: {
  label: string;
  value: string;
  color?: string;
  sub?: string;
}) {
  return (
    <div className="rounded-xl border border-gray-200 dark:border-gray-800 px-3 py-2">
      <div className="text-[11px] text-gray-500">{label}</div>
      <div className={`text-sm font-semibold tabular-nums ${color}`}>{value}</div>
      {sub ? <div className="text-[10px] text-gray-400">{sub}</div> : null}
    </div>
  );
}

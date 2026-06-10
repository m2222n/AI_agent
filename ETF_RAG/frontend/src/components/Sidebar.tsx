"use client";

import { useEffect, useMemo, useState } from "react";
import { useRouter } from "next/navigation";
import { getOverview, getVisitor } from "@/lib/api";
import type {
  InstrumentItem,
  OverviewResponse,
  VisitorResponse,
} from "@/lib/types";

function fmtDate(s: string | null): string {
  if (!s) return "-";
  if (s.length === 8) return `${s.slice(0, 4)}-${s.slice(4, 6)}-${s.slice(6, 8)}`;
  return s;
}
function fmtTradeValue(v: number): string {
  const 조 = 1e12,
    억 = 1e8;
  if (v >= 조) return `${(v / 조).toFixed(1)}조`;
  if (v >= 억) return `${Math.round(v / 억).toLocaleString("ko-KR")}억`;
  return v.toLocaleString("ko-KR");
}
function changeColor(v: number): string {
  return v > 0 ? "text-red-600" : v < 0 ? "text-blue-600" : "text-gray-400";
}

function ItemRow({
  it,
  onClick,
}: {
  it: InstrumentItem;
  onClick: () => void;
}) {
  return (
    <button
      type="button"
      onClick={onClick}
      className="flex w-full items-center justify-between gap-2 rounded px-2 py-1.5 text-left hover:bg-gray-100"
    >
      <span className="truncate text-xs text-gray-800">{it.name}</span>
      <span className="flex shrink-0 items-center gap-2">
        <span className="text-xs tabular-nums text-gray-500">
          {it.close.toLocaleString("ko-KR")}
        </span>
        <span className={`text-xs tabular-nums ${changeColor(it.change_pct)}`}>
          {it.change_pct > 0 ? "+" : ""}
          {it.change_pct}%
        </span>
      </span>
    </button>
  );
}

export default function Sidebar() {
  const router = useRouter();
  const [data, setData] = useState<OverviewResponse | null>(null);
  const [visitor, setVisitor] = useState<VisitorResponse | null>(null);
  const [tab, setTab] = useState<"etf" | "stock">("etf");
  const [q, setQ] = useState("");

  useEffect(() => {
    getOverview(20).then(setData);
    // 방문은 브라우저 세션당 1회만 기록(POST), 이후 마운트는 조회(GET)
    const recorded = sessionStorage.getItem("etfrag.visited") === "1";
    getVisitor(!recorded).then((v) => {
      if (v) setVisitor(v);
      if (!recorded) sessionStorage.setItem("etfrag.visited", "1");
    });
  }, []);

  const list = data ? (tab === "etf" ? data.top_etfs : data.top_stocks) : [];
  const filtered = useMemo(() => {
    const ql = q.trim().toLowerCase();
    if (!ql) return list;
    return list.filter(
      (it) => it.name.toLowerCase().includes(ql) || it.ticker.includes(ql),
    );
  }, [list, q]);

  const go = (it: InstrumentItem) =>
    router.push(`/technical?ticker=${encodeURIComponent(it.ticker)}`);

  return (
    <aside className="hidden w-72 shrink-0 overflow-y-auto border-r border-gray-200 p-3 text-sm lg:block">
      {/* 데이터 현황 */}
      <div className="mb-3">
        <div className="text-xs font-semibold text-gray-700">📊 데이터 현황</div>
        {data ? (
          <>
            <div className="mt-1 text-xs text-gray-600">
              ETF <b>{data.etf_count.toLocaleString("ko-KR")}</b> · 주식{" "}
              <b>{data.stock_count.toLocaleString("ko-KR")}</b> 종목
            </div>
            <div className="text-xs text-gray-400">
              📅 기준일 {fmtDate(data.as_of)} · 매일 18:30 업데이트
            </div>
          </>
        ) : (
          <div className="mt-1 text-xs text-gray-400">불러오는 중…</div>
        )}
        {visitor && visitor.total > 0 && (
          <div className="mt-1 text-xs text-gray-400">
            👤 오늘 <b>{visitor.daily.toLocaleString("ko-KR")}</b> · 누적{" "}
            <b>{visitor.total.toLocaleString("ko-KR")}</b>
          </div>
        )}
      </div>

      {/* ETF / 주식 탭 */}
      <div className="mb-2 flex gap-1">
        {(["etf", "stock"] as const).map((t) => (
          <button
            key={t}
            type="button"
            onClick={() => setTab(t)}
            className={[
              "flex-1 rounded-lg px-2 py-1 text-xs",
              tab === t ? "bg-blue-600 text-white" : "text-gray-600 hover:bg-gray-100",
            ].join(" ")}
          >
            {t === "etf" ? "📊 ETF" : "📈 주식"}
          </button>
        ))}
      </div>

      {/* 검색 */}
      <input
        value={q}
        onChange={(e) => setQ(e.target.value)}
        placeholder="종목 검색…"
        className="mb-2 w-full rounded-lg border border-gray-300 px-3 py-1.5 text-xs focus:border-blue-500 focus:outline-none"
      />

      {/* 목록 (거래대금 TOP) */}
      <div className="space-y-0.5">
        {filtered.length === 0 ? (
          <div className="px-2 py-3 text-center text-xs text-gray-400">
            {data ? "결과 없음" : "…"}
          </div>
        ) : (
          filtered.map((it) => <ItemRow key={it.ticker} it={it} onClick={() => go(it)} />)
        )}
      </div>

      {/* 투자 유의 */}
      <p className="mt-4 border-t border-gray-100 pt-3 text-[11px] leading-relaxed text-gray-400">
        ⚠️ 본 정보는 투자 참고용입니다. 투자 판단과 책임은 본인에게 있으며, 실제
        투자 시 추가 조사와 전문가 상담을 권장합니다.
      </p>
    </aside>
  );
}

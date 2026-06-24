"use client";

import { useEffect, useState } from "react";
import { useRouter } from "next/navigation";
import { getOverview, getVisitor, searchTickers } from "@/lib/api";
import PushToggle from "@/components/PushToggle";
import WatchlistStar from "@/components/WatchlistStar";
import { useWatchlist } from "@/lib/useWatchlist";
import { getWatchlistDetail } from "@/lib/auth";
import type {
  InstrumentItem,
  OverviewResponse,
  VisitorResponse,
  WatchlistDetailItem,
} from "@/lib/types";

// "삼성전자 (005930)" → {name, ticker}
function parseOption(opt: string): { name: string; ticker: string } {
  const m = opt.match(/^(.*)\s+\(([^)]+)\)\s*$/);
  return m ? { name: m[1], ticker: m[2] } : { name: opt, ticker: opt };
}

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
  showStar = false,
}: {
  it: InstrumentItem;
  onClick: () => void;
  showStar?: boolean;
}) {
  return (
    <div className="flex w-full items-center gap-1 rounded px-2 py-1.5 hover:bg-gray-100">
      <button
        type="button"
        onClick={onClick}
        className="flex flex-1 items-center justify-between gap-2 text-left"
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
      {showStar && <WatchlistStar ticker={it.ticker} />}
    </div>
  );
}

// 전체 검색 결과 행 (가격 없이 이름+⭐+클릭이동만 — searchTickers는 가격 미제공)
function SearchRow({
  name,
  ticker,
  onClick,
  showStar,
}: {
  name: string;
  ticker: string;
  onClick: () => void;
  showStar: boolean;
}) {
  return (
    <div className="flex w-full items-center gap-1 rounded px-2 py-1.5 hover:bg-gray-100">
      <button
        type="button"
        onClick={onClick}
        className="flex flex-1 items-center justify-between gap-2 text-left"
      >
        <span className="truncate text-xs text-gray-800">{name}</span>
        <span className="shrink-0 text-[10px] tabular-nums text-gray-400">{ticker}</span>
      </button>
      {showStar && <WatchlistStar ticker={ticker} />}
    </div>
  );
}

export default function Sidebar({
  open = false,
  onClose,
}: {
  open?: boolean;
  onClose?: () => void;
} = {}) {
  const router = useRouter();
  const watch = useWatchlist();
  const [data, setData] = useState<OverviewResponse | null>(null);
  const [visitor, setVisitor] = useState<VisitorResponse | null>(null);
  const [tab, setTab] = useState<"etf" | "stock" | "watch">("etf");
  const [q, setQ] = useState("");
  const [sector, setSector] = useState(""); // "" = 전체
  const [sectorStocks, setSectorStocks] = useState<InstrumentItem[] | null>(null);
  // 전체 종목 검색 결과 ("이름 (코드)" 문자열). 검색어 있을 때만 채움.
  const [searchHits, setSearchHits] = useState<string[]>([]);
  // 관심종목 + 종목명 (관심 탭 표시용)
  const [watchDetail, setWatchDetail] = useState<WatchlistDetailItem[]>([]);

  useEffect(() => {
    getOverview(20).then(setData);
    // 방문은 브라우저 세션당 1회만 기록(POST), 이후 마운트는 조회(GET)
    const recorded = sessionStorage.getItem("etfrag.visited") === "1";
    getVisitor(!recorded).then((v) => {
      if (v) setVisitor(v);
      if (!recorded) sessionStorage.setItem("etfrag.visited", "1");
    });
  }, []);

  // 업종 선택 시 해당 업종 종목 재조회(전체 목록 기준 거래대금 TOP). 전체면 기본 목록 사용.
  useEffect(() => {
    if (!sector) return;
    let alive = true;
    getOverview(20, sector).then((r) => {
      if (alive) setSectorStocks(r ? r.top_stocks : []);
    });
    return () => {
      alive = false;
    };
  }, [sector]);

  const onSectorChange = (s: string) => {
    setSector(s);
    setSectorStocks(null); // 새 선택 로딩 표시(전체면 그대로 null → 기본 목록 사용)
  };

  // 전체 종목 검색(디바운스 250ms) — 검색어 있으면 자동완성 API로 전 종목 조회.
  // 관심 탭은 검색 대신 관심종목 목록만 보여주므로 제외.
  useEffect(() => {
    const ql = q.trim();
    if (!ql || tab === "watch") {
      setSearchHits([]);
      return;
    }
    const assetType = tab === "etf" ? "etf" : "stock";
    const timer = setTimeout(() => {
      searchTickers(ql, 30, assetType).then(setSearchHits);
    }, 250);
    return () => clearTimeout(timer);
  }, [q, tab]);

  // 관심 탭 진입 또는 관심종목 변경 시 종목명 포함 상세 로드
  useEffect(() => {
    if (tab !== "watch" || !watch.enabled) {
      return;
    }
    getWatchlistDetail().then(setWatchDetail);
  }, [tab, watch.enabled, watch.tickers.length]);

  const stockList = sector ? sectorStocks ?? [] : data?.top_stocks ?? [];
  const list = data ? (tab === "etf" ? data.top_etfs : stockList) : [];

  const goTicker = (ticker: string) => {
    router.push(`/technical?ticker=${encodeURIComponent(ticker)}`);
    onClose?.(); // 모바일 드로워에서 종목 선택 시 닫기
  };
  const go = (it: InstrumentItem) => goTicker(it.ticker);

  const content = (
    <>
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

      {/* ETF / 주식 / 관심 탭 */}
      <div className="mb-2 flex gap-1">
        {(["etf", "stock", "watch"] as const).map((t) => (
          <button
            key={t}
            type="button"
            onClick={() => setTab(t)}
            className={[
              "flex-1 rounded-lg px-2 py-1 text-xs",
              tab === t ? "bg-blue-600 text-white" : "text-gray-600 hover:bg-gray-100",
            ].join(" ")}
          >
            {t === "etf" ? "📊 ETF" : t === "stock" ? "📈 주식" : "⭐ 관심"}
          </button>
        ))}
      </div>

      {/* 업종 필터 (주식 탭, 검색 안 할 때만) */}
      {tab === "stock" && !q.trim() && data && data.sectors.length > 0 && (
        <select
          value={sector}
          onChange={(e) => onSectorChange(e.target.value)}
          className="mb-2 w-full rounded-lg border border-gray-300 px-2 py-1.5 text-xs focus:border-blue-500 focus:outline-none"
        >
          <option value="">전체 업종</option>
          {data.sectors.map((s) => (
            <option key={s} value={s}>
              {s}
            </option>
          ))}
        </select>
      )}

      {/* 검색 (관심 탭 제외 — 전체 종목 검색) */}
      {tab !== "watch" && (
        <input
          value={q}
          onChange={(e) => setQ(e.target.value)}
          placeholder={`${tab === "etf" ? "ETF" : "주식"} 전체 검색…`}
          className="mb-2 w-full rounded-lg border border-gray-300 px-3 py-1.5 text-xs focus:border-blue-500 focus:outline-none"
        />
      )}

      {/* 목록 — 검색 중이면 전체 검색결과 / 관심 탭이면 관심종목 / 기본은 거래대금 TOP */}
      <div className="space-y-0.5">
        {tab === "watch" ? (
          !watch.enabled ? (
            <div className="px-2 py-3 text-center text-xs text-gray-400">
              로그인하면 관심종목을 저장할 수 있어요.
            </div>
          ) : watch.tickers.length === 0 ? (
            <div className="px-2 py-3 text-center text-xs text-gray-400">
              관심종목이 없어요. ETF·주식 탭에서 ⭐로 추가하세요.
            </div>
          ) : (
            // 종목명 로드 전이면 ticker로, 로드 후엔 종목명 표시
            (watchDetail.length ? watchDetail : watch.tickers.map((t) => ({ ticker: t, name: t }))).map(
              (it) => (
                <SearchRow
                  key={it.ticker}
                  name={it.name}
                  ticker={it.ticker}
                  onClick={() => goTicker(it.ticker)}
                  showStar
                />
              ),
            )
          )
        ) : q.trim() ? (
          searchHits.length === 0 ? (
            <div className="px-2 py-3 text-center text-xs text-gray-400">…</div>
          ) : (
            searchHits.map((opt) => {
              const { name, ticker } = parseOption(opt);
              return (
                <SearchRow
                  key={ticker}
                  name={name}
                  ticker={ticker}
                  onClick={() => goTicker(ticker)}
                  showStar={watch.enabled}
                />
              );
            })
          )
        ) : list.length === 0 ? (
          <div className="px-2 py-3 text-center text-xs text-gray-400">
            {!data || (tab === "stock" && sector && sectorStocks === null)
              ? "…"
              : "결과 없음"}
          </div>
        ) : (
          list.map((it) => (
            <ItemRow key={it.ticker} it={it} onClick={() => go(it)} showStar={watch.enabled} />
          ))
        )}
      </div>

      {/* 관심종목 푸시 알림 (로그인 + VAPID 활성 시) */}
      <PushToggle />

      {/* 투자 유의 */}
      <p className="mt-4 border-t border-gray-100 pt-3 text-[11px] leading-relaxed text-gray-400">
        ⚠️ 본 정보는 투자 참고용입니다. 투자 판단과 책임은 본인에게 있으며, 실제
        투자 시 추가 조사와 전문가 상담을 권장합니다.
      </p>
    </>
  );

  return (
    <>
      {/* 데스크톱: 고정 사이드바 (lg 이상) */}
      <aside className="hidden w-72 shrink-0 overflow-y-auto border-r border-gray-200 p-3 text-sm lg:block">
        {content}
      </aside>

      {/* 모바일/태블릿: 드로워 오버레이 (lg 미만, open일 때만) */}
      {open && (
        <div className="fixed inset-0 z-40 lg:hidden">
          {/* 배경 딤 — 클릭 시 닫기 */}
          <button
            type="button"
            aria-label="사이드바 닫기"
            onClick={onClose}
            className="absolute inset-0 bg-black/40"
          />
          {/* 드로워 패널 (좌측 슬라이드) */}
          <aside className="absolute left-0 top-0 h-full w-72 max-w-[85%] overflow-y-auto border-r border-gray-200 bg-white p-3 text-sm shadow-xl">
            <div className="mb-2 flex justify-end">
              <button
                type="button"
                onClick={onClose}
                className="rounded-lg px-2 py-1 text-xs text-gray-500 hover:bg-gray-100"
              >
                ✕ 닫기
              </button>
            </div>
            {content}
          </aside>
        </div>
      )}
    </>
  );
}

"use client";

import { useEffect, useRef, useState } from "react";
import { getSector } from "@/lib/api";
import type { SectorResponse } from "@/lib/types";
import ChartImage from "@/components/ChartImage";
import DataRangeNote from "@/components/DataRangeNote";
import { Loading, ErrorText } from "@/components/Feedback";

function jo(v: number): string {
  const 조 = 1_0000_0000_0000;
  if (v >= 조) return `${(v / 조).toFixed(1)}조`;
  return `${Math.round(v / 1_0000_0000).toLocaleString("ko-KR")}억`;
}

// 기간 옵션 (백엔드 /tabs/sector period 패턴과 일치)
const PERIODS: { value: string; label: string }[] = [
  { value: "1d", label: "1일" },
  { value: "1w", label: "1주" },
  { value: "1m", label: "1달" },
  { value: "3m", label: "3달" },
  { value: "6m", label: "6달" },
  { value: "1y", label: "1년" },
  { value: "2y", label: "2년" },
  { value: "3y", label: "3년" },
  { value: "5y", label: "5년" },
  { value: "10y", label: "10년" },
];

export default function SectorPage() {
  const [data, setData] = useState<SectorResponse | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [picked, setPicked] = useState<string>("");
  const [period, setPeriod] = useState<string>("1d");

  // 요청 순번 — mount 개요 로드와 섹터/기간 전환이 같은 setData를 쓰므로,
  // 빠른 전환 시 이전(느린) 응답이 최신을 덮어쓰지 않게 공유 순번으로 가드.
  const reqSeq = useRef(0);

  // mount 시 전체 개요 로드
  useEffect(() => {
    const seq = ++reqSeq.current;
    (async () => {
      try {
        const res = await getSector();
        if (seq !== reqSeq.current) return;
        if (!res) setError("섹터 데이터를 찾을 수 없어요.");
        setData(res);
      } catch {
        if (seq !== reqSeq.current) return;
        setError("데이터를 가져오지 못했어요.");
      } finally {
        if (seq === reqSeq.current) setLoading(false);
      }
    })();
  }, []);

  // 섹터/기간 변경 시 재조회 (섹터 미선택이면 전체 개요)
  const load = async (sector: string, p: string) => {
    const seq = ++reqSeq.current;
    setLoading(true);
    try {
      const res = sector ? await getSector(sector, p) : await getSector();
      if (seq !== reqSeq.current) return;
      if (res) setData(res);
    } finally {
      if (seq === reqSeq.current) setLoading(false);
    }
  };

  const onSector = (sector: string) => {
    setPicked(sector);
    load(sector, period);
  };
  const onPeriod = (p: string) => {
    setPeriod(p);
    if (picked) load(picked, p);
  };

  const stats = data?.stats ?? [];

  return (
    <main className="mx-auto w-full max-w-3xl px-3 py-5 sm:px-4">
      <h1 className="mb-1 text-lg font-bold text-gray-900">🏭 섹터 분석</h1>
      <p className="mb-4 text-xs text-gray-500">업종별 등락률·시가총액·밸류에이션</p>

      {loading && !data && <Loading text="불러오는 중…" />}
      {error && <ErrorText message={error} />}

      {data && (
        <div className="space-y-4">
          <ChartImage b64={data.overview_chart_b64} alt="섹터 개요" />

          {/* 섹터 선택 → 상세 */}
          {stats.length > 0 && (
            <select
              value={picked}
              onChange={(e) => onSector(e.target.value)}
              className="w-full rounded-xl border border-gray-300 dark:border-gray-700 px-3 py-2 text-sm"
            >
              <option value="">— 업종 상세 보기 —</option>
              {stats.map((s) => (
                <option key={s.sector} value={s.sector}>
                  {s.sector} ({s.count}종목)
                </option>
              ))}
            </select>
          )}

          {/* 기간 선택 (업종 선택 시에만) */}
          {picked && (
            <div className="flex flex-wrap gap-1.5">
              {PERIODS.map((p) => (
                <button
                  key={p.value}
                  type="button"
                  onClick={() => onPeriod(p.value)}
                  className={`rounded-full px-3 py-1 text-xs font-medium transition ${
                    period === p.value
                      ? "bg-blue-600 text-white"
                      : "bg-gray-100 text-gray-600 hover:bg-gray-200"
                  }`}
                >
                  {p.label}
                </button>
              ))}
            </div>
          )}

          {/* 기간 추이 차트 (1일 외 + 데이터 있을 때) */}
          {picked && period !== "1d" && data.trend_chart_b64 && (
            <div>
              <ChartImage
                b64={data.trend_chart_b64}
                alt={`${data.sector} 기간 추이`}
              />
              {typeof data.trend_return_pct === "number" && (
                <p className="mt-1 text-center text-xs text-gray-500">
                  시총 상위 {data.trend_constituents}종목 시총가중 지수 (기준일=100)
                </p>
              )}
            </div>
          )}
          {picked && period !== "1d" && !data.trend_chart_b64 && !loading && (
            <p className="text-center text-xs text-gray-400">
              이 기간의 추이 데이터가 충분하지 않아요.
            </p>
          )}

          {/* 업종 내 종목 상세 (1일 스냅샷) */}
          {data.detail_chart_b64 && (
            <ChartImage b64={data.detail_chart_b64} alt={`${data.sector} 상세`} />
          )}

          {/* 섹터 요약 표 (상위 20) */}
          <div className="overflow-x-auto">
            <table className="comparison-table text-xs">
              <thead>
                <tr>
                  <th className="text-left">업종</th>
                  <th className="text-right">종목수</th>
                  <th className="text-right">등락률</th>
                  <th className="text-right">시가총액</th>
                  <th className="text-right">PER중간</th>
                  <th className="text-right">상승/하락</th>
                </tr>
              </thead>
              <tbody>
                {stats.slice(0, 20).map((s) => (
                  <tr key={s.sector}>
                    <td className="text-left">{s.sector}</td>
                    <td className="text-right tabular-nums">{s.count}</td>
                    <td
                      className={`text-right tabular-nums ${s.change_pct > 0 ? "text-red-600" : s.change_pct < 0 ? "text-blue-600" : ""}`}
                    >
                      {s.change_pct > 0 ? "+" : ""}
                      {s.change_pct.toFixed(2)}%
                    </td>
                    <td className="text-right tabular-nums">{jo(s.market_cap)}</td>
                    <td className="text-right tabular-nums">{s.median_per}</td>
                    <td className="text-right tabular-nums">
                      {s.up_count}/{s.down_count}
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </div>
      )}

      <DataRangeNote />
    </main>
  );
}

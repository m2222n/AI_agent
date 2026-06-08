"use client";

import { useEffect, useState } from "react";
import { getSector } from "@/lib/api";
import type { SectorResponse } from "@/lib/types";
import ChartImage from "@/components/ChartImage";

function jo(v: number): string {
  const 조 = 1_0000_0000_0000;
  if (v >= 조) return `${(v / 조).toFixed(1)}조`;
  return `${Math.round(v / 1_0000_0000).toLocaleString("ko-KR")}억`;
}

export default function SectorPage() {
  const [data, setData] = useState<SectorResponse | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [picked, setPicked] = useState<string>("");

  // mount 시 전체 개요 로드
  useEffect(() => {
    (async () => {
      try {
        const res = await getSector();
        if (!res) setError("섹터 데이터를 찾을 수 없어요.");
        setData(res);
      } catch {
        setError("데이터를 가져오지 못했어요.");
      } finally {
        setLoading(false);
      }
    })();
  }, []);

  const loadDetail = async (sector: string) => {
    setPicked(sector);
    if (!sector) return;
    setLoading(true);
    try {
      const res = await getSector(sector);
      if (res) setData(res);
    } finally {
      setLoading(false);
    }
  };

  const stats = data?.stats ?? [];

  return (
    <main className="mx-auto w-full max-w-3xl px-3 py-5 sm:px-4">
      <h1 className="mb-1 text-lg font-bold text-gray-900">🏭 섹터 분석</h1>
      <p className="mb-4 text-xs text-gray-500">업종별 등락률·시가총액·밸류에이션</p>

      {loading && !data && (
        <p className="mt-6 text-center text-sm text-gray-400">불러오는 중…</p>
      )}
      {error && <p className="mt-6 text-center text-sm text-red-600">{error}</p>}

      {data && (
        <div className="space-y-4">
          <ChartImage b64={data.overview_chart_b64} alt="섹터 개요" />

          {/* 섹터 선택 → 상세 */}
          {stats.length > 0 && (
            <select
              value={picked}
              onChange={(e) => loadDetail(e.target.value)}
              className="w-full rounded-xl border border-gray-300 px-3 py-2 text-sm"
            >
              <option value="">— 업종 상세 보기 —</option>
              {stats.map((s) => (
                <option key={s.sector} value={s.sector}>
                  {s.sector} ({s.count}종목)
                </option>
              ))}
            </select>
          )}

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
    </main>
  );
}

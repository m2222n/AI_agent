"use client";

import { useState } from "react";
import { postComparison } from "@/lib/api";
import type { ComparisonResponse, ComparisonItem } from "@/lib/types";
import TickerSearch from "@/components/TickerSearch";
import ChartImage from "@/components/ChartImage";
import ComparisonTable from "@/components/ComparisonTable";

export default function ComparisonPage() {
  const [t1, setT1] = useState<{ name: string; ticker: string } | null>(null);
  const [t2, setT2] = useState<{ name: string; ticker: string } | null>(null);
  const [data, setData] = useState<ComparisonResponse | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const run = async (a: { ticker: string }, b: { ticker: string }) => {
    setLoading(true);
    setError(null);
    try {
      const res = await postComparison([a.ticker, b.ticker]);
      if (!res) setError("비교할 종목을 찾을 수 없어요.");
      setData(res);
    } catch {
      setError("데이터를 가져오지 못했어요.");
    } finally {
      setLoading(false);
    }
  };

  const pick1 = (sel: { name: string; ticker: string }) => {
    setT1(sel);
    if (t2) run(sel, t2);
  };
  const pick2 = (sel: { name: string; ticker: string }) => {
    setT2(sel);
    if (t1) run(t1, sel);
  };

  return (
    <main className="mx-auto w-full max-w-3xl px-3 py-5 sm:px-4">
      <h1 className="mb-1 text-lg font-bold text-gray-900">⚖️ 비교 분석</h1>
      <p className="mb-4 text-xs text-gray-500">두 종목의 시세·밸류에이션·상대 수익률</p>

      <div className="grid gap-3 sm:grid-cols-2">
        <div>
          <div className="mb-1 text-xs text-gray-500">종목 1</div>
          <TickerSearch onSelect={pick1} />
        </div>
        <div>
          <div className="mb-1 text-xs text-gray-500">종목 2</div>
          <TickerSearch onSelect={pick2} />
        </div>
      </div>

      {loading && <p className="mt-6 text-center text-sm text-gray-400">비교 중…</p>}
      {error && <p className="mt-6 text-center text-sm text-red-600">{error}</p>}

      {data && !loading && (
        <div className="mt-5">
          <ComparisonTable items={data.items as ComparisonItem[]} />
          <ChartImage b64={data.comparison_chart_b64} alt="상대 수익률 추이" />
          <ChartImage b64={data.valuation_chart_b64} alt="밸류에이션 비교" />
        </div>
      )}
    </main>
  );
}

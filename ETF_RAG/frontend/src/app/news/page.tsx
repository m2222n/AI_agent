"use client";

import { useState } from "react";
import { getNews } from "@/lib/api";
import type { NewsResponse } from "@/lib/types";
import TickerSearch from "@/components/TickerSearch";
import ChartImage from "@/components/ChartImage";
import DataRangeNote from "@/components/DataRangeNote";
import { Loading, ErrorText, EmptyState } from "@/components/Feedback";

// 감성 → 색상 배지
function sentBadge(s: string): string {
  if (s === "긍정") return "bg-red-50 text-red-600";
  if (s === "부정") return "bg-blue-50 text-blue-600";
  if (s === "중립") return "bg-gray-100 text-gray-500";
  return "bg-gray-50 text-gray-400";
}

function overallColor(s: string): string {
  if (s === "긍정") return "text-red-600";
  if (s === "부정") return "text-blue-600";
  if (s === "혼재") return "text-amber-600";
  return "text-gray-500";
}

export default function NewsPage() {
  const [data, setData] = useState<NewsResponse | null>(null);
  const [name, setName] = useState("");
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const onSelect = async (sel: { ticker: string; name: string }) => {
    setName(sel.name);
    setLoading(true);
    setError(null);
    setData(null);
    try {
      const res = await getNews(sel.ticker);
      if (!res) setError("종목을 찾을 수 없어요.");
      else setData(res);
    } catch {
      setError("뉴스를 가져오지 못했어요.");
    } finally {
      setLoading(false);
    }
  };

  return (
    <main className="mx-auto w-full max-w-3xl px-3 py-5 sm:px-4">
      <h1 className="mb-1 text-lg font-bold text-gray-900">📰 뉴스 감성</h1>
      <p className="mb-4 text-xs text-gray-500">
        종목 뉴스를 모아 긍정/부정을 분석하고, 일별 감성 추이를 보여줍니다 (최근 약 1주)
      </p>

      <TickerSearch onSelect={onSelect} />

      {loading && <Loading text="뉴스 분석 중…" />}
      {error && <ErrorText message={error} />}

      {data && !loading && (
        <div className="mt-5 space-y-4">
          {/* 전체 감성 요약 */}
          <div className="flex items-center gap-2">
            <h2 className="text-base font-semibold">{data.name}</h2>
            {data.ticker && <span className="text-xs text-gray-400">{data.ticker}</span>}
          </div>

          {data.articles.length === 0 ? (
            <EmptyState icon="📭" message="최근 관련 뉴스를 찾지 못했어요." />
          ) : (
            <>
              <div className="flex flex-wrap gap-2">
                <div className="rounded-xl border border-gray-200 px-3 py-2">
                  <div className="text-[11px] text-gray-500">전체 감성</div>
                  <div className={`text-sm font-semibold ${overallColor(data.overall_sentiment)}`}>
                    {data.overall_sentiment}
                  </div>
                </div>
                <div className="rounded-xl border border-gray-200 px-3 py-2">
                  <div className="text-[11px] text-gray-500">긍정 / 부정 / 중립</div>
                  <div className="text-sm font-semibold tabular-nums">
                    <span className="text-red-600">{data.positive_count}</span> /{" "}
                    <span className="text-blue-600">{data.negative_count}</span> /{" "}
                    <span className="text-gray-500">{data.neutral_count}</span>
                  </div>
                </div>
              </div>

              {data.summary && (
                <p className="rounded-xl bg-gray-50 p-3 text-xs leading-relaxed text-gray-700">
                  {data.summary}
                </p>
              )}

              {data.key_topics.length > 0 && (
                <div className="flex flex-wrap gap-1.5">
                  {data.key_topics.map((t) => (
                    <span key={t} className="rounded-full bg-blue-50 px-2.5 py-1 text-xs text-blue-700">
                      #{t}
                    </span>
                  ))}
                </div>
              )}

              {/* 감성 시계열 차트 (2일+ 누적 시) */}
              {data.chart_b64 ? (
                <ChartImage b64={data.chart_b64} alt={`${data.name} 뉴스 감성 추이`} />
              ) : (
                <p className="text-center text-xs text-gray-400">
                  감성 추이 차트는 이틀 이상의 뉴스가 쌓이면 표시돼요.
                </p>
              )}

              {/* 기사 목록 */}
              <div className="space-y-2">
                {data.articles.map((a, i) => (
                  <a
                    key={i}
                    href={a.link}
                    target="_blank"
                    rel="noopener noreferrer"
                    className="block rounded-xl border border-gray-200 p-3 hover:bg-gray-50"
                  >
                    <div className="flex items-start justify-between gap-2">
                      <span className="text-sm text-gray-800">{a.title}</span>
                      <span className={`shrink-0 rounded-full px-2 py-0.5 text-[10px] ${sentBadge(a.sentiment)}`}>
                        {a.sentiment}
                      </span>
                    </div>
                    <div className="mt-1 text-[11px] text-gray-400">
                      {a.source}{a.source && a.published ? " · " : ""}{a.published}
                    </div>
                  </a>
                ))}
              </div>
            </>
          )}
        </div>
      )}

      <DataRangeNote />
    </main>
  );
}

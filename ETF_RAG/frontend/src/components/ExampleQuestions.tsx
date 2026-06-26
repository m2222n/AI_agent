"use client";

import { useEffect, useState } from "react";
import { getMovers } from "@/lib/api";
import type { MoversResponse } from "@/lib/types";

// 하드코딩 fallback (movers 없을 때)
const FALLBACK = [
  "KODEX 200 수익률 알려줘",
  "삼성전자 기술적 분석해줘",
  "삼성전자랑 SK하이닉스 비교해줘",
  "반도체 ETF 추천해줘",
];

export default function ExampleQuestions({
  onPick,
}: {
  onPick: (q: string) => void;
}) {
  const [movers, setMovers] = useState<MoversResponse | null>(null);

  useEffect(() => {
    getMovers(2).then(setMovers);
  }, []);

  // movers 기반 동적 예시 (오늘 급등/급락/거래대금)
  const dynamic: string[] = [];
  if (movers) {
    movers.gainers.slice(0, 2).forEach((m) =>
      dynamic.push(`${m.name} 오늘 ${m.change_pct > 0 ? "+" : ""}${m.change_pct}% 왜 올랐어?`),
    );
    movers.losers.slice(0, 1).forEach((m) =>
      dynamic.push(`${m.name} ${m.change_pct}% 하락, 기술적 분석해줘`),
    );
    movers.most_traded.slice(0, 1).forEach((m) =>
      dynamic.push(`${m.name} 앞으로 어떨까?`),
    );
  }

  const dyn = dynamic.length > 0;
  const items = dyn ? dynamic : FALLBACK;

  return (
    <div className="mt-10">
      <p className="mb-2 text-center text-sm text-gray-400">
        궁금한 ETF나 주식을 물어보세요
      </p>
      <p className="mb-4 text-center text-xs text-gray-400">
        {dyn ? "🔥 오늘의 추천 질문" : "💡 이렇게 물어보세요"}
      </p>
      <div className="flex flex-wrap justify-center gap-2">
        {items.map((q, i) => (
          <button
            key={i}
            type="button"
            onClick={() => onPick(q)}
            className="rounded-full border border-gray-200 dark:border-gray-800 bg-gray-50 px-3 py-1.5 text-xs text-gray-700 hover:bg-blue-50 hover:text-blue-700"
          >
            {q}
          </button>
        ))}
      </div>
    </div>
  );
}

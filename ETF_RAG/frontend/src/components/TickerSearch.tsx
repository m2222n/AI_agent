"use client";

import { useEffect, useRef, useState } from "react";
import { searchTickers } from "@/lib/api";

// "삼성전자 (005930)" → {name, ticker}
function parseOption(opt: string): { name: string; ticker: string } {
  const m = opt.match(/^(.*)\s+\(([^)]+)\)\s*$/);
  if (m) return { name: m[1], ticker: m[2] };
  return { name: opt, ticker: opt };
}

export default function TickerSearch({
  onSelect,
  placeholder = "종목명 또는 종목코드 검색…",
  assetType,
}: {
  onSelect: (sel: { name: string; ticker: string; raw: string }) => void;
  placeholder?: string;
  assetType?: "stock" | "etf"; // 지정 시 해당 자산만 자동완성 (재무제표=주식)
}) {
  const [query, setQuery] = useState("");
  const [options, setOptions] = useState<string[]>([]);
  const [open, setOpen] = useState(false);
  const boxRef = useRef<HTMLDivElement>(null);

  // 디바운스 검색 (250ms)
  useEffect(() => {
    const q = query.trim();
    if (!q) {
      setOptions([]);
      return;
    }
    const timer = setTimeout(async () => {
      const opts = await searchTickers(q, 20, assetType);
      setOptions(opts);
      setOpen(true);
    }, 250);
    return () => clearTimeout(timer);
  }, [query, assetType]);

  // 바깥 클릭 시 닫기
  useEffect(() => {
    const onDoc = (e: MouseEvent) => {
      if (boxRef.current && !boxRef.current.contains(e.target as Node)) {
        setOpen(false);
      }
    };
    document.addEventListener("mousedown", onDoc);
    return () => document.removeEventListener("mousedown", onDoc);
  }, []);

  const pick = (opt: string) => {
    const { name, ticker } = parseOption(opt);
    setQuery(name);
    setOpen(false);
    onSelect({ name, ticker, raw: opt });
  };

  return (
    <div ref={boxRef} className="relative">
      <input
        value={query}
        onChange={(e) => setQuery(e.target.value)}
        onFocus={() => options.length && setOpen(true)}
        placeholder={placeholder}
        className="w-full rounded-xl border border-gray-300 px-4 py-2.5 text-sm focus:border-blue-500 focus:outline-none"
      />
      {open && options.length > 0 && (
        <ul className="absolute z-10 mt-1 max-h-72 w-full overflow-y-auto rounded-xl border border-gray-200 bg-white shadow-lg">
          {options.map((opt) => (
            <li key={opt}>
              <button
                type="button"
                onClick={() => pick(opt)}
                className="block w-full px-4 py-2 text-left text-sm hover:bg-blue-50"
              >
                {opt}
              </button>
            </li>
          ))}
        </ul>
      )}
    </div>
  );
}

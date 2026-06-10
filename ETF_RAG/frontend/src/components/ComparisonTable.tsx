import type { ComparisonItem } from "@/lib/types";

// 표에 보여줄 필드 정의 (라벨 + 값 추출 + 포맷터). 값이 모든 항목에서 없으면 행 생략.
type FieldDef = {
  label: string;
  get: (it: ComparisonItem) => unknown;
  fmt: (v: unknown) => string;
};

// returns 중첩 객체에서 기간별 수익률 추출 (it.returns["1m"] 등)
function ret(period: string) {
  return (it: ComparisonItem) => {
    const r = it.returns as Record<string, unknown> | undefined;
    return r && typeof r === "object" ? r[period] : undefined;
  };
}
const f = (key: string) => (it: ComparisonItem) => it[key];

const won = (v: unknown) =>
  typeof v === "number" ? `${v.toLocaleString("ko-KR")}원` : "-";
const pct = (v: unknown) =>
  typeof v === "number" ? `${v > 0 ? "+" : ""}${v.toFixed(2)}%` : "-";
const num = (v: unknown) =>
  typeof v === "number" ? v.toLocaleString("ko-KR") : "-";
const x = (v: unknown) =>
  typeof v === "number" ? `${v.toFixed(2)}배` : "-";

const marketCap = (v: unknown) => {
  if (typeof v !== "number") return "-";
  const 조 = 1_0000_0000_0000;
  const 억 = 1_0000_0000;
  if (v >= 조) return `${(v / 조).toFixed(1)}조원`;
  if (v >= 억) return `${Math.round(v / 억).toLocaleString("ko-KR")}억원`;
  return won(v);
};

const FIELDS: FieldDef[] = [
  { label: "종가", get: f("close"), fmt: won },
  { label: "등락률", get: f("change_pct"), fmt: pct },
  { label: "1주 수익률", get: ret("1w"), fmt: pct },
  { label: "1개월 수익률", get: ret("1m"), fmt: pct },
  { label: "3개월 수익률", get: ret("3m"), fmt: pct },
  { label: "1년 수익률", get: ret("1y"), fmt: pct },
  { label: "시가총액", get: f("market_cap"), fmt: marketCap },
  { label: "PER", get: f("per"), fmt: x },
  { label: "PBR", get: f("pbr"), fmt: x },
  { label: "배당수익률", get: f("div"), fmt: pct },
  { label: "NAV", get: f("nav"), fmt: won }, // ETF
  { label: "괴리율", get: f("deviation"), fmt: pct }, // ETF
  { label: "거래량", get: f("volume"), fmt: num },
];

export default function ComparisonTable({
  items,
}: {
  items: ComparisonItem[];
}) {
  if (!items?.length) return null;

  // 어느 항목에든 값이 있는 필드만 행으로
  const rows = FIELDS.filter((fd) =>
    items.some((it) => {
      const v = fd.get(it);
      return v !== undefined && v !== null;
    }),
  );

  return (
    <div className="overflow-x-auto">
      <table className="comparison-table text-xs">
        <thead>
          <tr>
            <th className="text-left">항목</th>
            {items.map((it, i) => (
              <th key={i} className="text-right">
                {it.name}
                {it.ticker ? (
                  <span className="block font-normal text-gray-400">
                    {it.ticker}
                  </span>
                ) : null}
              </th>
            ))}
          </tr>
        </thead>
        <tbody>
          {rows.map((fd) => (
            <tr key={fd.label}>
              <td className="text-left text-gray-500">{fd.label}</td>
              {items.map((it, i) => (
                <td key={i} className="text-right tabular-nums">
                  {fd.fmt(fd.get(it))}
                </td>
              ))}
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}

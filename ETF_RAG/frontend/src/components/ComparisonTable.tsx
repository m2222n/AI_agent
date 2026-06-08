import type { ComparisonItem } from "@/lib/types";

// 표에 보여줄 필드 정의 (라벨 + 포맷터). 값이 모든 항목에서 없으면 행 생략.
type FieldDef = {
  key: string;
  label: string;
  fmt: (v: unknown) => string;
};

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
  { key: "close", label: "종가", fmt: won },
  { key: "change_pct", label: "등락률", fmt: pct },
  { key: "return_1m", label: "1개월 수익률", fmt: pct },
  { key: "return_1y", label: "1년 수익률", fmt: pct },
  { key: "market_cap", label: "시가총액", fmt: marketCap },
  { key: "per", label: "PER", fmt: x },
  { key: "pbr", label: "PBR", fmt: x },
  { key: "div", label: "배당수익률", fmt: pct },
  { key: "nav", label: "NAV", fmt: won }, // ETF
  { key: "deviation", label: "괴리율", fmt: pct }, // ETF
  { key: "operating_margin", label: "영업이익률", fmt: pct },
  { key: "volume", label: "거래량", fmt: num },
];

export default function ComparisonTable({
  items,
}: {
  items: ComparisonItem[];
}) {
  if (!items?.length) return null;

  // 어느 항목에든 값이 있는 필드만 행으로
  const rows = FIELDS.filter((f) =>
    items.some((it) => it[f.key] !== undefined && it[f.key] !== null),
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
          {rows.map((f) => (
            <tr key={f.key}>
              <td className="text-left text-gray-500">{f.label}</td>
              {items.map((it, i) => (
                <td key={i} className="text-right tabular-nums">
                  {f.fmt(it[f.key])}
                </td>
              ))}
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}

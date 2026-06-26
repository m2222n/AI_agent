"use client";

/**
 * 가상투자 자산 배분 도넛 차트 (외부 라이브러리 없이 SVG).
 * 보유 종목 평가금액 + 현금을 비중으로 시각화.
 */

type Slice = { label: string; value: number };

// 구분 잘 되는 팔레트(현금은 회색 고정, 종목은 순환)
const COLORS = [
  "#2563eb", "#e8453c", "#34a853", "#fbbc04", "#9333ea",
  "#0891b2", "#db2777", "#65a30d", "#ea580c", "#4f46e5",
];
const CASH_COLOR = "#9ca3af";

function polar(cx: number, cy: number, r: number, deg: number) {
  const rad = ((deg - 90) * Math.PI) / 180;
  return { x: cx + r * Math.cos(rad), y: cy + r * Math.sin(rad) };
}

function arcPath(cx: number, cy: number, r: number, start: number, end: number) {
  // 단일 슬라이스가 100%면 SVG arc가 그려지지 않으므로 살짝 줄임
  const e = end - start >= 360 ? start + 359.999 : end;
  const s = polar(cx, cy, r, start);
  const f = polar(cx, cy, r, e);
  const large = e - start > 180 ? 1 : 0;
  return `M ${cx} ${cy} L ${s.x} ${s.y} A ${r} ${r} 0 ${large} 1 ${f.x} ${f.y} Z`;
}

export default function PortfolioPie({
  holdings,
  cash,
}: {
  holdings: { name: string; eval_value: number }[];
  cash: number;
}) {
  const slices: Slice[] = [
    ...holdings.map((h) => ({ label: h.name, value: h.eval_value })),
  ];
  if (cash > 0) slices.push({ label: "현금", value: cash });
  const total = slices.reduce((a, s) => a + s.value, 0);
  if (total <= 0) return null;

  const cx = 80, cy = 80, r = 76, hole = 46;
  let acc = 0;
  const arcs = slices.map((s, i) => {
    const start = (acc / total) * 360;
    acc += s.value;
    const end = (acc / total) * 360;
    const color = s.label === "현금" ? CASH_COLOR : COLORS[i % COLORS.length];
    return { ...s, start, end, color, pct: (s.value / total) * 100 };
  });

  return (
    <div className="flex flex-wrap items-center gap-4">
      <svg viewBox="0 0 160 160" width={160} height={160} role="img" aria-label="자산 배분">
        {arcs.map((a) => (
          <path key={a.label} d={arcPath(cx, cy, r, a.start, a.end)} fill={a.color} />
        ))}
        {/* 도넛 구멍 */}
        <circle cx={cx} cy={cy} r={hole} fill="white" />
      </svg>
      <ul className="space-y-1 text-xs">
        {arcs.map((a) => (
          <li key={a.label} className="flex items-center gap-2">
            <span className="inline-block h-3 w-3 shrink-0 rounded-sm" style={{ background: a.color }} />
            <span className="text-gray-700">{a.label}</span>
            <span className="ml-auto tabular-nums text-gray-500">{a.pct.toFixed(1)}%</span>
          </li>
        ))}
      </ul>
    </div>
  );
}

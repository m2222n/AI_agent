/* eslint-disable @next/next/no-img-element */
// base64 data URI는 next/image로 최적화 불가 → 일반 <img> 사용.
import type { StructuredData } from "@/lib/types";
import ComparisonTable from "./ComparisonTable";

function ChartImage({ b64, alt }: { b64: string; alt: string }) {
  return (
    <img
      src={`data:image/png;base64,${b64}`}
      alt={alt}
      className="mt-2 w-full rounded-lg border border-gray-200"
    />
  );
}

export default function StructuredDataView({ data }: { data: StructuredData }) {
  switch (data.__type__) {
    case "comparison_table":
      return (
        <div className="mt-2">
          <ComparisonTable items={data.items} />
          {data.comparison_chart_b64 && (
            <ChartImage b64={data.comparison_chart_b64} alt="상대 수익률 추이" />
          )}
        </div>
      );
    case "technical_chart":
      return <ChartImage b64={data.image_b64} alt={`${data.name} 기술적 분석 차트`} />;
    case "portfolio_chart":
      return (
        <ChartImage
          b64={data.image_b64}
          alt={`포트폴리오 차트 (${data.names.join(", ")})`}
        />
      );
    default:
      return null;
  }
}

/* eslint-disable @next/next/no-img-element */
// base64 data URI는 next/image로 최적화 불가 → 일반 <img>.

export default function ChartImage({
  b64,
  alt,
}: {
  b64: string | null | undefined;
  alt: string;
}) {
  if (!b64) return null;
  return (
    <img
      src={`data:image/png;base64,${b64}`}
      alt={alt}
      className="mt-3 w-full rounded-lg border border-gray-200"
    />
  );
}

// 분석 탭 공통 데이터 범위 안내. 일별 시세는 2014년 4월부터 보유(pykrx/KRX 12년 제한).
export default function DataRangeNote() {
  return (
    <p className="mt-4 text-[11px] leading-relaxed text-gray-400">
      ℹ️ 일별 시세(OHLCV) 데이터는 2014년 4월부터 보유합니다. 종목별 상장일에 따라
      실제 시작일은 더 늦을 수 있어요. 데이터 기반 참고 정보이며, 투자 판단의 책임은
      이용자 본인에게 있습니다.
    </p>
  );
}

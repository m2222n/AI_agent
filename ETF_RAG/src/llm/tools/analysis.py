"""
분석 도구 — get_realtime_price, analyze_sector, get_technical_indicators
"""

import json
import logging

from langchain_core.tools import tool

from src.llm.tools import _state
from src.llm.tools._helpers import _find_structured_data, _not_found_message, _fmt_date
from src.utils.formatters import format_market_cap

logger = logging.getLogger(__name__)


def _format_cap(value: int) -> str:
    """시가총액을 조/억 단위 문자열로 (접미사 '원' 없음)"""
    return format_market_cap(value, suffix=False)


def _calc_percentile(value: float, values: list[float]) -> float:
    """값이 리스트에서 몇 번째 백분위인지 계산 (0~100)."""
    if not values or value is None:
        return 50.0
    below = sum(1 for v in values if v < value)
    return round(below / len(values) * 100, 1)


def _format_valuation_position(stock_data: dict, sector_stocks: list) -> str:
    """종목의 업종 내 밸류에이션 상대 위치를 포맷팅."""
    ticker = stock_data.get("ticker", "")
    parts = []

    # PER 위치
    per = stock_data.get("per", 0)
    pers = [s["per"] for s in sector_stocks if s["per"] and s["per"] > 0]
    if per and per > 0 and len(pers) >= 3:
        pctile = _calc_percentile(per, pers)
        avg_per = sum(pers) / len(pers)
        diff_pct = (per - avg_per) / avg_per * 100
        if diff_pct > 20:
            label = "고평가"
        elif diff_pct < -20:
            label = "저평가"
        else:
            label = "평균 수준"
        parts.append(f"PER {per:.1f}배 (업종 평균 {avg_per:.1f}, "
                     f"상위 {100 - pctile:.0f}%, {label})")

    # PBR 위치
    pbr = stock_data.get("pbr", 0)
    pbrs = [s["pbr"] for s in sector_stocks if s["pbr"] and s["pbr"] > 0]
    if pbr and pbr > 0 and len(pbrs) >= 3:
        pctile = _calc_percentile(pbr, pbrs)
        avg_pbr = sum(pbrs) / len(pbrs)
        parts.append(f"PBR {pbr:.2f}배 (업종 평균 {avg_pbr:.2f}, "
                     f"상위 {100 - pctile:.0f}%)")

    # 배당수익률 위치
    div_rate = stock_data.get("div", 0)
    divs = [s["div"] for s in sector_stocks if s["div"] and s["div"] > 0]
    if div_rate and div_rate > 0 and len(divs) >= 3:
        pctile = _calc_percentile(div_rate, divs)
        avg_div = sum(divs) / len(divs)
        parts.append(f"배당 {div_rate:.2f}% (업종 평균 {avg_div:.2f}%, "
                     f"상위 {100 - pctile:.0f}%)")

    # 시가총액 순위
    mcap = stock_data.get("market_cap", 0)
    if mcap:
        caps_sorted = sorted(
            [s["market_cap"] for s in sector_stocks if s["market_cap"]],
            reverse=True
        )
        rank = next((i + 1 for i, c in enumerate(caps_sorted) if c <= mcap), len(caps_sorted))
        parts.append(f"시가총액 업종 내 {rank}/{len(caps_sorted)}위")

    if not parts:
        return ""
    return "[업종 내 밸류에이션 위치] " + " | ".join(parts)


def _format_sector_analysis(sector: str, stocks: list) -> str:
    """업종 분석 결과를 포맷팅 — 시가총액 상위 + PER/PBR 통계"""
    lines = [f"[{sector}] 업종 분석 ({len(stocks)}종목)\n"]

    # 시가총액 상위 10개
    lines.append("**시가총액 상위:**")
    for i, s in enumerate(stocks[:10], 1):
        per_str = f"PER {s['per']:.1f}" if s["per"] else "PER -"
        pbr_str = f"PBR {s['pbr']:.2f}" if s["pbr"] else "PBR -"
        cap_str = _format_cap(s["market_cap"])
        lines.append(
            f"{i}. [{s['ticker']}] {s['name']} | "
            f"종가 {s['close']:,}원 ({s['change_pct']:+.2f}%) | "
            f"시총 {cap_str} | {per_str} | {pbr_str}"
        )

    # 업종 PER/PBR 통계
    pers = [s["per"] for s in stocks if s["per"] and s["per"] > 0]
    pbrs = [s["pbr"] for s in stocks if s["pbr"] and s["pbr"] > 0]
    divs = [s["div"] for s in stocks if s["div"] and s["div"] > 0]

    lines.append(f"\n**업종 밸류에이션 통계 ({sector}):**")
    if pers:
        avg_per = sum(pers) / len(pers)
        min_per = min(pers)
        max_per = max(pers)
        sorted_pers = sorted(pers)
        median_per = sorted_pers[len(sorted_pers) // 2]
        lines.append(f"- PER: 평균 {avg_per:.1f}배, 중간값 {median_per:.1f}배 "
                     f"(최저 {min_per:.1f} ~ 최고 {max_per:.1f})")
        # PER 분포 구간
        ranges = [(0, 10), (10, 20), (20, 50), (50, float("inf"))]
        labels = ["0~10배", "10~20배", "20~50배", "50배 이상"]
        dist = [sum(1 for p in pers if lo <= p < hi) for lo, hi in ranges]
        dist_str = ", ".join(f"{l}: {c}개" for l, c in zip(labels, dist) if c > 0)
        lines.append(f"  분포: {dist_str}")
    if pbrs:
        avg_pbr = sum(pbrs) / len(pbrs)
        low_pbr = [s for s in stocks if s["pbr"] and 0 < s["pbr"] < 1]
        lines.append(f"- PBR: 평균 {avg_pbr:.2f}배 (PBR<1 저평가 {len(low_pbr)}종목)")
    if divs:
        avg_div = sum(divs) / len(divs)
        high_div = [s for s in stocks if s["div"] and s["div"] >= 3.0]
        lines.append(f"- 배당수익률: 평균 {avg_div:.2f}% (3% 이상 고배당 {len(high_div)}종목)")

    # 업종 시가총액 합계
    total_cap = sum(s["market_cap"] for s in stocks)
    lines.append(f"- 업종 시가총액 합계: {_format_cap(total_cap)}")

    return "\n".join(lines)


@tool
def get_realtime_price(name_or_ticker: str) -> str:
    """ETF나 주식의 현재 가격을 조회합니다. 장중에는 실시간(15분 지연) 데이터를,
    장 마감 후에는 가장 최근 종가 데이터를 반환합니다.
    "현재 가격", "지금 얼마", "실시간 시세" 등의 질문에 사용합니다.

    Args:
        name_or_ticker: ETF/주식 이름 또는 티커 (예: "KODEX 200", "069500", "삼성전자")
    """
    from config import REALTIME_PRICE

    # 종목 조회
    data = _find_structured_data(name_or_ticker)
    if not data:
        return _not_found_message(name_or_ticker)

    ticker = data.get("ticker", "")
    name = data.get("name", "")
    asset_type = "etf" if "nav" in data else "stock"

    # 장중 실시간 조회 시도
    if REALTIME_PRICE.get("enabled", True):
        try:
            from src.data.realtime import get_realtime_price as _get_rt
            rt = _get_rt(ticker, asset_type,
                         cache_ttl=REALTIME_PRICE.get("cache_ttl", 300))
            if rt:
                line = f"[{name} ({ticker})] 현재가: {rt['price']:,}원"
                if rt["change"] is not None:
                    line += f", 전일대비: {rt['change']:+,}원 ({rt['change_pct']:+.2f}%)"
                if rt.get("volume"):
                    line += f", 거래량: {rt['volume']:,}주"
                if rt.get("source") == "kis":
                    line += f"\n(한국투자증권 실시간 시세, 조회시각: {rt['timestamp']})"
                else:
                    line += f"\n(yfinance 15분 지연 데이터, 조회시각: {rt['timestamp']})"
                return line
        except Exception as e:
            logger.warning(f"실시간 가격 조회 실패: {e}")

    # Fallback: pykrx 구조화 데이터
    close = data.get("close", 0)
    change_pct = data.get("change_pct", 0)
    date = data.get("date", "")
    if len(date) == 8:
        date = f"{date[:4]}-{date[4:6]}-{date[6:]}"

    line = f"[{name} ({ticker})] 종가: {close:,}원, 등락률: {change_pct:+.2f}%"

    # 수익률 정보 추가
    returns = data.get("returns", {})
    if returns:
        labels = {"1d": "1일", "1w": "1주", "1m": "1개월", "3m": "3개월", "1y": "1년"}
        parts = []
        for k, label in labels.items():
            v = returns.get(k)
            if v is not None:
                parts.append(f"{label}: {v:+.2f}%")
        if parts:
            line += f"\n수익률: {', '.join(parts)}"

    # ETF 전용
    if "nav" in data:
        line += f"\nNAV: {data.get('nav', 0):,.0f}원"

    # 주식 전용
    if "per" in data:
        line += f"\nPER: {data.get('per', 0):.2f}배, PBR: {data.get('pbr', 0):.2f}배"

    try:
        from src.data.realtime import is_market_open
        if is_market_open():
            line += f"\n(실시간 데이터 조회 실패, 최근 수집 데이터 기준일: {date})"
        else:
            line += f"\n(장 마감 후 데이터, 기준일: {date})"
    except ImportError:
        line += f"\n(수집 데이터, 기준일: {date})"

    return line


@tool
def analyze_sector(query: str) -> str:
    """특정 종목이 포함된 ETF를 찾거나, 업종별 종목 분석(PER/PBR 비교, 시가총액 상위)을 수행합니다.
    "삼성전자 담고 있는 ETF", "전기전자 업종 분석", "은행 업종 PER 비교", "반도체 관련 ETF" 등의 질문에 사용합니다.

    Args:
        query: 종목명/티커, 업종명, 또는 섹터 키워드 (예: "삼성전자", "전기·전자", "반도체", "은행")
    """
    query_lower = query.lower().strip()

    # 1. 업종 인덱스 검색 — 정확 매칭 또는 부분 매칭
    if _state._sector_index:
        matched_sector = None
        matched_stocks = None

        # 정확 매칭
        for sector_name, stocks in _state._sector_index.items():
            if sector_name.lower() == query_lower or sector_name == query:
                matched_sector = sector_name
                matched_stocks = stocks
                break

        # 부분 매칭
        if not matched_sector:
            for sector_name, stocks in _state._sector_index.items():
                if query_lower in sector_name.lower():
                    matched_sector = sector_name
                    matched_stocks = stocks
                    break

        if matched_sector and matched_stocks:
            return _format_sector_analysis(matched_sector, matched_stocks)

    # 2. 보유종목 역인덱스 — 종목→ETF 매핑
    if not _state._holdings_reverse_index:
        return "보유종목/업종 데이터가 없습니다. 데이터 수집 후 이용 가능합니다."

    # 2-1. 정확 매칭 — 종목명 또는 티커로 직접 조회
    matches = _state._holdings_reverse_index.get(query_lower) or _state._holdings_reverse_index.get(query)
    if matches:
        stock_name = matches[0].get("stock_name", query)
        sorted_matches = sorted(matches, key=lambda x: x["weight"], reverse=True)
        seen = set()
        unique = []
        for m in sorted_matches:
            if m["etf_ticker"] not in seen:
                seen.add(m["etf_ticker"])
                unique.append(m)

        lines = [f"[{stock_name}]을(를) 보유한 ETF ({len(unique)}개):\n"]
        for m in unique[:15]:
            lines.append(
                f"- [{m['etf_ticker']}] {m['etf_name']} (비중: {m['weight']:.2f}%)"
            )
        if len(unique) > 15:
            lines.append(f"  ... 외 {len(unique) - 15}개")

        avg_weight = sum(m["weight"] for m in unique) / len(unique)
        max_m = unique[0]
        lines.append(f"\n[통계] 평균 비중: {avg_weight:.2f}%, "
                      f"최대 비중: {max_m['etf_name']} ({max_m['weight']:.2f}%)")

        # 해당 종목의 업종 정보 + 밸류에이션 위치
        stock_data = _state._stock_data_index.get(query_lower) or _state._stock_data_index.get(query)
        if stock_data and stock_data.get("sector"):
            sector = stock_data["sector"]
            lines.append(f"\n[업종] {stock_name}: {sector}")

            if sector in _state._sector_index:
                sector_stocks = _state._sector_index[sector]
                # 밸류에이션 상대 위치
                val_info = _format_valuation_position(stock_data, sector_stocks)
                if val_info:
                    lines.append(val_info)

                # 동일 업종 종목
                peers = [s for s in sector_stocks
                         if s["ticker"] != stock_data.get("ticker", "")][:5]
                if peers:
                    peer_names = ", ".join(p["name"] for p in peers)
                    lines.append(f"[동일 업종] {peer_names}")

        return "\n".join(lines)

    # 2-2. 부분 매칭 — 키워드로 종목명 검색
    keyword_matches = {}
    for key, entries in _state._holdings_reverse_index.items():
        if not key.replace(" ", "").isalpha() and not any(
            '\uac00' <= c <= '\ud7a3' for c in key
        ):
            continue
        if query_lower in key:
            for e in entries:
                st = e.get("stock_name", "")
                if st not in keyword_matches:
                    keyword_matches[st] = {"stock_name": st, "etfs": []}
                if e["etf_ticker"] not in [x["etf_ticker"] for x in keyword_matches[st]["etfs"]]:
                    keyword_matches[st]["etfs"].append(e)

    if keyword_matches:
        lines = [f"'{query}' 관련 종목을 보유한 ETF:\n"]
        for stock_name, info in sorted(
            keyword_matches.items(),
            key=lambda x: len(x[1]["etfs"]),
            reverse=True,
        )[:5]:
            etfs = sorted(info["etfs"], key=lambda x: x["weight"], reverse=True)
            lines.append(f"**{stock_name}** ({len(etfs)}개 ETF에 편입)")
            for e in etfs[:5]:
                lines.append(
                    f"  - [{e['etf_ticker']}] {e['etf_name']} (비중: {e['weight']:.2f}%)"
                )
            if len(etfs) > 5:
                lines.append(f"    ... 외 {len(etfs) - 5}개")
            lines.append("")
        return "\n".join(lines)

    return f"'{query}'에 해당하는 업종/보유종목 정보를 찾지 못했습니다."


@tool
def get_technical_indicators(name_or_ticker: str, days: int = 120) -> str:
    """ETF/주식의 기술적 지표를 분석합니다. 이동평균(MA), RSI, MACD, 볼린저 밴드, 골든크로스/데드크로스,
    스토캐스틱, 일목균형표, CCI, ADX, OBV, ATR 등 종합 기술적 분석을 제공합니다.
    차트 이미지도 함께 생성됩니다.
    "삼성전자 골든크로스 났어?", "KODEX 200 기술적 분석", "SK하이닉스 RSI", "삼성전자 일목균형표" 등의 질문에 사용합니다.

    Args:
        name_or_ticker: ETF/주식 이름 또는 티커 (예: "삼성전자", "005930", "KODEX 200")
        days: 차트 및 분석 기간 (영업일 수). 기본 120일(약 6개월). 1년=250, 3년=750, 5년=1250, 10년=2500.
    """
    # 종목 조회
    data = _find_structured_data(name_or_ticker)
    if not data:
        return _not_found_message(name_or_ticker)

    ticker = data.get("ticker", "")
    name = data.get("name", "")

    try:
        from src.data.technical import get_technical_summary
        analysis_days = max(days, 250)
        summary = get_technical_summary(ticker, days=analysis_days)
    except Exception as e:
        logger.warning(f"기술적 지표 계산 실패: {e}")
        return f"'{name}'의 기술적 지표 계산에 실패했습니다. (데이터 부족 또는 오류)"

    if not summary:
        return f"'{name}'의 일봉 데이터가 부족합니다 (최소 20일 필요)."

    # 포맷팅
    first_date = _fmt_date(summary.get("first_date", ""))
    last_date = _fmt_date(summary.get("last_date", summary["date"]))
    lines = [f"[{name} ({ticker})] 기술적 분석 (기준일: {last_date}, "
             f"종가: {summary['close']:,}원, 분석 기간: {summary['data_days']}일)"]
    lines.append(f"**데이터 범위:** {first_date} ~ {last_date} ({summary['data_days']}영업일)")
    lines.append("")

    # 이동평균
    ma = summary["ma"]
    lines.append("**이동평균(MA):**")
    for label, key in [("5일", "ma5"), ("20일", "ma20"), ("60일", "ma60"), ("120일", "ma120")]:
        val = ma.get(key)
        if val:
            diff = summary["close"] - val
            pct = diff / val * 100
            position = "위" if diff > 0 else "아래"
            lines.append(f"  - {label} MA: {val:,}원 (현재가 {position} {abs(pct):.1f}%)")

    # 추세
    lines.append(f"  - 추세 판정: **{summary['trend']}** (MA5 vs MA20 vs MA60 정배열 기준)")

    # 크로스
    cross = summary["cross"]
    cross_msgs = []
    for label, key in [("5일/20일", "5_20"), ("20일/60일", "20_60"), ("60일/120일", "60_120")]:
        val = cross.get(key)
        if val == "golden_cross":
            cross_msgs.append(f"  - ⚡ **{label} 골든크로스** 발생!")
        elif val == "dead_cross":
            cross_msgs.append(f"  - ⚠️ **{label} 데드크로스** 발생!")
    if cross_msgs:
        lines.append("\n**크로스 시그널:**")
        lines.extend(cross_msgs)
    else:
        lines.append("\n**크로스 시그널:** 최근 교차 없음")

    # RSI
    rsi = summary.get("rsi")
    if rsi is not None:
        if rsi >= 70:
            rsi_label = "과매수 구간 (매도 신호)"
        elif rsi <= 30:
            rsi_label = "과매도 구간 (매수 신호)"
        else:
            rsi_label = "중립 구간"
        lines.append(f"\n**RSI(14):** {rsi:.1f} — {rsi_label}")

    # MACD
    macd = summary.get("macd")
    if macd:
        macd_signal = "매수 우위" if macd["histogram"] > 0 else "매도 우위"
        lines.append(f"\n**MACD(12,26,9):**")
        lines.append(f"  - MACD: {macd['macd']:,.0f}, Signal: {macd['signal']:,.0f}, "
                     f"Histogram: {macd['histogram']:,.0f} ({macd_signal})")

    # 볼린저 밴드
    bb = summary.get("bollinger")
    if bb:
        lines.append(f"\n**볼린저 밴드(20,2):**")
        lines.append(f"  - 상단: {bb['upper']:,.0f}원, 중심: {bb['middle']:,.0f}원, "
                     f"하단: {bb['lower']:,.0f}원")
        lines.append(f"  - 밴드폭: {bb['width']:.1f}%, %B: {bb['pct_b']:.1f}%")
        if bb["pct_b"] > 100:
            lines.append("  - 상단 돌파 (과매수 가능성)")
        elif bb["pct_b"] < 0:
            lines.append("  - 하단 이탈 (과매도 가능성)")

    # 스토캐스틱
    stoch = summary.get("stochastic")
    if stoch:
        lines.append(f"\n**스토캐스틱(14,3):** %K {stoch['k']:.1f}, %D {stoch['d']:.1f} — {stoch['signal']}")

    # 일목균형표
    ichimoku = summary.get("ichimoku")
    if ichimoku:
        lines.append(f"\n**일목균형표:**")
        lines.append(f"  - 전환선: {ichimoku['tenkan']:,}원, 기준선: {ichimoku['kijun']:,}원")
        lines.append(f"  - 선행스팬1: {ichimoku['senkou_a']:,}원, 선행스팬2: {ichimoku['senkou_b']:,}원")
        lines.append(f"  - 현재가 **{ichimoku['cloud_status']}** "
                     f"({'강세 신호' if ichimoku['cloud_status'] == '구름대 위' else '약세 신호' if ichimoku['cloud_status'] == '구름대 아래' else '방향 탐색 중'})")

    # CCI
    cci = summary.get("cci")
    if cci:
        lines.append(f"\n**CCI(20):** {cci['cci']:+.1f} — {cci['signal']} (±100 기준)")

    # ADX
    adx = summary.get("adx")
    if adx:
        di_status = "+DI > -DI (상승 우위)" if adx["plus_di"] > adx["minus_di"] else "-DI > +DI (하락 우위)"
        lines.append(f"\n**ADX(14):** {adx['adx']:.1f} ({adx['trend_strength']}) — {di_status}")

    # OBV
    obv = summary.get("obv")
    if obv:
        lines.append(f"\n**OBV:** {obv['obv']:,} (20일 MA 대비 **{obv['trend']}** 구간)")

    # ATR
    atr = summary.get("atr")
    if atr:
        lines.append(f"\n**ATR(14):** {atr['atr']:,.0f}원 ({atr['atr_pct']:.1f}%, {atr['volatility']})")

    # ── 종합 기술적 판단 요약 ──
    bullish = 0
    bearish = 0
    # 추세
    if summary.get("trend") in ("상승 추세", "강한 상승"):
        bullish += 1
    elif summary.get("trend") in ("하락 추세", "강한 하락"):
        bearish += 1
    # RSI
    if rsi is not None:
        if rsi >= 70:
            bearish += 1
        elif rsi <= 30:
            bullish += 1
    # MACD
    if macd and macd.get("histogram", 0) > 0:
        bullish += 1
    elif macd and macd.get("histogram", 0) < 0:
        bearish += 1
    # 크로스
    for key in cross.values():
        if key == "golden_cross":
            bullish += 1
        elif key == "dead_cross":
            bearish += 1
    # 일목균형표
    if ichimoku and ichimoku.get("cloud_status") == "구름대 위":
        bullish += 1
    elif ichimoku and ichimoku.get("cloud_status") == "구름대 아래":
        bearish += 1
    # ADX
    if adx and adx.get("plus_di", 0) > adx.get("minus_di", 0):
        bullish += 1
    elif adx and adx.get("minus_di", 0) > adx.get("plus_di", 0):
        bearish += 1
    # OBV
    if obv and obv.get("trend") == "매집":
        bullish += 1
    elif obv and obv.get("trend") == "분산":
        bearish += 1

    total = bullish + bearish
    if total > 0:
        bull_pct = bullish / total * 100
        if bull_pct >= 70:
            overall = "강세 우위"
        elif bull_pct >= 55:
            overall = "약한 강세"
        elif bull_pct <= 30:
            overall = "약세 우위"
        elif bull_pct <= 45:
            overall = "약한 약세"
        else:
            overall = "중립 (혼조)"
        lines.append(f"\n**종합 판단:** {overall} (강세 {bullish}개 vs 약세 {bearish}개 지표)")

    text_result = "\n".join(lines)

    # 차트 이미지 생성
    try:
        from src.data.chart_generator import generate_technical_chart
        chart_days = max(days, 120)
        chart_b64 = generate_technical_chart(ticker, name, days=chart_days)
        if chart_b64:
            chart_json = json.dumps(
                {"__type__": "technical_chart", "image_b64": chart_b64, "name": name},
                ensure_ascii=False,
            )
            return f"{chart_json}\n\n---\n\n{text_result}"
    except Exception as e:
        logger.warning(f"차트 생성 실패: {e}")

    return text_result

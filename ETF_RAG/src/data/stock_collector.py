"""
주식 데이터 수집기 — pykrx 기반 일배치 수집

KOSPI + KOSDAQ 주식 전종목 시세 + 시가총액 + 펀더멘털(PER/PBR/EPS/DIV) 수집.
ETF collector.py와 동일한 패턴으로 KRX 로그인 + 일괄 API 활용.

사용법:
    python -m src.data.stock_collector              # 최근 영업일 기준 수집
    python -m src.data.stock_collector --date 20260403  # 특정일 수집
    python -m src.data.stock_collector --market KOSPI    # KOSPI만 수집
    python -m src.data.stock_collector --test       # 테스트 (10개만)

환경변수 (.env):
    KRX_ID: KRX Data Marketplace 로그인 ID
    KRX_PW: KRX Data Marketplace 로그인 비밀번호
"""

import json
import logging
import argparse
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional

from pykrx import stock

logger = logging.getLogger(__name__)

# 프로젝트 경로
DATA_DIR = Path(__file__).parent
OUTPUT_DIR = DATA_DIR / "collected"

# 수집 설정
REQUEST_DELAY = 1.5  # KRX 요청 간 딜레이 (초)

# 수익률 기간 정의 (ETF와 동일)
RETURN_PERIODS = [
    ("1d", 1),
    ("1w", 7),
    ("1m", 30),
    ("3m", 90),
    ("1y", 365),
]


# ── 유틸리티 ──────────────────────────────────────────────────


def _suppress_pykrx_logging_errors():
    """pykrx 내부 logging 포맷 에러 억제 필터 설치."""
    import logging as _logging

    class _PykrxFilter(_logging.Filter):
        def filter(self, record):
            if isinstance(record.args, dict):
                return False
            return True

    _logging.getLogger().addFilter(_PykrxFilter())


_suppress_pykrx_logging_errors()


def _safe_get_ticker_name(ticker: str) -> str:
    """pykrx get_market_ticker_name의 안전한 래퍼.

    pykrx 내부에서 존재하지 않는 종목 조회 시 에러 발생 가능.
    BaseException까지 잡아서 프로세스 크래시 방지.
    """
    try:
        name = stock.get_market_ticker_name(ticker)
        return name or ""
    except BaseException:
        return ""


def _is_weekend(dt: datetime) -> bool:
    """토요일(5) 또는 일요일(6)인지 확인."""
    return dt.weekday() >= 5


def find_latest_business_day(from_date: Optional[str] = None) -> str:
    """최근 영업일 찾기.

    1단계: 주말(토/일)이면 API 호출 없이 즉시 스킵 (불필요한 KRX 호출 방지)
    2단계: 평일이면 실제 시세 데이터로 확인 (공휴일 대응)
    """
    if from_date:
        dt = datetime.strptime(from_date, "%Y%m%d")
    else:
        dt = datetime.now()

    for _ in range(10):
        if _is_weekend(dt):
            dt -= timedelta(days=1)
            continue

        date_str = dt.strftime("%Y%m%d")
        try:
            df = stock.get_market_ohlcv_by_ticker(date_str, market="KOSPI")
            if not df.empty and (df["종가"] > 0).any():
                return date_str
        except Exception:
            pass
        dt -= timedelta(days=1)

    raise RuntimeError("최근 10일 내 영업일을 찾을 수 없습니다.")


# ── 일괄 수집 (전종목 한 번에) ────────────────────────────────

def collect_bulk_ohlcv(date: str, market: str = "ALL") -> dict:
    """전종목 시세(OHLCV) 일괄 수집

    Returns: {ticker: {open, high, low, close, volume, trade_value, change_pct}}
    """
    logger.info(f"주식 시세 일괄 수집 중... ({market}, 기준일: {date})")
    df = stock.get_market_ohlcv_by_ticker(date, market=market)

    result = {}
    for ticker, row in df.iterrows():
        result[ticker] = {
            "open": int(row.get("시가", 0)),
            "high": int(row.get("고가", 0)),
            "low": int(row.get("저가", 0)),
            "close": int(row.get("종가", 0)),
            "volume": int(row.get("거래량", 0)),
            "trade_value": int(row.get("거래대금", 0)),
            "change_pct": round(float(row.get("등락률", 0)), 2),
        }

    logger.info(f"시세 {len(result)}종목 수집 완료")
    return result


def collect_bulk_market_cap(date: str, market: str = "ALL") -> dict:
    """전종목 시가총액 + 상장주식수 일괄 수집

    Returns: {ticker: {market_cap, shares_outstanding}}
    """
    logger.info(f"시가총액 일괄 수집 중... ({market})")
    df = stock.get_market_cap_by_ticker(date, market=market)

    result = {}
    for ticker, row in df.iterrows():
        result[ticker] = {
            "market_cap": int(row.get("시가총액", 0)),
            "shares_outstanding": int(row.get("상장주식수", 0)),
        }

    logger.info(f"시가총액 {len(result)}종목 수집 완료")
    return result


def collect_bulk_fundamental(date: str, market: str = "ALL") -> dict:
    """전종목 펀더멘털(PER/PBR/EPS/BPS/DIV/DPS) 일괄 수집

    Returns: {ticker: {per, pbr, eps, bps, div, dps}}
    """
    logger.info(f"펀더멘털 일괄 수집 중... ({market})")

    result = {}
    # market="ALL"은 지원 안 될 수 있으므로 KOSPI + KOSDAQ 개별 수집
    markets = ["KOSPI", "KOSDAQ"] if market == "ALL" else [market]

    for mkt in markets:
        try:
            df = stock.get_market_fundamental_by_ticker(date, market=mkt)
            for ticker, row in df.iterrows():
                result[ticker] = {
                    "bps": round(float(row.get("BPS", 0)), 2),
                    "per": round(float(row.get("PER", 0)), 2),
                    "pbr": round(float(row.get("PBR", 0)), 2),
                    "eps": round(float(row.get("EPS", 0)), 2),
                    "div": round(float(row.get("DIV", 0)), 2),
                    "dps": round(float(row.get("DPS", 0)), 2),
                }
        except Exception as e:
            logger.warning(f"펀더멘털 수집 실패 ({mkt}): {e}")

    logger.info(f"펀더멘털 {len(result)}종목 수집 완료")
    return result


def collect_bulk_sector(date: str, market: str = "ALL") -> dict:
    """전종목 업종 분류 일괄 수집

    Returns: {ticker: sector_name}  (예: {"005930": "전기·전자"})
    """
    logger.info(f"업종 분류 수집 중... ({market})")
    markets = ["KOSPI", "KOSDAQ"] if market == "ALL" else [market]

    result = {}
    for mkt in markets:
        try:
            df = stock.get_market_sector_classifications(date, market=mkt)
            for ticker, row in df.iterrows():
                result[ticker] = row.get("업종명", "")
        except Exception as e:
            logger.warning(f"업종 분류 수집 실패 ({mkt}): {e}")

    logger.info(f"업종 분류 {len(result)}종목 수집 완료")
    return result


def collect_bulk_returns(date: str, market: str = "ALL") -> dict:
    """전종목 기간별 수익률 일괄 수집

    Returns: {ticker: {"1d": float, "1w": float, ...}}
    """
    dt_base = datetime.strptime(date, "%Y%m%d")
    result = {}

    markets = ["KOSPI", "KOSDAQ"] if market == "ALL" else [market]

    for label, days in RETURN_PERIODS:
        dt_from = dt_base - timedelta(days=days)
        fromdate = dt_from.strftime("%Y%m%d")

        logger.info(f"수익률 수집 중: {label} ({fromdate} ~ {date})")

        for mkt in markets:
            try:
                df = stock.get_market_price_change_by_ticker(fromdate, date, market=mkt)
                for ticker, row in df.iterrows():
                    if ticker not in result:
                        result[ticker] = {}
                    result[ticker][label] = round(float(row.get("등락률", 0)), 2)
            except Exception as e:
                logger.warning(f"수익률 수집 실패 ({label}, {mkt}): {e}")

        time.sleep(REQUEST_DELAY)

    logger.info(f"수익률 수집 완료: {len(result)}종목, {len(RETURN_PERIODS)}개 기간")
    return result


# ── 메인 수집 로직 ────────────────────────────────────────────

def collect_all(date: str, market: str = "ALL", max_stocks: int = 0) -> dict:
    """
    전체 주식 데이터 일괄 수집

    수집 전략 (모두 일괄 API):
        1) 시세(OHLCV) — 전종목 1회
        2) 시가총액 — 전종목 1회
        3) 펀더멘털(PER/PBR/EPS) — 전종목 1회
        4) 수익률(1d/1w/1m/3m/1y) — 일괄 API × 5기간

    Args:
        date: 기준일 (YYYYMMDD)
        market: "KOSPI", "KOSDAQ", "ALL" (기본: ALL)
        max_stocks: 수집할 최대 종목 수 (0=전체)
    """
    # 1) 종목 목록 수집
    logger.info(f"주식 목록 수집 중... (기준일: {date}, 시장: {market})")
    markets = ["KOSPI", "KOSDAQ"] if market == "ALL" else [market]
    tickers = []
    for mkt in markets:
        tickers.extend(stock.get_market_ticker_list(date, market=mkt))

    name_map = {}
    for t in tickers:
        name_map[t] = _safe_get_ticker_name(t)
    logger.info(f"주식 {len(tickers)}종목 목록 수집 완료")

    # 2) 시세 일괄 수집
    bulk_ohlcv = collect_bulk_ohlcv(date, market)
    time.sleep(REQUEST_DELAY)

    # 3) 시가총액 일괄 수집
    bulk_cap = collect_bulk_market_cap(date, market)
    time.sleep(REQUEST_DELAY)

    # 4) 펀더멘털 일괄 수집
    bulk_fund = collect_bulk_fundamental(date, market)
    time.sleep(REQUEST_DELAY)

    # 5) 업종 분류 일괄 수집
    bulk_sector = collect_bulk_sector(date, market)
    time.sleep(REQUEST_DELAY)

    # 6) 수익률 일괄 수집
    bulk_returns = collect_bulk_returns(date, market)

    # 7) 데이터 조립
    stocks = []
    for ticker in tickers:
        ohlcv = bulk_ohlcv.get(ticker, {})
        cap = bulk_cap.get(ticker, {})
        fund = bulk_fund.get(ticker, {})

        stocks.append({
            "ticker": ticker,
            "name": name_map.get(ticker, ""),
            "date": date,
            "sector": bulk_sector.get(ticker, ""),
            "ohlcv": ohlcv,
            "market_cap": cap.get("market_cap", 0),
            "shares_outstanding": cap.get("shares_outstanding", 0),
            "fundamental": fund,
            "returns": bulk_returns.get(ticker, {}),
        })

    # max_stocks 제한 (거래대금 상위 기준)
    if max_stocks > 0:
        stocks.sort(key=lambda s: s["ohlcv"].get("trade_value", 0), reverse=True)
        stocks = stocks[:max_stocks]

    result = {
        "metadata": {
            "collection_date": date,
            "collected_at": datetime.now().isoformat(),
            "total_stocks": len(stocks),
            "market": market,
            "source": "pykrx",
        },
        "stocks": stocks,
    }

    return result


def save_result(data: dict, output_dir: Path = OUTPUT_DIR) -> Path:
    """수집 결과를 JSON 파일로 저장"""
    output_dir.mkdir(parents=True, exist_ok=True)

    date = data["metadata"]["collection_date"]
    filename = f"stock_data_{date}.json"
    filepath = output_dir / filename

    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

    logger.info(f"수집 결과 저장: {filepath}")
    return filepath


def validate_result(data: dict) -> list[str]:
    """수집 결과 정합성 검증"""
    issues = []

    total = data["metadata"]["total_stocks"]
    actual = len(data["stocks"])

    if actual != total:
        issues.append(f"메타데이터 불일치: 예상 {total}개, 실제 {actual}개")

    # 시세 없는 종목 체크
    no_ohlcv = [s["name"] for s in data["stocks"] if not s.get("ohlcv")]
    if no_ohlcv:
        issues.append(f"시세 없는 종목 {len(no_ohlcv)}개: {', '.join(no_ohlcv[:5])}")

    # 종가가 0인 종목
    zero_close = [s["name"] for s in data["stocks"]
                  if s.get("ohlcv", {}).get("close", 0) == 0]
    if zero_close:
        issues.append(f"종가 0원 종목 {len(zero_close)}개: {', '.join(zero_close[:5])}")

    return issues


def main():
    parser = argparse.ArgumentParser(description="주식 데이터 수집기")
    parser.add_argument("--date", type=str, help="수집 기준일 (YYYYMMDD)")
    parser.add_argument("--market", type=str, default="ALL",
                        choices=["KOSPI", "KOSDAQ", "ALL"],
                        help="수집 대상 시장 (기본: ALL)")
    parser.add_argument("--max", type=int, default=0,
                        help="최대 수집 종목 수 (0=전체)")
    parser.add_argument("--test", action="store_true",
                        help="테스트 모드 (10개만)")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
    )

    # .env 로드
    try:
        from dotenv import load_dotenv
        load_dotenv(DATA_DIR.parent.parent / ".env")
    except ImportError:
        pass

    # KRX 로그인 (ETF collector와 동일한 세션 사용)
    from src.data.collector import ensure_krx_login
    ensure_krx_login()

    # 기준일 결정
    if args.date:
        date = args.date
    else:
        date = find_latest_business_day()

    logger.info(f"수집 기준일: {date}")

    # 수집
    max_stocks = 10 if args.test else args.max
    data = collect_all(date, market=args.market, max_stocks=max_stocks)

    # 검증
    issues = validate_result(data)
    if issues:
        logger.warning("정합성 이슈 발견:")
        for issue in issues:
            logger.warning(f"  - {issue}")

    # JSON 저장
    filepath = save_result(data)

    # SQLite 저장
    try:
        from src.data.database import init_db, upsert_stock_data, prune_old_data
        conn = init_db()
        upsert_stock_data(conn, data)
        prune_old_data(conn)
        conn.close()
        logger.info("SQLite 저장 완료")
    except Exception as e:
        logger.warning(f"SQLite 저장 실패 (JSON은 정상): {e}")

    logger.info(f"완료! {data['metadata']['total_stocks']}개 주식, 저장: {filepath}")


if __name__ == "__main__":
    main()

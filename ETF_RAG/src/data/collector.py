"""
ETF 데이터 수집기 — pykrx 기반 일배치 수집

Phase 1-1: 국내 ETF 전종목 목록 + 시세 + NAV + 괴리율 + 추적오차 + 보유종목 수집
데이터 소스: KRX (한국거래소) via pykrx

수집 전략:
    - 시세/NAV/등락률: 일괄 API (get_etf_ohlcv_by_ticker 등) → 전종목 1초
    - 보유종목: 개별 API (get_etf_portfolio_deposit_file) → 거래대금 상위 N개만

사용법:
    python -m src.data.collector              # 최근 영업일 기준 수집
    python -m src.data.collector --date 20260403  # 특정일 수집
    python -m src.data.collector --test       # 테스트 (10개 보유종목)
    python -m src.data.collector --holdings 100  # 보유종목 수집 대상 ETF 수

환경변수 (.env):
    KRX_ID: KRX Data Marketplace 로그인 ID
    KRX_PW: KRX Data Marketplace 로그인 비밀번호
"""

import json
import os
import time
import logging
import argparse
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional

import requests as req
from pykrx import stock
from pykrx.website.comm import webio

logger = logging.getLogger(__name__)

# 프로젝트 경로
DATA_DIR = Path(__file__).parent
OUTPUT_DIR = DATA_DIR / "collected"

# 수집 설정
REQUEST_DELAY = 1.5  # KRX 요청 간 딜레이 (초) — 너무 빠르면 차단됨
HOLDINGS_TOP_N = 20  # ETF당 보유종목 최대 수

# ── KRX 로그인 (2026-02 정책 변경: 로그인 필수) ──────────────────

_session = req.Session()


def _patch_pykrx_session():
    """pykrx 내부 HTTP 요청을 공유 세션으로 교체 (쿠키 유지)"""
    def _post_read(self, **params):
        return _session.post(self.url, headers=self.headers, data=params, timeout=30)

    def _get_read(self, **params):
        return _session.get(self.url, headers=self.headers, params=params, timeout=30)

    webio.Post.read = _post_read
    webio.Get.read = _get_read


def login_krx(login_id: str, login_pw: str) -> bool:
    """
    KRX data.krx.co.kr 로그인 후 세션 쿠키를 갱신합니다.
    Ref: https://github.com/sharebook-kr/pykrx/issues/276
    """
    _LOGIN_PAGE = "https://data.krx.co.kr/contents/MDC/COMS/client/MDCCOMS001.cmd"
    _LOGIN_JSP = "https://data.krx.co.kr/contents/MDC/COMS/client/view/login.jsp?site=mdc"
    _LOGIN_URL = "https://data.krx.co.kr/contents/MDC/COMS/client/MDCCOMS001D1.cmd"
    _UA = (
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
        "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/131.0.0.0 Safari/537.36"
    )

    # 초기 세션 발급
    _session.get(_LOGIN_PAGE, headers={"User-Agent": _UA}, timeout=15)
    _session.get(_LOGIN_JSP, headers={"User-Agent": _UA, "Referer": _LOGIN_PAGE}, timeout=15)

    payload = {
        "mbrNm": "", "telNo": "", "di": "", "certType": "",
        "mbrId": login_id, "pw": login_pw,
    }
    headers = {
        "User-Agent": _UA,
        "Referer": _LOGIN_PAGE,
        "X-Requested-With": "XMLHttpRequest",
    }

    resp = _session.post(_LOGIN_URL, data=payload, headers=headers, timeout=15)
    data = resp.json()
    error_code = data.get("_error_code", "")

    # CD011: 중복 로그인 → 기존 세션 끊고 재로그인
    if error_code == "CD011":
        payload["skipDup"] = "Y"
        resp = _session.post(_LOGIN_URL, data=payload, headers=headers, timeout=15)
        data = resp.json()
        error_code = data.get("_error_code", "")

    return error_code == "CD001"


def ensure_krx_login():
    """환경변수에서 KRX 계정을 읽어 로그인. 실패 시 RuntimeError."""
    krx_id = os.environ.get("KRX_ID", "")
    krx_pw = os.environ.get("KRX_PW", "")

    if not krx_id or not krx_pw:
        raise RuntimeError(
            "KRX 로그인 정보가 없습니다. .env에 KRX_ID, KRX_PW를 설정하세요.\n"
            "회원가입: https://data.krx.co.kr (무료)"
        )

    _patch_pykrx_session()

    if login_krx(krx_id, krx_pw):
        logger.info("KRX 로그인 성공")
    else:
        raise RuntimeError("KRX 로그인 실패. ID/PW를 확인하세요.")


# ── 유틸리티 ──────────────────────────────────────────────────

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

    for _ in range(10):  # 최대 10일 전까지 탐색
        # 주말은 API 호출 없이 즉시 스킵
        if _is_weekend(dt):
            dt -= timedelta(days=1)
            continue

        date_str = dt.strftime("%Y%m%d")
        try:
            # 실제 시세 데이터로 영업일 판별 (공휴일도 걸러짐)
            df = stock.get_etf_ohlcv_by_ticker(date_str)
            if not df.empty and (df["종가"] > 0).any():
                return date_str
        except Exception:
            pass
        dt -= timedelta(days=1)

    raise RuntimeError("최근 10일 내 영업일을 찾을 수 없습니다. KRX 서버 상태를 확인하세요.")


# ── 일괄 수집 (전종목 한 번에) ────────────────────────────────

def collect_bulk_ohlcv(date: str) -> dict:
    """전종목 시세 + NAV 일괄 수집

    Returns: {ticker: {open, high, low, close, volume, trade_value, nav, base_index}}
    """
    logger.info(f"전종목 시세/NAV 일괄 수집 중... (기준일: {date})")
    df = stock.get_etf_ohlcv_by_ticker(date)

    result = {}
    for ticker, row in df.iterrows():
        result[ticker] = {
            "open": int(row.get("시가", 0)),
            "high": int(row.get("고가", 0)),
            "low": int(row.get("저가", 0)),
            "close": int(row.get("종가", 0)),
            "volume": int(row.get("거래량", 0)),
            "trade_value": int(row.get("거래대금", 0)),
            "nav": round(float(row.get("NAV", 0)), 2),
            "base_index": round(float(row.get("기초지수", 0)), 2),
        }

    logger.info(f"시세/NAV {len(result)}종목 수집 완료")
    return result


def collect_bulk_change(date: str) -> dict:
    """전종목 등락률 일괄 수집

    Returns: {ticker: {change, change_pct}}
    """
    logger.info(f"전종목 등락률 일괄 수집 중...")
    try:
        df = stock.get_etf_price_change_by_ticker(date, date)
        result = {}
        for ticker, row in df.iterrows():
            result[ticker] = {
                "change": int(row.get("변동폭", 0)),
                "change_pct": round(float(row.get("등락률", 0)), 2),
            }
        logger.info(f"등락률 {len(result)}종목 수집 완료")
        return result
    except Exception as e:
        logger.warning(f"등락률 일괄 수집 실패: {e}")
        return {}


# ── 수익률 수집 (기간별 일괄) ──────────────────────────────────

# 수익률 기간 정의: (라벨, 일수)
RETURN_PERIODS = [
    ("1d", 1),
    ("1w", 7),
    ("1m", 30),
    ("3m", 90),
    ("1y", 365),
]


def collect_bulk_returns(date: str) -> dict:
    """전종목 기간별 수익률 일괄 수집

    각 기간(1일/1주/1개월/3개월/1년)에 대해 get_etf_price_change_by_ticker를 호출.
    Returns: {ticker: {"1d": float, "1w": float, "1m": float, "3m": float, "1y": float}}
    """
    dt_base = datetime.strptime(date, "%Y%m%d")
    result = {}

    for label, days in RETURN_PERIODS:
        dt_from = dt_base - timedelta(days=days)
        fromdate = dt_from.strftime("%Y%m%d")

        logger.info(f"수익률 수집 중: {label} ({fromdate} ~ {date})")
        try:
            df = stock.get_etf_price_change_by_ticker(fromdate, date)
            for ticker, row in df.iterrows():
                if ticker not in result:
                    result[ticker] = {}
                result[ticker][label] = round(float(row.get("등락률", 0)), 2)
        except Exception as e:
            logger.warning(f"수익률 수집 실패 ({label}): {e}")

        time.sleep(REQUEST_DELAY)

    logger.info(f"수익률 수집 완료: {len(result)}종목, {len(RETURN_PERIODS)}개 기간")
    return result


# ── 개별 수집 (ETF별) ─────────────────────────────────────────

def collect_etf_deviation(ticker: str, date: str) -> dict:
    """단일 ETF 괴리율 + 추적오차율 수집"""
    result = {}
    try:
        dev_df = stock.get_etf_price_deviation(date, date, ticker)
        if not dev_df.empty:
            row = dev_df.iloc[0]
            result["deviation"] = round(float(row.get("괴리율", 0)), 2)
    except BaseException as e:
        logger.warning(f"{ticker} 괴리율 수집 실패: {e}")

    try:
        te_df = stock.get_etf_tracking_error(date, date, ticker)
        if not te_df.empty:
            row = te_df.iloc[0]
            result["tracking_error"] = round(float(row.get("추적오차율", 0)), 2)
    except BaseException as e:
        logger.warning(f"{ticker} 추적오차율 수집 실패: {e}")

    return result


def _suppress_pykrx_logging_errors():
    """pykrx 내부 logging.info(args, kwargs) 포맷 에러 억제 필터 설치.

    pykrx는 에러 발생 시 logging.info(args, kwargs)를 호출하는데,
    args가 tuple이고 kwargs가 dict라서 Python logging의 % 포맷팅이 실패.
    이 '--- Logging error ---'가 stderr에 직접 출력되어 로그를 오염시킴.
    한 번만 호출하면 프로세스 수명 동안 유효.
    """
    import logging as _logging

    class _PykrxFilter(_logging.Filter):
        def filter(self, record):
            # pykrx util.py wrapper가 logging.info(args, kwargs) 호출 시
            # record.args가 dict인 경우 → pykrx 포맷 에러 → 무시
            if isinstance(record.args, dict):
                return False
            return True

    _logging.getLogger().addFilter(_PykrxFilter())


# 모듈 로드 시 필터 설치
_suppress_pykrx_logging_errors()


def _coerce_name(name) -> str:
    """pykrx 반환값을 string으로 강제 변환.

    pykrx 내부 DataFrame에 ticker가 중복되면 `.loc[ticker, '종목명']`이
    Series를 반환 → SQLite 바인딩에서 'type Series is not supported' 발생.
    Series면 첫 값만 추출.
    """
    if name is None:
        return ""
    if hasattr(name, "iloc"):
        try:
            name = name.iloc[0]
        except Exception:
            return ""
    return str(name) if name else ""


def _safe_get_ticker_name(ticker: str) -> str:
    """pykrx get_market_ticker_name의 안전한 래퍼.

    pykrx 내부에서 존재하지 않는 종목 조회 시 에러 발생 가능 (2026-04-13 장애).
    Series 반환 케이스도 방어 (티커 중복 시 발생).
    BaseException까지 잡아서 프로세스 크래시 방지.
    """
    try:
        return _coerce_name(stock.get_market_ticker_name(ticker))
    except BaseException:
        return ""


def _safe_get_etf_name(ticker: str) -> str:
    """pykrx get_etf_ticker_name의 안전한 래퍼 (Series 방어)."""
    try:
        return _coerce_name(stock.get_etf_ticker_name(ticker))
    except BaseException:
        return ""


def collect_etf_holdings(ticker: str, date: str) -> list[dict]:
    """단일 ETF 보유종목(PDF) 수집

    Note: stock.get_etf_portfolio_deposit_file(ticker, date) — ticker가 첫 번째 인자
    반환 컬럼: 계약수, 금액, 비중 (index=티커)
    """
    try:
        df = stock.get_etf_portfolio_deposit_file(ticker, date)
        if df.empty:
            return []

        # 금액 기준 내림차순 정렬 (상위 보유종목 우선)
        df = df.sort_values("금액", ascending=False)

        holdings = []
        for stock_ticker, row in df.head(HOLDINGS_TOP_N).iterrows():
            stock_name = _safe_get_ticker_name(str(stock_ticker))

            holdings.append({
                "stock_ticker": str(stock_ticker),
                "stock_name": stock_name,
                "shares": float(row["계약수"]),
                "amount": int(row["금액"]),
                "weight": round(float(row["비중"]), 2),
            })

        return holdings
    except Exception as e:
        logger.warning(f"{ticker} 보유종목 수집 실패: {e}")
        return []


# ── 메인 수집 로직 ────────────────────────────────────────────

def collect_all(date: str, max_etfs: int = 0, holdings_count: int = 100) -> dict:
    """
    전체 ETF 데이터 일괄 수집

    수집 전략:
        1) 시세/NAV/등락률 — 일괄 API로 전종목 한 번에 (2초)
        2) 수익률(1d/1w/1m/3m/1y) — 일괄 API × 5기간 (~8초)
        3) 괴리율/추적오차 — 개별 API, 전종목
        4) 보유종목 — 개별 API, 거래대금 상위 holdings_count개만

    Args:
        date: 기준일 (YYYYMMDD)
        max_etfs: 수집할 최대 ETF 수 (0=전체)
        holdings_count: 보유종목 수집 대상 ETF 수 (거래대금 상위)
    """
    # 1) ETF 목록 수집
    logger.info(f"ETF 목록 수집 중... (기준일: {date})")
    tickers = stock.get_etf_ticker_list(date)
    name_map = {}
    for t in tickers:
        name_map[t] = _safe_get_etf_name(t)
    logger.info(f"ETF {len(tickers)}종목 목록 수집 완료")

    # 2) 시세/NAV 일괄 수집
    bulk_ohlcv = collect_bulk_ohlcv(date)
    time.sleep(REQUEST_DELAY)

    # 3) 등락률 일괄 수집
    bulk_change = collect_bulk_change(date)
    time.sleep(REQUEST_DELAY)

    # 4) 수익률 일괄 수집 (1d/1w/1m/3m/1y)
    bulk_returns = collect_bulk_returns(date)

    # 5) ETF 데이터 조립
    etfs = []
    for ticker in tickers:
        ohlcv = bulk_ohlcv.get(ticker, {})
        change = bulk_change.get(ticker, {})

        # ohlcv에 등락 정보 병합
        if change:
            ohlcv["change"] = change.get("change", 0)
            ohlcv["change_pct"] = change.get("change_pct", 0.0)

        etfs.append({
            "ticker": ticker,
            "name": name_map.get(ticker, ""),
            "date": date,
            "ohlcv": ohlcv,
            "returns": bulk_returns.get(ticker, {}),
            "deviation": None,
            "tracking_error": None,
            "holdings": [],
        })

    # max_etfs 제한
    if max_etfs > 0:
        etfs = etfs[:max_etfs]

    # 6) 괴리율/추적오차 개별 수집 (전종목)
    logger.info(f"괴리율/추적오차율 수집 중... ({len(etfs)}종목)")
    for i, etf in enumerate(etfs):
        if (i + 1) % 100 == 0:
            logger.info(f"  괴리율/추적오차 진행: {i+1}/{len(etfs)}")

        dev = collect_etf_deviation(etf["ticker"], date)
        etf["deviation"] = dev.get("deviation")
        etf["tracking_error"] = dev.get("tracking_error")
        time.sleep(REQUEST_DELAY)

    # 7) 보유종목 수집 (거래대금 상위 N개만)
    # 거래대금 기준 정렬
    etfs_by_trade_value = sorted(
        etfs,
        key=lambda e: e.get("ohlcv", {}).get("trade_value", 0),
        reverse=True,
    )
    holdings_targets = set()
    for e in etfs_by_trade_value[:holdings_count]:
        holdings_targets.add(e["ticker"])

    logger.info(f"보유종목 수집 중... (거래대금 상위 {len(holdings_targets)}종목)")
    holdings_done = 0
    for etf in etfs:
        if etf["ticker"] not in holdings_targets:
            continue

        holdings_done += 1
        if holdings_done % 20 == 0:
            logger.info(f"  보유종목 진행: {holdings_done}/{len(holdings_targets)}")

        etf["holdings"] = collect_etf_holdings(etf["ticker"], date)
        time.sleep(REQUEST_DELAY)

    result = {
        "metadata": {
            "collection_date": date,
            "collected_at": datetime.now().isoformat(),
            "total_etfs": len(etfs),
            "holdings_collected": len(holdings_targets),
        },
        "etfs": etfs,
    }

    return result


def save_result(data: dict, output_dir: Path = OUTPUT_DIR) -> Path:
    """수집 결과를 JSON 파일로 저장"""
    output_dir.mkdir(parents=True, exist_ok=True)

    date = data["metadata"]["collection_date"]
    filename = f"etf_data_{date}.json"
    filepath = output_dir / filename

    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

    logger.info(f"수집 결과 저장: {filepath}")
    return filepath


def validate_result(data: dict) -> list[str]:
    """수집 결과 정합성 검증"""
    issues = []

    total = data["metadata"]["total_etfs"]
    actual = len(data["etfs"])

    if actual != total:
        issues.append(f"메타데이터 불일치: 예상 {total}개, 실제 {actual}개")

    # 시세 없는 ETF 체크
    no_ohlcv = [e["name"] for e in data["etfs"] if not e.get("ohlcv")]
    if no_ohlcv:
        issues.append(f"시세 없는 ETF {len(no_ohlcv)}개: {', '.join(no_ohlcv[:5])}")

    # 종가가 0인 ETF
    zero_close = [e["name"] for e in data["etfs"]
                  if e.get("ohlcv", {}).get("close", 0) == 0]
    if zero_close:
        issues.append(f"종가 0원 ETF {len(zero_close)}개: {', '.join(zero_close[:5])}")

    # 보유종목 수집 현황
    has_holdings = sum(1 for e in data["etfs"] if e.get("holdings"))
    logger.info(f"보유종목 수집 완료: {has_holdings}/{total}종목")

    return issues


def main():
    parser = argparse.ArgumentParser(description="ETF 데이터 수집기")
    parser.add_argument("--date", type=str, help="수집 기준일 (YYYYMMDD)")
    parser.add_argument("--max", type=int, default=0, help="최대 수집 ETF 수 (0=전체)")
    parser.add_argument("--holdings", type=int, default=100,
                        help="보유종목 수집 대상 ETF 수 (거래대금 상위, 기본 100)")
    parser.add_argument("--test", action="store_true",
                        help="테스트 모드 (10개만, 보유종목 5개)")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
    )

    # .env 로드 (있으면)
    try:
        from dotenv import load_dotenv
        load_dotenv(DATA_DIR.parent.parent / ".env")
    except ImportError:
        pass

    # KRX 로그인
    ensure_krx_login()

    # 기준일 결정
    if args.date:
        date = args.date
    else:
        date = find_latest_business_day()

    logger.info(f"수집 기준일: {date}")

    # 수집
    if args.test:
        max_etfs = 10
        holdings_count = 5
    else:
        max_etfs = args.max
        holdings_count = args.holdings

    data = collect_all(date, max_etfs=max_etfs, holdings_count=holdings_count)

    # 검증
    issues = validate_result(data)
    if issues:
        logger.warning("정합성 이슈 발견:")
        for issue in issues:
            logger.warning(f"  - {issue}")

    # JSON 저장 (하위 호환)
    filepath = save_result(data)

    # SQLite 저장
    try:
        from src.data.database import init_db, upsert_daily_data, prune_old_data
        conn = init_db()
        upsert_daily_data(conn, data)
        prune_old_data(conn)
        conn.close()
        logger.info("SQLite 저장 완료")
    except Exception as e:
        logger.warning(f"SQLite 저장 실패 (JSON은 정상): {e}")

    logger.info(f"완료! {data['metadata']['total_etfs']}개 ETF, 저장: {filepath}")


if __name__ == "__main__":
    main()

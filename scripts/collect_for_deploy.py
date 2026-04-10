"""
GitHub Actions용 경량 수집 스크립트 — deploy/ JSON 갱신 전용

pykrx로 ETF + 주식 데이터를 수집하고 deploy/ JSON을 업데이트합니다.
SQLite DB 없이 JSON만 생성하므로 GitHub Actions 환경에서 가볍게 실행 가능.

사용법:
    python scripts/collect_for_deploy.py              # 최근 영업일 자동 감지
    python scripts/collect_for_deploy.py --date 20260410  # 특정일 수집
"""

import json
import os
import sys
import time
import logging
import argparse
from datetime import datetime, timedelta
from pathlib import Path

import requests as req
from pykrx import stock
from pykrx.website.comm import webio

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)

# 프로젝트 경로
PROJECT_DIR = Path(__file__).parent.parent
DEPLOY_DIR = PROJECT_DIR / "ETF_RAG" / "src" / "data" / "deploy"

REQUEST_DELAY = 1.5


# ── KRX 로그인 ──────────────────────────────────────────────

_session = req.Session()


def _patch_pykrx_session():
    """pykrx 내부 HTTP 요청을 공유 세션으로 교체 (쿠키 유지)"""
    def _post_read(self, **params):
        return _session.post(self.url, headers=self.headers, data=params)

    def _get_read(self, **params):
        return _session.get(self.url, headers=self.headers, params=params)

    webio.Post.read = _post_read
    webio.Get.read = _get_read


def login_krx() -> bool:
    """KRX 로그인"""
    krx_id = os.environ.get("KRX_ID", "")
    krx_pw = os.environ.get("KRX_PW", "")

    if not krx_id or not krx_pw:
        logger.error("KRX_ID, KRX_PW 환경변수가 필요합니다.")
        return False

    _patch_pykrx_session()

    _LOGIN_PAGE = "https://data.krx.co.kr/contents/MDC/COMS/client/MDCCOMS001.cmd"
    _LOGIN_JSP = "https://data.krx.co.kr/contents/MDC/COMS/client/view/login.jsp?site=mdc"
    _LOGIN_URL = "https://data.krx.co.kr/contents/MDC/COMS/client/MDCCOMS001D1.cmd"
    _UA = (
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
        "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/131.0.0.0 Safari/537.36"
    )

    _session.get(_LOGIN_PAGE, headers={"User-Agent": _UA}, timeout=15)
    _session.get(_LOGIN_JSP, headers={"User-Agent": _UA, "Referer": _LOGIN_PAGE}, timeout=15)

    payload = {
        "mbrNm": "", "telNo": "", "di": "", "certType": "",
        "mbrId": krx_id, "pw": krx_pw,
    }
    headers = {
        "User-Agent": _UA,
        "Referer": _LOGIN_PAGE,
        "X-Requested-With": "XMLHttpRequest",
    }

    resp = _session.post(_LOGIN_URL, data=payload, headers=headers, timeout=15)
    data = resp.json()
    error_code = data.get("_error_code", "")

    if error_code == "CD011":
        payload["skipDup"] = "Y"
        resp = _session.post(_LOGIN_URL, data=payload, headers=headers, timeout=15)
        data = resp.json()
        error_code = data.get("_error_code", "")

    return error_code == "CD001"


# ── 영업일 탐색 ─────────────────────────────────────────────

def find_latest_business_day(from_date=None) -> str:
    """최근 영업일 찾기"""
    dt = datetime.strptime(from_date, "%Y%m%d") if from_date else datetime.now()

    for _ in range(10):
        date_str = dt.strftime("%Y%m%d")
        try:
            tickers = stock.get_etf_ticker_list(date_str)
            if len(tickers) > 0:
                return date_str
        except Exception:
            pass
        dt -= timedelta(days=1)

    raise RuntimeError("최근 10일 내 영업일을 찾을 수 없습니다.")


# ── ETF 수집 (deploy용 경량 버전) ────────────────────────────

def collect_etf_for_deploy(date: str) -> dict:
    """ETF 전종목 시세 + NAV + 등락률 수집 (deploy용, 보유종목/괴리율 생략)"""
    logger.info(f"ETF 수집 시작 (기준일: {date})")

    # 종목 목록
    tickers = stock.get_etf_ticker_list(date)
    name_map = {t: stock.get_etf_ticker_name(t) for t in tickers}
    logger.info(f"ETF {len(tickers)}종목")

    # 시세/NAV 일괄
    df_ohlcv = stock.get_etf_ohlcv_by_ticker(date)
    time.sleep(REQUEST_DELAY)

    # 등락률 일괄
    try:
        df_change = stock.get_etf_price_change_by_ticker(date, date)
    except Exception as e:
        logger.warning(f"등락률 수집 실패: {e}")
        df_change = None
    time.sleep(REQUEST_DELAY)

    # 수익률 (1d/1w/1m/3m/1y)
    dt_base = datetime.strptime(date, "%Y%m%d")
    returns_map = {}
    for label, days in [("1d", 1), ("1w", 7), ("1m", 30), ("3m", 90), ("1y", 365)]:
        fromdate = (dt_base - timedelta(days=days)).strftime("%Y%m%d")
        try:
            df_ret = stock.get_etf_price_change_by_ticker(fromdate, date)
            for ticker, row in df_ret.iterrows():
                if ticker not in returns_map:
                    returns_map[ticker] = {}
                returns_map[ticker][label] = round(float(row.get("등락률", 0)), 2)
        except Exception as e:
            logger.warning(f"수익률 수집 실패 ({label}): {e}")
        time.sleep(REQUEST_DELAY)

    # 조립
    etfs = []
    for ticker in tickers:
        ohlcv_data = {}
        if ticker in df_ohlcv.index:
            row = df_ohlcv.loc[ticker]
            ohlcv_data = {
                "open": int(row.get("시가", 0)),
                "high": int(row.get("고가", 0)),
                "low": int(row.get("저가", 0)),
                "close": int(row.get("종가", 0)),
                "volume": int(row.get("거래량", 0)),
                "trade_value": int(row.get("거래대금", 0)),
                "nav": round(float(row.get("NAV", 0)), 2),
                "base_index": round(float(row.get("기초지수", 0)), 2),
            }

        if df_change is not None and ticker in df_change.index:
            change_row = df_change.loc[ticker]
            ohlcv_data["change"] = int(change_row.get("변동폭", 0))
            ohlcv_data["change_pct"] = round(float(change_row.get("등락률", 0)), 2)

        etfs.append({
            "ticker": ticker,
            "name": name_map.get(ticker, ""),
            "date": date,
            "ohlcv": ohlcv_data,
            "returns": returns_map.get(ticker, {}),
            "deviation": None,
            "tracking_error": None,
            "holdings": [],
        })

    result = {
        "metadata": {
            "collection_date": date,
            "collected_at": datetime.now().isoformat(),
            "total_etfs": len(etfs),
            "holdings_collected": 0,
        },
        "etfs": etfs,
    }

    logger.info(f"ETF {len(etfs)}종목 수집 완료")
    return result


# ── 주식 수집 (deploy용 경량 버전) ────────────────────────────

def collect_stock_for_deploy(date: str) -> dict:
    """주식 전종목 시세 + 시가총액 + 펀더멘털 수집 (deploy용)"""
    logger.info(f"주식 수집 시작 (기준일: {date})")

    # 종목 목록 (KOSPI + KOSDAQ)
    tickers = []
    for mkt in ["KOSPI", "KOSDAQ"]:
        tickers.extend(stock.get_market_ticker_list(date, market=mkt))

    name_map = {}
    for t in tickers:
        try:
            name_map[t] = stock.get_market_ticker_name(t)
        except Exception:
            name_map[t] = ""
    logger.info(f"주식 {len(tickers)}종목")

    # 시세 일괄
    df_ohlcv = stock.get_market_ohlcv_by_ticker(date, market="ALL")
    time.sleep(REQUEST_DELAY)

    # 시가총액 일괄
    df_cap = stock.get_market_cap_by_ticker(date, market="ALL")
    time.sleep(REQUEST_DELAY)

    # 펀더멘털 일괄 (KOSPI + KOSDAQ 개별)
    fund_map = {}
    for mkt in ["KOSPI", "KOSDAQ"]:
        try:
            df_fund = stock.get_market_fundamental_by_ticker(date, market=mkt)
            for ticker, row in df_fund.iterrows():
                fund_map[ticker] = {
                    "bps": round(float(row.get("BPS", 0)), 2),
                    "per": round(float(row.get("PER", 0)), 2),
                    "pbr": round(float(row.get("PBR", 0)), 2),
                    "eps": round(float(row.get("EPS", 0)), 2),
                    "div": round(float(row.get("DIV", 0)), 2),
                    "dps": round(float(row.get("DPS", 0)), 2),
                }
        except Exception as e:
            logger.warning(f"펀더멘털 수집 실패 ({mkt}): {e}")
    time.sleep(REQUEST_DELAY)

    # 업종 분류
    sector_map = {}
    for mkt in ["KOSPI", "KOSDAQ"]:
        try:
            df_sector = stock.get_market_sector_classifications(date, market=mkt)
            for ticker, row in df_sector.iterrows():
                sector_map[ticker] = row.get("업종명", "")
        except Exception as e:
            logger.warning(f"업종 분류 수집 실패 ({mkt}): {e}")
    time.sleep(REQUEST_DELAY)

    # 수익률 (1d/1w/1m/3m/1y)
    dt_base = datetime.strptime(date, "%Y%m%d")
    returns_map = {}
    for label, days in [("1d", 1), ("1w", 7), ("1m", 30), ("3m", 90), ("1y", 365)]:
        fromdate = (dt_base - timedelta(days=days)).strftime("%Y%m%d")
        for mkt in ["KOSPI", "KOSDAQ"]:
            try:
                df_ret = stock.get_market_price_change_by_ticker(fromdate, date, market=mkt)
                for ticker, row in df_ret.iterrows():
                    if ticker not in returns_map:
                        returns_map[ticker] = {}
                    returns_map[ticker][label] = round(float(row.get("등락률", 0)), 2)
            except Exception as e:
                logger.warning(f"수익률 수집 실패 ({label}, {mkt}): {e}")
        time.sleep(REQUEST_DELAY)

    # 조립
    stocks = []
    for ticker in tickers:
        ohlcv_data = {}
        if ticker in df_ohlcv.index:
            row = df_ohlcv.loc[ticker]
            ohlcv_data = {
                "open": int(row.get("시가", 0)),
                "high": int(row.get("고가", 0)),
                "low": int(row.get("저가", 0)),
                "close": int(row.get("종가", 0)),
                "volume": int(row.get("거래량", 0)),
                "trade_value": int(row.get("거래대금", 0)),
                "change_pct": round(float(row.get("등락률", 0)), 2),
            }

        cap_data = {}
        if ticker in df_cap.index:
            cap_row = df_cap.loc[ticker]
            cap_data = {
                "market_cap": int(cap_row.get("시가총액", 0)),
                "shares_outstanding": int(cap_row.get("상장주식수", 0)),
            }

        stocks.append({
            "ticker": ticker,
            "name": name_map.get(ticker, ""),
            "date": date,
            "sector": sector_map.get(ticker, ""),
            "ohlcv": ohlcv_data,
            "market_cap": cap_data.get("market_cap", 0),
            "shares_outstanding": cap_data.get("shares_outstanding", 0),
            "fundamental": fund_map.get(ticker, {}),
            "returns": returns_map.get(ticker, {}),
        })

    result = {
        "metadata": {
            "collection_date": date,
            "collected_at": datetime.now().isoformat(),
            "total_stocks": len(stocks),
            "market": "ALL",
            "source": "pykrx",
        },
        "stocks": stocks,
    }

    logger.info(f"주식 {len(stocks)}종목 수집 완료")
    return result


# ── 메인 ─────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="GitHub Actions용 deploy 데이터 수집")
    parser.add_argument("--date", type=str, help="수집 기준일 (YYYYMMDD)")
    args = parser.parse_args()

    # KRX 로그인
    if not login_krx():
        logger.error("KRX 로그인 실패")
        sys.exit(1)
    logger.info("KRX 로그인 성공")

    # 기준일
    date = args.date if args.date else find_latest_business_day()
    logger.info(f"수집 기준일: {date}")

    # ETF 수집 → deploy/etf_data.json
    etf_data = collect_etf_for_deploy(date)
    etf_path = DEPLOY_DIR / "etf_data.json"
    with open(etf_path, "w", encoding="utf-8") as f:
        json.dump(etf_data, f, ensure_ascii=False, indent=2)
    logger.info(f"ETF deploy 저장: {etf_path} ({len(etf_data['etfs'])}종목)")

    # 주식 수집 → deploy/stock_data.json
    stock_data = collect_stock_for_deploy(date)
    stock_path = DEPLOY_DIR / "stock_data.json"
    with open(stock_path, "w", encoding="utf-8") as f:
        json.dump(stock_data, f, ensure_ascii=False, indent=2)
    logger.info(f"주식 deploy 저장: {stock_path} ({len(stock_data['stocks'])}종목)")

    # 검증
    etf_count = len(etf_data["etfs"])
    stock_count = len(stock_data["stocks"])
    zero_close_etf = sum(1 for e in etf_data["etfs"] if e.get("ohlcv", {}).get("close", 0) == 0)
    zero_close_stock = sum(1 for s in stock_data["stocks"] if s.get("ohlcv", {}).get("close", 0) == 0)

    logger.info(f"=== 수집 완료 ===")
    logger.info(f"  기준일: {date}")
    logger.info(f"  ETF: {etf_count}종목 (종가 0: {zero_close_etf})")
    logger.info(f"  주식: {stock_count}종목 (종가 0: {zero_close_stock})")

    if etf_count < 500 or stock_count < 1000:
        logger.error(f"수집 결과가 너무 적습니다! ETF={etf_count}, 주식={stock_count}")
        sys.exit(1)

    logger.info("정상 완료!")


if __name__ == "__main__":
    main()

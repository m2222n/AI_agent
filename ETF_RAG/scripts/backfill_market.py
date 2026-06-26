"""instruments.market(KOSPI/KOSDAQ) 일회성 백필.

기존 종목은 market이 비어 있어 yfinance .KS/.KQ 변환이 추측에 의존 →
pykrx로 시장별 종목 목록을 받아 instruments.market을 채운다. (ETF는 KOSPI 고정)

사용:
    python -m scripts.backfill_market
"""

import logging
import sys

from dotenv import load_dotenv

load_dotenv()  # KRX_ID/KRX_PW를 .env에서 로드 (로그인 필수)

from pykrx import stock

from src.data.database import init_db, DB_PATH

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)


def backfill() -> None:
    # KRX는 2026-02부터 로그인 필수 → 로그인 안 하면 종목 목록이 빈 응답으로
    # 온다(과거 "pykrx 외부 장애"로 오인했던 KOSDAQ 백필 보류의 진짜 원인).
    from src.data.collector import ensure_krx_login
    ensure_krx_login()

    # init_db가 _migrate를 호출 → instruments.market 컬럼 보장
    conn = init_db(DB_PATH)
    # 날짜: DB 최신 수집일(오늘이 영업일/장중이 아니면 pykrx가 빈 응답을 주므로)
    row = conn.execute("SELECT MAX(date) FROM daily_prices").fetchone()
    date = row[0] if row and row[0] else None
    if not date:
        logger.error("daily_prices가 비어 있어 기준일을 알 수 없습니다.")
        return

    # 1) 주식: 시장별 ticker 목록 (pykrx 빈 응답 잦아 재시도)
    import time
    updated = 0
    for mkt in ("KOSPI", "KOSDAQ"):
        tickers = []
        for attempt in range(4):
            try:
                tickers = stock.get_market_ticker_list(date, market=mkt)
                if tickers:
                    break
            except Exception as e:  # noqa: BLE001
                logger.warning(f"{mkt} 시도{attempt + 1} 실패: {e}")
            time.sleep(3)
        if not tickers:
            logger.warning(f"{mkt} 목록 비어있음 — 건너뜀(나중에 재실행 권장)")
            continue
        with conn:
            cur = conn.executemany(
                "UPDATE instruments SET market = ? WHERE ticker = ? AND type = 'stock'",
                [(mkt, t) for t in tickers],
            )
        logger.info(f"{mkt}: {len(tickers)}종목 market 설정")
        updated += len(tickers)

    # 2) ETF: 모두 KOSPI
    with conn:
        conn.execute(
            "UPDATE instruments SET market = 'KOSPI' "
            "WHERE type = 'etf' AND (market IS NULL OR market = '')"
        )

    # 3) 결과 요약
    rows = conn.execute(
        "SELECT market, COUNT(*) c FROM instruments GROUP BY market ORDER BY c DESC"
    ).fetchall()
    logger.info("=== 백필 결과 (market별 종목 수) ===")
    for r in rows:
        logger.info(f"  {r[0] or '(빈값)'}: {r[1]}")
    conn.close()


if __name__ == "__main__":
    try:
        backfill()
    except Exception as e:  # noqa: BLE001
        logger.error(f"백필 실패: {e}")
        sys.exit(1)

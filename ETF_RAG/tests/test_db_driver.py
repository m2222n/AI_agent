"""프로덕션 DATABASE_URL 스킴이 설치된 DBAPI 드라이버로 해석되는지 검증.

배경(2026-09-26 장애): SQLAlchemy 2.1부터 `postgresql://` 기본 드라이버가 psycopg2 →
psycopg(v3)로 바뀌었다. requirements에는 psycopg2-binary만 있어 api/db.py의 모듈 레벨
`create_engine(DATABASE_URL)`이 import 시점에 ModuleNotFoundError를 던지고, uvicorn이
앱을 로드하지 못해 프로덕션이 5일간 502였다. 로컬/테스트는 sqlite라 재현되지 않았다.

이 테스트는 실제 DB 없이 `create_engine`만 호출한다 — SQLAlchemy는 엔진 생성 시점에
DBAPI 모듈을 eager import 하므로, 드라이버가 없으면 여기서 바로 실패한다.
"""

from sqlalchemy import create_engine


def test_postgresql_url_resolves_to_installed_driver():
    """DEPLOY.md의 프로덕션 URL 형태(postgresql://)가 설치된 드라이버로 열려야 한다."""
    # api/db.py와 동일한 호출 형태(future=True). 연결은 하지 않는다.
    engine = create_engine("postgresql://u:p@localhost:5432/db", future=True)
    assert engine.dialect.name == "postgresql"
    # requirements.txt에 있는 드라이버는 psycopg2-binary뿐. 다른 값이면 requirements와
    # SQLAlchemy 기본값이 어긋난 것 → 프로덕션 부팅 실패로 이어진다.
    assert engine.dialect.driver == "psycopg2", (
        f"postgresql:// 기본 드라이버가 {engine.dialect.driver!r}로 바뀜 — "
        "requirements.txt의 psycopg2-binary와 불일치(SQLAlchemy 버전 핀 확인)"
    )

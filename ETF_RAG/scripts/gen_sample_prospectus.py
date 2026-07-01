"""샘플 ETF 투자설명서 PDF 생성 — PDF RAG end-to-end 검증/데모용.

실제 투자설명서는 저작권·배포 이슈가 있어, RAG 파이프라인 검증을 위한
**합성 샘플**을 생성한다. 구조화 데이터(시세·PER 등)에는 없고 PDF에만 있는
정보(총보수, 위험등급, 운용전략, 분배금 정책 등)를 담아 "PDF RAG가 정형
데이터로 못 답하는 질문에 답한다"는 차별점을 검증할 수 있게 한다.

파일명 규칙(pdf_loader가 메타 추출): {ticker}_{name}_{doc_type}.pdf

사용:
    python scripts/gen_sample_prospectus.py
출력: src/data/pdfs/*.pdf  (실제 투자설명서로 교체 가능)
"""

import sys
from pathlib import Path

from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import mm
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer

PDF_DIR = Path(__file__).parent.parent / "src" / "data" / "pdfs"
FONT_PATH = "/System/Library/Fonts/Supplemental/AppleGothic.ttf"

# 합성 투자설명서 데이터 — (ticker, name, doc_type, 본문 섹션들)
# 정형 데이터(시세/PER)엔 없는 항목 위주: 총보수·위험등급·운용전략·분배금·환헤지 등.
SAMPLES = [
    {
        "ticker": "069500", "name": "KODEX200", "doc_type": "투자설명서",
        "sections": [
            ("상품 개요",
             "KODEX 200은 코스피200 지수를 기초지수로 하는 패시브 ETF입니다. "
             "삼성자산운용이 운용하며, 국내 대표 대형주 200종목에 분산 투자합니다."),
            ("운용 보수 및 비용",
             "총보수는 연 0.15%이며, 운용보수 0.12% + 지정참가회사 보수 0.01% + "
             "수탁 0.01% + 사무관리 0.01%로 구성됩니다. 매매·중개 수수료는 별도이며, "
             "실부담 비용은 거래 빈도에 따라 달라질 수 있습니다."),
            ("투자 위험 등급",
             "본 ETF의 위험등급은 2등급(높은 위험)입니다. 코스피200 지수를 추종하므로 "
             "주식시장 변동에 직접 노출되며, 원금 손실이 발생할 수 있습니다. "
             "추적오차 위험, 괴리율 위험, 유동성 위험이 존재합니다."),
            ("운용 전략",
             "완전복제법을 원칙으로 코스피200 구성종목과 편입비중을 그대로 추종합니다. "
             "지수 정기변경(매년 6월·12월 선물 만기일) 시 구성종목을 조정하며, "
             "현금 비중을 최소화해 추적오차를 줄입니다."),
            ("분배금 정책",
             "분배금은 연 1회, 회계기간 종료일(매년 1월·4월·7월·10월 마지막 영업일) "
             "기준으로 지급 여부를 결정합니다. 보유 종목의 배당금과 이자수익을 재원으로 "
             "하며, 분배 가능 재원이 없으면 미지급될 수 있습니다."),
        ],
    },
    {
        "ticker": "133690", "name": "TIGER미국나스닥100", "doc_type": "투자설명서",
        "sections": [
            ("상품 개요",
             "TIGER 미국나스닥100은 나스닥100 지수를 추종하는 해외주식형 ETF입니다. "
             "미래에셋자산운용이 운용하며, 미국 나스닥 상장 대형 기술주 100종목에 투자합니다."),
            ("운용 보수 및 비용",
             "총보수는 연 0.07%로 국내 상장 나스닥100 ETF 중 낮은 수준입니다. "
             "해외 ETF 특성상 현지 거래·환전 비용이 기초자산 가격에 반영됩니다."),
            ("투자 위험 등급",
             "위험등급은 2등급(높은 위험)입니다. 기술주 집중으로 변동성이 크며, "
             "환율 변동 위험에 노출됩니다(환노출형, 환헤지 미적용). "
             "달러 강세 시 추가 수익, 약세 시 손실 요인이 됩니다."),
            ("환헤지 여부",
             "본 ETF는 환헤지를 하지 않는 환노출형(UH)입니다. 원/달러 환율 변동이 "
             "수익률에 직접 반영되므로, 환위험을 회피하려면 환헤지형(H) 상품을 고려해야 합니다."),
            ("분배금 정책",
             "분배금은 연 1회 이하로 지급되며, 해외 배당에 대한 현지 원천징수(미국 15%) "
             "후 재원으로 사용됩니다. 대부분 재투자되어 분배율은 낮은 편입니다."),
        ],
    },
]


def _register_font():
    pdfmetrics.registerFont(TTFont("KFont", FONT_PATH))


def _build(sample: dict):
    styles = getSampleStyleSheet()
    title_style = ParagraphStyle(
        "KTitle", parent=styles["Title"], fontName="KFont", fontSize=18, leading=24,
    )
    head_style = ParagraphStyle(
        "KHead", parent=styles["Heading2"], fontName="KFont", fontSize=13, leading=18,
        spaceBefore=10,
    )
    body_style = ParagraphStyle(
        "KBody", parent=styles["BodyText"], fontName="KFont", fontSize=10.5, leading=16,
    )

    PDF_DIR.mkdir(parents=True, exist_ok=True)
    fname = f"{sample['ticker']}_{sample['name']}_{sample['doc_type']}.pdf"
    path = PDF_DIR / fname
    doc = SimpleDocTemplate(str(path), pagesize=A4,
                            topMargin=20 * mm, bottomMargin=20 * mm)
    flow = [
        Paragraph(f"{sample['name']} ({sample['ticker']}) {sample['doc_type']}", title_style),
        Spacer(1, 6 * mm),
    ]
    for head, body in sample["sections"]:
        flow.append(Paragraph(head, head_style))
        flow.append(Paragraph(body, body_style))
        flow.append(Spacer(1, 3 * mm))
    doc.build(flow)
    return path


def main():
    try:
        _register_font()
    except Exception as e:  # noqa: BLE001
        print(f"한글 폰트 등록 실패({FONT_PATH}): {e}", file=sys.stderr)
        sys.exit(1)
    for s in SAMPLES:
        p = _build(s)
        print(f"생성: {p}")


if __name__ == "__main__":
    main()

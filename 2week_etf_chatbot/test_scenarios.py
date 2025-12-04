"""
ETF 챗봇 시나리오별 테스트 스크립트 (3주차)

테스트 시나리오:
1. 단순 질문 (simple) - 특정 ETF 정보 조회
2. 비교 질문 (compare) - ETF간 비교
3. 추천 질문 (recommend) - 조건에 맞는 ETF 추천
4. 위험 질문 (risk) - 위험도/주의사항 관련
5. Edge Case - 범위 외 질문, 모호한 질문
"""

import os
import sys
import json
import time
from datetime import datetime

# 프로젝트 경로 추가
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from openai import OpenAI
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document

# app.py에서 함수 import
from app import (
    load_etf_data,
    init_vector_db,
    retrieve_relevant_docs,
    classify_question_type,
    build_system_prompt
)

# -------------------------------------------------------------------
# 테스트 시나리오 정의
# -------------------------------------------------------------------
TEST_SCENARIOS = {
    "simple": [
        "KODEX 200 ETF에 대해 알려줘",
        "TIGER 미국S&P500의 수수료는 얼마야?",
        "KODEX 2차전지산업 ETF 투자전략이 뭐야?",
    ],
    "compare": [
        "KODEX 200과 TIGER 미국S&P500 비교해줘",
        "국내 주식형과 해외 주식형 ETF 차이가 뭐야?",
        "2차전지 ETF와 전기차 ETF 중에 뭐가 더 위험해?",
    ],
    "recommend": [
        "배당 수익률 높은 ETF 추천해줘",
        "안정적인 투자를 원하는데 어떤 ETF가 좋을까?",
        "미국 시장에 투자하고 싶은데 뭐가 좋아?",
        "수수료 낮은 ETF 알려줘",
    ],
    "risk": [
        "KODEX 2차전지산업의 위험도는?",
        "인버스 ETF 투자할 때 주의사항이 뭐야?",
        "ETF 투자 시 어떤 위험이 있어?",
    ],
    "edge_case": [
        "비트코인 ETF 있어?",
        "삼성전자 주가 알려줘",
        "내일 주식시장 어떻게 될까?",
        "가장 좋은 ETF 뭐야?",
    ]
}

# -------------------------------------------------------------------
# 테스트 실행 함수
# -------------------------------------------------------------------
def run_test_scenario(client, vectorstore, question: str, expected_type: str) -> dict:
    """
    단일 시나리오 테스트 실행
    """
    start_time = time.time()

    # 질문 유형 분류
    detected_type = classify_question_type(question)
    type_match = (detected_type == expected_type) or (expected_type == "edge_case")

    # 검색 실행
    search_start = time.time()
    context, sources = retrieve_relevant_docs(vectorstore, question)
    search_time = time.time() - search_start

    # 프롬프트 생성
    system_prompt = build_system_prompt(detected_type)

    # LLM 호출
    llm_start = time.time()
    try:
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": f"[검색된 ETF 문서]\n{context}\n\n[사용자 질문]\n{question}" if context else question}
            ],
            temperature=0.3,
            max_tokens=1000
        )
        answer = response.choices[0].message.content
        llm_success = True
    except Exception as e:
        answer = f"Error: {str(e)}"
        llm_success = False

    llm_time = time.time() - llm_start
    total_time = time.time() - start_time

    # 응답 품질 평가 (기본 체크)
    quality_checks = {
        "has_etf_reference": "[ETF-" in answer,
        "has_warning": "⚠️" in answer or "유의" in answer or "주의" in answer,
        "reasonable_length": 100 < len(answer) < 3000,
        "no_hallucination": "해당 정보" not in answer or "없습니다" in answer,
    }

    return {
        "question": question,
        "expected_type": expected_type,
        "detected_type": detected_type,
        "type_match": type_match,
        "sources_found": len(sources) if sources else 0,
        "answer_preview": answer[:300] + "..." if len(answer) > 300 else answer,
        "full_answer": answer,
        "performance": {
            "search_time_ms": round(search_time * 1000, 2),
            "llm_time_ms": round(llm_time * 1000, 2),
            "total_time_ms": round(total_time * 1000, 2)
        },
        "quality": quality_checks,
        "llm_success": llm_success
    }

def run_all_tests(client, vectorstore) -> dict:
    """
    전체 테스트 시나리오 실행
    """
    results = {
        "timestamp": datetime.now().isoformat(),
        "summary": {
            "total_tests": 0,
            "type_match_rate": 0,
            "avg_response_time_ms": 0,
            "success_rate": 0
        },
        "by_category": {},
        "details": []
    }

    all_times = []
    type_matches = 0
    successes = 0

    for category, questions in TEST_SCENARIOS.items():
        print(f"\n{'='*50}")
        print(f"Testing category: {category}")
        print(f"{'='*50}")

        category_results = []

        for i, question in enumerate(questions, 1):
            print(f"\n[{i}/{len(questions)}] Testing: {question[:50]}...")

            result = run_test_scenario(client, vectorstore, question, category)
            category_results.append(result)
            results["details"].append(result)

            # 통계 업데이트
            all_times.append(result["performance"]["total_time_ms"])
            if result["type_match"]:
                type_matches += 1
            if result["llm_success"]:
                successes += 1

            print(f"  - Type: {result['detected_type']} (expected: {category})")
            print(f"  - Time: {result['performance']['total_time_ms']:.0f}ms")
            print(f"  - Sources: {result['sources_found']}")

            # API 부하 방지를 위한 딜레이
            time.sleep(1)

        results["by_category"][category] = {
            "total": len(category_results),
            "avg_time_ms": round(sum(r["performance"]["total_time_ms"] for r in category_results) / len(category_results), 2)
        }

    # 최종 요약
    total_tests = len(results["details"])
    results["summary"] = {
        "total_tests": total_tests,
        "type_match_rate": round(type_matches / total_tests * 100, 1),
        "avg_response_time_ms": round(sum(all_times) / len(all_times), 2),
        "success_rate": round(successes / total_tests * 100, 1)
    }

    return results

# -------------------------------------------------------------------
# 메인 실행
# -------------------------------------------------------------------
def main():
    print("="*60)
    print("ETF 챗봇 시나리오 테스트 (3주차)")
    print("="*60)

    # API 키 확인
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        print("Error: OPENAI_API_KEY 환경변수가 설정되지 않았습니다.")
        sys.exit(1)

    # OpenAI 클라이언트 초기화
    client = OpenAI(api_key=api_key)

    # 벡터 DB 초기화
    print("\nInitializing vector database...")
    etf_data = load_etf_data()
    documents = []
    for etf in etf_data:
        content = f"""
ETF ID: {etf['id']}
상품명: {etf['name']} ({etf['ticker']})
카테고리: {etf['category']}
추종지수: {etf['index']}
운용사: {etf['asset_manager']}
총보수: {etf['total_expense_ratio']}
설명: {etf['description']}
위험등급: {etf['risk_level']}
투자전략: {etf['investment_strategy']}
배당정책: {etf['dividend_policy']}
투자자 유의사항: {etf['investor_caution']}
"""
        doc = Document(
            page_content=content,
            metadata={"id": etf["id"], "name": etf["name"], "ticker": etf["ticker"]}
        )
        documents.append(doc)

    embeddings = OpenAIEmbeddings(api_key=api_key)
    vectorstore = FAISS.from_documents(documents=documents, embedding=embeddings)

    print("Vector DB initialized successfully!")

    # 테스트 실행
    print("\nRunning tests...")
    results = run_all_tests(client, vectorstore)

    # 결과 저장
    report_path = os.path.join(os.path.dirname(__file__), "test_report.json")
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    # 요약 출력
    print("\n" + "="*60)
    print("TEST SUMMARY")
    print("="*60)
    print(f"Total Tests: {results['summary']['total_tests']}")
    print(f"Success Rate: {results['summary']['success_rate']}%")
    print(f"Type Match Rate: {results['summary']['type_match_rate']}%")
    print(f"Avg Response Time: {results['summary']['avg_response_time_ms']:.0f}ms")

    print("\nBy Category:")
    for cat, stats in results["by_category"].items():
        print(f"  - {cat}: {stats['total']} tests, avg {stats['avg_time_ms']:.0f}ms")

    print(f"\nFull report saved to: {report_path}")

if __name__ == "__main__":
    main()

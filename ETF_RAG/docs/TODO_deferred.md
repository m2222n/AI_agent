# Deferred Tasks (Phase 2 이후)

> Phase 2를 FAISS + Kiwi BM25 하이브리드로 먼저 구현한 뒤, 아래 항목을 순차 적용

## Vector DB 마이그레이션
- [ ] FAISS → **Pinecone** 마이그레이션 (free tier, 서버리스)
  - 인덱스 생성, 네임스페이스 설계 (ETF 메타/투자설명서 분리)
  - sparse-dense 하이브리드 검색 지원
  - persist 문제 해결 (현재 FAISS는 매번 재생성)

## Re-ranking
- [ ] **Cohere Rerank v3** 적용
  - 1차 검색 결과(hybrid) → Cohere로 재정렬 → top-k 선택
  - 한국어 지원 확인 필요

## 실시간 시세 (Phase 1-3)
- [ ] **한국투자증권 OpenAPI** 연동
  - KIS Developers 계좌 개설 + API 키 발급
  - REST 기반 실시간 시세 조회
  - 에러 핸들링 (timeout, retry, rate limit)
  - 추후 WebSocket 업그레이드 검토

## 평가 체계
- [ ] **RAGAS** 자동 평가 파이프라인
  - Faithfulness, Answer Relevancy, Context Recall
  - 평가 데이터셋 구축 (질문-정답-컨텍스트 쌍 50개+)
  - 변경 전후 정량 비교 기록

## 문서 처리
- [ ] ETF 투자설명서 PDF 파싱
  - PyPDFLoader + RecursiveCharacterTextSplitter
  - chunk_size=1000, overlap=100, tiktoken 기반
  - 메타데이터 태깅 (ETF ID, 카테고리, 문서 유형, 날짜)

## 임베딩 모델 비교
- [ ] OpenAI text-embedding-3-small vs BGE-M3 (한국어 특화) 비교 실험
  - RAGAS 평가 기반 정량 비교

## 데이터 구조
- [ ] 메타데이터(정적) vs 시세데이터(동적) 분리 설계
  - 정적: ETF명, 운용사, 카테고리, 설정일 등
  - 동적: 가격, NAV, 거래량, 수익률 등

---
_Created: 2026-04-07_

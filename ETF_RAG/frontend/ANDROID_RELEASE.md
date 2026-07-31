# 주선생 안드로이드 출시 체크리스트 (Google Play)

정태민 단독 진행(재우 하차, 2026-07-31). 목표: **안드로이드 앱 1개 Google Play 출시**.
장비: Intel 2019 맥 + macOS Tahoe 26.5.2 → **Android Studio로 이 맥에서 가능**(iOS와 달리 CPU 제약 없음).
앱명 "주선생", 번들 ID `ai.jusunsaeng.app`(iOS·Android 통일 완료).

기존 개념/빌드 절차는 `IOS_APP.md` Android 섹션 참조. 이 문서는 **출시(제출)까지 단계별 상태 추적**용.

---

## 진행 상태

| # | 단계 | 상태 | 비고 |
|---|------|------|------|
| 1 | Capacitor Android 프로젝트 생성 | ✅ 완료 | `android/`, 플러그인 4개 |
| 2 | 번들 ID 통일 `ai.jusunsaeng.app` | ✅ 완료 (2026-07-31) | 파일 4곳 + 디렉토리 `ai/jusunsaeng/app` 이동 |
| 3 | Android Studio 설치 | ⬜ 사용자 | 무료, `brew install --cask android-studio` 또는 공식 다운로드 |
| 4 | 에뮬레이터 빌드·동작 확인 | ⬜ | `cap open android` → Pixel 에뮬 Run → 체크리스트(로그인/챗봇/탭/오프라인) |
| 5 | 앱 서명 keystore 생성 | ⬜ | `keytool -genkey` → **분실 시 앱 업데이트 영구 불가, 반드시 백업** |
| 6 | 앱 아이콘·스플래시 최종 | 🔶 임시 있음 | 512 소스 사용 중. Play는 512×512 아이콘 필요(현 자산으로 가능) |
| 7 | 스크린샷 준비 | ⬜ | 폰 2장+ 필수(에뮬 캡처 가능). 태블릿은 선택 |
| 8 | 개인정보처리방침 URL | ⬜ | **Play 필수**. 간단한 페이지 1개 필요(수집 항목: 이메일/성별/나이대 명시) |
| 9 | Google Play Console 등록 | ⬜ | **$25 평생 1회**. 개인 계정 |
| 10 | 테스터 12명 14일 옵트인 | ⬜ | **2026 신규 개인계정 정식출시 요건**. 지인 12명 2주 연속 |
| 11 | AAB 빌드 → 업로드 → 심사 제출 | ⬜ | `./gradlew bundleRelease` 또는 Android Studio |

---

## 단계별 상세

### 3~4. Android Studio + 에뮬레이터 (이 맥에서)
```bash
# 설치 (택1)
brew install --cask android-studio
# 또는 https://developer.android.com/studio

cd frontend
npm run build:static      # out/ 정적 번들 (라이브 백엔드 URL 주입)
npx cap sync android
npx cap open android      # Android Studio 열림 → Device Manager로 Pixel 에뮬 생성 → ▶ Run
```
확인: 앱 아이콘/스플래시 → 로그인(라이브 백엔드) → 챗봇 SSE → 데이터탭 → 네트워크 끊고 오프라인 배너.

### 5. keystore (서명 키) — ⚠️ 가장 조심
```bash
keytool -genkey -v -keystore ~/jusunsaeng-release.keystore \
  -alias jusunsaeng -keyalg RSA -keysize 2048 -validity 10000
```
- 이 keystore + 비밀번호를 **분실하면 그 앱은 두 번 다시 업데이트 못 함**(새 앱으로 다시 내야 함).
- **백업 필수**: 안전한 곳(비밀번호 관리자/암호화 백업)에 keystore 파일 + alias + 비번 보관.
- Google Play App Signing(권장) 쓰면 업로드 키 분실 시 재설정 가능하나, 최초 등록 키는 여전히 신중히.

### 8. 개인정보처리방침
- 수집 항목(현재): **이메일(=ID), 비밀번호(해시), 성별, 나이대, 관심종목/가상투자 기록**.
- Play는 URL 형태 필수. 간단히 GitHub Pages나 프로젝트 라이브 사이트 `/privacy` 라우트로 1페이지 만들면 됨.

### 10. 테스터 12명 옵트인 (2026 정책, 개인 신규계정)
- 정식 프로덕션 공개 전, **비공개 테스트에 12명이 14일 연속 참여**해야 프로덕션 승격 자격.
- 지인 12명에게 테스트 링크 공유 → 2주간 앱 유지. **혼자 진행 시 이게 실질 최대 허들.**
- (APK 직접배포는 이 요건 skip하나 스토어 노출·신뢰 없음 — 출시 목적과 안 맞음)

---

## 비용 요약
- Google Play 등록: **$25 (평생 1회, 연회비 없음)**
- 안드로이드 유지 추가비용: **$0**
- (서비스 인프라 유지비는 별도 — Railway+OpenAI 등 월 $5~15, 앱화와 무관)

## 남은 리스크/메모
- iOS는 이 Intel 맥으로 영구 불가(Apple Silicon 필요) — 안드로이드만 출시. 상세 memory/project_ai_agent_ios_app.
- 브랜치 `feat/ios-capacitor`에서 진행 중(main 미머지).

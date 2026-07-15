# 주선생 모바일 앱 (Capacitor — iOS + Android)

기존 Next.js 웹앱을 코드 재작성 없이 iOS/Android 앱으로 래핑한다. 웹 배포(Railway)와
동일한 코드베이스를 쓰며, 앱은 정적 번들을 기기에 내장하고 데이터는 라이브
백엔드(Railway)를 호출한다. `ios/`·`android/` 프로젝트가 같은 `out/` 번들을 공유한다.

**요약**: iOS는 맥+Xcode(macOS 15↑) 필요 → 현 업무용 Intel 맥(macOS 14)에선 막힘.
Android는 **이 맥(macOS 14)에서 Android Studio만으로 가능**, macOS 업그레이드/윈도우 불필요.

## 구조

- `next.config.ts` — `BUILD_TARGET=static`이면 `output: "export"`(정적 번들 `out/`),
  기본은 `standalone`(Railway 웹 배포). 한 코드베이스로 둘 다 빌드.
- `capacitor.config.ts` — `webDir: "out"`, `server.url` 없음(원격 래퍼 아님 → Apple 4.2 안전).
- `ios/` — Capacitor가 생성한 Xcode 프로젝트(SPM 기반, CocoaPods 불필요).
- `src/components/OfflineBanner.tsx` — 오프라인 시 안내 배너.
- `src/components/NativeAppInit.tsx` — 상태바/스플래시 초기화(네이티브에서만).

## 앱 빌드 & 시뮬레이터 실행 (5단계)

### 사전 준비 (최초 1회)
Xcode 전체 앱이 필요하다(Command Line Tools만으로는 불가).

```bash
# 1) App Store에서 "Xcode" 설치 (~7GB, 시간 걸림)
# 2) 설치 후 커맨드라인이 전체 Xcode를 가리키도록 전환
sudo xcode-select -s /Applications/Xcode.app/Contents/Developer
sudo xcodebuild -license accept
xcodebuild -version   # 버전 나오면 성공
```

### 매번 (코드 바뀔 때마다)
```bash
cd frontend
npm run build:static     # out/ 생성 (라이브 백엔드 URL 주입)
npx cap sync ios         # out/ → ios 로 복사 + 플러그인 갱신
npx cap open ios         # Xcode 열림
```

Xcode에서:
1. 상단 타깃을 iPhone 시뮬레이터(예: iPhone 15)로 선택
2. ▶ Run

### 확인 체크리스트 (시뮬레이터)
- [ ] 앱 아이콘 · 스플래시 표시
- [ ] 로그인 성공 (라이브 백엔드 연결)
- [ ] 챗봇 답변 스트리밍(SSE) 동작
- [ ] 데이터 탭(기술/재무/비교/전망/섹터/뉴스/가상투자) 로딩
- [ ] 시뮬레이터 네트워크 끊고 → 오프라인 배너 표시

### CORS 문제가 나면
시뮬레이터 콘솔/네트워크에서 CORS 차단이 보이면, Railway 백엔드가
`CORS_ORIGINS`를 특정 웹 origin으로 제한하고 있어도 코드가
`capacitor://localhost`를 정규식으로 허용하도록 이미 처리돼 있다
(`api/main.py`의 `_capacitor_origin_regex`). 백엔드 재배포가 반영됐는지 확인.

## App Store 제출 전 남은 일 (6단계, 별도)

- **앱 아이콘 1024×1024**: 현재 512 소스를 업스케일해 씀. 제출용은 1024 원본 필요.
  `assets/icon.png`를 1024로 교체 후 `npx @capacitor/assets generate --ios` 재실행.
- **번들 ID**: 현재 임시 `com.example.etfrag`. 제출 전 확정(`capacitor.config.ts` + Xcode).
- **앱 이름**: 현재 "주선생"(후보: 주선생/주교수/영차). 언제든 변경 가능.
- Apple Developer Program 가입($99/년) → App Store Connect 앱 생성 → 아카이브/업로드.
- 스크린샷, 설명, 개인정보처리방침 URL, 심사 노트.

---

# Android 앱 (Capacitor)

iOS와 달리 **맥/윈도우/리눅스 어디서든** 빌드 가능하고 **macOS 업그레이드 불필요**.
현 업무용 Intel 맥(macOS 14) 그대로 진행할 수 있어 지금 열려 있는 유일한 스토어 경로.

## 사전 준비 (최초 1회)
Android Studio(무료)만 설치하면 된다. 자체 JDK·SDK·에뮬레이터 포함.

```bash
# 1) https://developer.android.com/studio 에서 Android Studio 다운로드·설치
#    (또는 brew install --cask android-studio)
# 2) 첫 실행 시 SDK·에뮬레이터 자동 설치 마법사 따라가기
# 3) 환경변수(터미널 빌드용, Android Studio만 쓸 거면 생략 가능)
#    export ANDROID_HOME=$HOME/Library/Android/sdk
```

## 매번 (코드 바뀔 때마다)
```bash
cd frontend
npm run build:static      # out/ 생성 (라이브 백엔드 URL 주입)
npx cap sync android      # out/ → android 로 복사 + 플러그인 갱신
npx cap open android      # Android Studio 열림
```

Android Studio에서:
1. 상단에서 에뮬레이터(가상 기기) 선택 — 없으면 Device Manager에서 생성(예: Pixel 7, API 34)
2. ▶ Run

### 실기기(공기계)로 테스트할 때 (선택)
- 갤럭시 등 안드로이드 폰에서 **설정 → 휴대전화 정보 → 빌드번호 7번 탭 → 개발자 모드** 켜기
- **USB 디버깅** 켜고 맥에 USB 연결 → Android Studio 기기 목록에 뜨면 ▶ Run
- **개통(유심) 불필요**, Wi-Fi만 있으면 앱(서버 통신) 정상 동작. 공기계는 **에뮬레이터로 대체 가능**하므로 필수 아님.

### 확인 체크리스트
- [ ] 앱 아이콘 · 스플래시 표시
- [ ] 로그인 성공 (라이브 백엔드 연결)
- [ ] 챗봇 답변 스트리밍(SSE) 동작
- [ ] 데이터 탭 로딩
- [ ] 네트워크 끊고 → 오프라인 배너 표시

### CORS 참고
Android WebView origin은 `http://localhost`(포트 없음), iOS는 `capacitor://localhost`.
`api/main.py`의 `_capacitor_origin_regex`가 **둘 다 이미 허용**한다(`CORS_ORIGINS`를
특정 origin으로 좁혀도 앱 요청은 통과). 기본 `CORS_ORIGINS=*`면 애초에 문제 없음.

## Google Play 제출 전 남은 일 (별도)
- **아이콘 512×512**(Play 스토어 등록용) — 현재 자산으로 충분, 필요 시 고해상도 교체.
- **번들 ID**: 임시 `com.example.etfrag` → 확정(`capacitor.config.ts`).
- Google Play Console 등록비 **$25(평생 1회)**.
- **2026년 정책**: 신규 개인 개발자 계정은 프로덕션 공개 전 **테스터 12명이 14일 연속 옵트인** 필요(요건 확인).
- 서명 키(keystore) 생성 + AAB 빌드 + 스크린샷/설명/개인정보처리방침.

"use client";

// 개인정보처리방침 — Google Play/App Store 제출 필수 + 회원가입 동의 링크 대상.
// 실제 수집 항목(api/auth.py 회원가입 스키마)과 일치시켜 유지할 것.
export default function PrivacyPage() {
  const updated = "2026-07-31";
  return (
    <main className="mx-auto w-full max-w-2xl px-4 py-8 text-sm leading-relaxed text-gray-700 dark:text-gray-300">
      <h1 className="mb-2 text-xl font-bold text-gray-900 dark:text-gray-100">
        개인정보처리방침
      </h1>
      <p className="mb-6 text-xs text-gray-500">최종 수정일: {updated}</p>

      <p className="mb-6">
        &lsquo;주선생&rsquo;(투자 AI 어시스턴트, 이하 &lsquo;서비스&rsquo;)은 이용자의 개인정보를
        중요하게 생각하며, 「개인정보 보호법」을 준수합니다. 본 방침은 서비스가 수집하는
        개인정보의 항목·목적·보관기간과 이용자의 권리를 안내합니다. <b>로그인 없이도 서비스의
        모든 기능을 이용할 수 있으며</b>, 아래 개인정보는 <b>회원가입(선택)</b> 시에만
        수집됩니다.
      </p>

      <Section title="1. 수집하는 개인정보 항목">
        <ul className="list-disc pl-5">
          <li><b>필수</b>: 이메일 주소(로그인 ID로 사용), 비밀번호(암호화 저장), 성별</li>
          <li><b>선택</b>: 나이대, 닉네임(표시용)</li>
          <li>
            <b>서비스 이용 중 생성</b>: 관심종목, 가상투자(모의투자) 기록, 대화 이력
          </li>
        </ul>
        <p className="mt-2 text-xs text-gray-500">
          ※ 이름·주민번호·연락처·금융계좌 등은 수집하지 않습니다. 가상투자는 실제 돈이
          아닌 모의 투자이며 실계좌와 연결되지 않습니다.
        </p>
      </Section>

      <Section title="2. 개인정보 수집·이용 목적">
        <ul className="list-disc pl-5">
          <li>회원 식별 및 로그인, 개인별 데이터(관심종목·가상투자·대화이력) 저장</li>
          <li>성별·나이대: 통계 및 맞춤 정보 제공(개인 식별 목적 아님)</li>
          <li>서비스 운영·개선 및 문의 응대</li>
        </ul>
      </Section>

      <Section title="3. 보관 및 이용 기간">
        <p>
          회원 탈퇴 시 수집한 개인정보 및 연계 데이터(관심종목·가상투자·대화이력·푸시
          구독)를 지체 없이 파기합니다. 법령에 따라 별도 보관이 요구되는 정보는 해당 기간
          동안 보관합니다.
        </p>
      </Section>

      <Section title="4. 제3자 제공 및 처리 위탁">
        <p>
          이용자의 개인정보를 제3자에게 판매·제공하지 않습니다. 다만 서비스 운영을 위해
          아래 처리자를 이용합니다.
        </p>
        <ul className="mt-2 list-disc pl-5">
          <li>클라우드 호스팅 및 데이터베이스(서비스 서버 운영)</li>
          <li>AI 응답 생성(입력하신 질문 텍스트가 처리되며, 계정 식별정보는 전달되지 않습니다)</li>
        </ul>
      </Section>

      <Section title="5. 이용자의 권리">
        <p>
          이용자는 언제든지 본인의 개인정보를 조회·수정할 수 있고(계정 설정), 회원 탈퇴로
          삭제를 요청할 수 있습니다. 비밀번호 변경은 계정 설정에서 가능합니다.
        </p>
      </Section>

      <Section title="6. 안전성 확보 조치">
        <p>
          비밀번호는 복호화 불가능한 해시(bcrypt)로 저장하며, 통신은 HTTPS로 암호화됩니다.
          인증에는 만료 기한이 있는 토큰(JWT)을 사용합니다.
        </p>
      </Section>

      <Section title="7. 문의처">
        <p>개인정보 관련 문의: jtm@flickdone.com</p>
      </Section>

      <Section title="8. 고지 의무">
        <p>
          본 방침의 내용이 변경될 경우 서비스 내 공지를 통해 안내합니다. 본 서비스는 투자
          참고 정보를 제공하며, 투자 판단과 책임은 이용자 본인에게 있습니다.
        </p>
      </Section>
    </main>
  );
}

function Section({ title, children }: { title: string; children: React.ReactNode }) {
  return (
    <section className="mb-5">
      <h2 className="mb-2 text-base font-semibold text-gray-900 dark:text-gray-100">
        {title}
      </h2>
      {children}
    </section>
  );
}

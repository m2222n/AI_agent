"use client";

import { useRouter } from "next/navigation";
import { useState } from "react";
import { useAuth } from "@/lib/AuthContext";
import { AGE_GROUPS, GENDERS } from "@/lib/auth";

export default function LoginPage() {
  const { login, signup } = useAuth();
  const router = useRouter();
  const [mode, setMode] = useState<"login" | "signup">("login");
  const [email, setEmail] = useState("");
  const [password, setPassword] = useState("");
  const [ageGroup, setAgeGroup] = useState(""); // 선택
  const [gender, setGender] = useState(""); // 필수(가입 시)
  const [showFind, setShowFind] = useState(false); // ID/비번 찾기 안내
  const [error, setError] = useState<string | null>(null);
  const [busy, setBusy] = useState(false);

  const submit = async (e: React.FormEvent) => {
    e.preventDefault();
    setError(null);
    setBusy(true);
    try {
      if (mode === "login") {
        await login(email, password);
      } else {
        if (!gender) {
          setError("성별을 선택하세요.");
          setBusy(false);
          return;
        }
        await signup(email, password, gender, ageGroup || undefined);
      }
      router.push("/"); // 채팅으로
    } catch (err) {
      setError(
        err instanceof Error ? err.message : "처리 중 오류가 발생했어요.",
      );
    } finally {
      setBusy(false);
    }
  };

  return (
    <main className="mx-auto w-full max-w-sm px-4 py-10">
      <h1 className="mb-1 text-lg font-bold text-gray-900">
        {mode === "login" ? "로그인" : "회원가입"}
      </h1>
      <p className="mb-5 text-xs text-gray-500">
        로그인하면 관심종목·대화 이력이 기기 간에 유지돼요. (선택 사항 — 로그인
        없이도 모든 기능 사용 가능)
      </p>

      <form onSubmit={submit} className="flex flex-col gap-3">
        <input
          type="email"
          required
          value={email}
          onChange={(e) => setEmail(e.target.value)}
          placeholder="이메일"
          className="rounded-xl border border-gray-300 dark:border-gray-700 px-4 py-2.5 text-sm focus:border-blue-500 focus:outline-none"
        />
        <input
          type="password"
          required
          minLength={mode === "signup" ? 8 : 1}
          value={password}
          onChange={(e) => setPassword(e.target.value)}
          placeholder={mode === "signup" ? "비밀번호 (8자 이상)" : "비밀번호"}
          className="rounded-xl border border-gray-300 dark:border-gray-700 px-4 py-2.5 text-sm focus:border-blue-500 focus:outline-none"
        />
        {mode === "signup" && (
          <label className="flex flex-col gap-1">
            <span className="text-xs text-gray-500">
              성별 <span className="text-red-500">*</span>{" "}
              <span className="text-gray-400">(맞춤 추천에 활용 — 이름은 받지 않아요)</span>
            </span>
            <select
              required
              value={gender}
              onChange={(e) => setGender(e.target.value)}
              className="rounded-xl border border-gray-300 dark:border-gray-700 px-4 py-2.5 text-sm focus:border-blue-500 focus:outline-none"
            >
              <option value="">선택하세요</option>
              {GENDERS.map((g) => (
                <option key={g} value={g}>{g}</option>
              ))}
            </select>
          </label>
        )}
        {mode === "signup" && (
          <label className="flex flex-col gap-1">
            <span className="text-xs text-gray-500">
              나이대 <span className="text-gray-400">(선택 — 맞춤 추천에 활용)</span>
            </span>
            <select
              value={ageGroup}
              onChange={(e) => setAgeGroup(e.target.value)}
              className="rounded-xl border border-gray-300 dark:border-gray-700 px-4 py-2.5 text-sm focus:border-blue-500 focus:outline-none"
            >
              <option value="">선택 안 함</option>
              {AGE_GROUPS.map((g) => (
                <option key={g} value={g}>{g}</option>
              ))}
            </select>
          </label>
        )}
        {error && <p className="text-xs text-red-600">{error}</p>}
        <button
          type="submit"
          disabled={busy}
          className="rounded-xl bg-blue-600 px-4 py-2.5 text-sm font-medium text-white hover:bg-blue-700 disabled:bg-gray-300"
        >
          {busy ? "처리 중…" : mode === "login" ? "로그인" : "가입하기"}
        </button>
      </form>

      <div className="mt-4 flex items-center gap-3">
        <button
          type="button"
          onClick={() => {
            setMode(mode === "login" ? "signup" : "login");
            setError(null);
            setShowFind(false);
          }}
          className="text-xs text-blue-600 hover:underline"
        >
          {mode === "login"
            ? "계정이 없으신가요? 회원가입"
            : "이미 계정이 있으신가요? 로그인"}
        </button>
        {mode === "login" && (
          <button
            type="button"
            onClick={() => setShowFind((v) => !v)}
            className="text-xs text-gray-500 hover:underline"
          >
            아이디·비밀번호 찾기
          </button>
        )}
      </div>

      {mode === "login" && showFind && (
        <div className="mt-3 rounded-xl border border-gray-200 dark:border-gray-800 bg-gray-50 p-3 text-xs leading-relaxed text-gray-600">
          <p className="mb-1">
            <b>아이디</b>: 가입하신 <b>이메일이 곧 아이디</b>예요. 이메일 주소로 로그인하세요.
          </p>
          <p>
            <b>비밀번호</b>: 현재 이메일 재설정은 준비 중이에요. 로그인이 되는 상태라면{" "}
            <b>계정 설정</b>에서 변경할 수 있고, 완전히 잊으셨다면 문의해 주세요.
          </p>
        </div>
      )}
    </main>
  );
}

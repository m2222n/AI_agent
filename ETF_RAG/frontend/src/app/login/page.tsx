"use client";

import { useRouter } from "next/navigation";
import { useState } from "react";
import { useAuth } from "@/lib/AuthContext";

export default function LoginPage() {
  const { login, signup } = useAuth();
  const router = useRouter();
  const [mode, setMode] = useState<"login" | "signup">("login");
  const [email, setEmail] = useState("");
  const [password, setPassword] = useState("");
  const [error, setError] = useState<string | null>(null);
  const [busy, setBusy] = useState(false);

  const submit = async (e: React.FormEvent) => {
    e.preventDefault();
    setError(null);
    setBusy(true);
    try {
      if (mode === "login") await login(email, password);
      else await signup(email, password);
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
          className="rounded-xl border border-gray-300 px-4 py-2.5 text-sm focus:border-blue-500 focus:outline-none"
        />
        <input
          type="password"
          required
          minLength={mode === "signup" ? 8 : 1}
          value={password}
          onChange={(e) => setPassword(e.target.value)}
          placeholder={mode === "signup" ? "비밀번호 (8자 이상)" : "비밀번호"}
          className="rounded-xl border border-gray-300 px-4 py-2.5 text-sm focus:border-blue-500 focus:outline-none"
        />
        {error && <p className="text-xs text-red-600">{error}</p>}
        <button
          type="submit"
          disabled={busy}
          className="rounded-xl bg-blue-600 px-4 py-2.5 text-sm font-medium text-white hover:bg-blue-700 disabled:bg-gray-300"
        >
          {busy ? "처리 중…" : mode === "login" ? "로그인" : "가입하기"}
        </button>
      </form>

      <button
        type="button"
        onClick={() => {
          setMode(mode === "login" ? "signup" : "login");
          setError(null);
        }}
        className="mt-4 text-xs text-blue-600 hover:underline"
      >
        {mode === "login"
          ? "계정이 없으신가요? 회원가입"
          : "이미 계정이 있으신가요? 로그인"}
      </button>
    </main>
  );
}

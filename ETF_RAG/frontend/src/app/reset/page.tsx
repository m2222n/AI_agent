"use client";

import { Suspense, useState } from "react";
import { useRouter, useSearchParams } from "next/navigation";
import { confirmPasswordReset } from "@/lib/auth";

function ResetInner() {
  const router = useRouter();
  const params = useSearchParams();
  const token = params.get("token") ?? "";

  const [pw, setPw] = useState("");
  const [pw2, setPw2] = useState("");
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [done, setDone] = useState(false);

  const submit = async (e: React.FormEvent) => {
    e.preventDefault();
    setError(null);
    if (pw !== pw2) {
      setError("두 비밀번호가 일치하지 않아요.");
      return;
    }
    setBusy(true);
    try {
      await confirmPasswordReset(token, pw);
      setDone(true);
    } catch (err) {
      setError(err instanceof Error ? err.message : "재설정에 실패했어요.");
    } finally {
      setBusy(false);
    }
  };

  if (!token) {
    return (
      <p className="text-sm text-red-600">
        유효한 재설정 링크가 아니에요. 로그인 화면에서 다시 요청해 주세요.
      </p>
    );
  }

  if (done) {
    return (
      <div className="flex flex-col gap-3">
        <p className="text-sm text-green-700">비밀번호를 변경했어요. 새 비밀번호로 로그인하세요.</p>
        <button
          onClick={() => router.push("/login")}
          className="rounded-xl bg-blue-600 px-4 py-2.5 text-sm font-medium text-white hover:bg-blue-700"
        >
          로그인하러 가기
        </button>
      </div>
    );
  }

  return (
    <form onSubmit={submit} className="flex flex-col gap-3">
      <input
        type="password"
        required
        minLength={8}
        value={pw}
        onChange={(e) => setPw(e.target.value)}
        placeholder="새 비밀번호 (8자 이상)"
        className="rounded-xl border border-gray-300 dark:border-gray-700 px-4 py-2.5 text-sm focus:border-blue-500 focus:outline-none"
      />
      <input
        type="password"
        required
        minLength={8}
        value={pw2}
        onChange={(e) => setPw2(e.target.value)}
        placeholder="새 비밀번호 확인"
        className="rounded-xl border border-gray-300 dark:border-gray-700 px-4 py-2.5 text-sm focus:border-blue-500 focus:outline-none"
      />
      {error && <p className="text-xs text-red-600">{error}</p>}
      <button
        type="submit"
        disabled={busy}
        className="rounded-xl bg-blue-600 px-4 py-2.5 text-sm font-medium text-white hover:bg-blue-700 disabled:bg-gray-300"
      >
        {busy ? "변경 중…" : "비밀번호 변경"}
      </button>
    </form>
  );
}

export default function ResetPage() {
  return (
    <main className="mx-auto w-full max-w-sm px-4 py-10">
      <h1 className="mb-5 text-lg font-bold text-gray-900">비밀번호 재설정</h1>
      <Suspense fallback={<p className="text-sm text-gray-500">불러오는 중…</p>}>
        <ResetInner />
      </Suspense>
    </main>
  );
}

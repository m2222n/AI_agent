"use client";

import { useRouter } from "next/navigation";
import { useState } from "react";
import Link from "next/link";
import { useAuth } from "@/lib/AuthContext";
import { changePassword, deleteAccount, updateNickname } from "@/lib/auth";
import { Notice } from "@/components/Feedback";

export default function AccountPage() {
  const { user, loading, logout, refresh } = useAuth();
  const router = useRouter();

  // 비로그인 안내
  if (!loading && !user) {
    return (
      <main className="mx-auto w-full max-w-sm px-4 py-10 text-center">
        <p className="mb-4 text-sm text-gray-600">
          계정 설정은 로그인 후 이용할 수 있어요.
        </p>
        <Link
          href="/login"
          className="rounded-lg bg-blue-600 px-4 py-2 text-sm font-medium text-white"
        >
          로그인하러 가기
        </Link>
      </main>
    );
  }
  if (loading || !user) return null;

  return (
    <main className="mx-auto w-full max-w-sm px-4 py-8">
      <h1 className="mb-1 text-lg font-bold text-gray-900">계정 설정</h1>
      <p className="mb-6 text-xs text-gray-500">
        로그인 계정: <span className="text-gray-700">{user.email}</span>
        <br />
        (이메일이 곧 아이디예요)
      </p>

      <NicknameSection
        current={user.nickname}
        onSaved={refresh}
      />
      <PasswordSection />
      <DeleteSection
        onDeleted={() => {
          logout();
          router.push("/");
        }}
      />
    </main>
  );
}

function Card({
  title,
  desc,
  children,
}: {
  title: string;
  desc?: string;
  children: React.ReactNode;
}) {
  return (
    <section className="mb-5 rounded-xl border border-gray-200 p-4">
      <h2 className="text-sm font-semibold text-gray-800">{title}</h2>
      {desc && <p className="mt-0.5 text-xs text-gray-500">{desc}</p>}
      <div className="mt-3">{children}</div>
    </section>
  );
}


const inputCls =
  "w-full rounded-lg border border-gray-300 px-3 py-2 text-sm focus:border-blue-500 focus:outline-none";
const btnCls =
  "rounded-lg bg-blue-600 px-4 py-2 text-sm font-medium text-white disabled:opacity-50";

function NicknameSection({
  current,
  onSaved,
}: {
  current: string;
  onSaved: () => Promise<void>;
}) {
  const [value, setValue] = useState(current);
  const [busy, setBusy] = useState(false);
  const [msg, setMsg] = useState<{ k: "ok" | "err"; t: string } | null>(null);

  const save = async (e: React.FormEvent) => {
    e.preventDefault();
    setMsg(null);
    setBusy(true);
    try {
      await updateNickname(value.trim());
      await onSaved();
      setMsg({ k: "ok", t: "닉네임을 변경했어요." });
    } catch (err) {
      setMsg({ k: "err", t: err instanceof Error ? err.message : "오류" });
    } finally {
      setBusy(false);
    }
  };

  return (
    <Card title="닉네임" desc="화면에 표시되는 이름이에요. 로그인에는 영향 없어요.">
      <form onSubmit={save} className="flex gap-2">
        <input
          value={value}
          onChange={(e) => setValue(e.target.value)}
          maxLength={40}
          placeholder="닉네임"
          className={inputCls}
        />
        <button
          type="submit"
          disabled={busy || !value.trim() || value.trim() === current}
          className={btnCls + " shrink-0"}
        >
          저장
        </button>
      </form>
      {msg && <Notice message={msg.t} kind={msg.k} />}
    </Card>
  );
}

function PasswordSection() {
  const [cur, setCur] = useState("");
  const [next, setNext] = useState("");
  const [confirm, setConfirm] = useState("");
  const [busy, setBusy] = useState(false);
  const [msg, setMsg] = useState<{ k: "ok" | "err"; t: string } | null>(null);

  const save = async (e: React.FormEvent) => {
    e.preventDefault();
    setMsg(null);
    if (next.length < 8) {
      setMsg({ k: "err", t: "새 비밀번호는 8자 이상이어야 해요." });
      return;
    }
    if (next !== confirm) {
      setMsg({ k: "err", t: "새 비밀번호 확인이 일치하지 않아요." });
      return;
    }
    setBusy(true);
    try {
      await changePassword(cur, next);
      setCur("");
      setNext("");
      setConfirm("");
      setMsg({ k: "ok", t: "비밀번호를 변경했어요." });
    } catch (err) {
      setMsg({ k: "err", t: err instanceof Error ? err.message : "오류" });
    } finally {
      setBusy(false);
    }
  };

  return (
    <Card title="비밀번호 변경">
      <form onSubmit={save} className="space-y-2">
        <input
          type="password"
          value={cur}
          onChange={(e) => setCur(e.target.value)}
          placeholder="현재 비밀번호"
          autoComplete="current-password"
          className={inputCls}
        />
        <input
          type="password"
          value={next}
          onChange={(e) => setNext(e.target.value)}
          placeholder="새 비밀번호 (8자 이상)"
          autoComplete="new-password"
          className={inputCls}
        />
        <input
          type="password"
          value={confirm}
          onChange={(e) => setConfirm(e.target.value)}
          placeholder="새 비밀번호 확인"
          autoComplete="new-password"
          className={inputCls}
        />
        <button
          type="submit"
          disabled={busy || !cur || !next || !confirm}
          className={btnCls + " w-full"}
        >
          비밀번호 변경
        </button>
      </form>
      {msg && <Notice message={msg.t} kind={msg.k} />}
    </Card>
  );
}

function DeleteSection({ onDeleted }: { onDeleted: () => void }) {
  const [open, setOpen] = useState(false);
  const [pw, setPw] = useState("");
  const [busy, setBusy] = useState(false);
  const [err, setErr] = useState<string | null>(null);

  const remove = async (e: React.FormEvent) => {
    e.preventDefault();
    setErr(null);
    setBusy(true);
    try {
      await deleteAccount(pw);
      onDeleted();
    } catch (e2) {
      setErr(e2 instanceof Error ? e2.message : "오류");
      setBusy(false);
    }
  };

  return (
    <Card
      title="회원 탈퇴"
      desc="계정과 관심종목·대화 이력·알림 설정이 모두 삭제돼요. 되돌릴 수 없어요."
    >
      {!open ? (
        <button
          type="button"
          onClick={() => setOpen(true)}
          className="rounded-lg border border-red-300 px-4 py-2 text-sm font-medium text-red-600 hover:bg-red-50"
        >
          회원 탈퇴
        </button>
      ) : (
        <form onSubmit={remove} className="space-y-2">
          <input
            type="password"
            value={pw}
            onChange={(e) => setPw(e.target.value)}
            placeholder="확인을 위해 비밀번호 입력"
            autoComplete="current-password"
            className={inputCls}
          />
          <div className="flex gap-2">
            <button
              type="submit"
              disabled={busy || !pw}
              className="rounded-lg bg-red-600 px-4 py-2 text-sm font-medium text-white disabled:opacity-50"
            >
              탈퇴하기
            </button>
            <button
              type="button"
              onClick={() => {
                setOpen(false);
                setPw("");
                setErr(null);
              }}
              className="rounded-lg px-4 py-2 text-sm text-gray-600 hover:bg-gray-100"
            >
              취소
            </button>
          </div>
        </form>
      )}
      {err && <Notice message={err} kind="err" />}
    </Card>
  );
}

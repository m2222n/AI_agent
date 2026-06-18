"use client";

import Link from "next/link";
import { usePathname } from "next/navigation";
import { useAuth } from "@/lib/AuthContext";

const TABS = [
  { href: "/", label: "💬 채팅" },
  { href: "/technical", label: "📊 기술적 분석" },
  { href: "/financial", label: "📑 재무제표" },
  { href: "/comparison", label: "⚖️ 비교" },
  { href: "/outlook", label: "🔮 전망" },
  { href: "/sector", label: "🏭 섹터" },
];

export default function NavBar({ onMenuClick }: { onMenuClick?: () => void } = {}) {
  const pathname = usePathname();
  const { user, loading, logout } = useAuth();

  return (
    <nav className="border-b border-gray-200">
      <div className="mx-auto flex max-w-3xl items-center gap-1 overflow-x-auto px-3 py-2 sm:px-4">
        {/* 모바일 햄버거 — 사이드바 드로워 열기 (lg 미만에서만) */}
        {onMenuClick && (
          <button
            type="button"
            aria-label="메뉴 열기"
            onClick={onMenuClick}
            className="shrink-0 rounded-lg px-2 py-1.5 text-base text-gray-600 hover:bg-gray-100 lg:hidden"
          >
            ☰
          </button>
        )}
        {TABS.map((t) => {
          const active = t.href === pathname;
          return (
            <Link
              key={t.href}
              href={t.href}
              className={[
                "shrink-0 rounded-lg px-3 py-1.5 text-xs font-medium transition-colors",
                active
                  ? "bg-blue-600 text-white"
                  : "text-gray-600 hover:bg-gray-100",
              ].join(" ")}
            >
              {t.label}
            </Link>
          );
        })}

        {/* 인증 영역 (오른쪽) */}
        <div className="ml-auto flex shrink-0 items-center gap-2 pl-2">
          {loading ? null : user ? (
            <>
              <Link
                href="/account"
                title="계정 설정"
                className={[
                  "shrink-0 rounded-lg px-2 py-1.5 text-xs font-medium",
                  pathname === "/account"
                    ? "bg-blue-600 text-white"
                    : "text-gray-700 hover:bg-gray-100",
                ].join(" ")}
              >
                ⚙️ {user.nickname}
              </Link>
              <button
                type="button"
                onClick={logout}
                className="shrink-0 rounded-lg px-3 py-1.5 text-xs text-gray-600 hover:bg-gray-100"
              >
                로그아웃
              </button>
            </>
          ) : (
            <Link
              href="/login"
              className={[
                "shrink-0 rounded-lg px-3 py-1.5 text-xs font-medium",
                pathname === "/login"
                  ? "bg-blue-600 text-white"
                  : "text-blue-600 hover:bg-blue-50",
              ].join(" ")}
            >
              로그인
            </Link>
          )}
        </div>
      </div>
    </nav>
  );
}

"use client";

import { useEffect, useState } from "react";

const KEY = "etfrag.theme"; // "dark" | "light"

/** html.dark 토글 + localStorage 영속. 초기값은 layout의 인라인 스크립트가 적용. */
export default function ThemeToggle() {
  const [dark, setDark] = useState(false);
  const [mounted, setMounted] = useState(false);

  useEffect(() => {
    setMounted(true);
    setDark(document.documentElement.classList.contains("dark"));
  }, []);

  const toggle = () => {
    const next = !dark;
    setDark(next);
    document.documentElement.classList.toggle("dark", next);
    try {
      localStorage.setItem(KEY, next ? "dark" : "light");
    } catch {
      /* localStorage 불가 환경 무시 */
    }
  };

  // SSR/하이드레이션 불일치 방지 — 마운트 전엔 빈 자리만
  if (!mounted) return <span className="inline-block h-8 w-8" aria-hidden="true" />;

  return (
    <button
      type="button"
      onClick={toggle}
      aria-label={dark ? "라이트 모드로 전환" : "다크 모드로 전환"}
      title={dark ? "라이트 모드" : "다크 모드"}
      className="rounded-lg p-1.5 text-base hover:bg-gray-100 dark:hover:bg-gray-800"
    >
      {dark ? "☀️" : "🌙"}
    </button>
  );
}

/** layout <head>에 인라인으로 넣어 FOUC(깜빡임) 방지하는 스크립트 문자열.
 * 저장된 테마가 dark면 즉시 html에 클래스 부여(저장 없으면 OS 설정 따름). */
export const THEME_INIT_SCRIPT = `(function(){try{var t=localStorage.getItem('${KEY}');if(t==='dark'||(!t&&window.matchMedia&&window.matchMedia('(prefers-color-scheme: dark)').matches)){document.documentElement.classList.add('dark');}}catch(e){}})();`;

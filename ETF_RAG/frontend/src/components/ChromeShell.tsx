"use client";

import { useState } from "react";
import NavBar from "@/components/NavBar";
import Sidebar from "@/components/Sidebar";

/**
 * NavBar + Sidebar + 본문을 감싸는 클라이언트 셸.
 * 모바일 드로워(사이드바) open 상태를 보유 — NavBar의 ☰ 버튼과 Sidebar 오버레이를 연결한다.
 * 데스크톱(lg+)에서는 Sidebar가 고정 표시되고 open 상태는 무시된다.
 */
export default function ChromeShell({ children }: { children: React.ReactNode }) {
  const [drawerOpen, setDrawerOpen] = useState(false);

  return (
    <>
      <NavBar onMenuClick={() => setDrawerOpen(true)} />
      <div className="flex flex-1 overflow-hidden">
        <Sidebar open={drawerOpen} onClose={() => setDrawerOpen(false)} />
        <div className="flex flex-1 flex-col overflow-y-auto">{children}</div>
      </div>
    </>
  );
}

import type { Metadata, Viewport } from "next";
import { Geist, Geist_Mono } from "next/font/google";
import "./globals.css";
import ChromeShell from "@/components/ChromeShell";
import { AuthProvider } from "@/lib/AuthContext";
import ServiceWorkerRegister from "@/components/ServiceWorkerRegister";
import { THEME_INIT_SCRIPT } from "@/components/ThemeToggle";

const geistSans = Geist({
  variable: "--font-geist-sans",
  subsets: ["latin"],
});

const geistMono = Geist_Mono({
  variable: "--font-geist-mono",
  subsets: ["latin"],
});

export const metadata: Metadata = {
  title: "투자 AI 어시스턴트",
  description: "ETF·주식 RAG 기반 투자 질의응답 챗봇",
  manifest: "/manifest.webmanifest",
  appleWebApp: {
    capable: true,
    statusBarStyle: "default",
    title: "투자AI",
  },
  icons: {
    icon: "/icons/icon-192.png",
    apple: "/icons/icon-192.png",
  },
};

export const viewport: Viewport = {
  themeColor: "#2563eb",
};

export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode;
}>) {
  return (
    <html
      lang="ko"
      className={`${geistSans.variable} ${geistMono.variable} h-full antialiased`}
      suppressHydrationWarning
    >
      <head>
        {/* FOUC 방지 — 첫 페인트 전에 저장된 테마를 html에 적용 */}
        <script dangerouslySetInnerHTML={{ __html: THEME_INIT_SCRIPT }} />
      </head>
      <body className="min-h-full flex flex-col">
        <AuthProvider>
          <ChromeShell>{children}</ChromeShell>
        </AuthProvider>
        <ServiceWorkerRegister />
      </body>
    </html>
  );
}

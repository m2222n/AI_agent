"use client";

import {
  createContext,
  useContext,
  useEffect,
  useState,
  type ReactNode,
} from "react";
import {
  clearToken,
  fetchMe,
  login as apiLogin,
  setToken,
  signup as apiSignup,
  type AuthUser,
} from "./auth";

interface AuthState {
  user: AuthUser | null;
  loading: boolean; // 초기 토큰 검증 중
  login: (email: string, password: string) => Promise<void>;
  signup: (email: string, password: string) => Promise<void>;
  logout: () => void;
}

const AuthCtx = createContext<AuthState | null>(null);

export function AuthProvider({ children }: { children: ReactNode }) {
  const [user, setUser] = useState<AuthUser | null>(null);
  const [loading, setLoading] = useState(true);

  // mount 시 저장된 토큰으로 사용자 복원
  useEffect(() => {
    let cancelled = false;
    fetchMe()
      .then((u) => {
        if (!cancelled) setUser(u);
      })
      .finally(() => {
        if (!cancelled) setLoading(false);
      });
    return () => {
      cancelled = true;
    };
  }, []);

  const login = async (email: string, password: string) => {
    const token = await apiLogin(email, password);
    setToken(token);
    setUser(await fetchMe());
  };

  const signup = async (email: string, password: string) => {
    const token = await apiSignup(email, password);
    setToken(token);
    setUser(await fetchMe());
  };

  const logout = () => {
    clearToken();
    setUser(null);
  };

  return (
    <AuthCtx.Provider value={{ user, loading, login, signup, logout }}>
      {children}
    </AuthCtx.Provider>
  );
}

export function useAuth(): AuthState {
  const ctx = useContext(AuthCtx);
  if (!ctx) throw new Error("useAuth must be used within AuthProvider");
  return ctx;
}

// 투자 AI 어시스턴트 — 최소 PWA 서비스 워커.
// 앱 셸(same-origin GET)만 network-first + 캐시 fallback. API/타 오리진은 손대지 않음.
const CACHE = "etfrag-shell-v1";

self.addEventListener("install", (event) => {
  self.skipWaiting();
  event.waitUntil(
    caches.open(CACHE).then((c) => c.addAll(["/", "/manifest.webmanifest"]).catch(() => {})),
  );
});

self.addEventListener("activate", (event) => {
  event.waitUntil(
    caches
      .keys()
      .then((keys) =>
        Promise.all(keys.filter((k) => k !== CACHE).map((k) => caches.delete(k))),
      )
      .then(() => self.clients.claim()),
  );
});

self.addEventListener("fetch", (event) => {
  const req = event.request;
  const url = new URL(req.url);

  // GET + same-origin만 처리. POST/SSE, 백엔드 API(다른 오리진)는 그대로 통과.
  if (req.method !== "GET" || url.origin !== self.location.origin) return;

  // network-first: 최신 우선, 오프라인이면 캐시 fallback (앱 셸 유지).
  event.respondWith(
    fetch(req)
      .then((res) => {
        const copy = res.clone();
        caches.open(CACHE).then((c) => c.put(req, copy)).catch(() => {});
        return res;
      })
      .catch(() => caches.match(req).then((cached) => cached || caches.match("/"))),
  );
});

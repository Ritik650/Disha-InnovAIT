// Single API client. In dev, vite proxies /api -> http://localhost:8000.
// In production builds the same /api path is served behind nginx.
const BASE = "/api";

async function getJson(path) {
  const r = await fetch(BASE + path);
  if (!r.ok) throw new Error(`${r.status} ${r.statusText}: ${path}`);
  return r.json();
}

async function postJson(path, body) {
  const r = await fetch(BASE + path, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
  });
  if (!r.ok) throw new Error(`${r.status} ${r.statusText}: ${path}`);
  return r.json();
}

export const api = {
  health: () => getJson("/health"),
  qini: () => getJson("/qini"),
  businessCase: () => getJson("/business_case"),
  plan: (repId, { date = "2025-11-17", arm = "real" } = {}) =>
    getJson(`/plan/${repId}?date=${date}&arm=${arm}`),
  retailerWhy: (retailerId, { date = "2025-11-17", arm = "real" } = {}) =>
    getJson(`/retailer/${retailerId}/why?date=${date}&arm=${arm}`),
  territorySignals: (repId, { date = "2025-11-17" } = {}) =>
    getJson(`/territory/${repId}/signals?date=${date}`),
  simRun: ({ arm = "real", date = "2025-11-17" } = {}) =>
    getJson(`/sim/run?arm=${arm}&date=${date}`),
  anomalies: ({ date = "2025-11-17", limit = 20 } = {}) =>
    getJson(`/anomalies?date=${date}&limit=${limit}`),
  learning: () => getJson("/learning"),
  postOutcomes: (outcomes) => postJson("/outcomes", { outcomes }),
};

import type { ORMessage } from "./openrouter";

/** Rolling conversation memory, persisted so it survives page reloads. */

const KEY = "octa-history";
const MAX_MESSAGES = 24; // last 12 exchanges

export function loadHistory(): ORMessage[] {
  try {
    return JSON.parse(localStorage.getItem(KEY) ?? "[]");
  } catch {
    return [];
  }
}

export function remember(role: "user" | "assistant", content: string) {
  const history = loadHistory();
  history.push({ role, content });
  localStorage.setItem(KEY, JSON.stringify(history.slice(-MAX_MESSAGES)));
}

export function clearHistory() {
  localStorage.removeItem(KEY);
}

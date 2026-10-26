import { loadHistory } from "./history";
import { settings } from "./openrouter";

/**
 * Prototype: TypeSafe's Jev (a "System One" decision model) judges whether a
 * captured utterance was meant for Octa or said to the other people in the
 * room. Called through OpenRouter's System One API with the same OpenRouter
 * key as everything else; the camera's findings go in as state fields.
 *
 * Modes: "off", "log" (ask and record, never drop), "gate" (drop clear no's).
 * Every verdict lands in jevLog (window.octa.jevLog) and the console.
 */

export type JevMode = "off" | "log" | "gate";

export const jevSettings = {
  get mode(): JevMode {
    const m = localStorage.getItem("octa-jev-mode");
    return m === "log" || m === "gate" ? m : "off";
  },
  set mode(v: JevMode) { localStorage.setItem("octa-jev-mode", v); },
};

const ENDPOINT = "https://openrouter.ai/api/v1/systemone";
const MODEL = "typesafe/jev-1.14"; // pinned: thresholds below are tuned against one version
const DROP_BELOW = 0.3; // noul under this = clearly not for Octa
const TIMEOUT_MS = 2500; // slower than this and we act as if Jev said yes

export interface JevVerdict {
  at: string; // ISO time
  utterance: string;
  /** Probability the utterance was meant for Octa; null if the call failed. */
  noul: number | null;
  addressed: boolean; // what the app acted on
  dropped: boolean;
  latencyMs: number;
  model?: string;
  inputTokens?: number;
  costUsd?: number;
  error?: string;
}

export const jevLog: JevVerdict[] = [];

export interface RoomContext {
  speaker: number | null; // floor holder, if any
  facingOcta: boolean | null; // null = no camera judgement
  lipsMoving: boolean | null;
  peopleInView: number;
}

/** Ask Jev whether `utterance` was addressed to Octa. Never throws. */
export async function checkAddressee(utterance: string, room: RoomContext): Promise<JevVerdict> {
  const started = performance.now();
  const base = { at: new Date().toISOString(), utterance };

  // recent turns help with follow-ups ("and the second one?") that only make sense as replies
  const recent = loadHistory()
    .slice(-4)
    .map((m) => `${m.role === "user" ? "Person" : "Octa"}: ${typeof m.content === "string" ? m.content.slice(0, 280) : ""}`);

  const ctrl = new AbortController();
  const timer = setTimeout(() => ctrl.abort(), TIMEOUT_MS);
  try {
    const res = await fetch(ENDPOINT, {
      method: "POST",
      signal: ctrl.signal,
      headers: {
        Authorization: `Bearer ${settings.key}`,
        "Content-Type": "application/json",
        "X-Title": "Octa",
      },
      body: JSON.stringify({
        model: MODEL,
        state: {
          setting:
            "A meeting room. Octa is a voice assistant on a shared screen; several people may be talking to each other. One microphone captured the utterance below.",
          utterance,
          speaker: room.speaker !== null ? `P${room.speaker}, who holds the floor` : "unknown",
          speaker_facing_octa: room.facingOcta,
          speaker_lips_moving: room.lipsMoving,
          people_in_view: room.peopleInView,
          recent_conversation_with_octa: recent,
        },
        questions: {
          addressed_to_octa: {
            type: "noul",
            instructions:
              "Was the utterance addressed to Octa, the voice assistant — as a question, request, or command for it — rather than said to other people in the room?",
            criteria: {
              true: "Asks Octa something, tells it to do or show something, or follows up on what Octa just said",
              false: "Remarks, asides, or questions aimed at colleagues; thinking aloud; talk about Octa rather than to it",
            },
          },
        },
      }),
    });
    const latencyMs = Math.round(performance.now() - started);
    if (!res.ok) {
      const detail = (await res.text()).slice(0, 200);
      return record({ ...base, noul: null, addressed: true, dropped: false, latencyMs, error: `HTTP ${res.status}: ${detail}` });
    }
    const data = await res.json();
    const noul: number | null = data?.answers?.addressed_to_octa?.noul ?? null;
    const dropped = jevSettings.mode === "gate" && noul !== null && noul < DROP_BELOW;
    return record({
      ...base, noul, addressed: !dropped, dropped, latencyMs,
      model: data?.model, inputTokens: data?.usage?.input_tokens, costUsd: data?.usage?.cost,
    });
  } catch (err) {
    const latencyMs = Math.round(performance.now() - started);
    const error = ctrl.signal.aborted ? `timeout after ${TIMEOUT_MS}ms` : String(err);
    return record({ ...base, noul: null, addressed: true, dropped: false, latencyMs, error });
  } finally {
    clearTimeout(timer);
  }
}

function record(v: JevVerdict): JevVerdict {
  jevLog.push(v);
  const p = v.noul === null ? "—" : v.noul.toFixed(2);
  console.info(
    `[jev] ${v.dropped ? "DROP" : "pass"} p(for Octa)=${p} ${v.latencyMs}ms${v.error ? ` (${v.error})` : ""} — "${v.utterance}"`
  );
  return v;
}

/** The log as CSV, for pasting into a sheet. */
export function jevLogCsv(): string {
  const cols: (keyof JevVerdict)[] = ["at", "utterance", "noul", "addressed", "dropped", "latencyMs", "model", "inputTokens", "costUsd", "error"];
  const esc = (x: unknown) => `"${String(x ?? "").replace(/"/g, '""')}"`;
  return [cols.join(","), ...jevLog.map((v) => cols.map((c) => esc(v[c])).join(","))].join("\n");
}

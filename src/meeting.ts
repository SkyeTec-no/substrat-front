import type { MicMeter } from "./speech";
import { toWavBase64, transcribe } from "./speech";
import type { VoiceBook } from "./voices";
import { chat, settings } from "./openrouter";

/**
 * Meeting log: Octa follows the whole meeting, not just what's said to it.
 * The room mic is cut into utterances (level VAD with pre-roll), each one is
 * transcribed and tied to a person — by voiceprint first, by whose lips were
 * moving second. Octa still only *answers* the floor holder, but every
 * answer is given with the meeting so far in mind, and older parts are
 * folded into a running summary so the context stays small.
 *
 * Off by default: it sends everything said in the room to the transcription
 * model, so it's switched on deliberately and shown on screen while it runs.
 */

export interface LogLine {
  at: number; // Date.now()
  who: string; // "P2", "Octa", or "?"
  text: string;
  toOcta?: boolean; // a request to Octa (vs. talk in the room)
  /** How many people's lips moved during this utterance (2+ = people talking over each other). */
  talkers?: number;
}

const START_LEVEL = 0.1; // mic level that opens an utterance
const END_QUIET_MS = 800; // this much quiet closes it
const MAX_SECONDS = 15;
const MIN_SECONDS = 0.6;
const RECENT_CHARS = 3500; // verbatim transcript handed to the model
const SUMMARIZE_AFTER = 6000; // fold older lines into the summary past this

export class MeetingLog {
  /** Keep minutes: store every line and hand the meeting to the model. */
  enabled = false;
  /** Listen for "Okta" in the room: transcribe, check, and drop (unless minutes are on). */
  wake = false;
  lines: LogLine[] = [];
  summary = "";
  private summarized = 0; // lines[0..summarized) are folded into the summary
  private talkers: { t: number; ids: number[] }[] = [];
  private seg: { blocks: Float32Array[]; t0: number; lastLoud: number } | null = null;
  private mic: MicMeter;
  private voices: VoiceBook | null;
  private paused: () => boolean;
  private summarizing = false;
  /** Every transcribed room line, as it lands (the app listens for "Okta"). */
  onLine: (line: LogLine) => void = () => {};

  constructor(mic: MicMeter, voices: VoiceBook | null, paused: () => boolean) {
    this.mic = mic;
    this.voices = voices;
    this.paused = paused;
    mic.listen((b) => {
      if (this.seg) this.seg.blocks.push(b);
    });
  }

  /** ~10×/s: who the camera sees talking right now (lip movement). */
  noteTalkers(ids: number[]) {
    const t = performance.now();
    this.talkers.push({ t, ids });
    while (this.talkers.length && t - this.talkers[0].t > 30_000) this.talkers.shift();
  }

  /** ~10×/s: the VAD. Pauses while Octa talks or takes a command itself. */
  tick(level: number) {
    if (!this.enabled && !this.wake) return;
    const now = performance.now();
    if (this.paused()) {
      this.seg = null; // Octa's own voice and its command captures aren't room talk
      return;
    }
    if (!this.seg) {
      if (level >= START_LEVEL) this.seg = { blocks: this.mic.preroll(0.4), t0: now - 400, lastLoud: now };
      return;
    }
    if (level >= START_LEVEL) this.seg.lastLoud = now;
    const len = (now - this.seg.t0) / 1000;
    if (now - this.seg.lastLoud > END_QUIET_MS || len > MAX_SECONDS) {
      const seg = this.seg;
      this.seg = null;
      if ((seg.lastLoud - seg.t0) / 1000 >= MIN_SECONDS) void this.finish(seg.blocks, seg.t0, now);
    }
  }

  /** Lines the app writes itself: requests to Octa and Octa's answers. */
  add(line: Omit<LogLine, "at">) {
    if (!this.enabled) return;
    this.lines.push({ at: Date.now(), ...line });
    void this.maybeSummarize();
  }

  /** The meeting so far, for the model: summary + recent lines verbatim. */
  context(): string {
    if (!this.lines.length && !this.summary) return "";
    const recent: string[] = [];
    let chars = 0;
    for (let i = this.lines.length - 1; i >= this.summarized; i--) {
      const l = this.lines[i];
      const row = `[${clock(l.at)}] ${l.who}${l.toOcta ? " → Octa" : ""}: ${l.text}`;
      if (chars + row.length > RECENT_CHARS) break;
      recent.unshift(row);
      chars += row.length;
    }
    return (
      (this.summary ? `Earlier in the meeting (summary):\n${this.summary}\n\n` : "") +
      (recent.length ? `Recent, verbatim:\n${recent.join("\n")}` : "")
    );
  }

  private async finish(blocks: Float32Array[], t0: number, t1: number) {
    const rate = this.mic.sampleRate;
    try {
      const [text, voice] = await Promise.all([
        transcribe(toWavBase64(blocks, rate)),
        this.voices?.identifyPcm(blocks) ?? Promise.resolve(null),
      ]);
      if (!text || text.length < 2) return;
      // who: a recognised voice first; else whoever's lips moved most meanwhile
      const counts = new Map<number, number>();
      for (const s of this.talkers) {
        if (s.t < t0 || s.t > t1) continue;
        for (const id of s.ids) counts.set(id, (counts.get(id) ?? 0) + 1);
      }
      let who = voice?.id != null ? `P${voice.id}` : "";
      if (!who) {
        const top = [...counts].sort((a, b) => b[1] - a[1])[0];
        who = top ? `P${top[0]}` : "?";
      }
      // several mouths moving for a fair share of the utterance = talking over each other
      const frames = this.talkers.filter((s) => s.t >= t0 && s.t <= t1).length || 1;
      const talkers = [...counts.values()].filter((n) => n / frames > 0.25).length;
      const line = { at: Date.now() - Math.round(performance.now() - t0), who, text, talkers };
      if (this.enabled) this.lines.push(line);
      this.onLine(line);
      console.info(`[meeting] ${who}: ${text}`);
      void this.maybeSummarize();
    } catch (err) {
      console.warn("meeting transcription failed", err);
    }
  }

  /** Fold the oldest unsummarized lines into the running summary. */
  private async maybeSummarize() {
    if (this.summarizing || !settings.key) return;
    const pending = this.lines.slice(this.summarized);
    const chars = pending.reduce((n, l) => n + l.text.length + 12, 0);
    if (chars < SUMMARIZE_AFTER) return;
    // keep the newest ~third verbatim; fold the rest
    const fold = pending.slice(0, Math.floor(pending.length * 0.66));
    this.summarizing = true;
    try {
      const r = await chat(
        [
          {
            role: "system",
            content:
              "You keep the running minutes of a meeting. Merge the existing summary with the new transcript lines " +
              "into one updated summary in Norwegian bokmål: who said what that matters, decisions, open questions, " +
              "numbers and names. Keep speaker labels (P1, P2, …). Under 250 words, plain text, no preamble.",
          },
          {
            role: "user",
            content:
              `Existing summary:\n${this.summary || "(none)"}\n\nNew lines:\n` +
              fold.map((l) => `[${clock(l.at)}] ${l.who}: ${l.text}`).join("\n"),
          },
        ],
        settings.fastModel
      );
      if (r.text.trim()) {
        this.summary = r.text.trim();
        this.summarized += fold.length;
      }
    } catch (err) {
      console.warn("meeting summary failed", err);
    } finally {
      this.summarizing = false;
    }
  }
}

function clock(ms: number): string {
  return new Date(ms).toLocaleTimeString("nb-NO", { hour: "2-digit", minute: "2-digit" });
}

import type { MicMeter } from "./speech";
import { embed, cosine, to16k, preloadVoiceModel } from "./voiceprint";

/**
 * Octa's ear for who is speaking. It learns each person's voice by itself:
 * whenever the camera sees exactly one person's lips moving while the mic
 * hears speech, that audio is theirs. Every ~3 s of it refines their
 * voiceprint (a running average of embeddings). Afterwards the voice alone
 * says who spoke — even when their face is turned away — and a voice that
 * matches nobody in view (someone behind the camera) stands out.
 *
 * Voiceprints live in memory for the session only.
 */

const ENROL_SECONDS = 3; // speech per refinement
const MATCH = 0.55; // cosine similarity that counts as "that's them"
const KEEP_SECONDS = 8; // mic history kept for identification
const LIVE_EVERY = 1500; // ms between live "who's talking" checks

export interface VoiceMatch {
  id: number | null; // best-matching person, null if nobody passes MATCH
  score: number;
  scores: Map<number, number>;
}

export class VoiceBook {
  private prints = new Map<number, { sum: Float32Array; n: number }>();
  private enrol = new Map<number, { blocks: Float32Array[]; len: number }>();
  private history: { t: number; pcm: Float32Array }[] = [];
  private lastObserve = performance.now();
  private busy = false;
  private liveAt = 0;
  /** Who the voice says is talking right now (live, ~1.5 s windows). */
  live: { id: number | null; at: number } = { id: null, at: 0 };

  private mic: MicMeter;

  constructor(mic: MicMeter) {
    this.mic = mic;
    preloadVoiceModel();
    mic.listen((pcm) => {
      const t = performance.now();
      this.history.push({ t, pcm });
      while (this.history.length && t - this.history[0].t > KEEP_SECONDS * 1000) this.history.shift();
    });
  }

  /** A number went to someone new: whatever was learned under it isn't theirs. */
  forget(id: number) {
    this.prints.delete(id);
    this.enrol.delete(id);
  }

  /** How many refinements a person's voiceprint has had (0 = unknown voice). */
  known(id: number): number {
    return this.prints.get(id)?.n ?? 0;
  }

  /** Called ~10×/s with the people whose lips are visibly moving right now. */
  observe(talking: number[], level: number) {
    const now = performance.now();
    const since = this.lastObserve;
    this.lastObserve = now;
    // only unambiguous moments teach: one talker, and actual speech on the mic
    if (talking.length !== 1 || level < 0.1) return;
    const id = talking[0];
    const e = this.enrol.get(id) ?? { blocks: [], len: 0 };
    for (const h of this.history) {
      if (h.t > since && h.t <= now) {
        e.blocks.push(h.pcm);
        e.len += h.pcm.length;
      }
    }
    this.enrol.set(id, e);
    if (e.len >= ENROL_SECONDS * this.mic.sampleRate && !this.busy) {
      this.enrol.delete(id);
      void this.refine(id, e.blocks);
    }
  }

  private async refine(id: number, blocks: Float32Array[]) {
    this.busy = true;
    try {
      const v = await embed(to16k(join(blocks), this.mic.sampleRate));
      if (!v) return;
      const p = this.prints.get(id) ?? { sum: new Float32Array(v.length), n: 0 };
      const len = Math.hypot(...v) || 1;
      for (let i = 0; i < v.length; i++) p.sum[i] += v[i] / len;
      p.n = Math.min(p.n + 1, 99);
      this.prints.set(id, p);
      console.info(`[voice] learned P${id} (${p.n} sample${p.n > 1 ? "s" : ""})`);
    } finally {
      this.busy = false;
    }
  }

  /** Whose voice was on the mic since `since` (performance.now()), last ~5 s of it. */
  async identify(since: number): Promise<VoiceMatch | null> {
    if (!this.prints.size) return null;
    const blocks = this.history.filter((h) => h.t >= since).map((h) => h.pcm);
    const rate = this.mic.sampleRate;
    let pcm = join(blocks);
    if (pcm.length > rate * 5) pcm = pcm.subarray(pcm.length - rate * 5);
    const v = await embed(to16k(pcm, rate));
    return v ? this.match(v) : null;
  }

  /** A live guess at who's talking, for the tiles. Cheap to call every frame. */
  tickLive(level: number) {
    const now = performance.now();
    if (!this.prints.size || this.busy || level < 0.1 || now - this.liveAt < LIVE_EVERY) return;
    this.liveAt = now;
    this.busy = true;
    void this.identify(now - LIVE_EVERY)
      .then((m) => {
        this.live = { id: m?.id ?? null, at: performance.now() };
      })
      .finally(() => (this.busy = false));
  }

  private match(v: Float32Array): VoiceMatch {
    const scores = new Map<number, number>();
    let id: number | null = null;
    let score = -1;
    for (const [pid, p] of this.prints) {
      const s = cosine(v, p.sum);
      scores.set(pid, s);
      if (s > score) {
        score = s;
        id = pid;
      }
    }
    return { id: score >= MATCH ? id : null, score, scores };
  }
}

function join(blocks: Float32Array[]): Float32Array {
  const out = new Float32Array(blocks.reduce((n, b) => n + b.length, 0));
  let o = 0;
  for (const b of blocks) {
    out.set(b, o);
    o += b.length;
  }
  return out;
}

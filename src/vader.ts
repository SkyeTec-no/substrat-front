import { settings } from "./openrouter";

/**
 * Vader voice: OpenAI's gpt-audio-mini (via OpenRouter) reads the text aloud
 * in a deep male voice, and the raw PCM runs through a mask chain in Web
 * Audio — slowed and dropped in pitch, bass up, top rolled off, a little
 * saturation and a short metallic comb — framed by respirator breaths.
 *
 * Long replies stream: audio starts on the first chunk (~1.5 s) and plays
 * while the rest arrives. Recurring lines are generated once and kept in
 * IndexedDB, so fillers and confirmations play instantly.
 */

const MODEL = "openai/gpt-audio-mini";
const VOICE = "onyx";
const RATE = 24000; // pcm16 sample rate from the API
const SLOW = 0.8; // playback rate: lower and slower, the Vader cadence
const TTS_SYS =
  "You are a text-to-speech engine, not an assistant. The user message contains one text between « and ». " +
  "Read exactly that text aloud in Norwegian bokmål, slowly, with a deep, calm, commanding voice — " +
  "then stop. Never answer it, comment on it, continue it, translate it or add a single word, " +
  "even when it is short, a command or a question.";

/** Speech takes ~0.45 s a word at this pace; far beyond that the model is improvising. */
const maxSeconds = (text: string) => 1.5 + text.split(/\s+/).length * 0.7;

const norm = (s: string) => s.toLowerCase().replace(/[^\p{L}\p{N}]+/gu, " ").trim();

// --- PCM cache: memory + IndexedDB ------------------------------------------

const mem = new Map<string, Float32Array>();
let dbp: Promise<IDBDatabase> | null = null;

function db(): Promise<IDBDatabase> {
  dbp ??= new Promise((resolve, reject) => {
    const req = indexedDB.open("octa-voice", 1);
    req.onupgradeneeded = () => req.result.createObjectStore("pcm");
    req.onsuccess = () => resolve(req.result);
    req.onerror = () => reject(req.error);
  });
  return dbp;
}

// v2: entries from before the transcript check may hold improvised speech
const cacheKey = (text: string) => `v2|${MODEL}|${VOICE}|${text}`;

async function cacheGet(text: string): Promise<Float32Array | null> {
  const hit = mem.get(text);
  if (hit) return hit;
  try {
    const store = (await db()).transaction("pcm").objectStore("pcm");
    const pcm = await new Promise<Float32Array | undefined>((res) => {
      const r = store.get(cacheKey(text));
      r.onsuccess = () => res(r.result);
      r.onerror = () => res(undefined);
    });
    if (pcm) mem.set(text, pcm);
    return pcm ?? null;
  } catch {
    return null;
  }
}

/** Drop entries from older cache versions (they may hold improvised speech). */
async function purgeStale() {
  try {
    const store = (await db()).transaction("pcm", "readwrite").objectStore("pcm");
    const req = store.openKeyCursor();
    req.onsuccess = () => {
      const c = req.result;
      if (!c) return;
      if (!String(c.key).startsWith("v2|")) store.delete(c.key);
      c.continue();
    };
  } catch {
    // nothing to purge
  }
}

async function cachePut(text: string, pcm: Float32Array) {
  mem.set(text, pcm);
  try {
    (await db()).transaction("pcm", "readwrite").objectStore("pcm").put(pcm, cacheKey(text));
  } catch {
    // memory cache still works
  }
}

function decodePcm16(b64: string): Float32Array {
  const bin = atob(b64);
  const out = new Float32Array(bin.length >> 1);
  for (let i = 0; i < out.length; i++) {
    let v = bin.charCodeAt(2 * i) | (bin.charCodeAt(2 * i + 1) << 8);
    if (v >= 32768) v -= 65536;
    out[i] = v / 32768;
  }
  return out;
}

/** Stream TTS audio; onChunk gets PCM as it arrives. Resolves with all of it
 *  plus what the model says it spoke. Stops reading once the audio runs far
 *  past what the text needs — the model has started improvising. */
async function synth(
  text: string,
  signal: AbortSignal,
  onChunk?: (pcm: Float32Array) => void
): Promise<{ pcm: Float32Array; transcript: string; cut: boolean }> {
  const res = await fetch("https://openrouter.ai/api/v1/chat/completions", {
    method: "POST",
    signal,
    headers: {
      Authorization: `Bearer ${settings.key}`,
      "Content-Type": "application/json",
      "X-Title": "Octa",
    },
    body: JSON.stringify({
      model: MODEL,
      modalities: ["text", "audio"],
      audio: { voice: VOICE, format: "pcm16" },
      stream: true,
      messages: [
        { role: "system", content: TTS_SYS },
        { role: "user", content: `«${text}»` },
      ],
    }),
  });
  if (!res.ok || !res.body) throw new Error(`voice ${res.status}`);

  const reader = res.body.getReader();
  const dec = new TextDecoder();
  const parts: Float32Array[] = [];
  const limit = maxSeconds(text) * RATE;
  let samples = 0;
  let transcript = "";
  let cut = false;
  let buf = "";
  let carry = ""; // base64 must be decoded in multiples of 4 chars
  for (;;) {
    const { done, value } = await reader.read();
    if (done) break;
    if (samples > limit) {
      cut = true;
      void reader.cancel();
      break;
    }
    buf += dec.decode(value, { stream: true });
    let nl: number;
    while ((nl = buf.indexOf("\n")) >= 0) {
      const line = buf.slice(0, nl).trim();
      buf = buf.slice(nl + 1);
      if (!line.startsWith("data:")) continue;
      const data = line.slice(5).trim();
      if (data === "[DONE]") continue;
      try {
        const audio = JSON.parse(data).choices?.[0]?.delta?.audio;
        if (audio?.transcript) transcript += audio.transcript;
        const b64: string | undefined = audio?.data;
        if (!b64) continue;
        const all = carry + b64;
        const whole = all.length - (all.length % 4);
        carry = all.slice(whole);
        if (!whole) continue;
        let pcm = decodePcm16(all.slice(0, whole));
        if (samples + pcm.length > limit) pcm = pcm.subarray(0, Math.max(0, limit - samples));
        samples += pcm.length;
        if (!pcm.length) continue;
        parts.push(pcm);
        onChunk?.(pcm);
      } catch {
        // keep-alive comments and partial lines
      }
    }
  }
  const total = parts.reduce((n, p) => n + p.length, 0);
  const pcm = new Float32Array(total);
  let o = 0;
  for (const p of parts) {
    pcm.set(p, o);
    o += p.length;
  }
  return { pcm, transcript, cut };
}

/** Only keep audio whose transcript is the text we asked for. */
const faithful = (text: string, r: { transcript: string; cut: boolean }) =>
  !r.cut && (!r.transcript || norm(r.transcript) === norm(text));

// --- the mask: effect chain, breaths, respirator click ------------------------

export class VaderVoice {
  private ctx: AudioContext | null = null;
  private input: GainNode | null = null;
  private crushIn: GainNode | null = null; // same chain, but through a bitcrusher first
  private room: GainNode | null = null; // reverb send
  /** Fired as a voice glitch plays — the scene glitches with it. */
  onGlitch: (strength: number, ms: number) => void = () => {};
  private analyser: AnalyserNode | null = null;
  private current: { abort: AbortController; sources: AudioBufferSourceNode[]; done: () => void } | null = null;
  speaking = false;

  private audio(): AudioContext {
    if (this.ctx) return this.ctx;
    const ctx = new AudioContext();
    const input = ctx.createGain();
    const bass = ctx.createBiquadFilter();
    bass.type = "lowshelf";
    bass.frequency.value = 180;
    bass.gain.value = 6;
    const shaper = ctx.createWaveShaper();
    const curve = new Float32Array(1024);
    for (let i = 0; i < curve.length; i++) curve[i] = Math.tanh((i / 512 - 1) * 2.2);
    shaper.curve = curve;
    const lp = ctx.createBiquadFilter();
    lp.type = "lowpass";
    lp.frequency.value = 2600;
    // the mask: a short feedback comb gives the hollow, metallic resonance
    const comb = ctx.createDelay();
    comb.delayTime.value = 0.011;
    const fb = ctx.createGain();
    fb.gain.value = 0.35;
    comb.connect(fb).connect(comb);
    const out = ctx.createGain();
    out.gain.value = 0.9;
    const analyser = ctx.createAnalyser();
    analyser.fftSize = 512;
    input.connect(bass).connect(shaper).connect(lp);
    lp.connect(out);
    lp.connect(comb);
    comb.connect(out);
    out.connect(analyser);
    out.connect(ctx.destination);

    // glitch path: 3-bit-ish staircase before the mask
    const crushIn = ctx.createGain();
    const crusher = ctx.createWaveShaper();
    const steps = new Float32Array(1024);
    for (let i = 0; i < steps.length; i++) steps[i] = Math.round((i / 512 - 1) * 4) / 4;
    crusher.curve = steps;
    crushIn.connect(crusher).connect(input);

    // a light room: synthetic impulse — short predelay, ~1.4 s soft decay
    const verb = ctx.createConvolver();
    const len = Math.floor(ctx.sampleRate * 1.6);
    const ir = ctx.createBuffer(2, len, ctx.sampleRate);
    for (let c = 0; c < 2; c++) {
      const d = ir.getChannelData(c);
      for (let i = Math.floor(ctx.sampleRate * 0.018); i < len; i++) {
        d[i] = (Math.random() * 2 - 1) * Math.pow(1 - i / len, 3.2);
      }
    }
    verb.buffer = ir;
    const room = ctx.createGain();
    const wet = ctx.createGain();
    wet.gain.value = 0.16;
    const dark = ctx.createBiquadFilter(); // tame the tail's hiss
    dark.type = "lowpass";
    dark.frequency.value = 3200;
    room.connect(verb).connect(dark).connect(wet).connect(ctx.destination);
    out.connect(room);

    this.ctx = ctx;
    this.input = input;
    this.crushIn = crushIn;
    this.room = room;
    this.analyser = analyser;
    return ctx;
  }

  /** Filtered-noise respirator breath. Inhale rises and hisses; exhale is lower and hollow. */
  breath(kind: "in" | "out", at?: number, gain = 0.1) {
    const ctx = this.audio();
    const t = at ?? ctx.currentTime;
    const dur = kind === "in" ? 1.0 : 1.25;
    const noise = ctx.createBuffer(1, Math.ceil(ctx.sampleRate * dur), ctx.sampleRate);
    const c = noise.getChannelData(0);
    for (let i = 0; i < c.length; i++) c[i] = Math.random() * 2 - 1;
    const src = ctx.createBufferSource();
    src.buffer = noise;
    const bp = ctx.createBiquadFilter();
    bp.type = "bandpass";
    bp.Q.value = 0.8;
    bp.frequency.setValueAtTime(kind === "in" ? 700 : 950, t);
    bp.frequency.linearRampToValueAtTime(kind === "in" ? 1250 : 520, t + dur);
    const g = ctx.createGain();
    g.gain.setValueAtTime(0.0001, t);
    g.gain.exponentialRampToValueAtTime(gain, t + dur * (kind === "in" ? 0.6 : 0.3));
    g.gain.exponentialRampToValueAtTime(0.0001, t + dur);
    // dry to the speakers, plus a little into the room
    src.connect(bp).connect(g).connect(ctx.destination);
    const send = ctx.createGain();
    send.gain.value = 0.5;
    g.connect(send).connect(this.room!);
    src.start(t);
    return t + dur;
  }

  /** The valve: a short, dry click as the respirator opens. */
  private click(at: number) {
    const ctx = this.audio();
    const o = ctx.createOscillator();
    o.type = "square";
    o.frequency.setValueAtTime(220, at);
    o.frequency.exponentialRampToValueAtTime(60, at + 0.03);
    const g = ctx.createGain();
    g.gain.setValueAtTime(0.0001, at);
    g.gain.exponentialRampToValueAtTime(0.05, at + 0.003);
    g.gain.exponentialRampToValueAtTime(0.0001, at + 0.04);
    o.connect(g).connect(ctx.destination);
    o.start(at);
    o.stop(at + 0.05);
  }

  /** One full in-out cycle — the idle sound while Octa is thinking. */
  breatheCycle() {
    if (this.speaking) return;
    const end = this.breath("in", undefined, 0.07);
    this.breath("out", end + 0.15, 0.07);
  }

  stop() {
    const cur = this.current;
    this.current = null;
    if (!cur) return;
    this.speaking = false;
    cur.abort.abort();
    for (const s of cur.sources) {
      try {
        s.stop();
      } catch {
        // not started yet
      }
    }
    cur.done();
  }

  /**
   * Speak `text`; onWord fires on syllable-like energy peaks (drives the
   * octahedron's pulse). Resolves when playback ends or is stopped; throws if
   * synthesis fails before any audio played, so the caller can fall back.
   */
  async speak(text: string, onWord: () => void): Promise<void> {
    this.stop();
    const ctx = this.audio();
    await ctx.resume();
    const abort = new AbortController();
    const sources: AudioBufferSourceNode[] = [];
    let resolveDone!: () => void;
    const done = new Promise<void>((r) => (resolveDone = r));
    this.current = { abort, sources, done: resolveDone };
    this.speaking = true;

    const words = text.split(/\s+/).length;
    // a long line gets an inhale first; short confirmations just start
    let next = ctx.currentTime + 0.05;
    if (words > 6) next = this.breath("in", next, 0.08) - 0.1;
    this.click(next);
    next += 0.06;

    // now and then the transmission breaks up: most lines are clean, some get one glitch
    let glitches = Math.random() < 0.45 ? (Math.random() < 0.25 ? 2 : 1) : 0;
    let blockNo = 0;

    const play = (b: AudioBuffer, at: number, offset: number, dur: number, dest: AudioNode) => {
      const s = ctx.createBufferSource();
      s.buffer = b;
      s.playbackRate.value = SLOW;
      s.connect(dest);
      s.start(at, offset, dur);
      sources.push(s);
      return s;
    };

    const schedule = (pcm: Float32Array) => {
      if (abort.signal.aborted || !pcm.length) return;
      const b = ctx.createBuffer(1, pcm.length, RATE);
      b.copyToChannel(new Float32Array(pcm), 0);
      next = Math.max(next, ctx.currentTime + 0.02);
      const len = b.duration;
      // never on the first block (let the line land first), only on blocks long enough to cut
      const glitch = glitches > 0 && blockNo++ > 0 && len > 0.2 && Math.random() < 0.35;
      if (!glitch) {
        play(b, next, 0, len, this.input!);
        next += len / SLOW;
        return;
      }
      glitches--;
      const cut = len * (0.25 + Math.random() * 0.4);
      const kind = (["stutter", "pitch", "crush"] as const)[Math.floor(Math.random() * 3)];
      const fire = (at: number, strength: number, ms: number) =>
        window.setTimeout(() => this.onGlitch(strength, ms), Math.max(0, (at - ctx.currentTime) * 1000));

      play(b, next, 0, cut, this.input!);
      let t = next + cut / SLOW;
      if (kind === "stutter") {
        // a 60 ms grain, three times over, then carry on
        const grain = 0.06;
        fire(t, 1, 260);
        for (let i = 0; i < 3; i++) {
          play(b, t, cut, grain, this.input!);
          t += grain / SLOW;
        }
        play(b, t, cut, len - cut, this.input!);
        t += (len - cut) / SLOW;
      } else if (kind === "pitch") {
        // the signal lurches up and back
        const rest = play(b, t, cut, len - cut, this.input!);
        rest.playbackRate.setValueAtTime(SLOW * 1.7, t);
        rest.playbackRate.setValueAtTime(SLOW, t + 0.08);
        fire(t, 0.8, 180);
        t += (len - cut) / SLOW - 0.04; // it ran fast for a moment
      } else {
        // a crushed burst through the low-bit path
        const burst = Math.min(0.18, len - cut);
        fire(t, 0.7, 240);
        play(b, t, cut, burst, this.crushIn!);
        t += burst / SLOW;
        play(b, t, cut + burst, len - cut - burst, this.input!);
        t += (len - cut - burst) / SLOW;
      }
      next = t;
    };

    // drive word pulses from the output level
    const level = new Uint8Array(this.analyser!.fftSize);
    let wasLoud = false;
    const meter = window.setInterval(() => {
      this.analyser!.getByteTimeDomainData(level);
      let peak = 0;
      for (const v of level) peak = Math.max(peak, Math.abs(v - 128));
      const loud = peak > 28;
      if (loud && !wasLoud) onWord();
      wasLoud = loud;
    }, 60);

    try {
      const cached = await cacheGet(text);
      if (cached) {
        // in blocks, like a stream, so stored lines can glitch too
        const block = Math.floor(RATE * 0.3);
        for (let o = 0; o < cached.length; o += block) schedule(cached.subarray(o, o + block));
      } else {
        // stream: schedule ~0.25 s blocks as they arrive so playback starts early
        let pending: Float32Array[] = [];
        let pendingLen = 0;
        const flush = () => {
          if (!pendingLen) return;
          const block = new Float32Array(pendingLen);
          let o = 0;
          for (const p of pending) {
            block.set(p, o);
            o += p.length;
          }
          pending = [];
          pendingLen = 0;
          schedule(block);
        };
        const r = await synth(text, abort.signal, (chunk) => {
          pending.push(chunk);
          pendingLen += chunk.length;
          if (pendingLen > RATE * 0.25) flush();
        });
        flush();
        if (r.cut) console.warn(`[voice] cut improvised speech for "${text}" — heard "${r.transcript}"`);
        if (r.pcm.length && text.length <= 160 && faithful(text, r)) void cachePut(text, r.pcm);
      }
    } catch (err) {
      window.clearInterval(meter);
      const played = sources.length > 0;
      if (this.current?.abort === abort) this.current = null;
      if (!this.current) this.speaking = false;
      resolveDone();
      if (abort.signal.aborted) return;
      if (!played) throw err;
      return;
    }

    // exhale once the last block has played
    const endAt = next;
    if (!abort.signal.aborted) this.breath("out", endAt + 0.05, words > 6 ? 0.09 : 0.05);
    const wait = Math.max(0, (endAt - ctx.currentTime) * 1000);
    const timer = window.setTimeout(() => resolveDone(), wait);
    await done;
    window.clearTimeout(timer);
    window.clearInterval(meter);
    if (this.current?.abort === abort) this.current = null;
    if (!this.current) this.speaking = false; // a newer line may already be speaking
  }

  /** Generate and store lines that recur, a couple at a time, so they play instantly. */
  async warm(lines: string[]) {
    await purgeStale();
    const todo: string[] = [];
    for (const l of new Set(lines)) if (!(await cacheGet(l))) todo.push(l);
    const worker = async () => {
      for (let l = todo.shift(); l; l = todo.shift()) {
        try {
          // two tries: the model occasionally improvises on short lines
          for (let attempt = 0; attempt < 2; attempt++) {
            const r = await synth(l, new AbortController().signal);
            if (r.pcm.length && faithful(l, r)) {
              await cachePut(l, r.pcm);
              break;
            }
          }
        } catch {
          // try again next session
        }
      }
    };
    await Promise.all([worker(), worker()]);
  }
}

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
const VOICE = "ash"; // warmer than onyx — a small friendly robot, not a Sith lord
const RATE = 24000; // pcm16 sample rate from the API
// playback faster than recorded: pitch and formants rise together, the way a
// small robot (WALL-E, not Vader) sounds
const SLOW = 1.13;
const CARRIER_HZ = 220; // the robot's home note (A3); it lilts around it as it talks
const CARRIER_RATIOS = [1, 2.006, 0.5]; // fundamental, a slightly sharp octave, a sub
const TTS_SYS =
  "You are a text-to-speech engine, not an assistant. The user message contains one text between « and ». " +
  "Read exactly that text aloud in Norwegian bokmål, in a warm, curious, playful and slightly shy voice — " +
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
  get context(): AudioContext | null {
    return this.ctx;
  }

  /** Called with each new AudioContext, so the app can route it to the chosen speaker. */
  onContext: (ctx: AudioContext) => void = () => {};
  private input: GainNode | null = null;
  private crushIn: GainNode | null = null; // same chain, but through a bitcrusher first
  private carrierOscs: OscillatorNode[] = [];
  private bandPeak = { low: 0.05, mid: 0.05, high: 0.05 };
  private room: GainNode | null = null; // reverb send
  /** Fired as a voice glitch plays — the scene glitches with it. */
  onGlitch: (strength: number, ms: number) => void = () => {};
  private analyser: AnalyserNode | null = null;
  private current: { abort: AbortController; sources: AudioBufferSourceNode[]; done: () => void } | null = null;
  speaking = false;

  private audio(): AudioContext {
    if (this.ctx) return this.ctx;
    const ctx = new AudioContext();
    const input = ctx.createGain(); // the spoken voice: the vocoder's modulator

    // --- carrier: what the robot "sings" with — a flat buzz at one pitch, an
    // octave a hair sharp for metal, and noise for the s/t/k sounds up top
    const carrier = ctx.createGain();
    const oscs = [
      { type: "sawtooth" as OscillatorType, f: CARRIER_HZ * CARRIER_RATIOS[0], g: 0.6 },
      { type: "sawtooth" as OscillatorType, f: CARRIER_HZ * CARRIER_RATIOS[1], g: 0.3 },
      { type: "square" as OscillatorType, f: CARRIER_HZ * CARRIER_RATIOS[2], g: 0.15 },
    ].map(({ type, f, g }) => {
      const o = ctx.createOscillator();
      o.type = type;
      o.frequency.value = f;
      const og = ctx.createGain();
      og.gain.value = g;
      o.connect(og).connect(carrier);
      o.start();
      return o;
    });
    const hiss = ctx.createBufferSource();
    const nb = ctx.createBuffer(1, ctx.sampleRate * 2, ctx.sampleRate);
    const nd = nb.getChannelData(0);
    for (let i = 0; i < nd.length; i++) nd[i] = Math.random() * 2 - 1;
    hiss.buffer = nb;
    hiss.loop = true;
    hiss.start();

    // --- 18-band channel vocoder: each band of the voice opens the same band
    // of the carrier. Third-octave bands from 120 Hz to 7 kHz.
    const rect = new Float32Array(1024);
    for (let i = 0; i < rect.length; i++) rect[i] = Math.abs(i / 512 - 1);
    const vocoded = ctx.createGain();
    vocoded.gain.value = 1.5; // make-up: band-limited envelopes are quiet
    const BANDS = 18;
    for (let b = 0; b < BANDS; b++) {
      const f = 120 * Math.pow(7000 / 120, b / (BANDS - 1));
      const q = 4.3;
      const mod = ctx.createBiquadFilter();
      mod.type = "bandpass";
      mod.frequency.value = f;
      mod.Q.value = q;
      const rectifier = ctx.createWaveShaper();
      rectifier.curve = rect;
      const env = ctx.createBiquadFilter(); // envelope follower
      env.type = "lowpass";
      env.frequency.value = 38;
      const envGain = ctx.createGain();
      envGain.gain.value = 9;
      const vca = ctx.createGain();
      vca.gain.value = 0;
      input.connect(mod).connect(rectifier).connect(env).connect(envGain).connect(vca.gain);
      const car = ctx.createBiquadFilter();
      car.type = "bandpass";
      car.frequency.value = f;
      car.Q.value = q;
      (f > 3800 ? hiss : carrier).connect(car);
      car.connect(vca).connect(vocoded);
    }

    // ring modulation: the voice wobbles at 47 Hz — the metallic, tinny edge
    const ring = ctx.createGain();
    ring.gain.value = 0.7;
    const ringOsc = ctx.createOscillator();
    ringOsc.frequency.value = 72;
    const ringDepth = ctx.createGain();
    ringDepth.gain.value = 0.1; // just a shimmer of metal
    ringOsc.connect(ringDepth).connect(ring.gain);
    ringOsc.start();
    vocoded.connect(ring);

    // a thin layer of the real voice underneath keeps the words intelligible
    const dry = ctx.createBiquadFilter();
    dry.type = "lowpass";
    dry.frequency.value = 3600;
    const dryGain = ctx.createGain();
    dryGain.gain.value = 0.34; // the real voice underneath: warmth and clear words
    input.connect(dry).connect(dryGain);

    const mix = ctx.createGain();
    ring.connect(mix);
    dryGain.connect(mix);
    const shaper = ctx.createWaveShaper();
    const curve = new Float32Array(1024);
    for (let i = 0; i < curve.length; i++) curve[i] = Math.tanh((i / 512 - 1) * 1.15);
    shaper.curve = curve;
    // the mask: a short feedback comb gives the hollow, metallic resonance
    const comb = ctx.createDelay();
    comb.delayTime.value = 0.006;
    const fb = ctx.createGain();
    fb.gain.value = 0.22;
    comb.connect(fb).connect(comb);
    const out = ctx.createGain();
    out.gain.value = 0.7;
    // a limiter last: comb feedback and glitch bursts must never clip the room
    const limiter = ctx.createDynamicsCompressor();
    limiter.threshold.value = -8;
    limiter.knee.value = 4;
    limiter.ratio.value = 14;
    limiter.attack.value = 0.002;
    limiter.release.value = 0.12;
    const analyser = ctx.createAnalyser();
    analyser.fftSize = 512;
    // a narrower dB window than the default (-100..-30), so bands track the
    // speech instead of sitting at full scale
    analyser.minDecibels = -75;
    analyser.maxDecibels = -15;
    mix.connect(shaper);
    shaper.connect(out);
    shaper.connect(comb);
    comb.connect(out);
    out.connect(limiter);
    limiter.connect(analyser);
    limiter.connect(ctx.destination);
    this.carrierOscs = oscs;

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
    this.onContext(ctx);
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


  /** The robot's note jumps for a moment — the carrier glitches with the voice. */
  private carrierJump(at: number, dur: number) {
    const factor = [1.5, 0.75, 2, 1.335][Math.floor(Math.random() * 4)];
    for (const [i, o] of this.carrierOscs.entries()) {
      const f = CARRIER_RATIOS[i] * CARRIER_HZ; // from the base, so jumps never drift
      o.frequency.setValueAtTime(f * factor, at);
      o.frequency.setValueAtTime(f, at + Math.min(0.25, dur));
    }
  }

  /** Octa's voice by band, 0..1 — what makes the body vibrate. */
  bands(): { low: number; mid: number; high: number } {
    if (!this.analyser || !this.ctx) return { low: 0, mid: 0, high: 0 };
    const bins = new Uint8Array(this.analyser.frequencyBinCount);
    this.analyser.getByteFrequencyData(bins);
    const hz = this.ctx.sampleRate / this.analyser.fftSize;
    const avg = (f0: number, f1: number) => {
      let sum = 0;
      let n = 0;
      for (let i = Math.floor(f0 / hz); i <= Math.ceil(f1 / hz) && i < bins.length; i++) {
        sum += bins[i];
        n++;
      }
      return n ? sum / n / 255 : 0;
    };
    // each band against its own recent peak: the vocoder puts far more energy
    // in the lows, so one fixed scale would leave mids and highs flat
    const norm = (key: "low" | "mid" | "high", raw: number) => {
      this.bandPeak[key] = Math.max(raw, this.bandPeak[key] * 0.995, 0.05);
      const x = raw / this.bandPeak[key];
      return Math.min(1, Math.max(0, (x - 0.35) / 0.65)); // below ~a third of peak = still
    };
    return {
      low: norm("low", avg(80, 300)),
      mid: norm("mid", avg(300, 2000)),
      high: norm("high", avg(2000, 6000)),
    };
  }

  /** Open the audio context inside a user gesture (the Begin click), so idle
   *  sounds can play before Octa has said anything. */
  prime() {
    void this.audio().resume();
  }

  /** Idle interference: a short crackle of crushed static through the room,
   *  sometimes with a ghost of a tone — the channel hiccuping on its own. */
  staticBurst(strength = 1) {
    if (this.speaking) return;
    const ctx = this.audio();
    if (ctx.state !== "running") return; // never resume audio without a user gesture
    const t = ctx.currentTime;
    const dur = 0.12 + Math.random() * 0.18 * strength;
    const noise = ctx.createBuffer(1, Math.ceil(ctx.sampleRate * dur), ctx.sampleRate);
    const c = noise.getChannelData(0);
    // sample-and-hold noise: chunky, digital rather than hissy
    const hold = 6 + Math.floor(Math.random() * 30);
    let v = 0;
    for (let i = 0; i < c.length; i++) {
      if (i % hold === 0) v = Math.random() * 2 - 1;
      c[i] = v * (Math.random() < 0.92 ? 1 : 0); // dropouts
    }
    const src = ctx.createBufferSource();
    src.buffer = noise;
    const bp = ctx.createBiquadFilter();
    bp.type = "bandpass";
    bp.frequency.value = 900 + Math.random() * 2400;
    bp.Q.value = 1.2;
    const g = ctx.createGain();
    g.gain.setValueAtTime(0.0001, t);
    g.gain.exponentialRampToValueAtTime(0.05 * strength, t + 0.01);
    g.gain.setValueAtTime(0.05 * strength, t + dur * 0.7);
    g.gain.exponentialRampToValueAtTime(0.0001, t + dur);
    src.connect(bp).connect(g).connect(ctx.destination);
    g.connect(this.room!);
    src.start(t);
    if (Math.random() < 0.4) {
      // a tone bleeding through the static
      const o = ctx.createOscillator();
      o.type = "square";
      o.frequency.setValueAtTime(110 + Math.random() * 220, t);
      const og = ctx.createGain();
      og.gain.setValueAtTime(0.0001, t);
      og.gain.exponentialRampToValueAtTime(0.02 * strength, t + 0.01);
      og.gain.exponentialRampToValueAtTime(0.0001, t + dur);
      o.connect(og).connect(this.room!);
      og.connect(ctx.destination);
      o.start(t);
      o.stop(t + dur + 0.02);
    }
  }

  /** One full in-out cycle — the idle sound while Octa is thinking. */
  breatheCycle() {
    if (this.speaking) return;
    // thinking: two soft rising notes, a small machine humming to itself
    const t = this.audio().currentTime;
    this.tone(t, 660, 0.12, 0.025);
    this.tone(t + 0.16, 880, 0.14, 0.02);
  }

  /** A short sine note into the room. */
  private tone(at: number, hz: number, dur: number, gain: number) {
    const ctx = this.audio();
    const o = ctx.createOscillator();
    o.type = "sine";
    o.frequency.value = hz;
    const g = ctx.createGain();
    g.gain.setValueAtTime(0.0001, at);
    g.gain.exponentialRampToValueAtTime(gain, at + 0.015);
    g.gain.exponentialRampToValueAtTime(0.0001, at + dur);
    o.connect(g).connect(ctx.destination);
    g.connect(this.room!);
    o.start(at);
    o.stop(at + dur + 0.02);
  }

  /** "I'm about to speak": a bright two-note chime. */
  private chime(at: number) {
    this.tone(at, 988, 0.09, 0.03);
    this.tone(at + 0.05, 1319, 0.12, 0.025);
  }

  /** Intonation: on each syllable the robot's note slides to a new step, so
   *  it speaks with a sing-song lilt ("woo-ah") instead of a monotone drone. */
  private intone() {
    const ctx = this.ctx;
    if (!ctx) return;
    const steps = [0, 3, 5, 7, -2, 4, 9, 2]; // semitones around home, mostly up
    const k = Math.pow(2, steps[Math.floor(Math.random() * steps.length)] / 12);
    const t = ctx.currentTime;
    for (const [i, o] of this.carrierOscs.entries()) {
      o.frequency.cancelScheduledValues(t);
      o.frequency.setTargetAtTime(CARRIER_RATIOS[i] * CARRIER_HZ * k, t, 0.05);
    }
    // and now and then a little "bwip" of delight
    if (Math.random() < 0.18) this.tone(t + 0.02, 1400 + Math.random() * 600, 0.07, 0.018);
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

    // a bright little chime, then the words — no Vader breathing any more
    let next = ctx.currentTime + 0.05;
    this.chime(next);
    next += 0.08;

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
      const kinds = ["stutter", "pitch", "crush", "freeze", "tapestop", "reverse"] as const;
      const kind = kinds[Math.floor(Math.random() * kinds.length)];
      const fire = (at: number, strength: number, ms: number) => {
        this.carrierJump(at, ms / 1000);
        window.setTimeout(() => this.onGlitch(strength, ms), Math.max(0, (at - ctx.currentTime) * 1000));
      };

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
      } else if (kind === "freeze") {
        // the signal locks on a 25 ms grain and buzzes, dying away
        const grain = 0.025;
        fire(t, 0.9, 300);
        for (let i = 0; i < 9; i++) {
          const g = ctx.createGain();
          g.gain.value = 1 - i / 10;
          g.connect(this.input!);
          play(b, t, cut, grain, g);
          t += grain / SLOW;
        }
        play(b, t, cut, len - cut, this.input!);
        t += (len - cut) / SLOW;
      } else if (kind === "tapestop") {
        // power failing: the voice sags in pitch and speed, then snaps back
        const sag = Math.min(0.2, len - cut);
        const s1 = play(b, t, cut, sag, this.input!);
        s1.playbackRate.setValueAtTime(SLOW, t);
        s1.playbackRate.linearRampToValueAtTime(SLOW * 0.3, t + 0.22);
        fire(t, 0.8, 320);
        t += 0.26;
        play(b, t, cut + sag, len - cut - sag, this.input!);
        t += (len - cut - sag) / SLOW;
      } else if (kind === "reverse") {
        // a 120 ms scrap of the word, played backwards
        const grain = Math.min(0.12, len - cut);
        const n = Math.floor(grain * RATE);
        const off = Math.floor(cut * RATE);
        const rev = ctx.createBuffer(1, n, RATE);
        const src = b.getChannelData(0);
        const dst = rev.getChannelData(0);
        for (let i = 0; i < n; i++) dst[i] = src[off + n - 1 - i] ?? 0;
        fire(t, 0.7, 200);
        play(rev, t, 0, grain, this.input!);
        t += grain / SLOW;
        play(b, t, cut + grain, len - cut - grain, this.input!);
        t += (len - cut - grain) / SLOW;
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
      if (loud && !wasLoud) {
        onWord();
        this.intone();
      }
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

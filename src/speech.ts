/** Web Speech API + mic metering, wrapped so the engines can be swapped later. */
import { settings } from "./openrouter";
import { VaderVoice } from "./vader";

export class MicMeter {
  private analyser: AnalyserNode | null = null;
  private data: Uint8Array<ArrayBuffer> | null = null;
  private stream: MediaStream | null = null;
  private ctx: AudioContext | null = null;
  private tap: ScriptProcessorNode | null = null;
  /** Last ~1 s of raw mic audio, so a capture can start before it was asked to. */
  private ring: Float32Array[] = [];
  private ringLen = 0;
  /** Live capture sinks (CloudRecognizer) — fed every audio block. */
  private sinks = new Set<(block: Float32Array) => void>();

  get sampleRate(): number {
    return this.ctx?.sampleRate ?? 48000;
  }

  /** Open the chosen mic ("" = system default). Re-callable to switch devices. */
  async init(deviceId = ""): Promise<void> {
    this.close();
    const stream = await navigator.mediaDevices.getUserMedia({
      audio: deviceId
        ? { deviceId: { exact: deviceId }, echoCancellation: true, noiseSuppression: true }
        : { echoCancellation: true, noiseSuppression: true },
    });
    const ctx = new AudioContext();
    const src = ctx.createMediaStreamSource(stream);
    this.analyser = ctx.createAnalyser();
    this.analyser.fftSize = 512;
    this.analyser.smoothingTimeConstant = 0.6;
    src.connect(this.analyser);
    this.data = new Uint8Array(this.analyser.fftSize);

    // raw tap for cloud transcription: keeps a 1 s pre-roll and feeds live captures
    const tap = ctx.createScriptProcessor(2048, 1, 1);
    tap.onaudioprocess = (e) => {
      const block = new Float32Array(e.inputBuffer.getChannelData(0));
      this.ring.push(block);
      this.ringLen += block.length;
      while (this.ringLen - this.ring[0].length > ctx.sampleRate) this.ringLen -= this.ring.shift()!.length;
      for (const sink of this.sinks) sink(block);
    };
    const mute = ctx.createGain();
    mute.gain.value = 0;
    src.connect(tap);
    tap.connect(mute).connect(ctx.destination); // a ScriptProcessor only runs when connected
    this.stream = stream;
    this.ctx = ctx;
    this.tap = tap;
  }

  close() {
    this.stream?.getTracks().forEach((t) => t.stop());
    this.tap?.disconnect();
    void this.ctx?.close();
    this.stream = this.ctx = this.tap = null;
    this.analyser = null;
    this.ring = [];
    this.ringLen = 0;
  }

  /** The last `seconds` of audio before now. */
  preroll(seconds: number): Float32Array[] {
    const want = seconds * this.sampleRate;
    const out: Float32Array[] = [];
    let got = 0;
    for (let i = this.ring.length - 1; i >= 0 && got < want; i--) {
      out.unshift(this.ring[i]);
      got += this.ring[i].length;
    }
    return out;
  }

  listen(sink: (block: Float32Array) => void): () => void {
    this.sinks.add(sink);
    return () => this.sinks.delete(sink);
  }

  /** RMS level 0..1, gently boosted so normal speech reads ~0.3–0.8. */
  level(): number {
    if (!this.analyser || !this.data) return 0;
    this.analyser.getByteTimeDomainData(this.data);
    let sum = 0;
    for (let i = 0; i < this.data.length; i++) {
      const v = (this.data[i] - 128) / 128;
      sum += v * v;
    }
    return Math.min(1, Math.sqrt(sum / this.data.length) * 4);
  }
}

/** Mono float PCM → 16 kHz 16-bit WAV, base64. */
function toWavBase64(blocks: Float32Array[], inRate: number): string {
  const total = blocks.reduce((n, b) => n + b.length, 0);
  const outRate = 16000;
  const ratio = inRate / outRate;
  const n = Math.floor(total / ratio);
  const flat = new Float32Array(total);
  let o = 0;
  for (const b of blocks) {
    flat.set(b, o);
    o += b.length;
  }
  const buf = new ArrayBuffer(44 + n * 2);
  const v = new DataView(buf);
  const w = (off: number, str: string) => [...str].forEach((c, i) => v.setUint8(off + i, c.charCodeAt(0)));
  w(0, "RIFF");
  v.setUint32(4, 36 + n * 2, true);
  w(8, "WAVE");
  w(12, "fmt ");
  v.setUint32(16, 16, true);
  v.setUint16(20, 1, true);
  v.setUint16(22, 1, true);
  v.setUint32(24, outRate, true);
  v.setUint32(28, outRate * 2, true);
  v.setUint16(32, 2, true);
  v.setUint16(34, 16, true);
  w(36, "data");
  v.setUint32(40, n * 2, true);
  for (let i = 0; i < n; i++) {
    // box-average each output sample over its input window: cheap anti-aliasing
    const a = Math.floor(i * ratio);
    const b = Math.min(total, Math.floor((i + 1) * ratio));
    let sum = 0;
    for (let j = a; j < b; j++) sum += flat[j];
    const x = Math.max(-1, Math.min(1, sum / Math.max(1, b - a)));
    v.setInt16(44 + i * 2, x * 32767, true);
  }
  const bytes = new Uint8Array(buf);
  let bin = "";
  for (let i = 0; i < bytes.length; i += 8192) bin += String.fromCharCode(...bytes.subarray(i, i + 8192));
  return btoa(bin);
}

export interface SpeechCapture {
  onInterim: (text: string) => void;
  onFinal: (text: string) => void;
  onEnd: () => void;
  start(): void;
  stop(): void;
}

/**
 * Transcription from the chosen meeting-room mic. Chrome's built-in
 * recognizer only ever hears the system default mic, so with a specific mic
 * selected the audio is captured here (with ~0.6 s of pre-roll so the first
 * word survives) and transcribed by gpt-audio-mini via OpenRouter.
 */
export class CloudRecognizer implements SpeechCapture {
  onInterim: (text: string) => void = () => {};
  onFinal: (text: string) => void = () => {};
  onEnd: () => void = () => {};
  private stopNow = false;
  private mic: MicMeter;

  constructor(mic: MicMeter) {
    this.mic = mic;
  }

  start() {
    const mic = this.mic;
    const blocks = mic.preroll(0.6);
    let heard = false;
    let lastLoud = performance.now();
    const started = performance.now();
    const unlisten = mic.listen((b) => blocks.push(b));
    this.onInterim("…");

    const done = async () => {
      window.clearInterval(vad);
      unlisten();
      if (!heard) {
        this.onEnd();
        return;
      }
      this.onInterim("transkriberer …");
      try {
        const text = await transcribe(toWavBase64(blocks, mic.sampleRate));
        if (text) this.onFinal(text);
      } catch (err) {
        console.warn("transcription failed", err);
      }
      this.onEnd();
    };

    // a simple level VAD: speech → wait for 0.8 s of quiet; nothing for 4 s → give up
    const vad = window.setInterval(() => {
      const now = performance.now();
      const loud = mic.level() > 0.12;
      if (loud) {
        heard = true;
        lastLoud = now;
      }
      if (
        this.stopNow ||
        (heard && now - lastLoud > 800) ||
        (!heard && now - started > 4000) ||
        now - started > 20000
      ) {
        void done();
      }
    }, 50);
  }

  stop() {
    this.stopNow = true;
  }
}

async function transcribe(wavB64: string): Promise<string> {
  const res = await fetch("https://openrouter.ai/api/v1/chat/completions", {
    method: "POST",
    headers: {
      Authorization: `Bearer ${settings.key}`,
      "Content-Type": "application/json",
      "X-Title": "Octa",
    },
    body: JSON.stringify({
      model: "openai/gpt-audio-mini",
      messages: [
        {
          role: "system",
          content:
            "Transcribe the audio verbatim. Output only the transcript, in the language spoken " +
            "(usually Norwegian bokmål). If there is no clear speech, output nothing.",
        },
        { role: "user", content: [{ type: "input_audio", input_audio: { data: wavB64, format: "wav" } }] },
      ],
    }),
  });
  if (!res.ok) throw new Error(`transcribe ${res.status}`);
  const data = await res.json();
  return String(data.choices?.[0]?.message?.content ?? "").trim();
}

// lib.dom has SpeechRecognitionEvent but not the recognizer itself
interface SpeechRecognition {
  lang: string;
  interimResults: boolean;
  continuous: boolean;
  onresult: ((e: SpeechRecognitionEvent) => void) | null;
  onend: (() => void) | null;
  onerror: (() => void) | null;
  start(): void;
  stop(): void;
}

type RecognitionCtor = new () => SpeechRecognition;

export class Recognizer implements SpeechCapture {
  private rec: SpeechRecognition | null = null;

  onInterim: (text: string) => void = () => {};
  onFinal: (text: string) => void = () => {};
  onEnd: () => void = () => {};

  static supported(): boolean {
    return "SpeechRecognition" in window || "webkitSpeechRecognition" in window;
  }

  start(lang = "nb-NO") {
    const w = window as unknown as {
      SpeechRecognition?: RecognitionCtor;
      webkitSpeechRecognition?: RecognitionCtor;
    };
    const Ctor = (w.SpeechRecognition ?? w.webkitSpeechRecognition)!;
    const rec = new Ctor();
    this.rec = rec;
    rec.lang = lang;
    rec.interimResults = true;
    rec.continuous = false;

    let final = "";
    rec.onresult = (e: SpeechRecognitionEvent) => {
      let interim = "";
      for (let i = e.resultIndex; i < e.results.length; i++) {
        const r = e.results[i];
        if (r.isFinal) final += r[0].transcript;
        else interim += r[0].transcript;
      }
      if (interim) this.onInterim(interim);
    };
    rec.onend = () => {
      if (final.trim()) this.onFinal(final.trim());
      this.onEnd();
    };
    rec.onerror = () => {}; // onend still fires; a silent no-result is fine
    rec.start();
  }

  stop() {
    this.rec?.stop();
  }
}

export class Speaker {
  readonly vader = new VaderVoice();

  /** True while either engine is talking. */
  get speaking(): boolean {
    return this.vader.speaking || speechSynthesis.speaking;
  }

  /** Robot-voice hooks: the app hangs its buzz and chirps on these. */
  onStartFx: () => void = () => {};
  onWordFx: () => void = () => {};
  onEndFx: () => void = () => {};

  /**
   * Deterministic voice choice: the user's saved pick first, then a fixed
   * priority list. (Never scan the voice list with a broad regex — Chrome
   * sorts voices alphabetically, so the match silently changes over time.)
   */
  static pickVoice(): SpeechSynthesisVoice | undefined {
    const voices = speechSynthesis.getVoices();
    if (settings.voice) {
      const saved = voices.find((v) => v.name === settings.voice);
      if (saved) return saved;
    }
    // Octa is a young male robot: a male Norwegian voice when the Mac has one
    // (Henrik is a free download under Spoken Content), else Nora pitched down
    for (const name of ["Henrik", "Nora", "Google norsk bokmål"]) {
      const v = voices.find((v) => v.name === name || v.name.startsWith(`${name} (`));
      if (v) return v;
    }
    const norwegian = (v: SpeechSynthesisVoice) => /^(nb|no|nn)\b/i.test(v.lang);
    return voices.find((v) => v.default && norwegian(v)) ?? voices.find(norwegian);
  }

  /** Speaks text; onWord fires per word boundary, resolves when done.
   *  Vader style goes through gpt-audio; without a key, or if it fails before
   *  any sound, the system voice takes over so Octa never goes mute. */
  async speak(text: string, onWord: () => void): Promise<void> {
    if (settings.voiceStyle === "vader" && settings.key) {
      speechSynthesis.cancel();
      try {
        await this.vader.speak(text, onWord);
        return;
      } catch (err) {
        console.warn("vader voice failed, falling back to the system voice", err);
      }
    }
    return this.speakSystem(text, onWord);
  }

  private speakSystem(text: string, onWord: () => void): Promise<void> {
    return new Promise((resolve) => {
      const u = new SpeechSynthesisUtterance(text);
      u.lang = "nb-NO";
      // robot: a young male register — dropped (further for a female base
      // voice, which otherwise reads as a woman), flat and a little quick
      const preferred = Speaker.pickVoice();
      if (preferred) u.voice = preferred;
      const robot = settings.robotVoice;
      const male = /henrik/i.test(u.voice?.name ?? "");
      u.rate = robot ? 1.15 : 1.0;
      u.pitch = robot ? (male ? 0.75 : 0.5) : 0.9;
      if (robot) u.onstart = () => this.onStartFx();

      u.onboundary = (e) => {
        if (e.name !== "word") return;
        onWord();
        if (settings.robotVoice) this.onWordFx();
      };
      u.onend = () => {
        if (robot) this.onEndFx();
        resolve();
      };
      u.onerror = () => {
        if (robot) this.onEndFx();
        resolve();
      };
      speechSynthesis.cancel();
      speechSynthesis.speak(u);
    });
  }

  stop() {
    this.vader.stop();
    speechSynthesis.cancel();
    this.onEndFx(); // cancel doesn't reliably fire onend — never leave the buzz running
  }
}

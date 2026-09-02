/** Web Speech API + mic metering, wrapped so the engines can be swapped later. */
import { settings } from "./openrouter";

export class MicMeter {
  private analyser: AnalyserNode | null = null;
  private data: Uint8Array<ArrayBuffer> | null = null;

  async init(): Promise<void> {
    const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
    const ctx = new AudioContext();
    const src = ctx.createMediaStreamSource(stream);
    this.analyser = ctx.createAnalyser();
    this.analyser.fftSize = 512;
    this.analyser.smoothingTimeConstant = 0.6;
    src.connect(this.analyser);
    this.data = new Uint8Array(this.analyser.fftSize);
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

export class Recognizer {
  private rec: SpeechRecognition | null = null;

  onInterim: (text: string) => void = () => {};
  onFinal: (text: string) => void = () => {};
  onEnd: () => void = () => {};

  static supported(): boolean {
    return "SpeechRecognition" in window || "webkitSpeechRecognition" in window;
  }

  start(lang = "en-US") {
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
    // Daniel first — the clipped, slightly robotic voice this UI shipped with
    for (const name of ["Daniel", "Google UK English Male", "Samantha", "Google US English"]) {
      const v = voices.find((v) => v.name === name);
      if (v) return v;
    }
    return (
      voices.find((v) => v.default && v.lang.startsWith("en")) ??
      voices.find((v) => v.lang.startsWith("en"))
    );
  }

  /** Speaks text; onWord fires per word boundary, resolves when done. */
  speak(text: string, onWord: () => void): Promise<void> {
    return new Promise((resolve) => {
      const u = new SpeechSynthesisUtterance(text);
      u.rate = 1.0;
      u.pitch = 0.9;

      const preferred = Speaker.pickVoice();
      if (preferred) u.voice = preferred;

      u.onboundary = (e) => {
        if (e.name === "word") onWord();
      };
      u.onend = () => resolve();
      u.onerror = () => resolve();
      speechSynthesis.cancel();
      speechSynthesis.speak(u);
    });
  }

  stop() {
    speechSynthesis.cancel();
  }
}

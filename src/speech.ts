/** Web Speech API + mic metering, wrapped so the engines can be swapped later. */
import { settings } from "./openrouter";
import { VaderVoice } from "./vader";

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

import * as ort from "onnxruntime-web/wasm";

/**
 * Voiceprints: who is speaking, from the sound alone. Runs fully in the
 * browser — WeSpeaker's ResNet34 (VoxCeleb, CC BY 4.0) via ONNX Runtime Web
 * turns a few seconds of speech into a 256-number embedding; two embeddings
 * of the same voice point the same way (cosine similarity).
 *
 * Octa never asks anyone to enrol: it learns each person's voice in the
 * moments the camera sees *their* lips moving while they speak, and only
 * keeps voiceprints for the session, in memory.
 */

const MODEL = "/models/voxceleb_resnet34_LM.onnx";
const RATE = 16000;

// the bundled build carries its own loader; only the .wasm binary is served
// from public/ (Vite won't let a page import modules out of public/)
ort.env.wasm.wasmPaths = { wasm: "/ort/ort-wasm-simd-threaded.wasm" };
ort.env.wasm.numThreads = 1; // no cross-origin isolation here, so no threads
// inference in a worker, so a 250 ms embedding never stalls the 3D scene
ort.env.wasm.proxy = true;

// --- Kaldi-style 80-bin log-mel fbank, as WeSpeaker was trained on -------------

const FRAME = 400; // 25 ms at 16 kHz
const SHIFT = 160; // 10 ms
const NFFT = 512;
const MELS = 80;

const povey = new Float32Array(FRAME).map((_, i) => Math.pow(0.5 - 0.5 * Math.cos((2 * Math.PI * i) / (FRAME - 1)), 0.85));

const melBank: { start: number; weights: Float32Array }[] = (() => {
  const mel = (f: number) => 1127 * Math.log(1 + f / 700);
  const lo = mel(20);
  const hi = mel(RATE / 2);
  const step = (hi - lo) / (MELS + 1);
  const bins = NFFT / 2 + 1;
  const banks = [];
  for (let m = 0; m < MELS; m++) {
    const l = lo + m * step;
    const c = l + step;
    const r = c + step;
    const w = new Float32Array(bins);
    let start = -1;
    for (let k = 0; k < bins; k++) {
      const f = mel((k * RATE) / NFFT);
      const v = f > l && f < r ? (f <= c ? (f - l) / (c - l) : (r - f) / (r - c)) : 0;
      w[k] = v;
      if (v > 0 && start < 0) start = k;
    }
    banks.push({ start: Math.max(0, start), weights: w });
  }
  return banks;
})();

/** In-place radix-2 FFT (re, im of length NFFT). */
function fft(re: Float32Array, im: Float32Array) {
  const n = re.length;
  for (let i = 1, j = 0; i < n; i++) {
    let bit = n >> 1;
    for (; j & bit; bit >>= 1) j ^= bit;
    j ^= bit;
    if (i < j) {
      [re[i], re[j]] = [re[j], re[i]];
      [im[i], im[j]] = [im[j], im[i]];
    }
  }
  for (let len = 2; len <= n; len <<= 1) {
    const ang = (-2 * Math.PI) / len;
    const wr = Math.cos(ang);
    const wi = Math.sin(ang);
    for (let i = 0; i < n; i += len) {
      let cr = 1;
      let ci = 0;
      for (let j = 0; j < len / 2; j++) {
        const ar = re[i + j + len / 2] * cr - im[i + j + len / 2] * ci;
        const ai = re[i + j + len / 2] * ci + im[i + j + len / 2] * cr;
        re[i + j + len / 2] = re[i + j] - ar;
        im[i + j + len / 2] = im[i + j] - ai;
        re[i + j] += ar;
        im[i + j] += ai;
        const t = cr * wr - ci * wi;
        ci = cr * wi + ci * wr;
        cr = t;
      }
    }
  }
}

/** 16 kHz mono float PCM → [frames × 80] log-mel, mean-normalized per bin. */
export function fbank(pcm: Float32Array): { data: Float32Array; frames: number } {
  const frames = Math.max(0, Math.floor((pcm.length - FRAME) / SHIFT) + 1);
  const out = new Float32Array(frames * MELS);
  const re = new Float32Array(NFFT);
  const im = new Float32Array(NFFT);
  for (let f = 0; f < frames; f++) {
    const off = f * SHIFT;
    let mean = 0;
    for (let i = 0; i < FRAME; i++) mean += pcm[off + i];
    mean /= FRAME;
    re.fill(0);
    im.fill(0);
    // int16 scale like Kaldi, DC removed, 0.97 pre-emphasis, Povey window
    let prev = (pcm[off] - mean) * 32768;
    for (let i = 0; i < FRAME; i++) {
      const x = (pcm[off + i] - mean) * 32768;
      re[i] = (x - 0.97 * (i ? prev : x)) * povey[i];
      prev = x;
    }
    fft(re, im);
    for (let m = 0; m < MELS; m++) {
      const { start, weights } = melBank[m];
      let e = 0;
      for (let k = start; k < weights.length && (weights[k] > 0 || k === start); k++) {
        e += weights[k] * (re[k] * re[k] + im[k] * im[k]);
      }
      out[f * MELS + m] = Math.log(Math.max(e, 1.19e-7));
    }
  }
  // cepstral mean normalization over the utterance
  for (let m = 0; m < MELS; m++) {
    let mu = 0;
    for (let f = 0; f < frames; f++) mu += out[f * MELS + m];
    mu /= frames || 1;
    for (let f = 0; f < frames; f++) out[f * MELS + m] -= mu;
  }
  return { data: out, frames };
}

/** Any-rate mono PCM → 16 kHz (box-averaged decimation; mics run at 44.1/48 kHz). */
export function to16k(pcm: Float32Array, rate: number): Float32Array {
  if (rate === RATE) return pcm;
  const ratio = rate / RATE;
  const n = Math.floor(pcm.length / ratio);
  const out = new Float32Array(n);
  for (let i = 0; i < n; i++) {
    const a = Math.floor(i * ratio);
    const b = Math.min(pcm.length, Math.floor((i + 1) * ratio));
    let s = 0;
    for (let j = a; j < b; j++) s += pcm[j];
    out[i] = s / Math.max(1, b - a);
  }
  return out;
}

export function cosine(a: Float32Array, b: Float32Array): number {
  let dot = 0, na = 0, nb = 0;
  for (let i = 0; i < a.length; i++) {
    dot += a[i] * b[i];
    na += a[i] * a[i];
    nb += b[i] * b[i];
  }
  return dot / (Math.sqrt(na * nb) || 1);
}

// --- the model --------------------------------------------------------------

let session: Promise<ort.InferenceSession> | null = null;

function load() {
  session ??= ort.InferenceSession.create(MODEL, { executionProviders: ["wasm"] });
  return session;
}

/** A voiceprint for 16 kHz speech (≥ ~1 s works; 2–4 s is better). */
export async function embed(pcm16k: Float32Array): Promise<Float32Array | null> {
  const { data, frames } = fbank(pcm16k);
  if (frames < 80) return null; // under ~0.8 s: too little to say who it is
  const s = await load();
  const feeds = { [s.inputNames[0]]: new ort.Tensor("float32", data, [1, frames, MELS]) };
  const out = await s.run(feeds);
  return out[s.outputNames[0]].data as Float32Array;
}

/** Warm the model up off the critical path (it's 26 MB). */
export function preloadVoiceModel() {
  void load().catch((err) => console.warn("voice model failed to load", err));
}

import type { OctaState } from "./scene";

interface StateVoicing {
  freq: number; // drone root
  ratio: number; // second oscillator ratio (1.06 = eerie minor second)
  cutoff: number; // lowpass base
  droneGain: number;
  delayFeedback: number;
}

const VOICINGS: Record<OctaState, StateVoicing> = {
  idle: { freq: 55, ratio: 1.498, cutoff: 320, droneGain: 0.045, delayFeedback: 0.25 },
  listening: { freq: 65.4, ratio: 2.0, cutoff: 700, droneGain: 0.06, delayFeedback: 0.3 },
  thinking: { freq: 49, ratio: 1.06, cutoff: 500, droneGain: 0.075, delayFeedback: 0.55 },
  speaking: { freq: 82.4, ratio: 1.5, cutoff: 900, droneGain: 0.055, delayFeedback: 0.3 },
  noting: { freq: 41.2, ratio: 2.0, cutoff: 260, droneGain: 0.05, delayFeedback: 0.2 },
};

/**
 * Fully procedural soundscape. Nothing is sampled — every parameter can be
 * driven continuously by the octahedron's motion and the user's voice level.
 */
export class Soundscape {
  private ctx: AudioContext;
  private master: GainNode;

  private oscA: OscillatorNode;
  private oscB: OscillatorNode;
  private droneFilter: BiquadFilterNode;
  private droneGain: GainNode;

  private shimmerGain: GainNode;
  private shimmerFilter: BiquadFilterNode;

  private delay: DelayNode;
  private feedback: GainNode;

  private state: OctaState = "idle";
  private blipTimer: number | null = null;

  constructor() {
    const ctx = new AudioContext();
    this.ctx = ctx;

    this.master = ctx.createGain();
    this.master.gain.value = 0;

    const compressor = ctx.createDynamicsCompressor();
    compressor.threshold.value = -24;

    // generated reverb tail
    const convolver = ctx.createConvolver();
    convolver.buffer = this.makeImpulse(2.8, 2.5);
    const wet = ctx.createGain();
    wet.gain.value = 0.35;
    const dry = ctx.createGain();
    dry.gain.value = 0.8;

    this.master.connect(dry).connect(compressor);
    this.master.connect(convolver).connect(wet).connect(compressor);
    compressor.connect(ctx.destination);

    // echo line for blips + shimmer
    this.delay = ctx.createDelay(1);
    this.delay.delayTime.value = 0.28;
    this.feedback = ctx.createGain();
    this.feedback.gain.value = 0.25;
    this.delay.connect(this.feedback).connect(this.delay);
    this.delay.connect(this.master);

    // drone: two detuned oscillators through a tracked lowpass
    this.droneFilter = ctx.createBiquadFilter();
    this.droneFilter.type = "lowpass";
    this.droneFilter.frequency.value = 320;
    this.droneFilter.Q.value = 2.5;

    this.droneGain = ctx.createGain();
    this.droneGain.gain.value = 0.045;

    this.oscA = ctx.createOscillator();
    this.oscA.type = "sawtooth";
    this.oscA.frequency.value = 55;
    this.oscB = ctx.createOscillator();
    this.oscB.type = "triangle";
    this.oscB.frequency.value = 55 * 1.498;
    this.oscB.detune.value = 7;

    this.oscA.connect(this.droneFilter);
    this.oscB.connect(this.droneFilter);
    this.droneFilter.connect(this.droneGain).connect(this.master);

    // slow LFO breathing the detune — keeps the drone alive
    const lfo = ctx.createOscillator();
    lfo.frequency.value = 0.07;
    const lfoAmt = ctx.createGain();
    lfoAmt.gain.value = 6;
    lfo.connect(lfoAmt).connect(this.oscB.detune);
    lfo.start();

    // shimmer: filtered noise that follows motion + voice
    const noise = ctx.createBufferSource();
    noise.buffer = this.makeNoise(2);
    noise.loop = true;
    this.shimmerFilter = ctx.createBiquadFilter();
    this.shimmerFilter.type = "bandpass";
    this.shimmerFilter.frequency.value = 1800;
    this.shimmerFilter.Q.value = 1.5;
    this.shimmerGain = ctx.createGain();
    this.shimmerGain.gain.value = 0;
    noise.connect(this.shimmerFilter).connect(this.shimmerGain).connect(this.master);
    noise.start();

    this.oscA.start();
    this.oscB.start();
  }

  private makeNoise(seconds: number): AudioBuffer {
    const len = this.ctx.sampleRate * seconds;
    const buf = this.ctx.createBuffer(1, len, this.ctx.sampleRate);
    const d = buf.getChannelData(0);
    for (let i = 0; i < len; i++) d[i] = Math.random() * 2 - 1;
    return buf;
  }

  private makeImpulse(seconds: number, decay: number): AudioBuffer {
    const len = this.ctx.sampleRate * seconds;
    const buf = this.ctx.createBuffer(2, len, this.ctx.sampleRate);
    for (let ch = 0; ch < 2; ch++) {
      const d = buf.getChannelData(ch);
      for (let i = 0; i < len; i++) {
        d[i] = (Math.random() * 2 - 1) * Math.pow(1 - i / len, decay);
      }
    }
    return buf;
  }

  async start() {
    await this.ctx.resume();
    this.master.gain.setTargetAtTime(1, this.ctx.currentTime, 1.5);
  }

  /** Browsers can suspend the context; call on any user interaction. */
  ensureRunning() {
    if (this.ctx.state !== "running") void this.ctx.resume();
  }

  setState(state: OctaState) {
    if (state === this.state) return;
    this.state = state;
    const v = VOICINGS[state];
    const t = this.ctx.currentTime;

    this.oscA.frequency.setTargetAtTime(v.freq, t, 0.6);
    this.oscB.frequency.setTargetAtTime(v.freq * v.ratio, t, 0.8);
    this.droneGain.gain.setTargetAtTime(v.droneGain, t, 0.5);
    this.feedback.gain.setTargetAtTime(v.delayFeedback, t, 0.4);

    this.whoosh(state === "idle" ? -1 : 1);

    if (this.blipTimer !== null) {
      clearTimeout(this.blipTimer);
      this.blipTimer = null;
    }
    if (state === "thinking") this.scheduleBlip();
  }

  /** Rising (1) or falling (-1) filtered-noise sweep on state changes. */
  private whoosh(direction: number) {
    const t = this.ctx.currentTime;
    const src = this.ctx.createBufferSource();
    src.buffer = this.makeNoise(0.8);
    const bp = this.ctx.createBiquadFilter();
    bp.type = "bandpass";
    bp.Q.value = 3;
    const from = direction > 0 ? 250 : 1600;
    const to = direction > 0 ? 1600 : 250;
    bp.frequency.setValueAtTime(from, t);
    bp.frequency.exponentialRampToValueAtTime(to, t + 0.55);
    const g = this.ctx.createGain();
    g.gain.setValueAtTime(0.0001, t);
    g.gain.exponentialRampToValueAtTime(0.12, t + 0.12);
    g.gain.exponentialRampToValueAtTime(0.0001, t + 0.7);
    src.connect(bp).connect(g).connect(this.master);
    src.start(t);
    src.stop(t + 0.8);
  }

  /** One eerie ping into the echo line. Public: fired when a sub-agent spawns. */
  ping(freq?: number, gain = 0.05 + Math.random() * 0.04) {
    const t = this.ctx.currentTime;
    // high, slightly inharmonic cluster
    const scale = [830, 996, 1245, 1494, 1868, 2490];
    const f =
      (freq ?? scale[Math.floor(Math.random() * scale.length)]) *
      (0.99 + Math.random() * 0.02);
    const osc = this.ctx.createOscillator();
    osc.type = "sine";
    osc.frequency.setValueAtTime(f, t);
    osc.frequency.exponentialRampToValueAtTime(f * 0.985, t + 0.25);
    const g = this.ctx.createGain();
    g.gain.setValueAtTime(0.0001, t);
    g.gain.exponentialRampToValueAtTime(gain, t + 0.015);
    g.gain.exponentialRampToValueAtTime(0.0001, t + 0.3);
    osc.connect(g).connect(this.delay);
    g.connect(this.master);
    osc.start(t);
    osc.stop(t + 0.35);
  }

  /** Random data-blips while thinking. */
  private scheduleBlip() {
    if (this.state !== "thinking") return;
    this.ping();
    this.blipTimer = window.setTimeout(() => this.scheduleBlip(), 130 + Math.random() * 420);
  }

  /** Called every frame — this is where the sound follows the object. */
  update(angularSpeed: number, level: number) {
    const t = this.ctx.currentTime;
    const v = VOICINGS[this.state];

    // spin opens the drone's filter; voice brightens it further
    const cutoff = v.cutoff + angularSpeed * 260 + level * 900;
    this.droneFilter.frequency.setTargetAtTime(Math.min(cutoff, 4000), t, 0.15);

    // shimmer rides motion + voice
    const shimmer = Math.min(0.06, angularSpeed * 0.006 + level * 0.05);
    this.shimmerGain.gain.setTargetAtTime(shimmer, t, 0.2);
    this.shimmerFilter.frequency.setTargetAtTime(1200 + level * 2600 + angularSpeed * 120, t, 0.25);

    // fast tumbling bends the drone slightly sharp — motion you can hear
    this.oscA.detune.setTargetAtTime(angularSpeed * 4, t, 0.3);
  }
}

import { FilesetResolver, GestureRecognizer, FaceDetector } from "@mediapipe/tasks-vision";

/**
 * Camera-based hand gesture control via MediaPipe GestureRecognizer.
 * Runs fully in-browser (WASM/GPU) — the camera feed never leaves the machine.
 *
 * Vocabulary:
 *  - one pinched hand, moving        → pan; push/pull toward camera → zoom
 *  - open palm held ~1s              → shelve everything to the bento grid
 *  - fist opened over a grid item    → pull it into focus
 *  - finger held to your ear         → start listening
 *  - hands crossed (an X)            → exit note-taker mode
 */

export interface HandState {
  x: number; // screen px, mirrored so moving right means right
  y: number;
  pinching: boolean;
}

export interface GestureEvents {
  onPan: (dxPx: number, dyPx: number) => void;
  onZoom: (factor: number) => void;
  onPalmHold: () => void;
  onEarTouch: () => void;
  onDoublePinch: (x: number, y: number) => void;
  /** Fist released into an open hand — a "grab and pull forth" at that position. */
  onFistOpen: (x: number, y: number) => void;
  /** Both hands crossed into an X and held. */
  onHandsCrossed: () => void;
  onHands: (hands: HandState[]) => void;
  /** Head position for parallax: mirrored offset from center + face size (0 = lost). */
  onHead?: (nx: number, ny: number, size: number) => void;
}

const PINCH_ON = 0.38; // thumb-index distance / hand size — with hysteresis
const PINCH_OFF = 0.52;

/** Grab one webcam frame as a JPEG data URL — reuses the gesture camera when
 *  it's running, otherwise opens the camera briefly just for the snapshot. */
export async function captureCameraFrame(active?: GestureControl | null): Promise<string | null> {
  const grab = (video: HTMLVideoElement) => {
    const s = Math.min(1, 640 / (video.videoWidth || 640));
    const c = document.createElement("canvas");
    c.width = Math.round((video.videoWidth || 640) * s);
    c.height = Math.round((video.videoHeight || 480) * s);
    c.getContext("2d")!.drawImage(video, 0, 0, c.width, c.height);
    return c.toDataURL("image/jpeg", 0.85);
  };

  if (active?.videoEl && active.videoEl.readyState >= 2) return grab(active.videoEl);

  try {
    const stream = await navigator.mediaDevices.getUserMedia({
      video: { width: 640, facingMode: "user" },
    });
    const video = document.createElement("video");
    video.playsInline = true;
    video.muted = true;
    video.srcObject = stream;
    await video.play();
    await new Promise((r) => setTimeout(r, 350)); // let exposure settle
    const url = grab(video);
    stream.getTracks().forEach((t) => t.stop());
    return url;
  } catch {
    return null;
  }
}

export class GestureControl {
  private recognizer: GestureRecognizer | null = null;
  private faceDetector: FaceDetector | null = null;
  private video: HTMLVideoElement | null = null;
  private stream: MediaStream | null = null;
  private running = false;

  private pinchState = [false, false];
  private pinchDownAt = [0, 0]; // for double-pinch detection
  private suppressPan = [false, false]; // after a double pinch, until release
  private lastMid: { x: number; y: number } | null = null;
  private lastSpread: number | null = null;
  private lastPinchSize: number | null = null;
  private palmSince = 0;
  private palmFired = false;
  private lastFistAt = [0, 0];
  private palmSuppressUntil = 0;

  // crossed-hands: learn which labeled hand normally sits on which side,
  // then "crossed" = that order flipping while both hands are up
  private orderEMA = 0;
  private crossedSince = 0;
  private crossedFired = false;

  private frameCount = 0;
  private ears: { x: number; y: number }[] = []; // tragion keypoints, video coords
  private faceWidth = 0;
  private earSince = 0;
  private earFired = false;

  async start(events: GestureEvents) {
    const fileset = await FilesetResolver.forVisionTasks("/mediapipe-wasm");
    this.recognizer = await GestureRecognizer.createFromOptions(fileset, {
      baseOptions: {
        modelAssetPath: "/models/gesture_recognizer.task",
        delegate: "GPU",
      },
      runningMode: "VIDEO",
      numHands: 2,
    });
    this.faceDetector = await FaceDetector.createFromOptions(fileset, {
      baseOptions: {
        modelAssetPath: "/models/blaze_face_short_range.tflite",
        delegate: "GPU",
      },
      runningMode: "VIDEO",
    });

    this.stream = await navigator.mediaDevices.getUserMedia({
      video: { width: 640, height: 480, facingMode: "user" },
    });
    const video = document.createElement("video");
    video.playsInline = true;
    video.muted = true;
    video.srcObject = this.stream;
    await video.play();
    this.video = video;

    this.running = true;
    const tick = () => {
      if (!this.running) return;
      this.processFrame(events);
      requestAnimationFrame(tick);
    };
    requestAnimationFrame(tick);
  }

  get videoEl(): HTMLVideoElement | null {
    return this.video;
  }

  stop() {
    this.running = false;
    this.stream?.getTracks().forEach((t) => t.stop());
    this.recognizer?.close();
    this.faceDetector?.close();
    this.recognizer = null;
    this.faceDetector = null;
    this.video = null;
    this.stream = null;
    this.lastMid = this.lastSpread = this.lastPinchSize = null;
  }

  private processFrame(events: GestureEvents) {
    if (!this.recognizer || !this.video || this.video.readyState < 2) return;
    const res = this.recognizer.recognizeForVideo(this.video, performance.now());

    // ears move slowly — refresh face keypoints every 5th frame
    if (this.faceDetector && this.frameCount++ % 5 === 0) {
      const face = this.faceDetector.detectForVideo(this.video, performance.now()).detections[0];
      const kp = face?.keypoints;
      if (kp && kp.length >= 6) {
        // keypoints 4 and 5 are the right/left ear tragions
        this.ears = [kp[4], kp[5]].map((k) => ({ x: k.x, y: k.y }));
        this.faceWidth = Math.hypot(kp[4].x - kp[5].x, kp[4].y - kp[5].y) || 0;
        // eyes midpoint → head position (mirrored x, like everything else)
        const cx = (kp[0].x + kp[1].x) / 2;
        const cy = (kp[0].y + kp[1].y) / 2;
        events.onHead?.(0.5 - cx, cy - 0.42, this.faceWidth);
      } else {
        this.ears = [];
        events.onHead?.(0, 0, 0); // face lost — drift back to center
      }
    }

    const hands: HandState[] = [];
    const labeled: { label: string; sx: number; sy: number }[] = [];
    let earTouch = false;
    const pinchPoints: { x: number; y: number; size: number }[] = [];
    let palmSeen = false;

    for (let i = 0; i < res.landmarks.length && i < 2; i++) {
      const lm = res.landmarks[i];
      const thumb = lm[4];
      const index = lm[8];
      // hand size reference: wrist to middle-finger base
      const size = Math.hypot(lm[0].x - lm[9].x, lm[0].y - lm[9].y) || 1e-3;
      const pinchDist = Math.hypot(thumb.x - index.x, thumb.y - index.y) / size;

      const was = this.pinchState[i];
      const pinching = was ? pinchDist < PINCH_OFF : pinchDist < PINCH_ON;
      this.pinchState[i] = pinching;

      // mirror x so moving your hand right moves things right
      const sx = (1 - (thumb.x + index.x) / 2) * innerWidth;
      const sy = ((thumb.y + index.y) / 2) * innerHeight;
      hands.push({ x: sx, y: sy, pinching });

      if (!was && pinching) {
        // pinch-release-pinch within 550ms = double pinch
        const now = performance.now();
        if (now - this.pinchDownAt[i] < 550) {
          this.pinchDownAt[i] = 0;
          this.suppressPan[i] = true; // hold still until this pinch releases
          events.onDoublePinch(sx, sy);
        } else {
          this.pinchDownAt[i] = now;
        }
      }
      if (!pinching) this.suppressPan[i] = false;

      if (pinching && !this.suppressPan[i]) pinchPoints.push({ x: sx, y: sy, size });

      const label = res.handedness?.[i]?.[0]?.categoryName;
      if (label) labeled.push({ label, sx, sy });

      const gesture = res.gestures[i]?.[0];
      if (gesture?.categoryName === "Closed_Fist" && gesture.score > 0.5) {
        this.lastFistAt[i] = performance.now();
      }
      if (gesture?.categoryName === "Open_Palm" && gesture.score > 0.6) {
        palmSeen = true;
        // fist just released into an open hand → grab-and-pull at this spot
        if (performance.now() - this.lastFistAt[i] < 700) {
          this.lastFistAt[i] = 0;
          this.palmSuppressUntil = performance.now() + 1500;
          events.onFistOpen(sx, sy);
        }
      }

      // fingertip near an ear tragion (in raw video coords — both unmirrored)
      if (this.ears.length && this.faceWidth > 0) {
        const tip = lm[8];
        for (const ear of this.ears) {
          if (Math.hypot(tip.x - ear.x, tip.y - ear.y) < this.faceWidth * 0.55) {
            earTouch = true;
          }
        }
      }
    }
    events.onHands(hands);

    // crossed hands: the learned left/right order flips and holds ~500ms
    if (labeled.length === 2 && labeled[0].label !== labeled[1].label) {
      const left = labeled.find((h) => h.label === "Left")!;
      const right = labeled.find((h) => h.label === "Right")!;
      const d = left.sx - right.sx;
      const sign = Math.sign(d);
      const aligned = Math.abs(left.sy - right.sy) < innerHeight * 0.3;
      const crossed =
        Math.abs(this.orderEMA) > 0.6 && sign !== 0 && sign !== Math.sign(this.orderEMA) && aligned;
      if (crossed) {
        if (!this.crossedSince) this.crossedSince = performance.now();
        if (!this.crossedFired && performance.now() - this.crossedSince > 500) {
          this.crossedFired = true;
          events.onHandsCrossed();
        }
      } else {
        this.orderEMA += (sign - this.orderEMA) * 0.02;
        this.crossedSince = 0;
        this.crossedFired = false;
      }
    } else {
      this.crossedSince = 0;
      this.crossedFired = false;
    }

    // finger held to the ear ~350ms → listen (release before it can re-fire)
    if (earTouch) {
      if (!this.earSince) this.earSince = performance.now();
      if (!this.earFired && performance.now() - this.earSince > 350) {
        this.earFired = true;
        events.onEarTouch();
      }
    } else {
      this.earSince = 0;
      this.earFired = false;
    }

    // open palm held → shelve all (suppressed right after a grab-and-pull,
    // so keeping the hand open doesn't immediately undo the recall)
    if (palmSeen && !pinchPoints.length && performance.now() > this.palmSuppressUntil) {
      if (!this.palmSince) this.palmSince = performance.now();
      if (!this.palmFired && performance.now() - this.palmSince > 900) {
        this.palmFired = true;
        events.onPalmHold();
      }
    } else {
      this.palmSince = 0;
      this.palmFired = false;
    }

    if (pinchPoints.length === 2) {
      // two-hand pinch: zoom by spread ratio (midpoint also pans)
      const spread = Math.hypot(
        pinchPoints[0].x - pinchPoints[1].x,
        pinchPoints[0].y - pinchPoints[1].y
      );
      if (this.lastSpread) events.onZoom(spread / this.lastSpread);
      this.lastSpread = spread;
      this.lastMid = null;
    } else if (pinchPoints.length === 1) {
      const mid = pinchPoints[0];
      if (this.lastMid) {
        const dx = mid.x - this.lastMid.x;
        const dy = mid.y - this.lastMid.y;
        if (Math.abs(dx) + Math.abs(dy) > 1) events.onPan(dx, -dy);
      }
      // hand size in frame is a depth proxy: pinch + push/pull = one-hand zoom
      if (this.lastPinchSize) {
        const ratio = Math.max(0.93, Math.min(1.07, mid.size / this.lastPinchSize));
        if (Math.abs(ratio - 1) > 0.006) events.onZoom(Math.pow(ratio, 1.6));
      }
      this.lastPinchSize = mid.size;
      this.lastMid = mid;
      this.lastSpread = null;
    } else {
      this.lastMid = null;
      this.lastSpread = null;
      this.lastPinchSize = null;
    }
  }
}

import { FilesetResolver, GestureRecognizer, FaceLandmarker } from "@mediapipe/tasks-vision";
import { Floor, type FloorState } from "./floor";

/**
 * Camera-based hand gesture control via MediaPipe GestureRecognizer.
 * Runs fully in-browser (WASM/GPU) — the camera feed never leaves the machine.
 *
 * Built for a room with several people: every face becomes a tracked person,
 * every hand is bound to the face it belongs to, and only the person holding
 * the floor (see floor.ts) steers. Raise a hand above your head to ask for it;
 * Octa picks from the raised hands, and the holder keeps the floor for as long
 * as they look at it.
 *
 * Vocabulary (floor holder only):
 *  - one pinched hand, moving        → pan; push/pull toward camera → zoom
 *  - open palm held ~1s              → shelve everything to the bento grid
 *  - fist opened over a grid item    → pull it into focus
 *  - finger held to your ear         → start listening
 *  - hands crossed (an X)            → exit note-taker mode
 * Anyone:
 *  - hand held above the head ~1s    → ask for the floor (Octa chooses)
 *
 * Face landmarks also give each person's lip gap and head direction over time,
 * so the app can check that the floor holder was looking at Octa with their
 * lips moving while speech was captured — one shared microphone can't tell
 * voices apart, the camera can.
 */

export interface HandState {
  x: number; // screen px, mirrored so moving right means right
  y: number;
  pinching: boolean;
  /** Tracked person this hand belongs to (null = no face to bind to). */
  person: number | null;
  /** True when this hand is allowed to steer. */
  active: boolean;
  /** Held above the head — asking for the floor. */
  raised: boolean;
}

export interface MeetingState extends FloorState {
  persons: number[];
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
  /** Who is in view, who holds the floor, who is waiting. */
  onMeeting?: (s: MeetingState) => void;
}

const PINCH_ON = 0.38; // thumb-index distance / hand size — with hysteresis
const PINCH_OFF = 0.52;

const MAX_HANDS = 8;
const PERSON_TTL = 2500; // ms a face may go unseen before the person is dropped
const HAND_TTL = 300; // ms a hand track survives a missed frame
const RAISE_HOLD = 1000; // ms a hand must stay above the head to count
const BIND_MAX = 4.5; // max hand-to-face distance, in face widths
const MAX_FACES = 6;
const MOUTH_LOG_MS = 60000; // how much lip history each person keeps
const MOUTH_WINDOW = 600; // ms — lips "moving" = gap range over this window…
const MOUTH_MOVING = 0.025; // …exceeds this share of face height
const MOUTH_SPOKE = 0.12; // share of face samples that must be "moving" to count as speaking
// "looking at Octa" = face turned toward the camera (which sits by the screen):
const GAZE_YAW = 0.13; // nose offset from the cheek midpoint, share of face width
const GAZE_PITCH: [number, number] = [0.22, 0.62]; // nose between eye line (0) and chin (1)
const GAZE_IRIS = 0.2; // irises off-center in their eyes by more than this → looking aside

interface Person {
  id: number;
  cx: number; // face center, video px
  cy: number;
  w: number; // face box size, video px
  h: number;
  top: number;
  ears: { x: number; y: number }[]; // normalized video coords
  lastSeen: number;
  /** Lip gap / face height per face frame, with whether the lips were moving
   *  and the face was turned to Octa then. */
  mouth: { t: number; gap: number; moving: boolean; looking: boolean }[];
  lookEma: number; // smoothed 0..1 "facing Octa"
  looking: boolean;
}

/** Did the floor holder's lips move during a capture? */
export interface MouthCheck {
  holder: number;
  spoke: boolean; // lips moving while facing Octa — what the gate uses
  facing: boolean; // mostly facing Octa during the capture
  moving: boolean; // lips moving at all during the capture
}

interface HandTrack {
  id: number;
  wx: number; // wrist, video px
  wy: number;
  lastSeen: number;
  owner: number | null;
  pinching: boolean;
  pinchDownAt: number; // for double-pinch detection
  suppressPan: boolean; // after a double pinch, until release
  lastFistAt: number;
  raiseSince: number;
  raiseFired: boolean;
}

export interface Crop {
  x: number; // video px
  y: number;
  w: number;
  h: number;
}

/** Head pose + iris check from face-mesh landmarks (normalized coords).
 *  1 nose tip · 234/454 face sides · 33/263 outer eye corners · 152 chin ·
 *  468/473 iris centers with 33–133 and 362–263 as their eye's corners. */
function facing(lm: { x: number; y: number }[]): boolean {
  const left = lm[234];
  const right = lm[454];
  const faceW = right.x - left.x;
  if (Math.abs(faceW) < 1e-4) return false;
  const yaw = (lm[1].x - left.x) / faceW - 0.5;
  const eyeY = (lm[33].y + lm[263].y) / 2;
  const pitch = (lm[1].y - eyeY) / (lm[152].y - eyeY || 1e-4);
  let iris = 0;
  if (lm.length > 473) {
    const a = (lm[468].x - lm[33].x) / (lm[133].x - lm[33].x || 1e-4) - 0.5;
    const b = (lm[473].x - lm[362].x) / (lm[263].x - lm[362].x || 1e-4) - 0.5;
    iris = (a + b) / 2;
  }
  return (
    Math.abs(yaw) < GAZE_YAW &&
    pitch > GAZE_PITCH[0] &&
    pitch < GAZE_PITCH[1] &&
    Math.abs(iris) < GAZE_IRIS
  );
}

/** Grab one webcam frame as a JPEG data URL — reuses the gesture camera when
 *  it's running, otherwise opens the camera briefly just for the snapshot.
 *  With a crop, only that region of the gesture camera's frame is kept. */
export async function captureCameraFrame(
  active?: GestureControl | null,
  crop?: Crop | null
): Promise<string | null> {
  const grab = (video: HTMLVideoElement, r?: Crop | null) => {
    const vw = video.videoWidth || 640;
    const vh = video.videoHeight || 480;
    const src = r ?? { x: 0, y: 0, w: vw, h: vh };
    const s = Math.min(1, 640 / src.w);
    const c = document.createElement("canvas");
    c.width = Math.round(src.w * s);
    c.height = Math.round(src.h * s);
    c.getContext("2d")!.drawImage(video, src.x, src.y, src.w, src.h, 0, 0, c.width, c.height);
    return c.toDataURL("image/jpeg", 0.85);
  };

  if (active?.videoEl && active.videoEl.readyState >= 2) return grab(active.videoEl, crop);

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
  private faceLandmarker: FaceLandmarker | null = null;
  private video: HTMLVideoElement | null = null;
  private stream: MediaStream | null = null;
  private running = false;

  private persons: Person[] = [];
  private nextPersonId = 1;
  private tracks: HandTrack[] = [];
  private nextHandId = 1;
  private floor: Floor | null = null;
  private lastHolder: number | null = null;
  private lastPersonIds = "";

  private lastMid: { x: number; y: number } | null = null;
  private lastSpread: number | null = null;
  private lastPinchSize: number | null = null;
  private palmSince = 0;
  private palmFired = false;
  private palmSuppressUntil = 0;

  // crossed-hands: learn which labeled hand normally sits on which side,
  // then "crossed" = that order flipping while both hands are up
  private orderEMA = 0;
  private crossedSince = 0;
  private crossedFired = false;

  private frameCount = 0;
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
      numHands: MAX_HANDS,
    });
    this.faceLandmarker = await FaceLandmarker.createFromOptions(fileset, {
      baseOptions: {
        modelAssetPath: "/models/face_landmarker.task",
        delegate: "GPU",
      },
      runningMode: "VIDEO",
      numFaces: MAX_FACES,
      minFaceDetectionConfidence: 0.4,
    });

    // wider frame: a meeting room has people at the edges, not just one face in the middle
    this.stream = await navigator.mediaDevices.getUserMedia({
      video: { width: 1280, height: 720, facingMode: "user" },
    });
    const video = document.createElement("video");
    video.playsInline = true;
    video.muted = true;
    video.srcObject = this.stream;
    await video.play();
    this.video = video;

    this.floor = new Floor((s) =>
      events.onMeeting?.({ ...s, persons: this.persons.map((p) => p.id) })
    );

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

  /** What the camera currently sees, in video px — for the debug camera view. */
  snapshot() {
    return {
      holder: this.floor?.holder ?? null,
      queue: [...(this.floor?.queue ?? [])],
      persons: this.persons.map((p) => ({
        id: p.id, x: p.cx - p.w / 2, y: p.cy - p.h / 2, w: p.w, h: p.h,
        talking: !!p.mouth.at(-1)?.moving && performance.now() - p.lastSeen < 400,
        looking: p.looking && performance.now() - p.lastSeen < 400,
        // a hand of theirs is above their head right now (asking for the floor)
        handUp: this.tracks.some((t) => t.owner === p.id && t.raiseSince > 0),
      })),
      hands: this.tracks.map((t) => ({ x: t.wx, y: t.wy, owner: t.owner, raised: t.raiseSince > 0 })),
    };
  }

  /** Whether the floor holder's lips moved since `since` (performance.now()).
   *  null when it can't be judged — no floor, or their face was mostly out of
   *  view — in which case the caller should give them the benefit of the doubt. */
  holderSpokeSince(since: number): MouthCheck | null {
    const holder = this.floor?.holder ?? null;
    const p = this.persons.find((q) => q.id === holder);
    if (!p || holder === null) return null;
    const samples = p.mouth.filter((m) => m.t >= since);
    const span = performance.now() - since;
    // face frames run at roughly 20/s; under a third of that coverage → can't tell
    if (samples.length < Math.max(3, (span / 1000) * 20 * 0.33)) return null;
    // it only counts as theirs if they spoke *to Octa*: lips moving while facing it
    const share = (f: (m: Person["mouth"][number]) => boolean) =>
      samples.filter(f).length / samples.length;
    return {
      holder,
      spoke: share((m) => m.moving && m.looking) >= MOUTH_SPOKE,
      facing: share((m) => m.looking) >= 0.5,
      moving: share((m) => m.moving) >= MOUTH_SPOKE,
    };
  }

  /** The floor holder, framed head and shoulders (video px) — for anything that
   *  should be about them rather than the whole room. null = no holder in view. */
  holderCrop(): { person: number; crop: Crop } | null {
    const holder = this.floor?.holder ?? null;
    const p = this.persons.find((q) => q.id === holder);
    if (!p || holder === null || !this.video) return null;
    const vw = this.video.videoWidth;
    const vh = this.video.videoHeight;
    // face box widened 3× and stretched down to take in hair, collar and clothing
    const x = Math.max(0, p.cx - p.w * 1.5);
    const y = Math.max(0, p.cy - p.h * 1.1);
    const w = Math.min(vw - x, p.w * 3);
    const h = Math.min(vh - y, p.h * 3.4);
    return { person: holder, crop: { x, y, w, h } };
  }

  /** Clear the floor and the queue — whoever raises a hand next takes it. */
  resetFloor() {
    this.floor?.reset();
  }

  stop() {
    this.running = false;
    this.stream?.getTracks().forEach((t) => t.stop());
    this.recognizer?.close();
    this.faceLandmarker?.close();
    this.recognizer = null;
    this.faceLandmarker = null;
    this.video = null;
    this.stream = null;
    this.floor = null;
    this.persons = [];
    this.tracks = [];
    this.lastPersonIds = "";
    this.lastMid = this.lastSpread = this.lastPinchSize = null;
  }

  /** Match this frame's faces to known persons (nearest center), minting new ones. */
  private updatePersons(now: number, vw: number, vh: number) {
    const faces = this.faceLandmarker!.detectForVideo(this.video!, now).faceLandmarks;
    const free = new Set(this.persons);
    for (const lm of faces) {
      let x0 = Infinity, y0 = Infinity, x1 = -Infinity, y1 = -Infinity;
      for (const l of lm) {
        x0 = Math.min(x0, l.x); x1 = Math.max(x1, l.x);
        y0 = Math.min(y0, l.y); y1 = Math.max(y1, l.y);
      }
      const b = { originX: x0 * vw, originY: y0 * vh, width: (x1 - x0) * vw, height: (y1 - y0) * vh };
      const cx = b.originX + b.width / 2;
      const cy = b.originY + b.height / 2;
      let best: Person | null = null;
      let bestDist = Infinity;
      for (const p of free) {
        const dist = Math.hypot(p.cx - cx, p.cy - cy);
        if (dist < Math.max(p.w, b.width) * 0.9 && dist < bestDist) {
          best = p;
          bestDist = dist;
        }
      }
      // 234 / 454 are the face-contour points beside the ears
      const ears = [lm[234], lm[454]].map((k) => ({ x: k.x, y: k.y }));
      // inner-lip gap (13 upper, 14 lower) over forehead-to-chin (10, 152)
      const faceH = Math.hypot((lm[10].x - lm[152].x) * vw, (lm[10].y - lm[152].y) * vh) || 1;
      const gap = Math.hypot((lm[13].x - lm[14].x) * vw, (lm[13].y - lm[14].y) * vh) / faceH;
      if (best) {
        free.delete(best);
        best.cx += (cx - best.cx) * 0.5;
        best.cy += (cy - best.cy) * 0.5;
        best.w += (b.width - best.w) * 0.5;
        best.h += (b.height - best.h) * 0.5;
        best.top = b.originY;
        best.ears = ears;
        best.lastSeen = now;
      } else {
        best = {
          id: this.nextPersonId++,
          cx, cy, w: b.width, h: b.height, top: b.originY, ears, lastSeen: now, mouth: [],
          lookEma: 0, looking: false,
        };
        this.persons.push(best);
      }
      // speech = lips opening and closing, not just an open mouth
      const recent = best.mouth.filter((m) => now - m.t < MOUTH_WINDOW).map((m) => m.gap);
      recent.push(gap);
      const moving = Math.max(...recent) - Math.min(...recent) > MOUTH_MOVING;
      // facing Octa: head turned to the camera and eyes not cut to the side
      best.lookEma += ((facing(lm) ? 1 : 0) - best.lookEma) * 0.35;
      best.looking = best.lookEma > (best.looking ? 0.35 : 0.65); // hysteresis
      best.mouth.push({ t: now, gap, moving, looking: best.looking });
      while (best.mouth.length && now - best.mouth[0].t > MOUTH_LOG_MS) best.mouth.shift();
    }
    this.persons = this.persons.filter((p) => now - p.lastSeen < PERSON_TTL);
  }

  /** Which person does a wrist belong to? Hands hang below and beside the face,
   *  so distance below counts half. Sticky: the current owner keeps the hand
   *  unless someone else is clearly closer. */
  private bindHand(t: HandTrack) {
    const cost = (p: Person) => {
      const dx = (t.wx - p.cx) / p.w;
      const dy = (t.wy - p.cy) / p.w;
      return Math.hypot(dx, dy > 0 ? dy * 0.5 : dy);
    };
    let best: Person | null = null;
    let bestCost = Infinity;
    for (const p of this.persons) {
      const c = cost(p);
      if (c < bestCost) {
        best = p;
        bestCost = c;
      }
    }
    const current = this.persons.find((p) => p.id === t.owner);
    if (current && cost(current) < BIND_MAX && cost(current) < bestCost * 1.35) return;
    t.owner = best && bestCost < BIND_MAX ? best.id : null;
  }

  private processFrame(events: GestureEvents) {
    if (!this.recognizer || !this.video || this.video.readyState < 2) return;
    const now = performance.now();
    const vw = this.video.videoWidth || 1280;
    const vh = this.video.videoHeight || 720;
    const res = this.recognizer.recognizeForVideo(this.video, now);

    // every 2nd frame: faces move slowly, but lips need ~20+ samples a second
    const faceFrame = this.faceLandmarker && this.frameCount++ % 2 === 0;
    if (faceFrame) {
      this.updatePersons(now, vw, vh);
      const ids = this.persons.map((p) => p.id);
      if (ids.join() !== this.lastPersonIds) {
        this.lastPersonIds = ids.join();
        const f = this.floor!;
        events.onMeeting?.({ holder: f.holder, queue: [...f.queue], persons: ids });
      }
    }

    // --- match detected hands to tracks by wrist position ---------------------
    const seen = new Set<HandTrack>();
    const frameHands: { t: HandTrack; i: number }[] = [];
    for (let i = 0; i < res.landmarks.length; i++) {
      const lm = res.landmarks[i];
      const wx = lm[0].x * vw;
      const wy = lm[0].y * vh;
      const sizePx = Math.hypot((lm[0].x - lm[9].x) * vw, (lm[0].y - lm[9].y) * vh);
      let best: HandTrack | null = null;
      let bestDist = Infinity;
      for (const t of this.tracks) {
        if (seen.has(t)) continue;
        const dist = Math.hypot(t.wx - wx, t.wy - wy);
        if (dist < Math.max(sizePx * 1.5, 40) && dist < bestDist) {
          best = t;
          bestDist = dist;
        }
      }
      if (!best) {
        best = {
          id: this.nextHandId++, wx, wy, lastSeen: now, owner: null,
          pinching: false, pinchDownAt: 0, suppressPan: false, lastFistAt: 0,
          raiseSince: 0, raiseFired: false,
        };
        this.tracks.push(best);
      }
      best.wx = wx;
      best.wy = wy;
      best.lastSeen = now;
      seen.add(best);
      this.bindHand(best);
      frameHands.push({ t: best, i });
    }
    this.tracks = this.tracks.filter((t) => now - t.lastSeen < HAND_TTL);

    // --- raise-to-speak: hand above the owner's head, held --------------------
    const raised = new Set<HandTrack>();
    for (const { t, i } of frameHands) {
      const p = this.persons.find((q) => q.id === t.owner);
      const tipY = res.landmarks[i][8].y * vh;
      const up = !!p && t.wy < p.top + p.h * 0.3 && tipY < p.top;
      if (!up) {
        t.raiseSince = 0;
        t.raiseFired = false;
        continue;
      }
      raised.add(t);
      if (!t.raiseSince) t.raiseSince = now;
      if (!t.raiseFired && now - t.raiseSince > RAISE_HOLD) {
        t.raiseFired = true;
        this.floor?.raise(p.id);
      }
    }

    // --- who steers -----------------------------------------------------------
    // With faces in view only the floor holder's hands count; with no faces at
    // all (dim room, camera too far) fall back to the first two hands, as before.
    const holder = this.floor?.holder ?? null;
    const anonymous = this.persons.length === 0;
    if (holder !== this.lastHolder) {
      // a new pair of hands — don't carry the previous holder's motion over
      this.lastHolder = holder;
      this.lastMid = this.lastSpread = this.lastPinchSize = null;
      this.orderEMA = 0;
    }
    const isActive = (t: HandTrack) =>
      !raised.has(t) && (anonymous ? true : holder !== null && t.owner === holder);
    const active = frameHands.filter(({ t }) => isActive(t)).slice(0, 2);

    const hands: HandState[] = [];
    const labeled: { label: string; sx: number; sy: number }[] = [];
    let earTouch = false;
    const pinchPoints: { x: number; y: number; size: number }[] = [];
    let palmSeen = false;
    const holderPerson = this.persons.find((p) => p.id === holder);

    for (const { t, i } of frameHands) {
      const lm = res.landmarks[i];
      const thumb = lm[4];
      const index = lm[8];
      // hand size reference: wrist to middle-finger base
      const size = Math.hypot(lm[0].x - lm[9].x, lm[0].y - lm[9].y) || 1e-3;
      const pinchDist = Math.hypot(thumb.x - index.x, thumb.y - index.y) / size;

      const was = t.pinching;
      const pinching = was ? pinchDist < PINCH_OFF : pinchDist < PINCH_ON;
      t.pinching = pinching;

      // mirror x so moving your hand right moves things right
      const sx = (1 - (thumb.x + index.x) / 2) * innerWidth;
      const sy = ((thumb.y + index.y) / 2) * innerHeight;
      const on = active.some((a) => a.t === t);
      hands.push({ x: sx, y: sy, pinching, person: t.owner, active: on, raised: raised.has(t) });
      if (!on) continue;

      if (!was && pinching) {
        // pinch-release-pinch within 550ms = double pinch
        if (now - t.pinchDownAt < 550) {
          t.pinchDownAt = 0;
          t.suppressPan = true; // hold still until this pinch releases
          events.onDoublePinch(sx, sy);
        } else {
          t.pinchDownAt = now;
        }
      }
      if (!pinching) t.suppressPan = false;

      if (pinching && !t.suppressPan) pinchPoints.push({ x: sx, y: sy, size });

      const label = res.handedness?.[i]?.[0]?.categoryName;
      if (label) labeled.push({ label, sx, sy });

      const gesture = res.gestures[i]?.[0];
      if (gesture?.categoryName === "Closed_Fist" && gesture.score > 0.5) {
        t.lastFistAt = now;
      }
      if (gesture?.categoryName === "Open_Palm" && gesture.score > 0.6) {
        palmSeen = true;
        // fist just released into an open hand → grab-and-pull at this spot
        if (now - t.lastFistAt < 700) {
          t.lastFistAt = 0;
          this.palmSuppressUntil = now + 1500;
          events.onFistOpen(sx, sy);
        }
      }

      // fingertip near one of the holder's ear tragions (raw video coords)
      const ears = holderPerson?.ears ?? [];
      if (ears.length) {
        const earSpan = Math.hypot(ears[0].x - ears[1].x, ears[0].y - ears[1].y);
        for (const ear of ears) {
          if (Math.hypot(index.x - ear.x, index.y - ear.y) < earSpan * 0.55) earTouch = true;
        }
      }
    }
    events.onHands(hands);

    this.floor?.update(
      this.persons.map((p) => p.id),
      new Set(this.persons.filter((p) => p.looking).map((p) => p.id)),
      now
    );

    // parallax follows whoever holds the floor (else the nearest face)
    if (faceFrame) {
      const head =
        holderPerson ?? [...this.persons].sort((a, b) => b.w - a.w)[0];
      if (head && now - head.lastSeen < 400) {
        events.onHead?.(0.5 - head.cx / vw, head.cy / vh - 0.42, head.w / vw);
      } else {
        events.onHead?.(0, 0, 0); // face lost — drift back to center
      }
    }

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
        if (!this.crossedSince) this.crossedSince = now;
        if (!this.crossedFired && now - this.crossedSince > 500) {
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
      if (!this.earSince) this.earSince = now;
      if (!this.earFired && now - this.earSince > 350) {
        this.earFired = true;
        events.onEarTouch();
      }
    } else {
      this.earSince = 0;
      this.earFired = false;
    }

    // open palm held → shelve all (suppressed right after a grab-and-pull,
    // so keeping the hand open doesn't immediately undo the recall)
    if (palmSeen && !pinchPoints.length && now > this.palmSuppressUntil) {
      if (!this.palmSince) this.palmSince = now;
      if (!this.palmFired && now - this.palmSince > 900) {
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

import { FilesetResolver, GestureRecognizer, FaceLandmarker, PoseLandmarker, type GestureRecognizerResult } from "@mediapipe/tasks-vision";
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
  /** The floor holder started talking while facing Octa — time to listen. */
  onHolderTalk?: (person: number) => void;
  /** With the floor free, someone faces Octa and starts talking. */
  onFreeTalk?: (person: number) => void;
  /** A number was given to someone Octa can't tie to anyone it saw before. */
  onNewPerson?: (person: number) => void;
}

const PINCH_ON = 0.38; // thumb-index distance / hand size — with hysteresis
const PINCH_OFF = 0.52;

const MAX_HANDS = 4;
// Vision runs at a fixed pace, not per display frame (a 120 Hz screen would
// double the work), and alternates: hands on one tick, people + faces on the next.
const VISION_MS = 50; // 20 ticks a second → 10 Hz hands, 10 Hz people/lips
const HAND_INPUT_W = 960; // the hand model gets a downscaled frame
const POSE_INPUT = 512; // pose tiles are downscaled to at most this (px, longest side)
// ms a person may go unseen before they're dropped: long enough to look down,
// lean out of frame or turn around without being mistaken for someone new
const PERSON_TTL = 6000;
const CROP = 256; // face crops are analysed at this size (px, square)
const FACE_FRESH = 500; // ms a face-mesh read stays the authority on gaze
const HAND_TTL = 300; // ms a hand track survives a missed frame
const RAISE_HOLD = 1000; // ms a hand must stay above the head to count
const BIND_MAX = 4.5; // max hand-to-face distance, in face widths
const MAX_FACES = 6;
const MOUTH_LOG_MS = 60000; // how much lip history each person keeps
const MOUTH_WINDOW = 600; // ms — lips "moving" = gap range over this window…
const MOUTH_MOVING = 0.025; // …exceeds this share of face height
// Speech is judged by how LONG the lips moved, not by what share of the capture
// they moved: a voice behind you keeps the recognizer listening, which stretches
// the capture and would dilute any share-based test.
const SPOKE_TO_OCTA_MS = 300; // lips moving while facing Octa
const SPOKE_LOOSE_MS = 450; // lips moving, facing Octa for part of it
const SPOKE_ALONE_MS = 250; // only one face in view: lips moving at all
// "looking at Octa" = face turned toward the camera (which sits by the screen):
// In a room people sit along the table and turn their heads to the screen at
// the end: faces aimed at the camera measured ±0.18-0.26 in a test frame.
const GAZE_YAW = 0.3; // nose offset from the cheek midpoint, share of face width
const GAZE_PITCH: [number, number] = [0.22, 0.62]; // nose between eye line (0) and chin (1)
const GAZE_IRIS = 0.2; // irises off-center in their eyes by more than this → looking aside

interface Person {
  id: number;
  cx: number; // head center, video px
  cy: number;
  w: number; // head box size, video px
  h: number;
  top: number;
  ears: { x: number; y: number }[]; // normalized video coords
  wrists: { x: number; y: number }[]; // visible wrists from the body pose, video px
  lastSeen: number;
  poseFacing: boolean; // head turned to the camera, judged from the body pose
  faceAt: number; // last time the face mesh read this person's face
  raiseSince: number; // wrist above the head (from the pose), held
  raiseFired: boolean;
  /** Elbows and wrists (video px): a raised arm can look like a small person. */
  arms: { x: number; y: number }[];
  /** Torso box (shoulders–hips, video px): stable when they turn or raise a hand. */
  box: { x0: number; y0: number; x1: number; y1: number };
  /** A still of their face, taken the moment they faced the camera — kept for the tile. */
  portrait: HTMLCanvasElement | null;
  portraitAt: number;
  /** Where the face mesh last found their face (video px) — the strongest
   *  evidence of who is who: two tracks reading one face are one person. */
  faceBox: { x0: number; y0: number; x1: number; y1: number } | null;
  /** Lip gap / face height per face frame, with whether the lips were moving
   *  and the face was turned to Octa then. */
  mouth: { t: number; gap: number; moving: boolean; looking: boolean }[];
  lookEma: number; // smoothed 0..1 "facing Octa"
  looking: boolean;
}

/** Did the floor holder's lips move during a capture? */
export interface MouthCheck {
  holder: number;
  spoke: boolean; // what the gate uses — see judgeSpeech
  facing: boolean; // facing Octa for a good part of the capture
  moving: boolean; // lips moved for a meaningful stretch
  alone: boolean; // the holder was the only face in view
  movingMs: number;
  toOctaMs: number; // lips moving while facing Octa
}

type MouthSample = { t: number; moving: boolean; looking: boolean };

/** Decide whether the floor holder spoke during a capture of `spanMs`. */
export function judgeSpeech(
  samples: MouthSample[],
  spanMs: number,
  alone: boolean
): Omit<MouthCheck, "holder"> {
  const dt = spanMs / Math.max(1, samples.length); // time each face sample stands for
  const count = (f: (m: MouthSample) => boolean) => samples.filter(f).length;
  const movingMs = count((m) => m.moving) * dt;
  const toOctaMs = count((m) => m.moving && m.looking) * dt;
  const facingShare = count((m) => m.looking) / Math.max(1, samples.length);
  const spoke =
    toOctaMs >= SPOKE_TO_OCTA_MS ||
    (movingMs >= SPOKE_LOOSE_MS && facingShare >= 0.25) ||
    (alone && movingMs >= SPOKE_ALONE_MS);
  return {
    spoke,
    facing: facingShare >= 0.5,
    moving: movingMs >= SPOKE_ALONE_MS,
    alone,
    movingMs: Math.round(movingMs),
    toOctaMs: Math.round(toOctaMs),
  };
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
  private poseLandmarker: PoseLandmarker | null = null;
  private crop: HTMLCanvasElement | null = null;
  private cropTurn = 0; // round-robin over people for face crops
  // The pose model shrinks its input to ~256 px, so on a whole 16:9 room frame
  // people at the far end vanish. It runs on overlapping tiles instead (3×2
  // plus the full frame), two tiles per pose frame, a full sweep every ~0.25 s.
  private tiles: { x: number; y: number; w: number; h: number }[] = [];
  private tileCanvas: HTMLCanvasElement | null = null;
  private tileTurn = 0;
  private video: HTMLVideoElement | null = null;
  private stream: MediaStream | null = null;
  private running = false;

  private persons: Person[] = [];
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
  private lastTick = 0;
  private poseTicks = 0;
  private candidates: { cx: number; cy: number; w: number; seen: number; at: number; first: number }[] = [];
  private gone: {
    id: number; cx: number; cy: number; w: number;
    portrait: HTMLCanvasElement | null; portraitAt: number; at: number;
  }[] = [];
  private handCanvas: HTMLCanvasElement | null = null;
  private lastHands: GestureRecognizerResult | null = null;
  private earSince = 0;
  private earFired = false;
  private talkSince = 0; // floor holder: lips moving while facing Octa since…
  private talkFired = false;

  private events: GestureEvents | null = null;

  /** cameraId: a meeting-room camera from the device list ("" = default). */
  async start(events: GestureEvents, cameraId = "") {
    this.events = events;
    const fileset = await FilesetResolver.forVisionTasks("/mediapipe-wasm");
    this.recognizer = await GestureRecognizer.createFromOptions(fileset, {
      baseOptions: {
        modelAssetPath: "/models/gesture_recognizer.task",
        delegate: "GPU",
      },
      runningMode: "VIDEO",
      numHands: MAX_HANDS,
    });
    // People are found by body pose — it sees a whole person across a meeting
    // room, where the face detector gives up beyond ~2 m. Each head is then
    // cropped from the full-resolution frame, enlarged, and read by the face
    // mesh for lips and gaze.
    this.poseLandmarker = await PoseLandmarker.createFromOptions(fileset, {
      baseOptions: {
        modelAssetPath: "/models/pose_landmarker_lite.task",
        delegate: "GPU",
      },
      // IMAGE mode: it runs on tiles of the frame, not one continuous stream
      runningMode: "IMAGE",
      numPoses: MAX_FACES,
      minPoseDetectionConfidence: 0.3,
      minPosePresenceConfidence: 0.3,
    });
    this.faceLandmarker = await FaceLandmarker.createFromOptions(fileset, {
      baseOptions: {
        modelAssetPath: "/models/face_landmarker.task",
        delegate: "GPU",
      },
      runningMode: "IMAGE",
      numFaces: 1,
      minFaceDetectionConfidence: 0.25,
      minFacePresenceConfidence: 0.25,
    });
    this.crop = document.createElement("canvas");
    this.crop.width = this.crop.height = CROP;

    // as many pixels as the camera gives (room cameras do 1080p): people at the
    // back of the room are small, and every pixel helps the detectors find them
    this.stream = await navigator.mediaDevices.getUserMedia({
      video: cameraId
        ? { deviceId: { exact: cameraId }, width: { ideal: 1920 }, height: { ideal: 1080 } }
        : { width: { ideal: 1920 }, height: { ideal: 1080 }, facingMode: "user" },
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
        talking: !!p.mouth.at(-1)?.moving && performance.now() - (p.mouth.at(-1)?.t ?? 0) < 400,
        looking: p.looking && performance.now() - p.lastSeen < 1000,
        // a hand of theirs is above their head right now (asking for the floor)
        handUp: p.raiseSince > 0 || this.tracks.some((t) => t.owner === p.id && t.raiseSince > 0),
      })),
      hands: this.tracks.map((t) => ({ x: t.wx, y: t.wy, owner: t.owner, raised: t.raiseSince > 0 })),
    };
  }

  /** Copy a person's kept portrait into a canvas. False until one was taken. */
  drawFace(id: number, canvas: HTMLCanvasElement): boolean {
    const p = this.persons.find((q) => q.id === id);
    if (!p?.portrait) return false;
    // only repaint when the portrait changed
    const stamp = String(p.portraitAt);
    if (canvas.dataset.stamp !== stamp) {
      canvas.dataset.stamp = stamp;
      canvas.getContext("2d")!.drawImage(p.portrait, 0, 0, canvas.width, canvas.height);
    }
    return true;
  }

  /** Head and a little shoulder, mirrored like everything else on screen. */
  private snapPortrait(p: Person, now: number, face: { x0: number; y0: number; x1: number; y1: number }) {
    const video = this.video!;
    // the mesh spans brow to chin: pad it so hair, ears and chin all fit
    const side = Math.min(video.videoHeight, Math.max(face.x1 - face.x0, face.y1 - face.y0) * 1.7);
    const fx = (face.x0 + face.x1) / 2;
    const fy = (face.y0 + face.y1) / 2 - side * 0.06; // a touch of headroom
    const sx = Math.max(0, Math.min(video.videoWidth - side, fx - side / 2));
    const sy = Math.max(0, Math.min(video.videoHeight - side, fy - side / 2));
    const c = (p.portrait ??= document.createElement("canvas"));
    c.width = c.height = 160;
    const ctx = c.getContext("2d")!;
    ctx.save();
    ctx.translate(c.width, 0);
    ctx.scale(-1, 1);
    ctx.drawImage(video, sx, sy, side, side, 0, 0, c.width, c.height);
    ctx.restore();
    p.portraitAt = now;
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
    return { holder, ...judgeSpeech(samples, span, this.persons.length === 1) };
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

  /** Give the free floor to this person (they addressed Octa by name). */
  giveFloor(id: number) {
    this.floor?.give(id);
  }

  /** Hand the floor to this person regardless (they called Octa by name). */
  takeFloor(id: number) {
    this.floor?.force(id);
  }

  /** Is this person looking at Octa right now? */
  isLooking(id: number): boolean {
    return !!this.persons.find((p) => p.id === id)?.looking;
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
    this.poseLandmarker?.close();
    this.recognizer = null;
    this.faceLandmarker = null;
    this.poseLandmarker = null;
    this.video = null;
    this.stream = null;
    this.floor = null;
    this.persons = [];
    this.candidates = [];
    this.gone = [];
    this.tracks = [];
    this.lastPersonIds = "";
    this.lastMid = this.lastSpread = this.lastPinchSize = null;
  }

  /** Find people by body pose, match them to known persons, and read a couple
   *  of faces (the floor holder always) from enlarged crops. */
  private updatePersons(now: number, vw: number, vh: number) {
    if (!this.tiles.length || this.tiles[0].w !== vw || this.tiles[0].h !== vh) {
      const tiles = [{ x: 0, y: 0, w: vw, h: vh }];
      const cols = 3;
      const rows = 2;
      const tw = Math.round((vw / cols) * 1.35);
      const th = Math.round((vh / rows) * 1.35);
      for (let r = 0; r < rows; r++) {
        for (let c = 0; c < cols; c++) {
          const x = Math.round(Math.min(vw - tw, Math.max(0, ((c + 0.5) * vw) / cols - tw / 2)));
          const y = Math.round(Math.min(vh - th, Math.max(0, ((r + 0.5) * vh) / rows - th / 2)));
          tiles.push({ x, y, w: tw, h: th });
        }
      }
      this.tiles = tiles;
      this.tileCanvas ??= document.createElement("canvas");
    }
    {
      const tile = this.tiles[this.tileTurn++ % this.tiles.length];
      const cv = this.tileCanvas!;
      // the model sees ~256 px anyway: shrink the tile before handing it over
      const k = Math.min(1, POSE_INPUT / Math.max(tile.w, tile.h));
      const cw = Math.round(tile.w * k);
      const ch = Math.round(tile.h * k);
      if (cv.width !== cw || cv.height !== ch) {
        cv.width = cw;
        cv.height = ch;
      }
      cv.getContext("2d")!.drawImage(this.video!, tile.x, tile.y, tile.w, tile.h, 0, 0, cw, ch);
      for (const lm of this.poseLandmarker!.detect(cv).landmarks) {
        this.seePose(
          lm.map((l) => ({ x: tile.x + l.x * tile.w, y: tile.y + l.y * tile.h, v: l.visibility ?? 1 })),
          now,
          vw,
          vh,
          tile
        );
      }
    }
    // people who've gone quiet for a while leave — remembered for a minute
    for (const p of this.persons) {
      if (now - p.lastSeen >= PERSON_TTL) {
        this.gone.push({ id: p.id, cx: p.cx, cy: p.cy, w: p.w, portrait: p.portrait, portraitAt: p.portraitAt, at: now });
      }
    }
    this.persons = this.persons.filter((p) => now - p.lastSeen < PERSON_TTL);
    this.gone = this.gone.filter((g) => now - g.at < 60_000);
    this.candidates = this.candidates.filter((c) => now - c.at < 1500);
    // two tracks on one head (overlapping tiles) — the older one stays
    for (const p of [...this.persons]) {
      const twin = this.persons.find(
        (q) =>
          q !== p &&
          q.id < p.id &&
          (Math.hypot(q.cx - p.cx, q.cy - p.cy) < Math.max(q.w, p.w) * 0.8 ||
            this.sameBody(q, p.box, { x: p.cx, y: p.cy }) ||
            this.sameFace(q, p, now))
      );
      if (twin) {
        twin.portrait ??= p.portrait;
        twin.lastSeen = Math.max(twin.lastSeen, p.lastSeen);
        this.persons = this.persons.filter((q) => q !== p);
      }
    }

    // face crops: the floor holder every time (their lips gate speech), plus
    // one other person in turn
    const holder = this.persons.find((p) => p.id === this.floor?.holder);
    // (a person is re-seen once per tile sweep, ~0.25 s — "recent" means within 1 s)
    const others = this.persons.filter((p) => p !== holder && now - p.lastSeen < 1000);
    // the holder's lips every tick; everyone else takes turns every other tick
    const pick = others.length && this.poseTicks++ % 2 === 0 ? others[this.cropTurn++ % others.length] : undefined;
    for (const p of [holder, pick]) if (p && now - p.lastSeen < 1000) this.readFace(p, now);
  }

  /** One detected body (landmarks in video px): match it to a person and update them. */
  private seePose(
    pts: { x: number; y: number; v: number }[],
    now: number,
    vw: number,
    vh: number,
    tile: { x: number; y: number; w: number; h: number }
  ) {
    const vis = (i: number) => (pts[i]?.v ?? 0) > 0.5;
    const dist = (a: number, b: number) => Math.hypot(pts[a].x - pts[b].x, pts[a].y - pts[b].y);
    // 0 nose · 2/5 eyes · 7/8 ears · 11/12 shoulders · 15/16 wrists
    // the torso (shoulders + hips) barely moves when someone raises a hand or
    // turns around — it's the anchor for "is this the same person?"
    const torsoPts = [11, 12, 23, 24].filter(vis).map((i) => pts[i]);
    const box = torsoPts.length >= 2
      ? {
          x0: Math.min(...torsoPts.map((q) => q.x)),
          y0: Math.min(...torsoPts.map((q) => q.y)),
          x1: Math.max(...torsoPts.map((q) => q.x)),
          y1: Math.max(...torsoPts.map((q) => q.y)),
        }
      : null;
    const face = [0, 2, 5].filter(vis);
    const head = face.length ? face : [7, 8].filter(vis);
    if (!head.length) {
      // turned away (no face, no ears): keep whoever owns this body alive —
      // by torso, or by any visible point inside their reach
      const seen = [0, 2, 5, 7, 8, 11, 12, 13, 14, 15, 16].filter(vis).map((i) => pts[i]);
      const owner =
        (box && this.persons.find((p) => this.sameBody(p, box, null))) ||
        this.persons.find((p) => seen.some((q) => this.inReach(p, q.x, q.y)));
      if (owner) owner.lastSeen = now;
      return;
    }
    const cx = head.reduce((a, i) => a + pts[i].x, 0) / head.length;
    const cy = head.reduce((a, i) => a + pts[i].y, 0) / head.length;
    const w = Math.max(
      vis(7) && vis(8) ? dist(7, 8) * 1.25 : 0,
      vis(2) && vis(5) ? dist(2, 5) * 2.4 : 0,
      vis(11) && vis(12) ? dist(11, 12) * 0.42 : 0,
      24
    );
    const h = w * 1.3;

    // A tile only sees part of the frame: a body cut by an inner tile edge
    // gives a wrong head, and someone close to the camera is too big for a
    // tile at all. The full frame (or the neighbouring tile) covers them.
    const full = tile.w === vw && tile.h === vh;
    // the pose model extrapolates: a "head" outside the image it was given is a
    // guess about a body it only partly saw (a sweater, a shoulder) — drop it
    if (cx < tile.x || cx > tile.x + tile.w || cy < tile.y || cy > tile.y + tile.h) return;
    if (!full) {
      if (w > tile.w * 0.3) return;
      const m = Math.max(w * 0.6, tile.w * 0.05);
      const nearInner =
        (cx - tile.x < m && tile.x > 0) ||
        (tile.x + tile.w - cx < m && tile.x + tile.w < vw) ||
        (cy - tile.y < m && tile.y > 0) ||
        (tile.y + tile.h - cy < m && tile.y + tile.h < vh);
      if (nearInner) return;
    }
    // facing the camera: nose centred between both visible ears
    let poseFacing = false;
    if (vis(0) && vis(7) && vis(8)) {
      const yaw = (pts[0].x - (pts[7].x + pts[8].x) / 2) / Math.max(1, dist(7, 8));
      poseFacing = Math.abs(yaw) < 0.35;
    }
    const ears = [7, 8].filter(vis).map((i) => ({ x: pts[i].x / vw, y: pts[i].y / vh }));
    const wrists = [15, 16].filter(vis).map((i) => ({ x: pts[i].x, y: pts[i].y }));
    const arms = [13, 14, 15, 16].filter(vis).map((i) => ({ x: pts[i].x, y: pts[i].y }));

    // same person if the torsos overlap, the head sits on their torso, or the
    // heads are close — overlapping tiles, raised arms and turned heads all
    // land here instead of minting someone new
    let best: Person | null = null;
    let bestDist = Infinity;
    for (const p of this.persons) {
      const d = Math.hypot(p.cx - cx, p.cy - cy);
      if ((d < Math.max(p.w, w) * 1.2 || this.sameBody(p, box, { x: cx, y: cy })) && d < bestDist) {
        best = p;
        bestDist = d;
      }
    }
    if (best) {
      if (box) best.box = box;
      if (full || !this.isBig(best, tile)) best.arms = arms;
      best.cx += (cx - best.cx) * 0.5;
      best.cy += (cy - best.cy) * 0.5;
      best.w += (w - best.w) * 0.5;
      best.h += (h - best.h) * 0.5;
    } else {
      // a stranger must keep showing up before they get a number — one stray
      // detection shouldn't mint a new person
      // anything sitting on someone's torso (a raised arm, a half-seen head)
      // is part of them, never a newcomer
      if (this.persons.some((p) => this.sameBody(p, box, { x: cx, y: cy }))) return;
      // a raised hand seen up close fools the pose model into a small "person"
      // whose head is the hand — anything at someone's elbow or wrist is theirs
      if (
        this.persons.some((p) =>
          p.arms.some((a) => Math.hypot(a.x - cx, a.y - cy) < Math.max(w * 1.5, p.w * 0.6))
        )
      ) {
        return;
      }
      // someone close to the camera fills several tiles: tile hits inside their
      // reach are fragments of them, never anyone new
      if (!full && this.persons.some((p) => this.isBig(p, tile) && this.inReach(p, cx, cy))) return;
      const cand = this.candidates.find((c) => Math.hypot(c.cx - cx, c.cy - cy) < Math.max(c.w, w) * 1.2);
      if (!cand) {
        this.candidates.push({ cx, cy, w, seen: 1, at: now, first: now });
        return;
      }
      cand.cx = cx;
      cand.cy = cy;
      cand.w = w;
      cand.at = now;
      // three sightings over most of a second: a lowering arm or a passing
      // shadow flickers out before that
      if (++cand.seen < 3 || now - cand.first < 800) return;
      this.candidates = this.candidates.filter((c) => c !== cand);
      // someone who just left and came back keeps their number and portrait
      const ghost = this.reclaim(cx, cy, w);
      // otherwise the lowest number not in use — alone in the room you're
      // always 1, however often Octa loses sight of you
      let id = ghost?.id;
      if (id === undefined) {
        id = 1;
        while (this.persons.some((p) => p.id === id)) id++;
        this.gone = this.gone.filter((g) => g.id !== id);
        this.events?.onNewPerson?.(id);
      }
      best = {
        id, cx, cy, w, h, top: 0, ears, wrists, lastSeen: now,
        poseFacing, faceAt: 0, raiseSince: 0, raiseFired: false,
        portrait: ghost?.portrait ?? null, portraitAt: ghost?.portraitAt ?? 0, faceBox: null,
        box: box ?? { x0: cx - w, y0: cy + h * 0.6, x1: cx + w, y1: cy + h * 2.5 },
        arms,
        mouth: [], lookEma: 0, looking: false,
      };
      this.persons.push(best);
    }
    best.top = best.cy - best.h * 0.55;
    if (ears.length) best.ears = ears;
    best.wrists = wrists;
    best.poseFacing = poseFacing;
    best.lastSeen = now;
    // without a fresh face read, gaze comes from head pose
    if (now - best.faceAt > FACE_FRESH) this.updateLook(best, poseFacing);

    // raise-to-speak from the pose: a wrist above the head, held
    const up = wrists.some((wr) => wr.y < best.top);
    if (!up) {
      best.raiseSince = 0;
      best.raiseFired = false;
    } else {
      if (!best.raiseSince) best.raiseSince = now;
      if (!best.raiseFired && now - best.raiseSince > RAISE_HOLD) {
        best.raiseFired = true;
        this.floor?.raise(best.id);
      }
    }
  }

  /** Both tracks recently read a face, and it's the same face in the frame. */
  private sameFace(a: Person, b: Person, now: number): boolean {
    if (!a.faceBox || !b.faceBox || now - a.faceAt > 1500 || now - b.faceAt > 1500) return false;
    const A = a.faceBox, B = b.faceBox;
    const ix = Math.max(0, Math.min(A.x1, B.x1) - Math.max(A.x0, B.x0));
    const iy = Math.max(0, Math.min(A.y1, B.y1) - Math.max(A.y0, B.y0));
    const inter = ix * iy;
    const union = (A.x1 - A.x0) * (A.y1 - A.y0) + (B.x1 - B.x0) * (B.y1 - B.y0) - inter;
    return union > 0 && inter / union > 0.3;
  }

  /** Big next to a tile — i.e. close to the camera. */
  private isBig(p: Person, tile: { w: number }): boolean {
    return p.w > tile.w * 0.2;
  }

  /** Inside everything person p occupies: head, torso and arms, plus a margin. */
  private inReach(p: Person, x: number, y: number): boolean {
    const xs = [p.cx - p.w, p.cx + p.w, p.box.x0, p.box.x1, ...p.arms.map((a) => a.x)];
    const ys = [p.cy - p.h, p.cy + p.h, p.box.y0, p.box.y1, ...p.arms.map((a) => a.y)];
    const x0 = Math.min(...xs), x1 = Math.max(...xs), y0 = Math.min(...ys), y1 = Math.max(...ys);
    const mx = (x1 - x0) * 0.1, my = (y1 - y0) * 0.1;
    return x > x0 - mx && x < x1 + mx && y > y0 - my && y < y1 + my;
  }

  /** Does this torso box / head point belong to person p? Torsos overlap, or
   *  the head sits within p's torso widened to the shoulders' reach. */
  private sameBody(
    p: Person,
    box: { x0: number; y0: number; x1: number; y1: number } | null,
    headPt: { x: number; y: number } | null
  ): boolean {
    const b = p.box;
    if (box) {
      const ix = Math.max(0, Math.min(b.x1, box.x1) - Math.max(b.x0, box.x0));
      const iy = Math.max(0, Math.min(b.y1, box.y1) - Math.max(b.y0, box.y0));
      const inter = ix * iy;
      const union = (b.x1 - b.x0) * (b.y1 - b.y0) + (box.x1 - box.x0) * (box.y1 - box.y0) - inter;
      if (union > 0 && inter / union > 0.3) return true;
    }
    if (headPt) {
      // the head floats above the shoulders — extend the box upward by a head
      const padX = (b.x1 - b.x0) * 0.15;
      const above = p.h * 1.6;
      if (headPt.x > b.x0 - padX && headPt.x < b.x1 + padX && headPt.y > b.y0 - above && headPt.y < b.y1) return true;
    }
    return false;
  }

  /** A recently departed person near (cx, cy) — or, if nobody is in view at all,
   *  whoever left last — so a returning person keeps their number. */
  private reclaim(cx: number, cy: number, w: number) {
    const near = this.gone
      .filter((g) => Math.hypot(g.cx - cx, g.cy - cy) < Math.max(g.w, w) * 3)
      .sort((a, b) => Math.hypot(a.cx - cx, a.cy - cy) - Math.hypot(b.cx - cx, b.cy - cy))[0];
    const pick = near ?? (this.persons.length === 0 ? this.gone.at(-1) : undefined);
    if (pick) this.gone = this.gone.filter((g) => g !== pick);
    return pick;
  }

  private updateLook(p: Person, facingNow: boolean) {
    p.lookEma += ((facingNow ? 1 : 0) - p.lookEma) * 0.35;
    p.looking = p.lookEma > (p.looking ? 0.35 : 0.65); // hysteresis
  }

  /** Enlarge one head from the full-res frame and read lips + gaze from the mesh. */
  private readFace(p: Person, now: number) {
    const video = this.video!;
    // generous crop: far heads are only ~30 px wide, and the mesh needs context
    const side = Math.max(p.w * 3, 96);
    const sx = Math.max(0, Math.min(video.videoWidth - side, p.cx - side / 2));
    const sy = Math.max(0, Math.min(video.videoHeight - side, p.cy - side / 2));
    const ctx = this.crop!.getContext("2d")!;
    ctx.drawImage(video, sx, sy, side, side, 0, 0, CROP, CROP);
    const lm = this.faceLandmarker!.detect(this.crop!).faceLandmarks[0];
    if (!lm) return; // too small or turned away — pose keeps the gaze estimate
    p.faceAt = now;
    // the mesh knows exactly where the face is: record it, and pull the track
    // onto it (the pose's head guess drifts, and two drifting guesses on one
    // head were how duplicates were born)
    let fx0 = Infinity, fy0 = Infinity, fx1 = -Infinity, fy1 = -Infinity;
    for (const l of lm) {
      fx0 = Math.min(fx0, l.x); fx1 = Math.max(fx1, l.x);
      fy0 = Math.min(fy0, l.y); fy1 = Math.max(fy1, l.y);
    }
    p.faceBox = { x0: sx + fx0 * side, y0: sy + fy0 * side, x1: sx + fx1 * side, y1: sy + fy1 * side };
    const fw = p.faceBox.x1 - p.faceBox.x0;
    p.cx += ((p.faceBox.x0 + p.faceBox.x1) / 2 - p.cx) * 0.5;
    p.cy += ((p.faceBox.y0 + p.faceBox.y1) / 2 - p.cy) * 0.5;
    p.w += (fw * 1.15 - p.w) * 0.5;
    p.h = p.w * 1.3;
    // inner-lip gap (13 upper, 14 lower) over forehead-to-chin (10, 152); the
    // crop is square, so normalized coordinates keep their proportions
    const faceH = Math.hypot(lm[10].x - lm[152].x, lm[10].y - lm[152].y) || 1;
    const gap = Math.hypot(lm[13].x - lm[14].x, lm[13].y - lm[14].y) / faceH;
    // speech = lips opening and closing, not just an open mouth
    const recent = p.mouth.filter((m) => now - m.t < MOUTH_WINDOW).map((m) => m.gap);
    recent.push(gap);
    const moving = Math.max(...recent) - Math.min(...recent) > MOUTH_MOVING;
    const facingNow = facing(lm);
    this.updateLook(p, facingNow);
    // portrait: snap one the moment they look straight at the camera, keep it
    // (re-taken after 10 min, so it follows people who change seats or light)
    if (facingNow && p.looking && (!p.portrait || now - p.portraitAt > 600_000)) {
      // frame the whole face from the mesh itself (crop → video px), not the
      // rough pose estimate — that clipped chins and foreheads up close
      let x0 = Infinity, y0 = Infinity, x1 = -Infinity, y1 = -Infinity;
      for (const l of lm) {
        x0 = Math.min(x0, l.x); x1 = Math.max(x1, l.x);
        y0 = Math.min(y0, l.y); y1 = Math.max(y1, l.y);
      }
      this.snapPortrait(p, now, { x0: sx + x0 * side, y0: sy + y0 * side, x1: sx + x1 * side, y1: sy + y1 * side });
    }
    p.mouth.push({ t: now, gap, moving, looking: p.looking });
    while (p.mouth.length && now - p.mouth[0].t > MOUTH_LOG_MS) p.mouth.shift();
  }

  /** Which person does a wrist belong to? Hands hang below and beside the face,
   *  so distance below counts half. Sticky: the current owner keeps the hand
   *  unless someone else is clearly closer. */
  private bindHand(t: HandTrack) {
    // best evidence: the body pose puts one of this person's wrists right here
    let byPose: Person | null = null;
    let byPoseDist = Infinity;
    for (const p of this.persons) {
      for (const wr of p.wrists) {
        const d = Math.hypot(wr.x - t.wx, wr.y - t.wy);
        if (d < Math.max(p.w * 1.2, 50) && d < byPoseDist) {
          byPose = p;
          byPoseDist = d;
        }
      }
    }
    if (byPose) {
      t.owner = byPose.id;
      return;
    }
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
    if (now - this.lastTick < VISION_MS) return;
    this.lastTick = now;
    const vw = this.video.videoWidth || 1280;
    const vh = this.video.videoHeight || 720;

    const handTick = this.frameCount++ % 2 === 1;
    if (handTick || !this.lastHands) {
      // hands on a downscaled copy (landmarks come back normalized, so nothing else changes)
      const hc = (this.handCanvas ??= document.createElement("canvas"));
      const hw = Math.min(vw, HAND_INPUT_W);
      const hh = Math.round((hw * vh) / vw);
      if (hc.width !== hw || hc.height !== hh) {
        hc.width = hw;
        hc.height = hh;
      }
      hc.getContext("2d")!.drawImage(this.video, 0, 0, hw, hh);
      this.lastHands = this.recognizer.recognizeForVideo(hc, now);
    }
    const res = this.lastHands;

    const faceFrame = this.poseLandmarker && !handTick;
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

    // talk-to-listen: the holder faces Octa and their lips move for a beat —
    // or, with the floor free, anyone does (Octa then answers only if it's
    // addressed by name)
    const lipsNow = (p: Person) => !!p.mouth.at(-1)?.moving && now - (p.mouth.at(-1)?.t ?? 0) < 200;
    const talker =
      holderPerson ?? (this.floor?.holder == null ? this.persons.find((p) => p.looking && lipsNow(p)) : undefined);
    const last = talker?.mouth.at(-1);
    const talkingToOcta = !!talker && talker.looking && lipsNow(talker);
    if (talkingToOcta) {
      if (!this.talkSince) this.talkSince = now;
      if (!this.talkFired && now - this.talkSince > 250) {
        this.talkFired = true;
        if (holderPerson) events.onHolderTalk?.(talker!.id);
        else events.onFreeTalk?.(talker!.id);
      }
    } else if (now - (last?.t ?? 0) > 1500 || !talker?.looking) {
      // a pause or a look away re-arms it for the next thing they say
      this.talkSince = 0;
      this.talkFired = false;
    }

    // parallax follows whoever holds the floor (else the nearest face)
    if (faceFrame) {
      const head =
        holderPerson ?? [...this.persons].sort((a, b) => b.w - a.w)[0];
      if (head && now - head.lastSeen < 1000) {
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

/**
 * The talking stick: in a meeting, exactly one person "has the floor" and
 * only their hands — and their voice — steer Octa.
 *
 * Rules:
 *  - nobody gets the floor without raising a hand
 *  - raised hands wait in line; Octa picks from it, preferring whoever is
 *    looking at it, then whoever raised first (a short window gathers
 *    hands that go up together, so it's a choice, not a race)
 *  - the holder keeps the floor for as long as they look at Octa — others
 *    can only get it once the holder looks away (or leaves the camera)
 *  - a holder who looks away with nobody waiting keeps it for a while,
 *    then the floor is freed
 */

export interface FloorState {
  holder: number | null;
  queue: number[];
}

const LOOK_AWAY_MS = 1500; // holder looking away this long → the next in line may take over
const RELEASE_MS = 15000; // …or with nobody waiting, this long → the floor is freed
const PICK_WINDOW_MS = 700; // gather hands raised together before Octa chooses

export class Floor {
  holder: number | null = null;
  queue: number[] = [];
  private awaySince = 0;
  private pickSince = 0;
  private onChange: (s: FloorState) => void;

  constructor(onChange: (s: FloorState) => void) {
    this.onChange = onChange;
  }

  /** A person held a hand above their head. */
  raise(id: number) {
    if (id === this.holder || this.queue.includes(id)) return;
    this.queue.push(id);
    this.emit();
  }

  /** Called each frame with who is in view and who is looking at Octa. */
  update(present: number[], looking: Set<number>, now: number) {
    const before = this.queue.length;
    this.queue = this.queue.filter((id) => present.includes(id));
    if (this.queue.length !== before) this.emit();

    if (this.holder !== null) {
      if (!present.includes(this.holder)) {
        this.release();
      } else if (looking.has(this.holder)) {
        this.awaySince = 0; // eyes on Octa: the floor is locked
      } else {
        if (!this.awaySince) this.awaySince = now;
        const away = now - this.awaySince;
        if ((this.queue.length && away > LOOK_AWAY_MS) || away > RELEASE_MS) this.release();
      }
    }

    if (this.holder === null && this.queue.length) {
      if (!this.pickSince) this.pickSince = now;
      if (now - this.pickSince >= PICK_WINDOW_MS) {
        // Octa's choice: someone looking at it first, then first come
        const pick = this.queue.find((id) => looking.has(id)) ?? this.queue[0];
        this.grant(pick);
      }
    } else {
      this.pickSince = 0;
    }
  }

  reset() {
    this.holder = null;
    this.queue = [];
    this.awaySince = this.pickSince = 0;
    this.emit();
  }

  private release() {
    this.holder = null;
    this.awaySince = 0;
    this.emit();
  }

  private grant(id: number) {
    this.holder = id;
    this.queue = this.queue.filter((q) => q !== id);
    this.awaySince = this.pickSince = 0;
    this.emit();
  }

  private emit() {
    this.onChange({ holder: this.holder, queue: [...this.queue] });
  }
}

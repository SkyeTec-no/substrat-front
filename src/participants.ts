/**
 * Participant strip: one square tile per person the camera sees, along the
 * bottom edge — whether they're looking at Octa, talking, have a hand up,
 * hold the floor or wait in line. Tiles are keyed by person id and updated
 * in place, so their CSS transitions and pulses never restart.
 */

export interface ParticipantView {
  holder: number | null;
  /** Octa is listening right now (to the holder). */
  listening?: boolean;
  queue: number[];
  persons: { id: number; looking: boolean; talking: boolean; handUp: boolean }[];
}

// 24×24 stroke paths, same line language as the slide icons
const EYE = "M2 12s4-7 10-7 10 7 10 7-4 7-10 7S2 12 2 12Z M12 9a3 3 0 1 0 0 6a3 3 0 1 0 0-6Z";
const EYE_OFF = "M2 12s4-7 10-7 10 7 10 7-4 7-10 7S2 12 2 12Z M4 4l16 16";
const MIC = "M12 3a3 3 0 0 0-3 3v6a3 3 0 0 0 6 0V6a3 3 0 0 0-3-3Z M5 11a7 7 0 0 0 14 0 M12 18v3";
const HAND =
  "M8 13V5.5a1.5 1.5 0 0 1 3 0V11 M11 10.5V4a1.5 1.5 0 0 1 3 0v6.5 M14 10.5V5.5a1.5 1.5 0 0 1 3 0V14c0 4-2.6 7-6.2 7-2.5 0-4-1.4-5.4-3.8L3.8 14c-.8-1.3.9-2.6 2-1.6L8 14.5";

const svg = (d: string, cls: string) =>
  `<svg class="${cls}" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.5" stroke-linecap="square" stroke-linejoin="miter"><path d="${d}"/></svg>`;

const tiles = new Map<number, HTMLDivElement>();

function createTile(id: number): HTMLDivElement {
  const el = document.createElement("div");
  el.className = "pt";
  el.style.setProperty("--who", `var(--person-${(id - 1) % 6})`);
  // numeral-led square: status lives in the corners, words only for the floor
  el.innerHTML = `
    <canvas class="pt-face" width="136" height="136"></canvas>
    <span class="pt-gaze" title="ser på Octa">${svg(EYE, "on")}${svg(EYE_OFF, "off")}</span>
    <span class="pt-hand" title="hånd oppe">${svg(HAND, "")}</span>
    <span class="pt-mic" title="Octa lytter">${svg(MIC, "")}</span>
    <span class="pt-num">${id}</span>
    <span class="pt-meter"><i></i><i></i><i></i></span>
    <span class="pt-role"></span>`;
  return el;
}

/** drawFace paints a person's live face crop into their tile (false = no face). */
export function renderParticipants(
  root: HTMLElement,
  view: ParticipantView | null,
  drawFace?: (id: number, canvas: HTMLCanvasElement) => boolean
) {
  const persons = view?.persons ?? [];
  root.classList.toggle("visible", persons.length > 0);

  const seen = new Set<number>();
  for (const [index, p] of [...persons].sort((a, b) => a.id - b.id).entries()) {
    seen.add(p.id);
    let el = tiles.get(p.id);
    if (!el) {
      el = createTile(p.id);
      tiles.set(p.id, el);
    }
    // keep id order, but only touch the DOM when a tile is out of place —
    // re-inserting would restart its pulse animations
    if (root.children[index] !== el) root.insertBefore(el, root.children[index] ?? null);

    const holder = view!.holder === p.id;
    const queuePos = view!.queue.indexOf(p.id);
    el.classList.toggle("looking", p.looking);
    el.classList.toggle("talking", p.talking);
    el.classList.toggle("hand-up", p.handUp);
    el.classList.toggle("holder", holder);
    el.classList.toggle("queued", queuePos >= 0);
    const face = el.querySelector<HTMLCanvasElement>(".pt-face")!;
    el.classList.toggle("has-face", !!drawFace?.(p.id, face));
    el.classList.toggle("listened", holder && !!view!.listening);

    const role = el.querySelector<HTMLSpanElement>(".pt-role")!;
    const roleText = holder ? (view!.listening ? "lytter" : "ordet") : queuePos >= 0 ? `kø ${queuePos + 1}` : "";
    if (role.textContent !== roleText) role.textContent = roleText;
  }

  for (const [id, el] of tiles) {
    if (!seen.has(id)) {
      el.remove();
      tiles.delete(id);
    }
  }
}

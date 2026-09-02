/** Local voice-command matchers for panel recall and dismissal. */

export function isRecallCommand(text: string): boolean {
  return (
    /\b(bring\s+(forth|back|up)|recall|pull\s+up)\b/i.test(text) ||
    (/\b(show|display)\b/i.test(text) && /\bagain\b/i.test(text))
  );
}

/** "generate a report / make a PDF" — assemble the canvas into a document. */
export function isReportCommand(text: string): boolean {
  return (
    /\b(generate|create|make|build|write|compile|assemble)\b.{0,40}\b(report|pdf|document|deck|analysis|strategy)\b/i.test(text) ||
    /\breport\b.{0,20}\bpdf\b/i.test(text)
  );
}

/** "take notes" — enter note-taker mode. */
export function isNoteCommand(text: string): boolean {
  return /\b(take notes?|note[- ]?taker|start (taking )?notes?|note mode|dictation)\b/i.test(text);
}

/** Spoken fallback for leaving note-taker mode (crossed hands is the primary). */
export function isStopNotesCommand(text: string): boolean {
  return /\b(stop|end|exit|finish|done)\b.{0,16}\bnot(es|ing)\b/i.test(text) ||
    /^\s*(stop|done|finished)\s*[.!]?\s*$/i.test(text);
}

/** "open this/that" — acts on whatever panel currently has highlight focus. */
export function isOpenCommand(text: string): boolean {
  return (
    /\b(open|focus on|expand)\b.{0,16}\b(this|that|it|one)\b/i.test(text) ||
    /^\s*open\s*[.!]?\s*$/i.test(text)
  );
}

/** "put this back / shelve it" — sends the focused panel to the bento grid. */
export function isShelveCommand(text: string): boolean {
  return (
    /\b(put|send)\b.{0,12}\b(back|away)\b/i.test(text) ||
    /\bshelve\b/i.test(text) ||
    /\bback to the (shelf|grid)\b/i.test(text)
  );
}

export function isDismissCommand(text: string): boolean {
  return (
    /\b(remove|dismiss|hide|clear)\b/i.test(text) &&
    /\b(it|that|this|them|panel|panels|image|picture|diagram|chart|table|all|everything|screen)\b/i.test(text)
  );
}

export function isDismissAll(text: string): boolean {
  return /\b(all|everything|screen)\b/i.test(text);
}

// words that describe the command, not the content being referred to
const STOP = new Set([
  "bring", "forth", "back", "up", "recall", "pull", "show", "display", "again",
  "remove", "dismiss", "hide", "clear", "the", "that", "this", "it", "them",
  "me", "please", "can", "you", "image", "picture", "photo", "diagram", "chart",
  "table", "panel", "visual", "one", "and", "with", "for", "from",
]);

/** Pick the panel whose description best matches the request; latest wins ties. */
export function matchPanel(
  text: string,
  panels: { id: string; desc: string }[]
): string | null {
  const words = text.toLowerCase().match(/[\p{L}\p{N}]+/gu) ?? [];
  const keys = words.filter((w) => w.length > 2 && !STOP.has(w));

  let best: string | null = null;
  let bestScore = 0;
  for (const p of panels) {
    const d = p.desc.toLowerCase();
    let score = 0;
    for (const k of keys) if (d.includes(k)) score++;
    if (score >= bestScore && score > 0) {
      bestScore = score;
      best = p.id;
    }
  }
  return best ?? panels.at(-1)?.id ?? null;
}

/** Local voice-command matchers for panel recall and dismissal — English and
 *  Norwegian. JS \b doesn't treat æ/ø/å as word characters, so Norwegian words
 *  that start with them are anchored with (?:^|\s) instead. */

export function isRecallCommand(text: string): boolean {
  return (
    /\b(bring\s+(forth|back|up)|recall|pull\s+up)\b/i.test(text) ||
    (/\b(show|display)\b/i.test(text) && /\bagain\b/i.test(text)) ||
    /\bhent\s+(frem|fram|tilbake|opp)\b/i.test(text) ||
    /\bta\s+(frem|fram|opp)\b/i.test(text) ||
    (/\bvis\b/i.test(text) && /\bigjen\b/i.test(text)) ||
    /\bvis\s+(meg\s+)?(bildet|diagrammet|grafen|tabellen|lysbildet|notatene|notatet)\b/i.test(text)
  );
}

/** Spoken to Octa by name ("Okta, …", "hva tenker du, Octa?"). The STT
 *  writes the name many ways; \b doesn't see "å", so use letter lookarounds. */
export function addressesOcta(text: string): boolean {
  return /(?<!\p{L})(okta|octa|åkta|okka|oktan)(?!\p{L})/iu.test(text);
}

/** "Okta, stopp" / "avbryt" / "glem det" — drop the answer in progress. Short
 *  lines only, so "hvorfor stopper toget?" isn't a cancel. */
export function isCancelCommand(text: string): boolean {
  const rest = text.replace(/(?<!\p{L})(okta|octa|åkta|okka|oktan)(?!\p{L})/giu, " ").trim();
  if (rest.split(/\s+/).filter(Boolean).length > 5) return false;
  return /(?<!\p{L})(stopp|stop|avbryt|glem det|glem det der|hold opp|la være|drit i det|kanseller|cancel|never mind|nok nå|det holder)(?!\p{L})/iu.test(rest);
}

/** "this / that" — the command means whatever is being pointed at. */
export function isDeictic(text: string): boolean {
  return /\b(denne|dette|den her|det her|this|that one|this one)\b/i.test(text);
}

/** "forstørr / zoom inn" etc. — size of the panel in focus. */
export function zoomCommand(text: string): "in" | "out" | "reset" | null {
  if (/\b(tilbakestill|nullstill|vanlig størrelse|reset( the)? (zoom|view))/i.test(text)) return "reset";
  if (/\b(forstørr|zoom(e)? inn|zoom in|enlarge)/i.test(text)) return "in";
  if (/\b(forminsk|zoom(e)? ut|zoom out)/i.test(text)) return "out";
  // "større" / "mindre" are everyday words — only a command when short
  // ("større!") or aimed at the panel ("gjør den større")
  const short = text.trim().split(/\s+/).length <= 3;
  const aimed = /\bgjør\s+(den|det|denne|dette|bildet|grafen|diagrammet|tabellen|lysbildet)\b/i.test(text) || /\bmake it\b/i.test(text);
  if (short || aimed) {
    if (/\b(større|nærmere|bigger|closer)\b/i.test(text)) return "in";
    if (/\b(mindre|smaller)\b/i.test(text)) return "out";
  }
  return null;
}

/** What kind of panel a request names, as the suffix panel ids carry. */
export function panelKind(text: string): string | null {
  if (/\b(bilde|bildet|foto|image|picture|photo)/i.test(text)) return "img";
  if (/\b(diagram|flytskjema|flowchart|mindmap|tankekart)/i.test(text)) return "svg";
  if (/\b(graf|grafen|stolpe|chart|graph|plot)/i.test(text)) return "chart";
  if (/\b(tabell|table)/i.test(text)) return "tbl";
  if (/\b(lysbilde|slide|punktene|keynote)/i.test(text)) return "slide";
  if (/\b(notat|notes|note)/i.test(text)) return "note";
  return null;
}

/** "generate a report / make a PDF" — assemble the canvas into a document. */
export function isReportCommand(text: string): boolean {
  return (
    /\b(generate|create|make|build|write|compile|assemble)\b.{0,40}\b(report|pdf|document|deck|analysis|strategy)\b/i.test(text) ||
    /\breport\b.{0,20}\bpdf\b/i.test(text) ||
    /\b(lag|generer|skriv|bygg|sett sammen)\b.{0,40}\b(rapport|pdf|dokument|analyse|strategi)/i.test(text)
  );
}

/** "take notes" — enter note-taker mode. */
export function isNoteCommand(text: string): boolean {
  return (
    /\b(take notes?|note[- ]?taker|start (taking )?notes?|note mode|dictation)\b/i.test(text) ||
    /\b(ta notater|notatmodus|start notat|diktat)/i.test(text)
  );
}

/** Spoken fallback for leaving note-taker mode (crossed hands is the primary). */
export function isStopNotesCommand(text: string): boolean {
  return /\b(stop|end|exit|finish|done)\b.{0,16}\bnot(es|ing)\b/i.test(text) ||
    /^\s*(stop|done|finished)\s*[.!]?\s*$/i.test(text) ||
    /\b(stopp|avslutt|ferdig)\b.{0,16}\bnotat/i.test(text) ||
    /^\s*(stopp|ferdig)\s*[.!]?\s*$/i.test(text);
}

/** "open this/that" — acts on whatever panel currently has highlight focus. */
export function isOpenCommand(text: string): boolean {
  return (
    /\b(se|kikk)\s+på\s+(denne|dette|den|det)\b/i.test(text) ||
    /\b(vis|ta\s+frem|ta\s+fram)\s+(denne|dette)\b/i.test(text) ||
    /\b(open|focus on|expand)\b.{0,16}\b(this|that|it|one)\b/i.test(text) ||
    /^\s*open\s*[.!]?\s*$/i.test(text) ||
    /(?:^|\s)(åpne|fokuser på|utvid)\s.{0,16}\b(denne|dette|den|det)\b/i.test(text) ||
    /^\s*åpne\s*[.!]?\s*$/i.test(text)
  );
}

/** "put this back / shelve it" — sends the focused panel to the bento grid. */
export function isShelveCommand(text: string): boolean {
  return (
    /\b(put|send)\b.{0,12}\b(back|away)\b/i.test(text) ||
    /\bshelve\b/i.test(text) ||
    /\bback to the (shelf|grid)\b/i.test(text) ||
    /\b(legg|send|flytt|sett)\b.{0,20}\b(tilbake|bort|bak|bakover|bakerst|vekk|unna|til side)\b/i.test(text) ||
    /\b(i hylla|i bakgrunnen)\b/i.test(text) ||
    /\b(put|move|send) (them|it|everything|the panels)\b.{0,12}\b(back|behind|aside)\b/i.test(text)
  );
}

/** "legg panelene bak" / "legg alle tilbake" — every focused panel, not just one. */
export function isShelveAll(text: string): boolean {
  return /\b(alle|alt|panelene|dem|de|everything|them|all)\b/i.test(text);
}

export function isDismissCommand(text: string): boolean {
  return (
    (/\b(remove|dismiss|hide|clear)\b/i.test(text) &&
      /\b(it|that|this|them|panel|panels|image|picture|diagram|chart|table|all|everything|screen)\b/i.test(text)) ||
    (/\b(fjern|skjul|lukk|rydd( bort| opp)?)\b/i.test(text) &&
      /\b(den|denne|det|dette|dem|panel|panelet|paneler|panelene|bildet|diagrammet|grafen|tabellen|lysbildet|notatene|alt|alle|skjermen)\b/i.test(text))
  );
}

export function isDismissAll(text: string): boolean {
  return /\b(all|everything|screen|alt|alle|skjermen)\b/i.test(text);
}

/** "use a smarter model" / "back to the fast one" — switch Octa's brain.
 *  Indirect phrasings are left to the router's tier branch. */
export function tierCommand(text: string): "smart" | "fast" | null {
  if (
    /\b(smartere|klokere|sterkere|kraftigere)\s+(modell|modellen|hjerne|hjernen|ai)\b/i.test(text) ||
    // "bedre/beste/smarte" alone is too common ("den beste modellen for regresjon") — needs a switch verb
    /\b(bruke?|bytt( over)? til|skift til|ta|aktiver)\s+(en\s+|den\s+)?(bedre|beste|smarte|smart)\s+(modell|modellen|hjerne|hjernen)\b/i.test(text) ||
    /\btenk\s+(grundigere|dypere|hardere|nøyere)/i.test(text) ||
    /\bsmart\s*modus\b/i.test(text) ||
    /\b(smarter|stronger|better|best|smart)\s+(model|brain)\b|\bthink harder\b/i.test(text)
  ) {
    return "smart";
  }
  if (
    /\b(raskere|kjappere|billigere|enklere|vanlig|vanlige|rask|raske)\s+(modell|modellen|hjerne|hjernen)\b/i.test(text) ||
    /\btilbake til\s+(den\s+)?(raske|vanlige|standard)/i.test(text) ||
    /\brask\s*modus\b/i.test(text) ||
    /\b(faster|cheaper|fast)\s+(model|brain)\b/i.test(text)
  ) {
    return "fast";
  }
  return null;
}

// words that describe the command, not the content being referred to
const STOP = new Set([
  "bring", "forth", "back", "up", "recall", "pull", "show", "display", "again",
  "remove", "dismiss", "hide", "clear", "the", "that", "this", "it", "them",
  "me", "please", "can", "you", "image", "picture", "photo", "diagram", "chart",
  "table", "panel", "visual", "one", "and", "with", "for", "from",
  "hent", "frem", "fram", "tilbake", "opp", "vis", "igjen", "fjern", "skjul",
  "lukk", "den", "det", "dette", "dem", "meg", "kan", "du", "bildet",
  "diagrammet", "grafen", "tabellen", "panelet", "og", "med", "fra",
  "ta", "frem", "fram", "lukk", "denne", "dette", "på", "se", "alle", "paneler",
  "panelene", "bilde", "lysbildet", "notatene", "forstørr", "forminsk",
]);

/** Strip common Norwegian noun endings: notatene → notat, bildet → bild. */
function stem(word: string): string {
  return word.length > 5 ? word.replace(/(ene|ane|en|et|er|a)$/, "") : word;
}

/** The panel the words actually name (by kind and description) — null if none. */
export function findPanel(
  text: string,
  panels: { id: string; desc: string }[]
): string | null {
  const words = text.toLowerCase().match(/[\p{L}\p{N}]+/gu) ?? [];
  const keys = words.filter((w) => w.length > 2 && !STOP.has(w));
  // "bildet av hunden": only images are candidates — ids end in their kind
  const kind = panelKind(text);
  const ofKind = kind ? panels.filter((p) => new RegExp(`-${kind}\\d*$`).test(p.id)) : [];
  if (ofKind.length) panels = ofKind;

  let best: string | null = null;
  let bestScore = 0;
  for (const p of panels) {
    const d = p.desc.toLowerCase();
    let score = 0;
    // Norwegian inflects nouns ("notatene", "bildet") — also try the bare stem
    for (const k of keys) if (d.includes(k) || d.includes(stem(k))) score++;
    if (score >= bestScore && score > 0) {
      bestScore = score;
      best = p.id;
    }
  }
  // a named kind with no word match: the latest panel of that kind
  return best ?? (ofKind.length ? ofKind.at(-1)!.id : null);
}

/** Pick the panel whose description best matches the request; latest wins ties. */
export function matchPanel(text: string, panels: { id: string; desc: string }[]): string | null {
  return findPanel(text, panels) ?? panels.at(-1)?.id ?? null;
}

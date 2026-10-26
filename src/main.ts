import "./style.css";
import { OctaScene, type OctaState } from "./scene";
import { Soundscape } from "./sound";
import { MicMeter, Recognizer, Speaker } from "./speech";
import { think } from "./brain";
import { settings, listModels, type ChatResult, type ModelTier, type VoiceStyle } from "./openrouter";
import { respond, keynote, repairMermaid, generateReport, generateCompliments, generateCommentary } from "./agents";
import { remember, clearHistory } from "./history";
import { renderMermaidSvg } from "./visuals";
import { isDownloadCommand, downloadArtifacts } from "./download";
import { tierCommand, isRecallCommand, isDismissCommand, isDismissAll, isOpenCommand, isShelveCommand, isNoteCommand, isStopNotesCommand, isReportCommand, matchPanel } from "./commands";
import { buildReportPdf, type ReportArtifact } from "./report";
import { loadImage, svgToCanvas, createTableRenderer, createWipeRenderer, createSlideRenderer, noteToCanvas } from "./panels";
import { GestureControl, captureCameraFrame, type HandState, type MeetingState } from "./gestures";
import { theme, setDarkMode, initTheme, onThemeChange } from "./theme";
import { renderParticipants, type ParticipantView } from "./participants";
import { jevSettings, jevLog, jevLogCsv, checkAddressee, type JevMode } from "./jev";

const canvas = document.querySelector<HTMLCanvasElement>("#scene")!;
const statusEl = document.querySelector<HTMLDivElement>("#status")!;
const transcriptEl = document.querySelector<HTMLDivElement>("#transcript")!;
const visualsEl = document.querySelector<HTMLDivElement>("#visuals")!;
const agentLabelsEl = document.querySelector<HTMLDivElement>("#agent-labels")!;
const overlay = document.querySelector<HTMLDivElement>("#overlay")!;
const startBtn = document.querySelector<HTMLButtonElement>("#start")!;

const scene = new OctaScene(canvas);
scene.setShape(settings.tier === "smart" ? "icosahedron" : "octahedron", false);
initTheme();
scene.setTheme(theme());
onThemeChange((t) => scene.setTheme(t));
const speaker = new Speaker();
speaker.onStartFx = () => sound?.robotStart();
speaker.onWordFx = () => sound?.robotWord();
speaker.onEndFx = () => sound?.robotStop();
// a glitch in the voice tears the picture too
speaker.vader.onGlitch = (strength, ms) => scene.glitch(strength, ms);
const mic = new MicMeter();

let sound: Soundscape | null = null;
let state: OctaState = "idle";
let speakLevel = 0; // synthetic envelope while TTS plays (its audio isn't tappable)

function setState(next: OctaState) {
  state = next;
  scene.setState(next);
  sound?.setState(next);
  statusEl.textContent = next;
  statusEl.className = next;
}

function showTranscript(text: string, interim = false) {
  transcriptEl.innerHTML = "";
  const span = document.createElement("span");
  if (interim) span.className = "interim";
  span.textContent = text;
  transcriptEl.appendChild(span);
}

// what each panel actually contains — the raw material for report assembly
const panelArtifacts = new Map<string, ReportArtifact & { desc: string }>();

/** Turn a result's artifacts into floating 3D panels (existing ones shelve first). */
async function presentArtifacts(result: ChatResult, heard: string) {
  const desc = `${heard} — ${result.text}`;
  let shelved = false;
  const ensureShelved = () => {
    if (!shelved) {
      shelved = true;
      scene.shelvePanels();
    }
  };

  for (const [i, url] of result.images.entries()) {
    try {
      const img = await loadImage(url);
      ensureShelved();
      const imgId = `panel-${Date.now()}-img${i}`;
      scene.addPanel(imgId, img, desc, { framed: true });
      panelArtifacts.set(imgId, { kind: "image", desc, source: img });
    } catch (err) {
      console.error("image panel failed", err);
    }
  }

  if (result.visual?.kind === "mermaid") {
    let code = result.visual.code;
    for (let attempt = 0; attempt < 2; attempt++) {
      try {
        const { svgEl, markup } = await renderMermaidSvg(code);
        lastSvgMarkup = markup;
        result.visual.code = code; // keep the working source for download/memory
        const wipe = createWipeRenderer(await svgToCanvas(svgEl));
        ensureShelved();
        const svgId = `panel-${Date.now()}-svg`;
        scene.addPanel(svgId, wipe.canvas, desc, { animate: wipe.draw });
        panelArtifacts.set(svgId, { kind: "diagram", desc, source: wipe.canvas, svgMarkup: markup });
        break;
      } catch (err) {
        console.error("mermaid render failed", err);
        if (attempt === 0 && settings.key) {
          // one repair round: send the parse error back to the model
          try {
            code = await repairMermaid(code, String(err));
            continue;
          } catch (repairErr) {
            console.error(repairErr);
          }
        }
        showTranscript(`${result.text} (diagrammet kunne ikke vises)`);
      }
    }
  } else if (result.visual) {
    // model-authored SVG chart → same floating-card treatment as diagrams
    try {
      const holder = document.createElement("div");
      holder.innerHTML = result.visual.code;
      const svgEl = holder.querySelector("svg");
      if (!svgEl) throw new Error("no <svg> in visual output");
      lastSvgMarkup = svgEl.outerHTML;
      const wipe = createWipeRenderer(await svgToCanvas(svgEl));
      ensureShelved();
      const chartId = `panel-${Date.now()}-chart`;
      scene.addPanel(chartId, wipe.canvas, desc, { animate: wipe.draw });
      panelArtifacts.set(chartId, { kind: "chart", desc, source: wipe.canvas, svgMarkup: svgEl.outerHTML });
    } catch (err) {
      showTranscript(`${result.text} (grafen kunne ikke vises)`);
      console.error(err);
    }
  }

  if (result.slide) {
    ensureShelved();
    const slide = createSlideRenderer(result.slide);
    const slideId = `panel-${Date.now()}-slide`;
    scene.addPanel(slideId, slide.canvas, desc, { animate: slide.draw });
    panelArtifacts.set(slideId, { kind: "slide", desc, source: slide.canvas, slide: result.slide });
  }

  if (result.table) {
    ensureShelved();
    const stream = createTableRenderer(result.table);
    const tblId = `panel-${Date.now()}-tbl`;
    scene.addPanel(tblId, stream.canvas, desc, { animate: stream.draw });
    panelArtifacts.set(tblId, { kind: "table", desc, source: stream.canvas, table: result.table });
  }
}

// --- sub-agent visualization -----------------------------------------------

const AGENT_COLORS = ["#1d4433", "#d94018", "#3a4044", "#4a9272"]; // matches scene palette
let agentIndex = 0;

// spoken when a specialist agent activates — keyed by its spawn label
const AGENT_VOICE_LINES: Record<string, string[]> = {
  searching: [
    "Jeg søker på nettet.",
    "Henter ferske kilder.",
    "Sjekker nettet for deg.",
    "Henter ferske data.",
    "Spør omverdenen.",
  ],
  imagining: [
    "Kunstneren er i gang.",
    "Kunstneren jobber.",
    "Maler det for deg.",
    "Henter inn kunstneren.",
  ],
  repainting: [
    "Maler det om nå.",
    "Kunstneren justerer bildet.",
    "Retusjerer bildet.",
    "Tilbake til lerretet.",
  ],
  diagramming: [
    "Tegner diagrammet.",
    "Arkitekten skisserer.",
    "Legger opp strukturen.",
    "Tegner det opp nå.",
  ],
  visualizing: [
    "Bygger visualiseringen.",
    "Setter opp grafen.",
    "Plotter dataene.",
    "Dataskulptøren er på saken.",
  ],
  looking: [
    "Tar en titt på deg.",
    "La meg se.",
    "Åpner øyet.",
    "Observerer.",
  ],
  reporting: [
    "Setter sammen rapporten.",
    "Analytikeren skriver.",
    "Bygger dokumentet.",
    "Tolker lerretet.",
  ],
};

// spoken once when a parallel fan-out begins (labels are model-chosen there)
// spoken fillers while the LLM round-trip is in flight
const FILLERS = ["Et øyeblikk.", "Jobber med det.", "Er på saken.", "La meg tenke.", "Fortsatt i gang."];

/** Lines Octa says again and again: pre-generated in the Vader voice and kept
 *  in IndexedDB, so they play the instant they're needed. */
function stockLines(): string[] {
  return [
    ...FILLERS,
    ...Object.values(AGENT_VOICE_LINES).flat(),
    ...FANOUT_LINES,
    "Mørk modus.", "Lys modus.",
    "Tilbake i hylla.", "Lagt bort.", "Ryddet.",
    "Åpner.", "Her er den.", "I fokus.",
    "Henter den tilbake.", "På skjermen.", "Vær så god.",
    "Alt fjernet.", "Skjermen er tom.", "Borte.", "Fjernet.", "Vekk.",
    "Ingenting er i fokus.", "Det er ingenting å åpne.", "Det er ingenting å hente tilbake ennå.",
    "Skjermen er allerede tom.", "Det er ingenting å laste ned ennå.",
    "Lagret den i nedlastinger.",
    "Bytter til den smarte modellen.", "Tilbake til den raske modellen.",
    "Jeg bruker allerede den smarte modellen.", "Jeg bruker allerede den raske modellen.",
    "Notatmodus på. Kryss hendene eller si «stopp notater» når du er ferdig.",
    "Notatene er lagret.", "Notatmodus av.", "Gestestyring er på.",
    "Jeg så ikke at du snakket. Si det gjerne igjen.",
    "Jeg fikk ikke kontakt med modellen. Sjekk innstillingene.",
    ...[1, 2, 3, 4, 5, 6].map((n) => `P${n}, værsågod.`),
  ];
}

let warming = false;
function warmVoice() {
  if (warming || settings.voiceStyle !== "vader" || !settings.key) return;
  warming = true;
  void speaker.vader.warm(stockLines()).finally(() => (warming = false));
}

const FANOUT_LINES = [
  "Jeg kaller inn underagentene mine.",
  "Deler dette på agentene mine.",
  "Sender ut teamet.",
  "Sender ut agentene.",
  "Sprer oss ut på denne.",
  "Agentene mine er på saken.",
];

const pickLine = (arr: string[]) => arr[Math.floor(Math.random() * arr.length)];
let fanoutAnnounced = false;

/** Switch Octa's brain; the body morphs to match. Returns false if already there. */
function setTier(tier: ModelTier): boolean {
  if (settings.tier === tier) return false;
  settings.tier = tier;
  scene.setShape(tier === "smart" ? "icosahedron" : "octahedron");
  sound?.ping(tier === "smart" ? 1320 : 660, 0.08);
  console.info(`[tier] ${tier} → ${settings.model}`);
  return true;
}

function announce(text: string) {
  speaker.speak(text, () => {
    scene.pulse(0.7);
    speakLevel = 0.4;
  });
}

const agentEvents = {
  onSpawn(id: string, label: string) {
    scene.addSatellite(id);
    sound?.ping(620 + (agentIndex % 4) * 180, 0.09);
    const lines = AGENT_VOICE_LINES[label];
    if (lines) {
      announce(pickLine(lines));
    } else if (!fanoutAnnounced) {
      fanoutAnnounced = true;
      announce(pickLine(FANOUT_LINES));
    }
    const chip = document.createElement("span");
    chip.dataset.agent = id;
    chip.textContent = label;
    chip.style.borderColor = chip.style.color = AGENT_COLORS[agentIndex++ % 4];
    agentLabelsEl.appendChild(chip);
  },
  onDone(id: string) {
    scene.removeSatellite(id);
    agentLabelsEl.querySelector(`[data-agent="${id}"]`)?.remove();
  },
};

// --- ambient compliments: camera-grounded filler lines ----------------------

// In a meeting each person gets their own pool, drawn from a crop of just
// them — compliments go to whoever holds the floor. Key 0 = no floor holder
// (solo, or no faces in view): the whole frame, as before.
const complimentPools = new Map<number, string[]>();
const complimentsAt = new Map<number, number>();

function complimentTarget() {
  return gestures?.holderCrop() ?? null;
}

async function refreshCompliments() {
  if (!settings.key || !gestures) return;
  const target = complimentTarget();
  const key = target?.person ?? 0;
  const pool = complimentPools.get(key) ?? [];
  if (pool.length >= 2 || Date.now() - (complimentsAt.get(key) ?? 0) < 240_000) return;
  complimentsAt.set(key, Date.now());
  try {
    const frame = await captureCameraFrame(gestures, target?.crop);
    if (!frame) return;
    const lines = await generateCompliments(frame);
    complimentPools.set(key, [...(complimentPools.get(key) ?? []), ...lines]);
  } catch (err) {
    console.error("compliment refresh failed", err);
  }
}

/** Next compliment for whoever holds the floor right now. */
function takeCompliment(): string | undefined {
  return complimentPools.get(complimentTarget()?.person ?? 0)?.shift();
}

// --- the brain: OpenRouter when a key is set, local echo otherwise ---------

async function reply(heard: string, events: typeof agentEvents): Promise<ChatResult> {
  if (!settings.key) {
    return { text: await think(heard), images: [] };
  }
  const result = await respond(heard, events, {
    lastImage: lastResult?.images[0],
    captureFrame: () => captureCameraFrame(gestures),
  });
  return keynote(result);
}

/** Rescue chart/diagram code that leaked into reply text: extract it into a
 *  renderable visual and keep only the prose for the voice. */
function salvageVisualFromText(result: ChatResult) {
  if (result.visual || !result.text) return;
  const text = result.text;
  let code: string | undefined;
  let kind: "mermaid" | "svg" | undefined;
  let matched = "";

  const fence = text.match(/```[a-z]*\s*\n([\s\S]*?)```/i);
  if (fence) {
    code = fence[1].trim();
    matched = fence[0];
    kind = /<svg/i.test(code) ? "svg" : "mermaid";
  } else {
    const svgm = text.match(/<svg[\s\S]*<\/svg>/i);
    if (svgm) {
      code = svgm[0];
      matched = svgm[0];
      kind = "svg";
    } else {
      const mm = text.match(
        /\b(flowchart\s+\w\w?|graph\s+\w\w?|sequenceDiagram|classDiagram|stateDiagram(?:-v2)?|erDiagram|journey|mindmap|timeline|gantt)\b[\s\S]*/
      );
      if (mm) {
        code = mm[0];
        matched = mm[0];
        kind = "mermaid";
      }
    }
  }
  if (!code || !kind) return;
  result.visual = { kind, code };
  result.text = text.replace(matched, "").replace(/\s+/g, " ").trim() || "Her er grafen.";
}

/** Strip light markdown so TTS doesn't read symbols aloud. */
function speakable(text: string): string {
  return text
    .replace(/[*_#`>|]/g, "")
    .replace(/\[([^\]]*)\]\([^)]*\)/g, "$1")
    .trim();
}

// --- note-taker mode --------------------------------------------------------

let noteMode = false;
let noteLines: string[] = [];
let noteRec: Recognizer | null = null;

function enterNoteMode() {
  if (noteMode) return;
  noteMode = true;
  noteLines = [];
  setState("noting");
  // wait for the announcement to actually finish so we don't transcribe ourselves
  speaker
    .speak("Notatmodus på. Kryss hendene eller si «stopp notater» når du er ferdig.", () => {
      scene.pulse(0.7);
      speakLevel = 0.4;
    })
    .then(() => setTimeout(noteLoop, 300));
}

// lines that are just Octa hearing itself — never note these
const SELF_ECHO = /note ?taker (online|off)|cross your hands|notes stored|notatmodus (på|av)|kryss hendene|notatene er lagret/i;

function noteLoop() {
  if (!noteMode) return;
  const rec = new Recognizer();
  noteRec = rec;
  rec.onInterim = (t) => showTranscript(t, true);
  rec.onFinal = (t) => {
    if (isStopNotesCommand(t)) {
      exitNoteMode();
      return;
    }
    if (SELF_ECHO.test(t)) return; // Octa hearing its own announcements
    noteLines.push(t);
    showTranscript(t);
    scene.pulse(0.5);
  };
  rec.onEnd = () => {
    noteRec = null;
    if (noteMode) setTimeout(noteLoop, 250); // keep the scribe listening
  };
  rec.start();
}

function exitNoteMode() {
  if (!noteMode) return;
  noteMode = false;
  noteRec?.stop();
  noteRec = null;
  setState("idle");
  showTranscript("");
  if (noteLines.length) {
    const stamp = new Date().toLocaleDateString("nb-NO", { month: "short", day: "numeric" });
    const canvas = noteToCanvas(noteLines, `Notater — ${stamp}`);
    const id = `panel-${Date.now()}-note`;
    const noteDesc = `notater notes ${noteLines.join(" ").slice(0, 120)}`;
    scene.addPanel(id, canvas, noteDesc);
    panelArtifacts.set(id, {
      kind: "note",
      desc: noteDesc,
      source: canvas,
      noteLines: [...noteLines],
      noteTitle: `Notater — ${stamp}`,
    });
    setTimeout(() => scene.shelfPanel(id), 1800); // show it, then tuck into the grid
    remember("assistant", `Notater tatt:\n${noteLines.join("\n")}`);
    sound?.ping(996, 0.08);
    announce("Notatene er lagret.");
  } else {
    announce("Notatmodus av.");
  }
}

// --- conversation loop ------------------------------------------------------

let turnId = 0;
let activeRec: Recognizer | null = null;
let lastResult: ChatResult | null = null; // artifacts "download it" refers to
let lastSvgMarkup: string | null = null; // rendered mermaid, for SVG download

/** Drop the current turn: silence TTS, clear satellites, invalidate in-flight work. */
function interrupt() {
  turnId++;
  speaker.stop();
  scene.clearSatellites();
  agentLabelsEl.innerHTML = "";
  speakLevel = 0;
}

async function converse() {
  if (noteMode) return; // the scribe owns the microphone
  sound?.ensureRunning();

  if (state === "listening") {
    activeRec?.stop(); // second press: finish the capture now
    return;
  }
  if (state === "thinking" || state === "speaking") {
    interrupt(); // barge-in: drop the current turn and listen right away
  }

  const myTurn = ++turnId;
  setState("listening");
  showTranscript("");
  // artifacts stay on screen while listening — "download it" needs them

  const rec = new Recognizer();
  activeRec = rec;
  const listenSince = performance.now();
  const heard = await new Promise<string | null>((resolve) => {
    rec.onInterim = (t) => showTranscript(t, true);
    rec.onFinal = (t) => resolve(t);
    rec.onEnd = () => resolve(null);
    rec.start();
  });
  activeRec = null;
  if (myTurn !== turnId) return;

  if (!heard) {
    setState("idle");
    showTranscript("");
    return;
  }

  // in a meeting: only act on speech when the floor holder's lips were moving
  const mouth = settings.handGestures ? gestures?.holderSpokeSince(listenSince) : null;
  mouthVerdict = {
    text: !mouth
      ? "heard · lips not checked"
      : mouth.spoke
        ? `heard P${mouth.holder} ✓`
        : `dropped · P${mouth.holder} lips ${mouth.movingMs}ms`,
    ok: !mouth || mouth.spoke,
    at: performance.now(),
  };
  if (mouth && !mouth.spoke) {
    console.info(
      `[floor] dropped "${heard}" — P${mouth.holder}: lips ${mouth.movingMs}ms, to Octa ${mouth.toOctaMs}ms, alone ${mouth.alone}`
    );
    setState("speaking");
    // alone in view: the voice came from off camera — don't tell anyone to raise a hand
    const msg = mouth.alone
      ? "Jeg så ikke at du snakket. Si det gjerne igjen."
      : `Jeg hørte noen, men P${mouth.holder} har ordet. De andre kan rekke opp hånden.`;
    showTranscript(msg);
    await speaker.speak(msg, () => {
      scene.pulse(1);
      speakLevel = 0.5;
    });
    if (myTurn === turnId) setState("idle");
    return;
  }

  // prototype: Jev judges whether this was said to Octa or to the room
  if (jevSettings.mode !== "off" && settings.key) {
    setState("thinking");
    const v = await checkAddressee(heard, {
      speaker: mouth?.holder ?? null,
      facingOcta: mouth ? mouth.facing : null,
      lipsMoving: mouth ? mouth.moving : null,
      peopleInView: gestures?.snapshot().persons.length ?? 0,
    });
    if (myTurn !== turnId) return;
    const p = v.noul === null ? "?" : v.noul.toFixed(2);
    mouthVerdict = {
      text: `jev ${p} · ${v.latencyMs}ms${v.dropped ? " · dropped" : ""}`,
      ok: !v.dropped,
      at: performance.now(),
    };
    if (v.dropped) {
      // said to the room, not to Octa — stay quiet, just show what was set aside
      showTranscript(`(ikke til meg) ${heard}`, true);
      setState("idle");
      setTimeout(() => {
        if (state === "idle") showTranscript("");
      }, 3000);
      return;
    }
  }
  showTranscript(heard);

  // panel + download commands are handled locally — no model round-trip
  const localReply = async (msg: string) => {
    setState("speaking");
    showTranscript(msg);
    await speaker.speak(msg, () => {
      scene.pulse(1);
      speakLevel = 0.5;
    });
    if (myTurn === turnId) setState("idle");
  };

  // "bruk en smartere modell" — switch brains locally, no model round-trip;
  // anything said beyond the command is then answered by the new brain
  const tier = tierCommand(heard);
  if (tier) {
    const switched = setTier(tier);
    if (heard.split(/\s+/).length <= 8) {
      await localReply(
        !switched
          ? tier === "smart" ? "Jeg bruker allerede den smarte modellen." : "Jeg bruker allerede den raske modellen."
          : tier === "smart" ? "Bytter til den smarte modellen." : "Tilbake til den raske modellen."
      );
      return;
    }
  }

  if (isDownloadCommand(heard)) {
    const hasArtifacts =
      lastResult && (lastResult.images.length || lastResult.visual || lastResult.table);
    const n = hasArtifacts ? await downloadArtifacts(lastResult!, lastSvgMarkup ?? undefined) : 0;
    await localReply(
      n
        ? `Lagret ${n === 1 ? "den" : `${n} filer`} i nedlastinger.`
        : "Det er ingenting å laste ned ennå."
    );
    return;
  }

  if (/\b(dark|night) mode\b|\bm(ø|o)rk ?modus|\bnattmodus/i.test(heard)) {
    setDarkMode(true);
    darkCheck.checked = true;
    await localReply("Mørk modus.");
    return;
  }
  if (/\b(light|day) mode\b|\blys ?modus|\bdagmodus/i.test(heard)) {
    setDarkMode(false);
    darkCheck.checked = false;
    await localReply("Lys modus.");
    return;
  }

  if (isNoteCommand(heard)) {
    enterNoteMode();
    return;
  }

  if (isReportCommand(heard)) {
    if (!settings.key) {
      await localReply("Jeg trenger en OpenRouter-nøkkel for å skrive rapporter.");
      return;
    }
    setState("thinking");
    const live = new Set(scene.listPanels().map((p) => p.id));
    const inventory = [...panelArtifacts.entries()]
      .filter(([id]) => live.has(id))
      .map(([id, a]) => ({ id, kind: a.kind, desc: a.desc.slice(0, 140) }));
    const events = {
      onSpawn: (id: string, label: string) => {
        if (myTurn === turnId) agentEvents.onSpawn(id, label);
      },
      onDone: (id: string) => agentEvents.onDone(id),
    };
    const spec = await generateReport(heard, inventory, events);
    if (myTurn !== turnId) return;
    if (!spec) {
      await localReply("Jeg fikk ikke satt sammen rapporten. Prøv igjen.");
      return;
    }
    const usable = new Map<string, ReportArtifact>();
    for (const [id, a] of panelArtifacts) {
      if (live.has(id)) usable.set(id, a);
    }
    const filename = await buildReportPdf(spec, usable);
    remember("user", heard);
    remember("assistant", `Report generated: "${spec.title}" (${filename}).`);
    await localReply(`Rapporten er klar: ${spec.title}. Lagret i nedlastinger.`);
    return;
  }

  if (isShelveCommand(heard)) {
    const id = scene.focusedPanelId();
    if (id) {
      scene.shelfPanel(id);
      sound?.ping(996, 0.06);
      await localReply(pickLine(["Tilbake i hylla.", "Lagt bort.", "Ryddet."]));
    } else {
      await localReply("Ingenting er i fokus.");
    }
    return;
  }

  if (isOpenCommand(heard)) {
    // "open this" = whatever has highlight focus (hand hover or mouse)
    const target =
      scene.highlightedPanel() ??
      [...scene.listPanels()].reverse().find((p) => p.id !== scene.focusedPanelId())?.id;
    if (target && scene.recallPanel(target)) {
      sound?.ping(880, 0.07);
      await localReply(pickLine(["Åpner.", "Her er den.", "I fokus."]));
    } else {
      await localReply("Det er ingenting å åpne.");
    }
    return;
  }

  if (isRecallCommand(heard)) {
    const target = matchPanel(heard, scene.listPanels());
    if (target && scene.recallPanel(target)) {
      await localReply(pickLine(["Her er den.", "Henter den tilbake.", "På skjermen.", "Vær så god."]));
    } else {
      await localReply("Det er ingenting å hente tilbake ennå.");
    }
    return;
  }

  if (isDismissCommand(heard)) {
    const panels = scene.listPanels();
    if (!panels.length) {
      await localReply("Skjermen er allerede tom.");
    } else if (isDismissAll(heard)) {
      scene.clearPanels();
      visualsEl.innerHTML = "";
      await localReply(pickLine(["Alt fjernet.", "Skjermen er tom.", "Borte."]));
    } else {
      const target = matchPanel(heard, panels);
      if (target) scene.removePanel(target);
      await localReply(pickLine(["Borte.", "Fjernet.", "Vekk."]));
    }
    return;
  }

  setState("thinking");
  void refreshCompliments(); // top up the personal filler pool in the background

  // Octa thinks out loud: a fast side call writes curious / darkly funny lines
  // about the request, spoken between fillers while the real answer is made
  const commentary: string[] = [];
  if (settings.key) {
    generateCommentary(heard)
      .then((lines) => {
        if (myTurn === turnId) commentary.push(...lines);
      })
      .catch(() => {});
  }

  // spoken fillers while the LLM round-trip is in flight — same low-latency TTS
  let fillerCount = 0;
  let fillerTimer = 0;
  const scheduleFiller = (delay: number) => {
    fillerTimer = window.setTimeout(() => {
      if (myTurn !== turnId || state !== "thinking") return;
      // agent announcements outrank generic fillers — never talk over them
      if (!speaker.speaking) {
        // Octa's own commentary on the topic as soon as it's written (a stored
        // filler covers the gap until then), with a compliment for whoever has
        // the floor slipped in every third beat
        const compliment = fillerCount > 0 && fillerCount % 3 === 2 ? takeCompliment() : undefined;
        const remark = compliment ? undefined : commentary.shift();
        const line = remark ?? compliment ?? FILLERS[Math.min(fillerCount, FILLERS.length - 1)];
        if (remark) showTranscript(remark, true); // the aside shows, dimmed, under the orb
        fillerCount++;
        speaker.speak(line, () => {
          scene.pulse(0.6);
          speakLevel = 0.35;
        });
      }
      // most answers land within a few seconds — keep the early beats close together
      scheduleFiller(fillerCount < 3 ? 3000 + Math.random() * 1200 : 5000 + Math.random() * 3000);
    }, delay);
  };
  scheduleFiller(1400);
  fanoutAnnounced = false;

  // stale turns must not spawn satellites, but finished ones may always clean up
  const events = {
    onSpawn: (id: string, label: string) => {
      if (myTurn === turnId) agentEvents.onSpawn(id, label);
    },
    onDone: (id: string) => agentEvents.onDone(id),
  };
  let result: ChatResult;
  try {
    result = await reply(heard, events);
    remember("user", heard);
    // keep visual code / table data in memory so follow-ups can iterate on them
    let memText = result.text;
    if (result.visual) {
      memText += `\n\n\`\`\`${result.visual.kind}\n${result.visual.code}\n\`\`\``;
    } else if (result.table) {
      memText +=
        `\n\n${result.table.title ?? ""}\n${result.table.headers.join(" | ")}\n` +
        result.table.rows.map((r) => r.join(" | ")).join("\n");
    }
    if (result.slide) {
      // the slide holds the substance — follow-ups ("tell me more about point 2") need it
      const sl = result.slide;
      memText +=
        `\n\nSlide: ${sl.headline}` +
        (sl.stat ? `\n${sl.stat.value} — ${sl.stat.label}` : "") +
        sl.points.map((pt, i) => `\n${i + 1}. ${pt.title}${pt.detail ? `: ${pt.detail}` : ""}`).join("");
    }
    remember("assistant", memText);
  } catch (err) {
    result = { text: "Jeg fikk ikke kontakt med modellen. Sjekk innstillingene.", images: [] };
    console.error(err);
  }
  clearTimeout(fillerTimer);
  if (myTurn !== turnId) return;
  agentLabelsEl.innerHTML = "";
  salvageVisualFromText(result);
  if (result.tier) {
    // the router heard an indirect "think harder" / "keep it quick"
    const switched = setTier(result.tier);
    if (!result.text) {
      result.text = !switched
        ? "Den modellen bruker jeg allerede."
        : result.tier === "smart" ? "Bytter til den smarte modellen." : "Tilbake til den raske modellen.";
    }
  }
  lastResult = result;

  setState("speaking");
  showTranscript(result.text);
  await presentArtifacts(result, heard);
  await speaker.speak(speakable(result.text), () => {
    scene.pulse(0.8 + Math.random() * 0.5);
    speakLevel = 0.45 + Math.random() * 0.35;
  });
  if (myTurn !== turnId) return;

  setState("idle");
  setTimeout(() => {
    if (state === "idle") showTranscript("");
  }, 6000);
}

// --- settings panel ---------------------------------------------------------

const settingsBtn = document.querySelector<HTMLButtonElement>("#settings-btn")!;
const settingsPanel = document.querySelector<HTMLDivElement>("#settings")!;
const keyInput = document.querySelector<HTMLInputElement>("#or-key")!;
const modelInput = document.querySelector<HTMLInputElement>("#or-model")!;
const modelList = document.querySelector<HTMLDataListElement>("#model-list")!;
const smartModelInput = document.querySelector<HTMLInputElement>("#or-smart-model")!;
const fastEffortSelect = document.querySelector<HTMLSelectElement>("#or-fast-effort")!;
fastEffortSelect.value = settings.fastEffort;
fastEffortSelect.addEventListener("change", () => (settings.fastEffort = fastEffortSelect.value));
const imageModelInput = document.querySelector<HTMLInputElement>("#or-image-model")!;
const imageModelList = document.querySelector<HTMLDataListElement>("#image-model-list")!;
const agentsCheck = document.querySelector<HTMLInputElement>("#or-agents")!;
const handsCheck = document.querySelector<HTMLInputElement>("#or-hands")!;
const voiceSelect = document.querySelector<HTMLSelectElement>("#or-voice")!;
const voiceStyleSelect = document.querySelector<HTMLSelectElement>("#or-voice-style")!;
voiceStyleSelect.value = settings.voiceStyle;
voiceStyleSelect.addEventListener("change", () => {
  settings.voiceStyle = voiceStyleSelect.value as VoiceStyle;
  if (settings.voiceStyle === "vader") warmVoice();
});

const jevModeSelect = document.querySelector<HTMLSelectElement>("#jev-mode")!;
jevModeSelect.value = jevSettings.mode;
jevModeSelect.addEventListener("change", () => (jevSettings.mode = jevModeSelect.value as JevMode));

const darkCheck = document.querySelector<HTMLInputElement>("#or-dark")!;
darkCheck.checked = theme().dark;
darkCheck.addEventListener("change", () => setDarkMode(darkCheck.checked));

// --- camera view: what the gesture camera sees, with faces and hands marked --

const camCheck = document.querySelector<HTMLInputElement>("#or-camview")!;
const camView = document.querySelector<HTMLCanvasElement>("#camview")!;
const camCtx = camView.getContext("2d")!;
// last lip-check verdict, shown in the camera view for a few seconds
let mouthVerdict: { text: string; ok: boolean; at: number } | null = null;

function setCamView(on: boolean) {
  camCheck.checked = on;
  camView.classList.toggle("visible", on);
}
camCheck.addEventListener("change", () => setCamView(camCheck.checked));

function drawCamView() {
  const video = gestures?.videoEl;
  if (!camCheck.checked) return;
  const w = 320;
  const h = video ? Math.round((w * video.videoHeight) / (video.videoWidth || 1)) || 180 : 180;
  if (camView.width !== w || camView.height !== h) {
    camView.width = w;
    camView.height = h;
  }
  camCtx.clearRect(0, 0, w, h);
  camCtx.font = "11px ui-monospace, Menlo, monospace";
  if (!video || video.readyState < 2 || !gestures) {
    camCtx.fillStyle = "#8c8677";
    camCtx.fillText(settings.handGestures ? "starting camera…" : "camera off — enable hand gestures", 10, h / 2);
    return;
  }
  const k = w / video.videoWidth;
  // mirrored, like the cursors: move right, see yourself move right
  camCtx.save();
  camCtx.translate(w, 0);
  camCtx.scale(-1, 1);
  camCtx.drawImage(video, 0, 0, w, h);
  camCtx.restore();
  const mx = (x: number) => w - x * k;
  const css = getComputedStyle(document.documentElement);
  const tint = (id: number | null) =>
    id === null ? "#8c8677" : css.getPropertyValue(`--person-${(id - 1) % 6}`).trim();

  const snap = gestures.snapshot();
  for (const p of snap.persons) {
    camCtx.strokeStyle = tint(p.id);
    camCtx.lineWidth = p.id === snap.holder ? 3 : 1.5;
    camCtx.strokeRect(mx(p.x + p.w), p.y * k, p.w * k, p.h * k);
    camCtx.fillStyle = tint(p.id);
    const tags =
      (p.id === snap.holder ? " ●" : "") + (p.looking ? " looking" : "") + (p.talking ? " speaking" : "");
    camCtx.fillText(`P${p.id}${tags}`, mx(p.x + p.w), p.y * k - 4);
  }
  for (const hd of snap.hands) {
    camCtx.fillStyle = tint(hd.owner);
    camCtx.beginPath();
    camCtx.arc(mx(hd.x), hd.y * k, hd.raised ? 7 : 4, 0, Math.PI * 2);
    camCtx.fill();
  }
  camCtx.fillStyle = "rgba(0,0,0,0.55)";
  camCtx.fillRect(0, h - 18, w, 18);
  camCtx.fillStyle = "#fff";
  camCtx.fillText(`faces ${snap.persons.length} · hands ${snap.hands.length}`, 8, h - 6);
  // mic state on the right: is Octa listening right now, and what did the lip check decide?
  let mic = state === "listening" ? "● listening" : state;
  camCtx.fillStyle = state === "listening" ? "#7fd4a0" : "#ccc";
  if (state !== "listening" && mouthVerdict && performance.now() - mouthVerdict.at < 5000) {
    mic = mouthVerdict.text;
    camCtx.fillStyle = mouthVerdict.ok ? "#7fd4a0" : "#ff8a65";
  }
  camCtx.textAlign = "right";
  camCtx.fillText(mic, w - 8, h - 6);
  camCtx.textAlign = "left";
}

handsCheck.checked = settings.handGestures;
handsCheck.addEventListener("change", () => {
  settings.handGestures = handsCheck.checked;
  if (handsCheck.checked) enableHandGestures();
  else disableHandGestures();
});

function populateVoices() {
  const current = settings.voice;
  voiceSelect.innerHTML = '<option value="">auto</option>';
  for (const v of speechSynthesis.getVoices().filter((v) => /^(nb|no|nn|en)/i.test(v.lang))) {
    const opt = document.createElement("option");
    opt.value = v.name;
    opt.textContent = `${v.name} (${v.lang})`;
    opt.selected = v.name === current;
    voiceSelect.appendChild(opt);
  }
}
speechSynthesis.addEventListener("voiceschanged", populateVoices);
populateVoices();
voiceSelect.addEventListener("change", () => (settings.voice = voiceSelect.value));

keyInput.value = settings.key;
modelInput.value = settings.fastModel;
smartModelInput.value = settings.smartModel;
imageModelInput.value = settings.imageModel;
agentsCheck.checked = settings.agentsEnabled;

let modelsLoaded = false;
settingsBtn.addEventListener("click", async () => {
  settingsPanel.classList.toggle("hidden");
  if (!modelsLoaded && !settingsPanel.classList.contains("hidden")) {
    modelsLoaded = true;
    try {
      for (const m of await listModels()) {
        const opt = document.createElement("option");
        opt.value = m.id;
        opt.label = m.name;
        modelList.appendChild(opt);
        if (m.imageOutput) imageModelList.appendChild(opt.cloneNode(true));
      }
    } catch {
      modelsLoaded = false; // retry next open
    }
  }
});

keyInput.addEventListener("change", () => {
  settings.key = keyInput.value.trim();
  warmVoice();
});
modelInput.addEventListener("change", () => (settings.fastModel = modelInput.value.trim()));
smartModelInput.addEventListener("change", () => (settings.smartModel = smartModelInput.value.trim()));
imageModelInput.addEventListener("change", () => (settings.imageModel = imageModelInput.value.trim()));
agentsCheck.addEventListener("change", () => (settings.agentsEnabled = agentsCheck.checked));

const clearBtn = document.querySelector<HTMLButtonElement>("#clear-memory")!;
clearBtn.addEventListener("click", () => {
  clearHistory();
  clearBtn.textContent = "forgotten";
  setTimeout(() => (clearBtn.textContent = "forget conversation"), 1500);
});

// --- boot -------------------------------------------------------------------

startBtn.addEventListener("click", async () => {
  overlay.classList.add("hidden");

  sound = new Soundscape();
  await sound.start();

  try {
    await mic.init();
  } catch {
    showTranscript("mikrofonen er utilgjengelig — bare visuelt");
  }

  if (!Recognizer.supported()) {
    showTranscript("talegjenkjenning krever Chrome — bare visuelt");
  }

  // warm up the TTS voice list (loads async in most browsers)
  speechSynthesis.getVoices();
  warmVoice(); // fillers and confirmations ready before they're needed
  speaker.vader.prime();

  if (settings.handGestures) enableHandGestures();
});

// --- gestures: drag to pan, wheel/pinch to zoom, tap shelf to recall --------

let dragPanelId: string | null = null;
let dragDistance = 0;
let lastPointerX = 0;
let lastPointerY = 0;
let downHit: { id: string; focused: boolean } | null = null;

canvas.addEventListener("pointerdown", (e) => {
  downHit = scene.hitPanel(e.clientX, e.clientY);
  dragDistance = 0;
  lastPointerX = e.clientX;
  lastPointerY = e.clientY;
  if (downHit?.focused) {
    dragPanelId = downHit.id;
    canvas.setPointerCapture(e.pointerId);
    canvas.style.cursor = "grabbing";
  }
});

canvas.addEventListener("pointermove", (e) => {
  if (dragPanelId) {
    const dx = e.clientX - lastPointerX;
    const dy = e.clientY - lastPointerY;
    dragDistance += Math.abs(dx) + Math.abs(dy);
    scene.panPanel(dragPanelId, dx, -dy);
    lastPointerX = e.clientX;
    lastPointerY = e.clientY;
  } else {
    const hover = scene.hitPanel(e.clientX, e.clientY);
    canvas.style.cursor = hover?.focused ? "grab" : "pointer";
    claimHighlight("mouse", hover && !hover.focused ? hover.id : null);
  }
});

canvas.addEventListener("pointerup", () => {
  const wasDrag = dragDistance > 6;
  const hit = downHit;
  const draggedId = dragPanelId;
  dragPanelId = null;
  downHit = null;
  canvas.style.cursor = "pointer";
  if (wasDrag) {
    // flicked upward past the threshold → back to the bento grid
    if (draggedId && scene.maybeShelfAfterDrag(draggedId)) sound?.ping(996, 0.06);
    return;
  }

  if (!settingsPanel.classList.contains("hidden")) {
    settingsPanel.classList.add("hidden");
    return;
  }
  if (hit && !hit.focused) {
    // tap a shelved panel to recall it
    scene.recallPanel(hit.id);
    sound?.ping(880, 0.07);
    return;
  }
  if (hit?.focused) return; // taps on the focused panel are gesture territory
  converse();
});

canvas.addEventListener("dblclick", (e) => {
  const hit = scene.hitPanel(e.clientX, e.clientY);
  if (hit?.focused) scene.resetPanelView(hit.id);
});

// --- camera hand gestures: pinch-pan, two-hand zoom, palm-hold reset --------

// one dot per visible hand, tinted by the person it belongs to
const handDots: HTMLDivElement[] = [];
const meetingEl = document.querySelector<HTMLDivElement>("#meeting")!;

/** Stable tint per tracked person. */
function personColor(id: number | null) {
  return id === null ? "var(--dim)" : `var(--person-${(id - 1) % 6})`;
}

function renderHandDots(hands: HandState[]) {
  while (handDots.length < hands.length) {
    const dot = document.createElement("div");
    dot.className = "hand-dot";
    document.body.appendChild(dot);
    handDots.push(dot);
  }
  for (const [i, dot] of handDots.entries()) {
    const h = hands[i];
    dot.classList.toggle("visible", !!h);
    if (!h) continue;
    dot.style.transform = `translate(${h.x}px, ${h.y}px)`;
    dot.style.setProperty("--who", personColor(h.person));
    dot.classList.toggle("pinch", h.pinching && h.active);
    dot.classList.toggle("passive", !h.active);
    dot.classList.toggle("raised", h.raised);
  }
}

function renderMeeting(m: MeetingState) {
  meetingEl.innerHTML = "";
  meetingEl.classList.toggle("visible", m.persons.length > 0 || m.queue.length > 0);
  const chip = (id: number, role: string) => {
    const el = document.createElement("span");
    el.className = `who ${role}`;
    el.style.setProperty("--who", personColor(id));
    el.textContent = `P${id}`;
    return el;
  };
  const label = (text: string) => {
    const el = document.createElement("span");
    el.className = "label";
    el.textContent = text;
    return el;
  };
  meetingEl.append(label("floor"));
  meetingEl.append(m.holder !== null ? chip(m.holder, "holder") : label("— raise a hand"));
  if (m.queue.length) {
    meetingEl.append(label("next"));
    for (const id of m.queue) meetingEl.append(chip(id, "queued"));
  }
}

let gestures: GestureControl | null = null;
let wasPinching = false;
let floorHolder: number | null = null;

// One mediator for highlight focus: hand hover beats mouse hover.
// Each source only releases its own claim.
const highlightClaims: { hand: string | null; mouse: string | null } = {
  hand: null,
  mouse: null,
};

function claimHighlight(source: "hand" | "mouse", id: string | null) {
  highlightClaims[source] = id;
  scene.highlightPanel(highlightClaims.hand ?? highlightClaims.mouse);
}

async function enableHandGestures() {
  if (gestures) return;
  gestures = new GestureControl();
  try {
    await gestures.start({
      onPan: (dx, dy) => {
        if (!settings.handGestures) return;
        const id = scene.focusedPanelId();
        if (id) scene.panPanel(id, dx, dy);
      },
      onZoom: (factor) => {
        if (!settings.handGestures) return;
        const id = scene.focusedPanelId();
        if (id) scene.zoomPanel(id, factor);
      },
      onPalmHold: () => {
        if (!settings.handGestures) return;
        // open palm = sweep everything back into the bento grid
        if (scene.focusedPanelId()) {
          scene.shelvePanels();
          sound?.ping(996, 0.07);
        }
      },
      onEarTouch: () => {
        if (!settings.handGestures) return;
        // hand to ear = "I'm listening" — barge in from any state except capture
        if (state === "listening") return;
        sound?.ping(1660, 0.06);
        converse();
      },
      onHandsCrossed: () => {
        if (noteMode) exitNoteMode();
      },
      onFistOpen: (x, y) => {
        if (!settings.handGestures) return;
        // open the fist over a grid item → pull it forth into focus
        const hit = scene.hitPanel(x, y);
        const target =
          (hit && !hit.focused ? hit.id : null) ??
          scene.nearestShelfPanel(x, y) ??
          scene.highlightedPanel();
        if (target && scene.recallPanel(target)) sound?.ping(880, 0.07);
      },
      onDoublePinch: (x, y) => {
        if (!settings.handGestures) return;
        const hit = scene.hitPanel(x, y);
        if (hit && !hit.focused) {
          scene.recallPanel(hit.id);
          sound?.ping(880, 0.07);
        } else if (hit?.focused) {
          scene.resetPanelView(hit.id);
        }
      },
      onHead: (nx, ny, size) => scene.setHeadPosition(nx, ny, size),
      onHands: (all) => {
        const hands = settings.handGestures ? all : [];
        renderHandDots(hands);
        const steering = hands.filter((h) => h.active);
        // pinch release after an upward pinch-drag also shelves the panel
        const anyPinch = steering.some((h) => h.pinching);
        if (wasPinching && !anyPinch && settings.handGestures) {
          const id = scene.focusedPanelId();
          if (id && scene.maybeShelfAfterDrag(id)) sound?.ping(996, 0.06);
        }
        wasPinching = anyPinch;
        let hover: string | null = null;
        for (const h of steering) {
          // an open hand near a shelf tile marks it as the candidate (magnetic)
          const hit = scene.hitPanel(h.x, h.y);
          hover = hit && !hit.focused ? hit.id : scene.nearestShelfPanel(h.x, h.y);
          if (hover) break;
        }
        claimHighlight("hand", hover);
      },
      onMeeting: (m) => {
        renderMeeting(m);
        if (m.holder !== floorHolder) {
          // the talking stick changed hands — a short rising chirp
          if (m.holder !== null && settings.handGestures) {
            sound?.ping(1320, 0.06);
            // Octa made its pick — name the person, but never talk over a turn
            if (state === "idle") announce(`P${m.holder}, værsågod.`);
          }
          floorHolder = m.holder;
          // have compliments ready for the new holder before they ask anything
          if (m.holder !== null) setTimeout(refreshCompliments, 1500);
        }
      },
    });
    announce("Gestestyring er på.");
    setTimeout(refreshCompliments, 5000);
  } catch (err) {
    gestures = null;
    settings.handGestures = false;
    handsCheck.checked = false;
    showTranscript("kameraet er utilgjengelig — gester av");
    console.error(err);
  }
}

function disableHandGestures() {
  gestures?.stop();
  gestures = null;
  scene.setHeadPosition(0, 0, 0); // ease the parallax camera back to center
  for (const dot of handDots) dot.classList.remove("visible", "pinch");
  floorHolder = null;
  renderMeeting({ holder: null, queue: [], persons: [] });
}

canvas.addEventListener(
  "wheel",
  (e) => {
    const hit = scene.hitPanel(e.clientX, e.clientY);
    if (!hit?.focused) return;
    e.preventDefault();
    // trackpad pinch arrives as ctrl+wheel — give it a stronger response
    const factor = Math.exp(-e.deltaY * (e.ctrlKey ? 0.01 : 0.0022));
    scene.zoomPanel(hit.id, factor, hit.uv);
  },
  { passive: false }
);
addEventListener("keydown", (e) => {
  if (!overlay.classList.contains("hidden")) return;
  if (e.code === "Escape" && noteMode) {
    exitNoteMode();
    return;
  }
  if (e.code === "Escape" && state !== "idle") {
    interrupt();
    activeRec?.stop();
    setState("idle");
    showTranscript("");
    return;
  }
  if (e.code === "KeyC" && !(document.activeElement instanceof HTMLInputElement)) {
    setCamView(!camCheck.checked);
    return;
  }
  if (e.code !== "Space") return;
  if (document.activeElement instanceof HTMLInputElement) return;
  e.preventDefault();
  converse();
});

// debug hook: octa.agentEvents.onSpawn("x", "test") from the console
Object.assign(window, {
  octa: {
    scene, agentEvents, setState, presentArtifacts, salvageVisualFromText, visualsEl,
    jevLog, jevLogCsv, checkAddressee, converse, setTier,
    // for scripted runs: patch Recognizer.prototype / speaker to feed text in
    Recognizer, speaker, idleGlitch,
    previewParticipants: (v: ParticipantView | null) => (participantsPreview = v),
    get state() { return state; },
    get tier() { return settings.tier; },
    get sound() { return sound; },
  },
});

const participantsEl = document.querySelector<HTMLDivElement>("#participants")!;
let participantsAt = 0;
let breathAt = 0;
// idle interference: every 25-70 s the channel hiccups on its own
let idleGlitchAt = performance.now() + 25000 + Math.random() * 45000;

function idleGlitch() {
  const strength = 0.5 + Math.random() * 0.5;
  const ms = 140 + Math.random() * 200;
  scene.glitch(strength, ms);
  if (settings.voiceStyle === "vader") speaker.vader.staticBurst(strength);
  // now and then it stutters twice
  if (Math.random() < 0.3) {
    setTimeout(() => {
      scene.glitch(strength * 0.7, ms * 0.7);
      if (settings.voiceStyle === "vader") speaker.vader.staticBurst(strength * 0.6);
    }, ms + 90 + Math.random() * 160);
  }
}
// set from the console (octa.previewParticipants) to style the strip without a camera
let participantsPreview: ParticipantView | null = null;

function frame() {
  speakLevel *= 0.94;
  scene.level =
    state === "listening" || state === "noting"
      ? mic.level()
      : state === "speaking"
        ? speakLevel
        : 0;

  scene.render();
  drawCamView();
  // participant tiles: 10 updates a second is plenty for gaze and hands
  const now = performance.now();
  if (
    state === "idle" &&
    now > idleGlitchAt &&
    overlay.classList.contains("hidden") &&
    document.visibilityState === "visible" &&
    !speaker.speaking
  ) {
    idleGlitchAt = now + 25000 + Math.random() * 45000;
    idleGlitch();
  }
  // the respirator keeps going while Octa thinks
  if (state === "thinking" && settings.voiceStyle === "vader" && !speaker.speaking && now - breathAt > 4800) {
    breathAt = now;
    speaker.vader.breatheCycle();
  }
  if (now - participantsAt > 100) {
    participantsAt = now;
    renderParticipants(
      participantsEl,
      participantsPreview ?? (gestures && settings.handGestures ? gestures.snapshot() : null)
    );
  }
  const m = scene.getMotion();
  sound?.update(m.angularSpeed, m.level);

  requestAnimationFrame(frame);
}
frame();

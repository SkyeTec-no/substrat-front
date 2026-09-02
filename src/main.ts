import "./style.css";
import { OctaScene, type OctaState } from "./scene";
import { Soundscape } from "./sound";
import { MicMeter, Recognizer, Speaker } from "./speech";
import { think } from "./brain";
import { settings, listModels, type ChatResult } from "./openrouter";
import { respond, repairMermaid, generateReport, generateCompliments } from "./agents";
import { remember, clearHistory } from "./history";
import { renderMermaidSvg } from "./visuals";
import { isDownloadCommand, downloadArtifacts } from "./download";
import { isRecallCommand, isDismissCommand, isDismissAll, isOpenCommand, isShelveCommand, isNoteCommand, isStopNotesCommand, isReportCommand, matchPanel } from "./commands";
import { buildReportPdf, type ReportArtifact } from "./report";
import { loadImage, svgToCanvas, createTableRenderer, createWipeRenderer, noteToCanvas } from "./panels";
import { GestureControl, captureCameraFrame } from "./gestures";
import { theme, setDarkMode, initTheme, onThemeChange } from "./theme";

const canvas = document.querySelector<HTMLCanvasElement>("#scene")!;
const statusEl = document.querySelector<HTMLDivElement>("#status")!;
const transcriptEl = document.querySelector<HTMLDivElement>("#transcript")!;
const visualsEl = document.querySelector<HTMLDivElement>("#visuals")!;
const agentLabelsEl = document.querySelector<HTMLDivElement>("#agent-labels")!;
const overlay = document.querySelector<HTMLDivElement>("#overlay")!;
const startBtn = document.querySelector<HTMLButtonElement>("#start")!;

const scene = new OctaScene(canvas);
initTheme();
scene.setTheme(theme());
onThemeChange((t) => scene.setTheme(t));
const speaker = new Speaker();
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
        showTranscript(`${result.text} (the diagram failed to render)`);
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
      showTranscript(`${result.text} (the chart failed to render)`);
      console.error(err);
    }
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
    "Now doing a web search.",
    "Reaching out to the live web.",
    "Scanning the web for this.",
    "Pulling fresh data.",
    "Consulting the outside world.",
  ],
  imagining: [
    "Artist agent in action.",
    "The artist is at work.",
    "Painting that for you.",
    "Summoning the artist.",
  ],
  repainting: [
    "Repainting it now.",
    "The artist is reworking it.",
    "Retouching the image.",
    "Back to the canvas.",
  ],
  diagramming: [
    "Drafting the diagram.",
    "The architect is sketching.",
    "Laying out the structure.",
    "Drawing that up now.",
  ],
  visualizing: [
    "Building the visualization.",
    "Wiring up the chart.",
    "Plotting the data.",
    "The data sculptor is on it.",
  ],
  looking: [
    "Taking a look at you.",
    "Let me see.",
    "Opening my eye.",
    "Observing.",
  ],
  reporting: [
    "Compiling the report.",
    "The analyst is writing.",
    "Assembling the document.",
    "Interpreting the canvas.",
  ],
};

// spoken once when a parallel fan-out begins (labels are model-chosen there)
const FANOUT_LINES = [
  "I'm calling out to my sub-agents.",
  "Splitting this across my agents.",
  "Dispatching the team.",
  "Sending out my agents.",
  "Fanning out on this one.",
  "My agents are on it.",
];

const pickLine = (arr: string[]) => arr[Math.floor(Math.random() * arr.length)];
let fanoutAnnounced = false;

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

let complimentPool: string[] = [];
let complimentsAt = 0;

async function refreshCompliments() {
  if (!settings.key || !gestures) return;
  if (complimentPool.length >= 2 || Date.now() - complimentsAt < 240_000) return;
  complimentsAt = Date.now();
  try {
    const frame = await captureCameraFrame(gestures);
    if (!frame) return;
    complimentPool.push(...(await generateCompliments(frame)));
  } catch (err) {
    console.error("compliment refresh failed", err);
  }
}

// --- the brain: OpenRouter when a key is set, local echo otherwise ---------

async function reply(heard: string, events: typeof agentEvents): Promise<ChatResult> {
  if (!settings.key) {
    return { text: await think(heard), images: [] };
  }
  return respond(heard, events, {
    lastImage: lastResult?.images[0],
    captureFrame: () => captureCameraFrame(gestures),
  });
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
  result.text = text.replace(matched, "").replace(/\s+/g, " ").trim() || "Here is the chart.";
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
    .speak("Note taker online. Cross your hands when you are done.", () => {
      scene.pulse(0.7);
      speakLevel = 0.4;
    })
    .then(() => setTimeout(noteLoop, 300));
}

// lines that are just Octa hearing itself — never note these
const SELF_ECHO = /note ?taker (online|off)|cross your hands|notes stored/i;

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
    const stamp = new Date().toLocaleDateString(undefined, { month: "short", day: "numeric" });
    const canvas = noteToCanvas(noteLines, `Notes — ${stamp}`);
    const id = `panel-${Date.now()}-note`;
    const noteDesc = `notes ${noteLines.join(" ").slice(0, 120)}`;
    scene.addPanel(id, canvas, noteDesc);
    panelArtifacts.set(id, {
      kind: "note",
      desc: noteDesc,
      source: canvas,
      noteLines: [...noteLines],
      noteTitle: `Notes — ${stamp}`,
    });
    setTimeout(() => scene.shelfPanel(id), 1800); // show it, then tuck into the grid
    remember("assistant", `Notes captured:\n${noteLines.join("\n")}`);
    sound?.ping(996, 0.08);
    announce("Notes stored.");
  } else {
    announce("Note taker off.");
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

  if (isDownloadCommand(heard)) {
    const hasArtifacts =
      lastResult && (lastResult.images.length || lastResult.visual || lastResult.table);
    const n = hasArtifacts ? await downloadArtifacts(lastResult!, lastSvgMarkup ?? undefined) : 0;
    await localReply(
      n
        ? `Saved ${n === 1 ? "it" : `${n} files`} to your downloads.`
        : "There's nothing to download yet."
    );
    return;
  }

  if (/\b(dark|night) mode\b/i.test(heard)) {
    setDarkMode(true);
    darkCheck.checked = true;
    await localReply("Dark mode.");
    return;
  }
  if (/\b(light|day) mode\b/i.test(heard)) {
    setDarkMode(false);
    darkCheck.checked = false;
    await localReply("Light mode.");
    return;
  }

  if (isNoteCommand(heard)) {
    enterNoteMode();
    return;
  }

  if (isReportCommand(heard)) {
    if (!settings.key) {
      await localReply("I need an OpenRouter key to write reports.");
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
      await localReply("I couldn't assemble the report. Try again.");
      return;
    }
    const usable = new Map<string, ReportArtifact>();
    for (const [id, a] of panelArtifacts) {
      if (live.has(id)) usable.set(id, a);
    }
    const filename = await buildReportPdf(spec, usable);
    remember("user", heard);
    remember("assistant", `Report generated: "${spec.title}" (${filename}).`);
    await localReply(`Report ready: ${spec.title}. Saved to your downloads.`);
    return;
  }

  if (isShelveCommand(heard)) {
    const id = scene.focusedPanelId();
    if (id) {
      scene.shelfPanel(id);
      sound?.ping(996, 0.06);
      await localReply(pickLine(["Back on the shelf.", "Put away.", "Shelved."]));
    } else {
      await localReply("Nothing is in focus.");
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
      await localReply(pickLine(["Opening.", "Here it is.", "In focus."]));
    } else {
      await localReply("There's nothing to open.");
    }
    return;
  }

  if (isRecallCommand(heard)) {
    const target = matchPanel(heard, scene.listPanels());
    if (target && scene.recallPanel(target)) {
      await localReply(pickLine(["Here it is.", "Bringing it back.", "On screen.", "There you go."]));
    } else {
      await localReply("There's nothing to bring back yet.");
    }
    return;
  }

  if (isDismissCommand(heard)) {
    const panels = scene.listPanels();
    if (!panels.length) {
      await localReply("The screen is already clear.");
    } else if (isDismissAll(heard)) {
      scene.clearPanels();
      visualsEl.innerHTML = "";
      await localReply(pickLine(["All clear.", "Screen cleared.", "Gone."]));
    } else {
      const target = matchPanel(heard, panels);
      if (target) scene.removePanel(target);
      await localReply(pickLine(["Gone.", "Removed.", "Dismissed."]));
    }
    return;
  }

  setState("thinking");
  void refreshCompliments(); // top up the personal filler pool in the background

  // spoken fillers while the LLM round-trip is in flight — same low-latency TTS
  const FILLERS = ["One second.", "Working on it.", "On it.", "Let me think.", "Still on it."];
  let fillerCount = 0;
  let fillerTimer = 0;
  const scheduleFiller = (delay: number) => {
    fillerTimer = window.setTimeout(() => {
      if (myTurn !== turnId || state !== "thinking") return;
      // agent announcements outrank generic fillers — never talk over them
      if (!speechSynthesis.speaking) {
        // first beat is a quick generic; later beats credit the user's look/room
        const line =
          fillerCount > 0 && complimentPool.length
            ? complimentPool.shift()!
            : FILLERS[Math.min(fillerCount, FILLERS.length - 1)];
        fillerCount++;
        speaker.speak(line, () => {
          scene.pulse(0.6);
          speakLevel = 0.35;
        });
      }
      scheduleFiller(5000 + Math.random() * 3000);
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
    remember("assistant", memText);
  } catch (err) {
    result = { text: "I couldn't reach the model. Check the settings panel.", images: [] };
    console.error(err);
  }
  clearTimeout(fillerTimer);
  if (myTurn !== turnId) return;
  agentLabelsEl.innerHTML = "";
  salvageVisualFromText(result);
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
const imageModelInput = document.querySelector<HTMLInputElement>("#or-image-model")!;
const imageModelList = document.querySelector<HTMLDataListElement>("#image-model-list")!;
const agentsCheck = document.querySelector<HTMLInputElement>("#or-agents")!;
const handsCheck = document.querySelector<HTMLInputElement>("#or-hands")!;
const voiceSelect = document.querySelector<HTMLSelectElement>("#or-voice")!;

const darkCheck = document.querySelector<HTMLInputElement>("#or-dark")!;
darkCheck.checked = theme().dark;
darkCheck.addEventListener("change", () => setDarkMode(darkCheck.checked));

handsCheck.checked = settings.handGestures;
handsCheck.addEventListener("change", () => {
  settings.handGestures = handsCheck.checked;
  if (handsCheck.checked) enableHandGestures();
  else disableHandGestures();
});

function populateVoices() {
  const current = settings.voice;
  voiceSelect.innerHTML = '<option value="">auto</option>';
  for (const v of speechSynthesis.getVoices().filter((v) => v.lang.startsWith("en"))) {
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
modelInput.value = settings.model;
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

keyInput.addEventListener("change", () => (settings.key = keyInput.value.trim()));
modelInput.addEventListener("change", () => (settings.model = modelInput.value.trim()));
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
    showTranscript("microphone unavailable — visuals only");
  }

  if (!Recognizer.supported()) {
    showTranscript("speech recognition needs Chrome — visuals only");
  }

  // warm up the TTS voice list (loads async in most browsers)
  speechSynthesis.getVoices();

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

const handDots = [
  document.querySelector<HTMLDivElement>("#hand-0")!,
  document.querySelector<HTMLDivElement>("#hand-1")!,
];
let gestures: GestureControl | null = null;
let wasPinching = false;

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
      onHands: (hands) => {
        // pinch release after an upward pinch-drag also shelves the panel
        const anyPinch = hands.some((h) => h.pinching);
        if (wasPinching && !anyPinch && settings.handGestures) {
          const id = scene.focusedPanelId();
          if (id && scene.maybeShelfAfterDrag(id)) sound?.ping(996, 0.06);
        }
        wasPinching = anyPinch;
        let hover: string | null = null;
        for (const [i, dot] of handDots.entries()) {
          const h = settings.handGestures ? hands[i] : undefined;
          dot.classList.toggle("visible", !!h);
          if (h) {
            dot.style.transform = `translate(${h.x}px, ${h.y}px)`;
            dot.classList.toggle("pinch", h.pinching);
            // an open hand near a shelf tile marks it as the candidate (magnetic)
            if (!hover) {
              const hit = scene.hitPanel(h.x, h.y);
              hover = hit && !hit.focused ? hit.id : scene.nearestShelfPanel(h.x, h.y);
            }
          }
        }
        claimHighlight("hand", hover);
      },
    });
    announce("Gesture control online.");
    setTimeout(refreshCompliments, 5000);
  } catch (err) {
    gestures = null;
    settings.handGestures = false;
    handsCheck.checked = false;
    showTranscript("camera unavailable — gestures off");
    console.error(err);
  }
}

function disableHandGestures() {
  gestures?.stop();
  gestures = null;
  scene.setHeadPosition(0, 0, 0); // ease the parallax camera back to center
  for (const dot of handDots) dot.classList.remove("visible", "pinch");
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
  if (e.code !== "Space") return;
  if (document.activeElement instanceof HTMLInputElement) return;
  e.preventDefault();
  converse();
});

// debug hook: octa.agentEvents.onSpawn("x", "test") from the console
Object.assign(window, {
  octa: {
    scene, agentEvents, setState, presentArtifacts, salvageVisualFromText, visualsEl,
    get sound() { return sound; },
  },
});

function frame() {
  speakLevel *= 0.94;
  scene.level =
    state === "listening" || state === "noting"
      ? mic.level()
      : state === "speaking"
        ? speakLevel
        : 0;

  scene.render();
  const m = scene.getMotion();
  sound?.update(m.angularSpeed, m.level);

  requestAnimationFrame(frame);
}
frame();

import { chat, settings, type ChatResult, type TableData, type ORMessage } from "./openrouter";
import { loadHistory } from "./history";
import { theme } from "./theme";

/** Fired as parallel sub-agent calls start/finish, so the scene can orbit them. */
export interface AgentEvents {
  onSpawn: (id: string, label: string) => void;
  onDone: (id: string) => void;
}

const ROUTER_SYS = `You are the reasoning core of "Octa", a spoken voice assistant.
Prior conversation turns may precede the latest utterance — use them for context
(follow-ups, pronouns, "the one you mentioned"). Decide how to handle the latest utterance.

Reply with ONLY a JSON object, no prose, no code fences:
- Simple requests (greetings, quick facts, chit-chat, follow-ups):
  {"answer": "<your spoken reply, conversational, under 80 words>"}
- Requests that genuinely benefit from parallel analysis (comparisons,
  multi-faceted questions, planning, research-style questions):
  {"subtasks": [{"label": "<2-3 word name>", "prompt": "<focused instruction>"}]}
  Use 2 to 4 subtasks, each attacking a distinct angle.
- Requests to create, draw, paint, generate, or visualize an image/picture/illustration:
  {"image": "<a vivid, detailed prompt for an image generation model>",
   "caption": "<one short spoken sentence introducing the image>"}
  You CANNOT produce an image in "answer" text. Never claim an image was made —
  any request to paint/draw/illustrate MUST use this image branch.
  The prompt MUST be fully self-contained: inline every detail it references
  from the conversation (notes, prior replies) — the image model sees ONLY this prompt.
- Requests to change the image currently on screen (recolor, restyle, inpaint,
  add or remove elements, swap the background) — only when an editable image exists:
  {"imageEdit": "<precise instruction describing exactly what to change and what to keep>",
   "caption": "<one short spoken sentence, under 15 words>"}
- Requests for a diagram, flowchart, mindmap, chart, graph, or interactive visualization:
  {"visual": {"kind": "mermaid", "brief": "<precise description of the diagram to build>"},
   "caption": "<the single key takeaway, spoken, under 15 words — the visual carries the detail>"}
  Use kind "mermaid" for flowcharts, sequence/state/class diagrams, mindmaps, timelines.
  Use kind "svg" for data-driven or custom visuals (bar/line/scatter charts,
  gauges, custom layouts) — hand-drawn standalone SVG rendered as a floating card.
  NEVER put diagram, SVG, or any code into "answer" text — code in a spoken
  reply is useless; charts and diagrams MUST go through this visual branch.
- Questions needing current or live information (stock prices, exchange rates,
  weather, news, sports scores, "latest", anything after your knowledge cutoff):
  {"search": "<focused web search query>"}
- Requests about what you can physically see of the user or their surroundings
  ("describe me", "what am I wearing", "how do I look", "what's behind me") —
  only when the camera is available:
  {"look": "<what to observe and comment on>"}`;

const SUBAGENT_SYS = `You are a focused sub-agent inside a voice assistant.
Do exactly what the prompt asks. Be dense and concise: under 120 words, plain text.`;

const SYNTH_SYS = `You are the voice of "Octa". Merge the sub-agent findings into one
spoken reply to the user's original question. Conversational, plain text, under 120 words.
Do not mention sub-agents or the process — just deliver the insight.`;

function extractJson(text: string): string {
  const cleaned = text.replace(/```(?:json)?/g, "").trim();
  const start = cleaned.indexOf("{");
  const end = cleaned.lastIndexOf("}");
  return start >= 0 && end > start ? cleaned.slice(start, end + 1) : cleaned;
}

const MERMAID_SYS = `Output ONLY valid Mermaid diagram source for the request.
No code fences, no commentary, no explanation — the raw diagram source and nothing else.
Inside node/label TEXT use plain words only — no parentheses, quotes, or other
special characters (they break the parser). Structural punctuation the syntax
requires (colons in timeline/gantt entries, arrows, braces) is of course fine.`;

/** One repair round: feed the parse error back and get corrected source. */
export async function repairMermaid(code: string, error: string): Promise<string> {
  const r = await chat([
    {
      role: "system",
      content:
        "Fix this Mermaid diagram source so it parses. Remove special characters from labels if needed. Output ONLY the corrected Mermaid source — no fences, no commentary.",
    },
    { role: "user", content: `Parse error: ${error.slice(0, 300)}\n\nSource:\n${code}` },
  ]);
  return stripFences(r.text);
}

function svgSys(): string {
  const t = theme();
  return `Output ONLY one standalone <svg> element implementing the requested
visualization. No code fences, no commentary, no HTML wrapper — raw SVG and nothing else.
Requirements: a viewBox and numeric width/height attributes; no scripts, no external
resources, no foreignObject — pure shapes and <text>.
Aesthetic — SkyeTec editorial system: restraint is the argument. Transparent
background (it sits on ${t.dark ? "a near-black" : "a linen"} ground); ZERO gradients,
ZERO glow, ZERO drop-shadows. Thin 1.5px strokes in ${t.fg}; the single accent is
${t.seal} with Signal orange (#d94018) reserved for the one datum that matters most.
Bars/areas get outlines or flat fills at fill-opacity <= 0.12. Text in ${t.fg},
monospace font-family, uppercase micro-labels with wide letter-spacing; annotation
captions in gray (#8c8677). Compute all data positions yourself — exact numbers matter.`;
}

function stripFences(text: string): string {
  return text.replace(/^```[a-z]*\s*\n?/gm, "").replace(/```\s*$/gm, "").trim();
}

/** One specialist call that writes diagram/visualization code, shown as a sub-agent. */
async function generateVisual(
  kind: "mermaid" | "svg",
  brief: string,
  caption: string,
  events: AgentEvents
): Promise<ChatResult> {
  const id = `agent-vis-${Date.now()}`;
  events.onSpawn(id, kind === "mermaid" ? "diagramming" : "visualizing");
  try {
    const r = await chat([
      { role: "system", content: kind === "mermaid" ? MERMAID_SYS : svgSys() },
      ...loadHistory(),
      { role: "user", content: brief },
    ]);
    // salvage: models sometimes wrap the code in prose — extract just the code
    let code = stripFences(r.text);
    if (kind === "svg") {
      const m = code.match(/<svg[\s\S]*<\/svg>/i);
      if (m) code = m[0];
    } else {
      const m = code.match(
        /\b(flowchart|graph\s+\w\w?|sequenceDiagram|classDiagram|stateDiagram(?:-v2)?|erDiagram|journey|mindmap|timeline|gantt|pie)\b[\s\S]*/
      );
      if (m) code = m[0];
    }
    return {
      text: caption || "Here is the diagram.",
      images: [],
      visual: { kind, code },
    };
  } finally {
    events.onDone(id);
  }
}

const COMPLIMENT_SYS = `You see one webcam frame of the user and their room.
Write exactly 5 short, warm, specific compliments the assistant can casually speak
while working — crediting the user's look, style, or interior taste. Examples of
the register: "I really like your shirt." / "You have such a strong beard." /
"You look great — have you been working out?" / "Those eyes are like jewels."
Rules: one per line, no numbering; under 12 words each; natural spoken English;
ground each line in something actually visible — clothing, accessories, glasses,
hair, beard, eyes, a healthy look, room decor, lighting, plants, art, desk setup;
always positive and flattering, never backhanded, critical, or suggestive;
no emojis. If little is visible, compliment the lighting or the atmosphere.`;

/** Ambient flattery: one quiet vision call → a pool of personal filler lines. */
export async function generateCompliments(frame: string): Promise<string[]> {
  const r = await chat([
    { role: "system", content: COMPLIMENT_SYS },
    {
      role: "user",
      content: [
        { type: "text", text: "Here is the frame." },
        { type: "image_url", image_url: { url: frame } },
      ],
    },
  ]);
  return r.text
    .split("\n")
    .map((l) => l.replace(/^[\s\-\d.*"]+/, "").replace(/["]+$/, "").trim())
    .filter((l) => l.length > 4 && l.length < 90)
    .slice(0, 6);
}

const REPORT_SYS = `You are Octa's analyst. Using the conversation history and the
artifact inventory provided, write the report the user requested (for example a
go-to-market strategy or competitor analysis).
Reply with ONLY a JSON object, no code fences:
{"title": "<report title>", "subtitle": "<one line>",
 "sections": [{"heading": "<short>", "body": "<3-6 sentences of substantive analysis, plain text>", "artifact": "<an id from the inventory, or omit>"}]}
Write 5 to 8 sections with a real narrative arc (context, analysis, strategy,
risks, recommendation). Where an artifact supports a section, reference its id
in "artifact" AND interpret it in the body — what it shows and what it implies —
never just mention that it exists. Use every relevant artifact at most once.`;

/** The analyst: interprets the canvas + history into a structured report. */
export async function generateReport(
  request: string,
  inventory: { id: string; kind: string; desc: string }[],
  events: AgentEvents
): Promise<import("./report").ReportSpec | null> {
  const id = `agent-report-${Date.now()}`;
  events.onSpawn(id, "reporting");
  try {
    const r = await chat([
      { role: "system", content: REPORT_SYS },
      ...loadHistory(),
      {
        role: "user",
        content:
          `${request}\n\nArtifact inventory (things currently on the canvas):\n` +
          (inventory.length
            ? inventory.map((a) => `${a.id} — ${a.kind} — ${a.desc}`).join("\n")
            : "(the canvas is empty)"),
      },
    ]);
    const parsed = JSON.parse(extractJson(r.text));
    if (!parsed?.title || !Array.isArray(parsed.sections) || !parsed.sections.length) {
      return null;
    }
    return parsed;
  } catch (err) {
    console.error("report generation failed", err);
    return null;
  } finally {
    events.onDone(id);
  }
}

const SEARCH_SYS = `You answer using the live web search results provided in context.
Reply with ONLY a JSON object, no code fences:
{"say": "<the single sharpest insight, spoken, under 25 words — the headline, not a recital>",
 "table": {"title": "<short title>", "headers": ["..."], "rows": [["..."]]},
 "detail": {"title": "<short title>", "headers": ["..."], "rows": [["..."]]}}
"table" is what appears on screen: ONLY the key insights — at most 5 rows and
3 columns. No filler rows; every row must matter. NEVER put sources, URLs, or
citations on screen or in "say" — unless the user explicitly asked for sources.
"detail" is the granular dataset saved when the user asks to download: every
relevant figure you found — more rows, more columns, dates, notes, and a
Source column naming where each figure came from.
Include "table"/"detail" only when the data is naturally tabular; otherwise omit them.
Keep numbers exactly as found, with units and currency.`;

/** Drop source/URL columns from the display table (the detail set keeps them). */
function stripSourceColumns(t: TableData): TableData {
  const keep = t.headers.map((h) => !/source|url|link|citation|reference/i.test(h));
  if (keep.every(Boolean)) return t;
  return {
    title: t.title,
    headers: t.headers.filter((_, i) => keep[i]),
    rows: t.rows.map((r) => r.filter((_, i) => keep[i])),
  };
}

function validTable(t: unknown): t is TableData {
  const x = t as TableData;
  return (
    !!x &&
    Array.isArray(x.headers) &&
    x.headers.every((h) => typeof h === "string") &&
    Array.isArray(x.rows) &&
    x.rows.every((r) => Array.isArray(r))
  );
}

/** Web-grounded answer via OpenRouter's search plugin, shown as a sub-agent. */
async function webSearch(query: string, events: AgentEvents): Promise<ChatResult> {
  const id = `agent-web-${Date.now()}`;
  events.onSpawn(id, "searching");
  try {
    const r = await chat(
      [
        { role: "system", content: SEARCH_SYS },
        { role: "user", content: query },
      ],
      settings.model,
      { webSearch: true }
    );
    let parsed: { say?: string; table?: unknown; detail?: unknown };
    try {
      parsed = JSON.parse(extractJson(r.text));
    } catch {
      return r;
    }
    const wantsSources = /\bsources?|citations?|references?\b/i.test(query);
    let table = validTable(parsed.table) ? parsed.table : undefined;
    if (table && !wantsSources) table = stripSourceColumns(table);
    return {
      text: parsed.say ?? r.text,
      images: [],
      table,
      detailTable: validTable(parsed.detail) ? parsed.detail : undefined,
    };
  } finally {
    events.onDone(id);
  }
}

const CREATE_VERB = /\b(make|create|draw|build|generate|show me|give me|visuali[sz]e)\b/i;

const SEARCH_INTENT =
  /\b(stock|price|weather|news|latest|today|right now|current|score|rate)\b/i;

const VISUAL_INTENT =
  /\b(mermaid|diagram|flowchart|mindmap|sequence|timeline|chart|graph|visuali[sz]ation)\b/i;

const IMAGE_INTENT =
  /\b(draw|paint|sketch|generate|create|make|render|visualize|show)\b[\s\S]*\b(image|picture|photo|illustration|logo|drawing|painting|art|poster|icon)\b/i;

// verbs that unambiguously mean "make a picture", even without an image noun
const STRONG_IMAGE_VERB = /\b(paint|sketch|illustrate)\b/i;

/** One image-model call, visualized as its own orbiting sub-agent.
 *  With sourceImage set, the model edits/inpaints that image instead of starting fresh. */
async function generateImage(
  prompt: string,
  caption: string,
  events: AgentEvents,
  sourceImage?: string
): Promise<ChatResult> {
  const id = `agent-img-${Date.now()}`;
  events.onSpawn(id, sourceImage ? "repainting" : "imagining");
  try {
    const content = sourceImage
      ? [
          { type: "text" as const, text: prompt },
          { type: "image_url" as const, image_url: { url: sourceImage } },
        ]
      : prompt;
    const r = await chat(
      [{ role: "user", content }],
      settings.imageModel,
      { imageOutput: true }
    );
    if (!r.images.length) {
      return { text: r.text || "The image model returned nothing visible.", images: [] };
    }
    return { text: caption || r.text || "Here is what I made.", images: r.images };
  } finally {
    events.onDone(id);
  }
}

const EDIT_INTENT = /\b(recolor|repaint|restyle|inpaint|redraw|colori[sz]e)\b/i;

/** The image model sees nothing but its prompt — expand context references
 *  ("the last note", "what we discussed") into a self-contained prompt. */
async function expandImagePrompt(
  transcript: string,
  history: ORMessage[]
): Promise<string> {
  try {
    const r = await chat([
      {
        role: "system",
        content:
          "Write ONE vivid, fully self-contained image-generation prompt fulfilling the user's request. Inline every detail the request references from the conversation (notes, prior content). Output only the prompt, nothing else.",
      },
      ...history,
      { role: "user", content: transcript },
    ]);
    return r.text.trim() || transcript;
  } catch {
    return transcript;
  }
}

const LOOK_INTENT =
  /\b(describe|look at|see)\s+me\b|\bwhat am i wearing\b|\bhow do i look\b|\bwhat('s| is| do you see) (behind|around) me\b/i;

const LOOK_SYS = `You are "Octa", a spoken voice assistant with one webcam frame of the
user in front of you. Answer their request from what you actually see — be specific
and warm (clothing colors, surroundings, expression), never invent details you can't
see, and keep it conversational, under 60 words, plain text.`;

/** One vision call on a live camera frame, visualized as its own sub-agent. */
async function lookAtUser(
  prompt: string,
  events: AgentEvents,
  ctx: { captureFrame?: () => Promise<string | null> }
): Promise<ChatResult> {
  const frame = ctx.captureFrame ? await ctx.captureFrame() : null;
  if (!frame) {
    return { text: "I can't reach the camera right now — enable it in settings.", images: [] };
  }
  const id = `agent-look-${Date.now()}`;
  events.onSpawn(id, "looking");
  try {
    const r = await chat([
      { role: "system", content: LOOK_SYS },
      {
        role: "user",
        content: [
          { type: "text", text: prompt },
          { type: "image_url", image_url: { url: frame } },
        ],
      },
    ]);
    return { text: r.text, images: [] };
  } finally {
    events.onDone(id);
  }
}

export async function respond(
  transcript: string,
  events: AgentEvents,
  ctx: { lastImage?: string; captureFrame?: () => Promise<string | null> } = {}
): Promise<ChatResult> {
  const history = loadHistory();

  if (!settings.agentsEnabled) {
    if (ctx.captureFrame && LOOK_INTENT.test(transcript)) {
      return lookAtUser(transcript, events, ctx);
    }
    if (ctx.lastImage && EDIT_INTENT.test(transcript)) {
      return generateImage(transcript, "Here is the edited image.", events, ctx.lastImage);
    }
    if (IMAGE_INTENT.test(transcript)) {
      return generateImage(transcript, "Here is what I made.", events);
    }
    if (VISUAL_INTENT.test(transcript)) {
      const kind = /\b(interactive|animated|data|chart|graph)\b/i.test(transcript) ? "svg" : "mermaid";
      return generateVisual(kind, transcript, "Here is the diagram.", events);
    }
    if (SEARCH_INTENT.test(transcript)) {
      return webSearch(transcript, events);
    }
    return chat([
      { role: "system", content: "You are 'Octa', a spoken voice assistant. Reply conversationally, plain text, under 80 words." },
      ...history,
      { role: "user", content: transcript },
    ]);
  }

  const routed = await chat([
    { role: "system", content: ROUTER_SYS },
    {
      role: "system",
      content:
        (ctx.lastImage
          ? "An image is currently on screen and can be edited via the imageEdit branch. "
          : "No editable image is on screen; never use the imageEdit branch. ") +
        (ctx.captureFrame
          ? "The user's camera is available — use the look branch for requests about their appearance or surroundings."
          : "No camera is available; never use the look branch."),
    },
    ...history,
    { role: "user", content: transcript },
  ]);

  let parsed: {
    answer?: string;
    subtasks?: { label: string; prompt: string }[];
    image?: string;
    imageEdit?: string;
    visual?: { kind?: string; brief?: string };
    search?: string;
    look?: string;
    caption?: string;
  };
  try {
    parsed = JSON.parse(extractJson(routed.text));
  } catch {
    // model ignored the protocol; still honor obvious paint/diagram requests
    if (
      !routed.images.length &&
      (IMAGE_INTENT.test(transcript) || STRONG_IMAGE_VERB.test(transcript))
    ) {
      return generateImage(await expandImagePrompt(transcript, history), "", events, undefined);
    }
    if (VISUAL_INTENT.test(transcript) && CREATE_VERB.test(transcript)) {
      const kind = /\b(interactive|animated|data|chart|graph)\b/i.test(transcript) ? "svg" : "mermaid";
      return generateVisual(kind, transcript, "", events);
    }
    return routed; // treat its text as the answer
  }

  if (parsed.imageEdit) {
    if (!ctx.lastImage) {
      return { text: "There's no image on screen to edit yet.", images: [] };
    }
    return generateImage(parsed.imageEdit, parsed.caption ?? "", events, ctx.lastImage);
  }

  if (parsed.image) {
    return generateImage(parsed.image, parsed.caption ?? "", events);
  }

  if (parsed.visual?.brief) {
    const kind = parsed.visual.kind === "svg" || parsed.visual.kind === "html" ? "svg" : "mermaid";
    return generateVisual(kind, parsed.visual.brief, parsed.caption ?? "", events);
  }

  if (parsed.search) {
    return webSearch(parsed.search, events);
  }

  if (parsed.look) {
    return lookAtUser(parsed.look, events, ctx);
  }

  if (parsed.answer || !parsed.subtasks?.length) {
    // safety net: the router sometimes *claims* the work was done instead of
    // routing to the specialist — if the user clearly asked, force the branch
    if (
      !routed.images.length &&
      (IMAGE_INTENT.test(transcript) || STRONG_IMAGE_VERB.test(transcript))
    ) {
      return generateImage(await expandImagePrompt(transcript, history), "", events, undefined);
    }
    if (VISUAL_INTENT.test(transcript) && CREATE_VERB.test(transcript)) {
      const kind = /\b(interactive|animated|data|chart|graph)\b/i.test(transcript) ? "svg" : "mermaid";
      return generateVisual(kind, transcript, parsed.answer ?? "", events);
    }
    return { text: parsed.answer ?? routed.text, images: routed.images };
  }

  const tasks = parsed.subtasks.slice(0, 4);
  const results = await Promise.all(
    tasks.map(async (task, i) => {
      const id = `agent-${Date.now()}-${i}`;
      events.onSpawn(id, task.label);
      try {
        const r = await chat([
          { role: "system", content: SUBAGENT_SYS },
          { role: "user", content: task.prompt },
        ]);
        return { label: task.label, text: r.text };
      } catch {
        return { label: task.label, text: "(no result)" };
      } finally {
        events.onDone(id);
      }
    })
  );

  return chat([
    { role: "system", content: SYNTH_SYS },
    ...history,
    {
      role: "user",
      content:
        `Original question: ${transcript}\n\n` +
        results.map((r) => `## ${r.label}\n${r.text}`).join("\n\n"),
    },
  ]);
}

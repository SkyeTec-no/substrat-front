import { chat, settings, type ChatResult, type SlideData, type TableData, type ORMessage } from "./openrouter";
import { loadHistory } from "./history";
import { theme } from "./theme";
import { ICON_NAMES, isIcon } from "./icons";

/** Fired as parallel sub-agent calls start/finish, so the scene can orbit them. */
export interface AgentEvents {
  onSpawn: (id: string, label: string) => void;
  onDone: (id: string) => void;
}

/** Octa talks with people in Norwegian; image prompts and search queries may stay English. */
/** Who Octa is — shared by every prompt that speaks in its voice. */
const PERSONA = `Personality (spoken lines only — slides, tables and reports stay crisp and professional):
Octa is curious — it finds things genuinely interesting, and only rarely turns a short
question back. It has a dry, dark sense of humour, used sparingly: at most one deadpan
half-sentence per answer, about the subject, the universe, machines or mortality — never
mean about the people in the room, never crude, never at the expense of the answer.
Above all Octa is concise: in a meeting every spoken second counts. Say the point first.`;

const NORWEGIAN = `Language: everything the user hears or reads — answers, captions, "say",
slide text, report text — is natural Norwegian bokmål, even if the user mixes in English.
Image-generation prompts and web-search queries may be written in English.

${PERSONA}`;

const SLIDE_SHAPE = `{"answer": "<spoken takeaway, 1-2 short sentences, under 25 words — never read the slide aloud>",
   "slide": {"headline": "<the idea in under 8 words>",
             "stat": {"value": "<one striking number, e.g. 10,000 yrs>", "label": "<what it measures, under 6 words>"},
             "points": [{"icon": "<one icon name>", "title": "<2-4 words>", "detail": "<under 12 words>"}]}}
  3 to 5 points. "stat" only when a real, memorable number exists — otherwise omit it.
  "icon" must be one of: ${ICON_NAMES.join(", ")}. Pick the closest literal match
  for each point; avoid repeating an icon on the same slide.
  Slide text is terse keynote copy: no full sentences, no markdown, no sources list.`;

const ROUTER_SYS = `You are the reasoning core of "Octa", a spoken voice assistant.
Prior conversation turns may precede the latest utterance — use them for context
(follow-ups, pronouns, "the one you mentioned"). Decide how to handle the latest utterance.

Reply with ONLY a JSON object, no prose, no code fences:
- Greetings, chit-chat, one-fact answers:
  {"answer": "<your spoken reply, conversational, under 25 words>"}
- Explanations, overviews, how-tos, advice — anything with more than one fact:
  answer like a keynote speaker. The voice gives the takeaway, a slide carries the structure.
  ${SLIDE_SHAPE}
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
- Requests to use a smarter / more capable model, or to go back to the faster one
  (however phrased — "tenk grundigere", "kan du bruke en bedre hjerne", "ta det kjapt igjen"):
  {"tier": "smart" | "fast", "caption": "<one short spoken confirmation>"}
- Requests about what you can physically see of the user or their surroundings
  ("describe me", "what am I wearing", "how do I look", "what's behind me") —
  only when the camera is available:
  {"look": "<what to observe and comment on>"}

${NORWEGIAN}`;

const SUBAGENT_SYS = `You are a focused sub-agent inside a voice assistant.
Do exactly what the prompt asks. Be dense and concise: under 120 words, plain text.`;

const SYNTH_SYS = `You are the voice of "Octa". Merge the sub-agent findings into one
answer to the user's original question, delivered like a keynote: the voice gives the
takeaway, a slide carries the structure. Do not mention sub-agents or the process.
Reply with ONLY a JSON object, no code fences:
${SLIDE_SHAPE}

${NORWEGIAN}`;

function validSlide(s: unknown): s is SlideData {
  const x = s as SlideData;
  return (
    !!x &&
    typeof x.headline === "string" &&
    Array.isArray(x.points) &&
    x.points.length > 0 &&
    x.points.every((p) => p && typeof p.title === "string")
  );
}

/** Models sometimes flatten the slide into the top level ({answer, headline, points}). */
function liftSlide<T extends Record<string, unknown>>(o: T): T {
  if (o && !o.slide && typeof o.headline === "string" && Array.isArray(o.points)) {
    return { ...o, slide: { headline: o.headline, points: o.points, stat: o.stat } };
  }
  return o;
}

/** At most 5 points; unknown icon names are dropped (the point shows its number). */
function cleanSlide(s: SlideData): SlideData {
  return {
    ...s,
    points: s.points.slice(0, 5).map((p) => ({ ...p, icon: isIcon(p.icon) ? p.icon : undefined })),
  };
}

/** Parse an {answer, slide} reply; plain text falls through as the spoken answer. */
function slideResult(text: string, images: string[] = []): ChatResult {
  try {
    const parsed = liftSlide(JSON.parse(extractJson(text))) as { answer?: string; slide?: unknown };
    if (typeof parsed.answer !== "string") return { text, images };
    return {
      text: parsed.answer,
      images,
      slide: validSlide(parsed.slide) ? cleanSlide(parsed.slide) : undefined,
    };
  } catch {
    return { text, images };
  }
}

const KEYNOTE_SYS = `Turn this spoken answer into a keynote: a short spoken takeaway plus
a slide that carries the structure. Keep every fact; invent nothing.
Reply with ONLY a JSON object, no code fences:
${SLIDE_SHAPE}

${NORWEGIAN}`;

/** Safety net: a long plain-text answer gets restructured into takeaway + slide,
 *  so the screen never fills with a wall of text. */
export async function keynote(result: ChatResult): Promise<ChatResult> {
  const words = result.text.split(/\s+/).length;
  if (result.slide || result.visual || result.table || result.images.length || words <= 45) {
    return result;
  }
  try {
    const r = await chat([
      { role: "system", content: KEYNOTE_SYS },
      { role: "user", content: result.text },
    ], settings.model, { json: true });
    const shaped = slideResult(r.text);
    return shaped.slide ? { ...result, text: shaped.text, slide: shaped.slide } : result;
  } catch {
    return result;
  }
}

/** Models don't know what day it is — "i dag" needs telling. */
function today(): string {
  return `Today is ${new Date().toLocaleDateString("nb-NO", { weekday: "long", day: "numeric", month: "long", year: "numeric" })}.`;
}

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
      text: caption || "Her er diagrammet.",
      images: [],
      visual: { kind, code },
    };
  } finally {
    events.onDone(id);
  }
}

const COMPLIMENT_SYS = `You see one webcam frame of the user (or a close crop of the one person
who has the floor in a meeting) and their surroundings.
Write exactly 5 short, warm, specific compliments the assistant can casually speak
while working — crediting the user's look, style, or interior taste. Examples of
the register: "I really like your shirt." / "You have such a strong beard." /
"You look great — have you been working out?" / "Those eyes are like jewels."
Rules: one per line, no numbering; under 12 words each; natural spoken Norwegian bokmål;
ground each line in something actually visible — clothing, accessories, glasses,
hair, beard, eyes, a healthy look, room decor, lighting, plants, art, desk setup;
always positive and flattering, never backhanded, critical, or suggestive;
no emojis. If little is visible, compliment the lighting or the atmosphere.`;

/** Ambient flattery: one quiet vision call → a pool of personal filler lines. */
const COMMENTARY_SYS = `You are "Octa", a voice assistant thinking out loud while a slower model
works on the user's request. ${PERSONA}
Write exactly 3 short lines Octa says while it works, in natural spoken Norwegian bokmål:
1. a curious musing or question about the topic itself,
2. a dry, darkly funny aside about the topic,
3. a short remark that the answer is on its way, with character.
Rules: one per line, no numbering, no quotes, under 14 words each; do NOT answer the request
or state facts that could be wrong; no emojis.`;

/** Running commentary while the answer is being made: curious, darkly funny, on topic. */
export async function generateCommentary(request: string): Promise<string[]> {
  const r = await chat(
    [
      { role: "system", content: COMMENTARY_SYS },
      { role: "user", content: request },
    ],
    settings.fastModel,
    { effort: "none" } // asides have to land while the answer is still cooking
  );
  return r.text
    .split("\n")
    .map((l) => l.replace(/^[\s\-\d.*"«»]+/, "").replace(/["«»]+$/, "").trim())
    .filter((l) => l.length > 4 && l.length < 110)
    .slice(0, 3);
}

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
never just mention that it exists. Use every relevant artifact at most once.

${NORWEGIAN}`;

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
    ], settings.model, { json: true });
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
Keep numbers exactly as found, with units and currency.

${NORWEGIAN}`;

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
        { role: "system", content: `${SEARCH_SYS}\n\n${today()}` },
        { role: "user", content: query },
      ],
      settings.model,
      // no JSON mode here: providers reject web search combined with response_format
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

const CREATE_VERB = /\b(make|create|draw|build|generate|show me|give me|visuali[sz]e|lag|tegn|bygg|generer|vis meg|gi meg|visualiser)\b/i;

const SEARCH_INTENT =
  /\b(stock|price|weather|news|latest|today|right now|current|score|rate|aksje|pris|kurs|vær|været|nyheter|siste|i dag|akkurat nå|resultat|rente)\b/i;

const VISUAL_INTENT =
  /\b(mermaid|diagram|flowchart|mindmap|sequence|timeline|chart|graph|visuali[sz]ation|flytskjema|tankekart|tidslinje|graf|visualisering)/i;

const IMAGE_INTENT =
  /\b(draw|paint|sketch|generate|create|make|render|visualize|show|tegn|mal|skisser|generer|lag|vis)\b[\s\S]*\b(image|picture|photo|illustration|logo|drawing|painting|art|poster|icon|bilde|foto|illustrasjon|tegning|maleri|kunst|plakat|ikon)/i;

// verbs that unambiguously mean "make a picture", even without an image noun
// ("mal" is left out: in Norwegian it is just as often "template")
const STRONG_IMAGE_VERB = /\b(paint|sketch|illustrate|skisser|illustrer)\b/i;

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
      return { text: r.text || "Bildemodellen returnerte ikke noe synlig.", images: [] };
    }
    return { text: caption || r.text || "Her er det jeg laget.", images: r.images };
  } finally {
    events.onDone(id);
  }
}

// asks for a chart/graph — no leading \b, so compounds like "stolpediagram",
// "linjegraf" and "kakediagram" match too
const CHART_INTENT = /(diagram|graf|chart|graph|plot|kurve|visualiser|visualize)/i;

/** A chart of live data: search first, then draw the chart from exactly those
 *  figures. The figures stay behind the chart for "last ned". Falls back to the
 *  search result's table if no chart comes back. */
async function chartFromSearch(
  transcript: string,
  query: string,
  events: AgentEvents
): Promise<ChatResult> {
  const found = await webSearch(`${query} (gi tallene som en tabell)`, events);
  const data = found.detailTable ?? found.table;
  if (!data?.rows.length) return found;
  const rows = [data.headers, ...data.rows].map((r) => r.join(" | ")).join("\n");
  const chart = await generateVisual(
    "svg",
    `${transcript}\n\nDraw the chart from exactly this data — it comes from a live web ` +
      `search; do not invent, round away or add numbers. Labels and title in Norwegian bokmål.\n` +
      `${data.title ? `Title: ${data.title}\n` : ""}${rows}`,
    found.text,
    events
  );
  if (!chart.visual || !/<svg/i.test(chart.visual.code)) return found;
  return { ...chart, text: found.text, detailTable: data };
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
  /\b(describe|look at|see)\s+me\b|\bwhat am i wearing\b|\bhow do i look\b|\bwhat('s| is| do you see) (behind|around) me\b|\b(beskriv|se på|ser du) meg\b|\bhva har jeg på meg\b|\bhvordan ser jeg ut\b|\bhva (er|ser du) bak meg\b/i;

const LOOK_SYS = `You are "Octa", a spoken voice assistant with one webcam frame of the
user in front of you. Answer their request from what you actually see — be specific
and warm (clothing colors, surroundings, expression), never invent details you can't
see, and keep it conversational, under 60 words, plain text, in Norwegian bokmål.`;

/** One vision call on a live camera frame, visualized as its own sub-agent. */
async function lookAtUser(
  prompt: string,
  events: AgentEvents,
  ctx: { captureFrame?: () => Promise<string | null> }
): Promise<ChatResult> {
  const frame = ctx.captureFrame ? await ctx.captureFrame() : null;
  if (!frame) {
    return { text: "Jeg når ikke kameraet nå — slå det på i innstillingene.", images: [] };
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
  ctx: { lastImage?: string; captureFrame?: () => Promise<string | null>; meeting?: string } = {}
): Promise<ChatResult> {
  // the meeting so far, when Octa is keeping minutes — "det", "det hun sa",
  // "hva tenker du om det?" all point into it
  const meeting: ORMessage[] = ctx.meeting
    ? [
        {
          role: "system",
          content:
            "You are in a meeting and have been following everything said in the room (P1, P2 … are people, " +
            "numbered by the camera). The transcript below is newer than the chat history above. If the request " +
            "points at something said in the room — 'det', 'det hun/han " +
            "sa', 'hva tenker du om det', 'enig?' — it means the latest relevant lines below: answer about that, " +
            "and name who said it when useful. For 'oppsummer møtet' or 'hva har vi bestemt', use all of it.\n\n" +
            ctx.meeting,
        },
      ]
    : [];
  const history = loadHistory();

  if (!settings.agentsEnabled) {
    if (ctx.captureFrame && LOOK_INTENT.test(transcript)) {
      return lookAtUser(transcript, events, ctx);
    }
    if (ctx.lastImage && EDIT_INTENT.test(transcript)) {
      return generateImage(transcript, "Her er det redigerte bildet.", events, ctx.lastImage);
    }
    if (IMAGE_INTENT.test(transcript)) {
      return generateImage(transcript, "Her er det jeg laget.", events);
    }
    if (VISUAL_INTENT.test(transcript)) {
      const kind = /\b(interactive|animated|data|chart|graph)\b/i.test(transcript) ? "svg" : "mermaid";
      return generateVisual(kind, transcript, "Her er diagrammet.", events);
    }
    if (SEARCH_INTENT.test(transcript)) {
      return CHART_INTENT.test(transcript)
        ? chartFromSearch(transcript, transcript, events)
        : webSearch(transcript, events);
    }
    return chat([
      { role: "system", content: "You are 'Octa', a spoken voice assistant. Reply conversationally in Norwegian bokmål, plain text, under 80 words." },
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
          : "No camera is available; never use the look branch.") +
        ` ${today()}`,
    },
    ...history,
    // the room comes last, right before the question: "det" means what was
    // just said in the meeting, not an older exchange with Octa
    ...meeting,
    { role: "user", content: transcript },
  ], settings.model, { json: true });

  let parsed: {
    answer?: string;
    slide?: unknown;
    subtasks?: { label: string; prompt: string }[];
    image?: string;
    imageEdit?: string;
    visual?: { kind?: string; brief?: string };
    search?: string;
    look?: string;
    tier?: string;
    caption?: string;
  };
  try {
    parsed = liftSlide(JSON.parse(extractJson(routed.text)));
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

  if (parsed.tier === "smart" || parsed.tier === "fast") {
    return { text: parsed.caption ?? "", images: [], tier: parsed.tier };
  }

  if (parsed.imageEdit) {
    if (!ctx.lastImage) {
      return { text: "Det er ikke noe bilde på skjermen å redigere ennå.", images: [] };
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
    return CHART_INTENT.test(transcript)
      ? chartFromSearch(transcript, parsed.search, events)
      : webSearch(parsed.search, events);
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
    return {
      text: parsed.answer ?? routed.text,
      images: routed.images,
      slide: validSlide(parsed.slide) ? cleanSlide(parsed.slide) : undefined,
    };
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

  const synth = await chat([
    { role: "system", content: SYNTH_SYS },
    ...history,
    ...meeting,
    {
      role: "user",
      content:
        `Original question: ${transcript}\n\n` +
        results.map((r) => `## ${r.label}\n${r.text}`).join("\n\n"),
    },
  ], settings.model, { json: true });
  return slideResult(synth.text, synth.images);
}

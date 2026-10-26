/** Minimal OpenRouter client — browser-direct, key lives in localStorage. */

const BASE = "https://openrouter.ai/api/v1";

export type ContentPart =
  | { type: "text"; text: string }
  | { type: "image_url"; image_url: { url: string } };

export interface ORMessage {
  role: "system" | "user" | "assistant";
  content: string | ContentPart[];
}

export interface Visual {
  kind: "mermaid" | "svg";
  code: string;
}

/** A keynote-style answer card: one headline, an optional hero number, a few short points. */
export interface SlideData {
  headline: string;
  stat?: { value: string; label: string };
  points: { title: string; detail?: string; icon?: string }[];
}

export interface TableData {
  title?: string;
  headers: string[];
  rows: string[][];
}

export interface ChatResult {
  text: string;
  images: string[]; // data/https URLs, present for image-output models
  visual?: Visual; // renderable diagram/visualization code
  table?: TableData; // distilled insights shown on the panel
  slide?: SlideData; // keynote card — the voice gives the takeaway, the card the structure
  tier?: ModelTier; // the user asked Octa to switch brains
  detailTable?: TableData; // granular dataset — what "download it" saves
}

export type ModelTier = "fast" | "smart";

export const settings = {
  get key() { return localStorage.getItem("octa-or-key") ?? ""; },
  set key(v: string) { localStorage.setItem("octa-or-key", v); },
  // two brains: a fast one by default, a smart one on request ("bruk en smartere modell")
  get fastModel() { return localStorage.getItem("octa-fast-model") || "openai/gpt-6-luna"; },
  set fastModel(v: string) { localStorage.setItem("octa-fast-model", v); },
  get smartModel() { return localStorage.getItem("octa-smart-model") || "openai/gpt-6.1-sol"; },
  set smartModel(v: string) { localStorage.setItem("octa-smart-model", v); },
  /** Reasoning effort for the fast brain — "low" measured ~30% quicker than the
   *  model default with the same answers; "default" sends nothing. */
  get fastEffort() { return localStorage.getItem("octa-fast-effort") || "low"; },
  set fastEffort(v: string) { localStorage.setItem("octa-fast-effort", v); },
  get tier(): ModelTier { return localStorage.getItem("octa-tier") === "smart" ? "smart" : "fast"; },
  set tier(v: ModelTier) { localStorage.setItem("octa-tier", v); },
  /** The model every text call uses — whichever tier is active. */
  get model() { return this.tier === "smart" ? this.smartModel : this.fastModel; },
  get agentsEnabled() { return localStorage.getItem("octa-or-agents") !== "0"; },
  set agentsEnabled(v: boolean) { localStorage.setItem("octa-or-agents", v ? "1" : "0"); },
  get imageModel() { return localStorage.getItem("octa-or-image-model") || "google/gemini-3.1-flash-image"; },
  set imageModel(v: string) { localStorage.setItem("octa-or-image-model", v); },
  get voice() { return localStorage.getItem("octa-voice") ?? ""; }, // "" = auto
  set voice(v: string) { localStorage.setItem("octa-voice", v); },
  get handGestures() { return localStorage.getItem("octa-hand-gestures") === "1"; },
  set handGestures(v: boolean) { localStorage.setItem("octa-hand-gestures", v ? "1" : "0"); },
};

export async function chat(
  messages: ORMessage[],
  model = settings.model,
  opts: { imageOutput?: boolean; webSearch?: boolean; json?: boolean } = {}
): Promise<ChatResult> {
  const res = await fetch(`${BASE}/chat/completions`, {
    method: "POST",
    headers: {
      Authorization: `Bearer ${settings.key}`,
      "Content-Type": "application/json",
      "X-Title": "Octa",
    },
    body: JSON.stringify({
      model,
      messages,
      // image-output models only generate when the request asks for it
      ...(opts.imageOutput ? { modalities: ["image", "text"] } : {}),
      // OpenRouter's web plugin injects live search results before the model runs
      ...(opts.webSearch ? { plugins: [{ id: "web", max_results: 5 }] } : {}),
      // JSON-protocol calls: without this, a long plain-text history drags the
      // model into answering in plain text and the protocol (slides!) is lost
      ...(opts.json ? { response_format: { type: "json_object" } } : {}),
      // the fast brain answers sooner with less thinking; the smart one keeps its default
      ...(model === settings.fastModel && settings.fastEffort !== "default" && !opts.imageOutput
        ? { reasoning: { effort: settings.fastEffort } }
        : {}),
    }),
  });
  if (!res.ok) {
    throw new Error(`OpenRouter ${res.status}: ${(await res.text()).slice(0, 300)}`);
  }
  const data = await res.json();
  const msg = data.choices?.[0]?.message ?? {};
  const images: string[] = (msg.images ?? [])
    .map((i: { image_url?: { url?: string } }) => i.image_url?.url)
    .filter(Boolean);
  return { text: msg.content ?? "", images };
}

export interface ModelInfo {
  id: string;
  name: string;
  imageOutput: boolean;
}

let modelCache: ModelInfo[] | null = null;

export async function listModels(): Promise<ModelInfo[]> {
  if (modelCache) return modelCache;
  const res = await fetch(`${BASE}/models`);
  if (!res.ok) throw new Error(`OpenRouter models ${res.status}`);
  const data = await res.json();
  type RawModel = {
    id: string;
    name: string;
    architecture?: { output_modalities?: string[] };
  };
  modelCache = (data.data as RawModel[])
    .map((m) => ({
      id: m.id,
      name: m.name,
      imageOutput: m.architecture?.output_modalities?.includes("image") ?? false,
    }))
    .sort((a, b) => a.id.localeCompare(b.id));
  return modelCache;
}

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
  detailTable?: TableData; // granular dataset — what "download it" saves
}

export const settings = {
  get key() { return localStorage.getItem("octa-or-key") ?? ""; },
  set key(v: string) { localStorage.setItem("octa-or-key", v); },
  get model() { return localStorage.getItem("octa-or-model") || "openrouter/auto"; },
  set model(v: string) { localStorage.setItem("octa-or-model", v); },
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
  opts: { imageOutput?: boolean; webSearch?: boolean } = {}
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

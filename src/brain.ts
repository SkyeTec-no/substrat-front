/**
 * Stub brain: fakes a thinking delay and returns canned-but-varied replies.
 * Swap this for a real LLM call later — the interface is just text in, text out.
 */

const OPENERS = [
  "I heard you.",
  "Interesting.",
  "Let me reflect that back.",
  "Noted.",
  "Understood.",
];

const REFLECTIONS = [
  (t: string) => `You said: ${t}.`,
  (t: string) => `So, ${t.toLowerCase()} — is that right?`,
  (t: string) => `"${t}" — I'll hold onto that.`,
];

const pick = <T>(arr: T[]): T => arr[Math.floor(Math.random() * arr.length)];

export async function think(transcript: string): Promise<string> {
  const delay = 1400 + Math.random() * 1800;
  await new Promise((r) => setTimeout(r, delay));
  return `${pick(OPENERS)} ${pick(REFLECTIONS)(transcript)}`;
}

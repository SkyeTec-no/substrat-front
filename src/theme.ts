/** SkyeTec theme tokens — mirrors skyetec.ai: `dark` class on <html>, token overrides. */

import type { OctaState } from "./scene";

export interface ThemeTokens {
  dark: boolean;
  bg: string;
  panel: string;
  fg: string;
  dim: string;
  seal: string; // brand green (patina) at a contrast fit for the ground
  signal: string;
  graphite: string;
  ruleStrong: string;
  octaBody: string;
  frame: string;
  stateColors: Record<OctaState, string>;
  satelliteColors: string[];
}

export const LIGHT: ThemeTokens = {
  dark: false,
  bg: "#e5e8de",
  panel: "#e4ded0",
  fg: "#0c1210",
  dim: "#8c8677",
  seal: "#1d4433",
  signal: "#d94018",
  graphite: "#3a4044",
  ruleStrong: "rgba(12, 18, 16, 0.3)",
  octaBody: "#0e1412",
  frame: "#0c1210",
  stateColors: {
    idle: "#3a4044",
    listening: "#1d4433",
    thinking: "#d94018",
    speaking: "#4a9272",
    noting: "#8c8677",
  },
  satelliteColors: ["#1d4433", "#d94018", "#3a4044", "#4a9272"],
};

export const DARK: ThemeTokens = {
  dark: true,
  bg: "#0b0e12",
  panel: "#13171e",
  fg: "#e4ded0",
  dim: "#8c8677",
  seal: "#4a9272",
  signal: "#d94018",
  graphite: "#4e4a40",
  ruleStrong: "rgba(242, 236, 224, 0.3)",
  octaBody: "#181d25",
  frame: "#e4ded0",
  stateColors: {
    idle: "#4e4a40",
    listening: "#4a9272",
    thinking: "#d94018",
    speaking: "#5f7266",
    noting: "#8c8677",
  },
  satelliteColors: ["#4a9272", "#d94018", "#8c8677", "#5f7266"],
};

let current: ThemeTokens =
  localStorage.getItem("octa-dark") === "1" ? DARK : LIGHT;

const listeners: ((t: ThemeTokens) => void)[] = [];

export function theme(): ThemeTokens {
  return current;
}

export function onThemeChange(cb: (t: ThemeTokens) => void) {
  listeners.push(cb);
}

export function setDarkMode(dark: boolean) {
  current = dark ? DARK : LIGHT;
  localStorage.setItem("octa-dark", dark ? "1" : "0");
  document.documentElement.classList.toggle("dark", dark);
  for (const cb of listeners) cb(current);
}

/** Apply the persisted choice at boot, exactly like the main page does. */
export function initTheme() {
  document.documentElement.classList.toggle("dark", current.dark);
}

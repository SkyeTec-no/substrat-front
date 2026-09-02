# Octa

A voice interaction surface — a SkyeTec surface, styled after the [Subspear](https://subspear.vercel.app/) design system.

Octa is a floating octahedron rendered in three.js that listens, thinks, and speaks, backed by an OpenRouter-driven reasoning layer. Answers, diagrams, charts, tables, images, and dictated notes appear as translucent 3D panels arranged in a bento grid, controllable by voice, mouse, or one-handed camera gestures.

## Features

- **Voice loop** — Web Speech API for speech-to-text and text-to-speech, zero network latency in the speech path itself. Barge-in (interrupt mid-reply), spoken fillers while the model thinks, and camera-grounded ambient compliments about your look or room.
- **Reasoning** — OpenRouter as the model backend (bring your own key, stored in `localStorage` only). A router step decides between a direct answer, a parallel sub-agent fan-out (visualized as satellites orbiting the octahedron with comet tails), web search, image generation/editing, diagram/chart generation, "look at me" vision, or PDF report assembly.
- **Artifacts as 3D panels** — images, mermaid diagrams, hand-authored SVG charts, tables, and note cards render as floating holographic cards, not DOM overlays. New artifacts take focus in front of the octahedron; older ones shelve into a structured bento grid behind it.
- **Gestures** — one hand does everything: point to highlight a grid tile, fist-open (or double-pinch) to pull it into focus, pinch-drag to pan, pinch + push/pull to zoom, flick up to shelve a panel, palm-hold to shelve everything, finger-to-ear to start listening. Runs fully client-side via MediaPipe (WASM/GPU), served locally — the camera feed never leaves the machine.
- **Head-tracked parallax** — subtle camera dolly/pan driven by head position, giving the panel layers real depth.
- **Note-taker mode** — say "take notes" for a dedicated dictation persona (distinct color, motion, and drone); cross your hands to exit. The transcript becomes a note card in the grid.
- **Reports** — "generate a go-to-market report" assembles conversation history and everything currently on the canvas into a structured, illustrated PDF — always rendered in the light/linen theme regardless of the active UI theme.
- **Theming** — light (Linen/Ink) and dark modes matching skyetec.ai, toggleable by voice ("dark mode") or in settings.
- **Downloads** — "download it" saves the last artifact: native-format images, rendered SVGs for diagrams/charts, CSV for tables (a separate detail dataset with full granularity, e.g. sources, backs every on-screen table).

## Getting started

```bash
npm install
npm run dev
```

Open the app, click **Begin**, then open the settings gear to paste an [OpenRouter](https://openrouter.ai/) API key. Without a key, Octa runs in a local echo/stub mode — the full voice and 3D loop works, but responses are canned.

## Scripts

- `npm run dev` — start the Vite dev server
- `npm run build` — type-check and build for production
- `npm run preview` — preview the production build locally

## Stack

three.js · GSAP · Web Speech API · Web Audio API · OpenRouter · MediaPipe Tasks Vision · mermaid · jsPDF · Vite + TypeScript

## Notes

- `public/mediapipe-wasm/` and `public/models/` are the MediaPipe runtime and models (gesture recognizer, face detector), served locally rather than from a CDN so gesture tracking works offline and never phones home.
- Gesture control and the "look at me" / ambient-compliment features require camera permission and are opt-in via the settings panel.

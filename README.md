# Octa

A voice interaction surface — a SkyeTec surface, styled after the [Subspear](https://subspear.vercel.app/) design system.

Octa is a floating octahedron rendered in three.js that listens, thinks, and speaks, backed by an OpenRouter-driven reasoning layer. Answers, diagrams, charts, tables, images, and dictated notes appear as translucent 3D panels arranged in a bento grid, controllable by voice, mouse, or one-handed camera gestures.

## Features

- **Voice loop** — Web Speech API for speech-to-text and text-to-speech, zero network latency in the speech path itself. Barge-in (interrupt mid-reply), spoken fillers while the model thinks, and camera-grounded ambient compliments about your look or room.
- **Norwegian** — Octa listens (`nb-NO`) and speaks Norwegian bokmål (macOS voice Nora when available); voice commands work in Norwegian and English.
- **Two brains** — a fast model (default `openai/gpt-6-luna`) and a smart one (default `openai/gpt-6.1-sol`). Say "bruk en smartere modell" / "tenk grundigere" or "tilbake til den raske modellen" (the router also catches indirect phrasings). The body follows the brain: octahedron for fast, icosahedron for smart, with a GSAP spring on every switch.
- **Reasoning** — OpenRouter as the model backend (bring your own key, stored in `localStorage` only). A router step decides between a direct answer, a parallel sub-agent fan-out (visualized as satellites orbiting the octahedron with comet tails), web search, image generation/editing, diagram/chart generation, "look at me" vision, or PDF report assembly.
- **Artifacts as 3D panels** — images, mermaid diagrams, hand-authored SVG charts, tables, and note cards render as floating holographic cards, not DOM overlays. New artifacts take focus in front of the octahedron; older ones shelve into a structured bento grid behind it.
- **Gestures** — one hand does everything: point to highlight a grid tile, fist-open (or double-pinch) to pull it into focus, pinch-drag to pan, pinch + push/pull to zoom, flick up to shelve a panel, palm-hold to shelve everything, finger-to-ear to start listening. Runs fully client-side via MediaPipe (WASM/GPU), served locally — the camera feed never leaves the machine.
- **Meetings** — several people can share the camera. Every face is a tracked person (P1, P2…) and every hand is bound to its face. Raise a hand above your head to ask for the floor — Octa picks from the raised hands (whoever is looking at it first) and names who goes next. The holder keeps the floor as long as they look at Octa; only when they look away can someone waiting take over. Only the holder's hands steer, and speech only counts when the holder was facing Octa with their lips moving — otherwise Octa says so instead of acting on it. Press `c` for a camera view showing who it sees.
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

- `public/mediapipe-wasm/` and `public/models/` are the MediaPipe runtime and models (gesture recognizer, face landmarker), served locally rather than from a CDN so gesture tracking works offline and never phones home.
- Gesture control and the "look at me" / ambient-compliment features require camera permission and are opt-in via the settings panel.
- `public/models/voxceleb_resnet34_LM.onnx` is [WeSpeaker](https://github.com/wenet-e2e/wespeaker)'s ResNet34 speaker-embedding model trained on VoxCeleb ([Wespeaker/wespeaker-voxceleb-resnet34-LM](https://huggingface.co/Wespeaker/wespeaker-voxceleb-resnet34-LM), CC BY 4.0). Octa runs it in the browser via ONNX Runtime Web (`public/ort/`) to recognize who is speaking; voiceprints are learned from moments when the camera sees a person's lips move, kept in memory for the session only, and never leave the machine.

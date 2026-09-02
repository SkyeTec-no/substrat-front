import type { TableData } from "./openrouter";
import { theme } from "./theme";

// While set, renderers use light (linen) tokens regardless of the UI theme —
// used when re-rendering artifacts for the always-light PDF.
let pdfLight = false;
export function setPdfLightRendering(v: boolean) {
  pdfLight = v;
}
function pal() {
  if (pdfLight) {
    return {
      dark: false,
      fg: "#0c1210",
      dim: "#8c8677",
      seal: "#1d4433",
      signal: "#d94018",
      ruleStrong: "rgba(12, 18, 16, 0.3)",
    };
  }
  const t = theme();
  return { dark: t.dark, fg: t.fg, dim: t.dim, seal: t.seal, signal: t.signal, ruleStrong: t.ruleStrong };
}

/** Rasterizers that turn artifacts into canvases/images for 3D panel textures. */

export function loadImage(url: string): Promise<HTMLImageElement> {
  return new Promise((resolve, reject) => {
    const img = new Image();
    img.crossOrigin = "anonymous";
    img.onload = () => resolve(img);
    img.onerror = reject;
    img.src = url;
  });
}

/** Serialize an SVG and draw it onto a dark-backed canvas. */
export async function svgToCanvas(svg: SVGElement, scale = 2): Promise<HTMLCanvasElement> {
  const clone = svg.cloneNode(true) as SVGElement;
  clone.setAttribute("xmlns", "http://www.w3.org/2000/svg");
  const vb = clone.getAttribute("viewBox")?.split(/[\s,]+/).map(Number);
  const w = vb?.[2] || 800;
  const h = vb?.[3] || 500;
  clone.setAttribute("width", String(w));
  clone.setAttribute("height", String(h));
  clone.style.maxWidth = "";

  const blob = new Blob([new XMLSerializer().serializeToString(clone)], {
    type: "image/svg+xml",
  });
  const url = URL.createObjectURL(blob);
  try {
    const img = await loadImage(url);
    const canvas = document.createElement("canvas");
    canvas.width = Math.round(w * scale);
    canvas.height = Math.round(h * scale);
    const ctx = canvas.getContext("2d")!;
    // clean single pass — no glow; restraint is the argument
    ctx.drawImage(img, 0, 0, canvas.width, canvas.height);
    return canvas;
  } finally {
    URL.revokeObjectURL(url);
  }
}

export interface PanelRender {
  canvas: HTMLCanvasElement;
  /** Redraw for the given elapsed seconds; returns true once the animation settled. */
  draw: (elapsed: number) => boolean;
}

const SCRAMBLE = "01<>[]{}/\\|=+*#$%&@!?";

/**
 * Animated table: characters stream in matrix-style — scrambling through
 * random glyphs before settling. Thin neon CLI type, no backing, no borders.
 */
export function createTableRenderer(t: TableData, scale = 2): PanelRender {
  const mono = "ui-monospace, Menlo, monospace";
  const pad = 30;
  const rowH = 36;
  const headH = 34;
  const titleH = t.title ? 46 : 0;
  const gap = 44;

  const meas = document.createElement("canvas").getContext("2d")!;
  const colWidths = t.headers.map((h, c) => {
    meas.font = `12px ${mono}`;
    let w = meas.measureText(h.toUpperCase()).width;
    meas.font = `300 15px ${mono}`;
    for (const row of t.rows) {
      w = Math.max(w, meas.measureText(String(row[c] ?? "")).width);
    }
    return w + gap;
  });

  // the letter-spaced title can outgrow the columns — never clip it
  meas.font = `11px ${mono}`;
  meas.letterSpacing = "5px";
  const titleW = t.title ? meas.measureText(t.title.toUpperCase()).width + 16 : 0;
  meas.letterSpacing = "0px";
  const W = Math.max(pad * 2 + colWidths.reduce((a, b) => a + b, 0) - gap + 10, pad * 2 + titleW);
  const H = pad * 2 + titleH + headH + t.rows.length * rowH;

  const canvas = document.createElement("canvas");
  canvas.width = Math.round(W * scale);
  canvas.height = Math.round(H * scale);
  const ctx = canvas.getContext("2d")!;

  // per-character reveal: char i of an item starting at `start` lands at start + i*0.022
  const CHAR_DT = 0.022;
  const drawStreamed = (
    text: string,
    x: number,
    y: number,
    start: number,
    elapsed: number,
    style: { fill: string; glow: number; font: string; spacing?: string }
  ): boolean => {
    ctx.font = style.font;
    ctx.letterSpacing = style.spacing ?? "1px";
    let cx = x;
    let settled = true;
    for (let i = 0; i < text.length; i++) {
      const ch = text[i];
      const revealAt = start + i * CHAR_DT;
      if (elapsed >= revealAt) {
        ctx.fillStyle = style.fill;
        ctx.shadowBlur = 0;
        void style.glow;
        ctx.fillText(ch, cx, y);
      } else if (elapsed >= revealAt - 0.25 && ch !== " ") {
        settled = false;
        ctx.fillStyle = pal().dark ? "rgba(74, 146, 114, 0.45)" : "rgba(29, 68, 51, 0.4)";
        ctx.shadowBlur = 0;
        ctx.fillText(SCRAMBLE[Math.floor(Math.random() * SCRAMBLE.length)], cx, y);
      } else if (elapsed < revealAt) {
        settled = false;
      }
      cx += ctx.measureText(ch).width;
    }
    ctx.shadowBlur = 0;
    ctx.letterSpacing = "0px";
    return settled;
  };

  const draw = (elapsed: number): boolean => {
    ctx.setTransform(1, 0, 0, 1, 0, 0);
    ctx.clearRect(0, 0, canvas.width, canvas.height);
    ctx.scale(scale, scale);
    let settled = true;
    let y = pad;

    if (t.title) {
      settled =
        drawStreamed(t.title.toUpperCase(), pad, y + 14, 0, elapsed, {
          fill: pal().seal,
          glow: 0,
          font: `11px ${mono}`,
          spacing: "5px",
        }) && settled;
      y += titleH;
    }

    let x = pad;
    for (const [c, h] of t.headers.entries()) {
      settled =
        drawStreamed(h.toUpperCase(), x, y + 14, 0.15, elapsed, {
          fill: pal().dim,
          glow: 0,
          font: `12px ${mono}`,
          spacing: "2px",
        }) && settled;
      x += colWidths[c];
    }
    y += headH - 8;

    // header underline sweeps in with the stream
    const lineP = Math.min(1, Math.max(0, (elapsed - 0.15) / 0.5));
    if (lineP > 0) {
      ctx.strokeStyle = pal().ruleStrong;
      ctx.beginPath();
      ctx.moveTo(pad, y);
      ctx.lineTo(pad + (W - pad * 2) * lineP, y);
      ctx.stroke();
    }
    if (lineP < 1) settled = false;
    y += 8;

    for (const [r, row] of t.rows.entries()) {
      const rowStart = 0.3 + r * 0.13;
      x = pad;
      for (const [c, cell] of row.entries()) {
        settled =
          drawStreamed(String(cell ?? ""), x, y + 22, rowStart + c * 0.05, elapsed, {
            fill: c === 0 ? pal().dim : pal().fg,
            glow: 0,
            font: `500 15px ${mono}`,
          }) && settled;
        x += colWidths[c] ?? 0;
      }
      y += rowH;
    }
    return settled;
  };

  draw(0);
  return { canvas, draw };
}

/** A captured dictation as a note card: warm title, wrapped luminous lines. */
export function noteToCanvas(lines: string[], title: string, scale = 2): HTMLCanvasElement {
  const mono = "ui-monospace, Menlo, monospace";
  const pad = 34;
  const maxW = 560;
  const lineH = 26;
  const titleH = 48;

  const meas = document.createElement("canvas").getContext("2d")!;
  meas.font = `300 15px ${mono}`;
  const wrapped: { text: string; first: boolean }[] = [];
  for (const line of lines) {
    let current = "";
    let first = true;
    for (const word of line.split(/\s+/)) {
      const trial = current ? `${current} ${word}` : word;
      if (meas.measureText(trial).width > maxW - 20 && current) {
        wrapped.push({ text: current, first });
        first = false;
        current = word;
      } else {
        current = trial;
      }
    }
    if (current) wrapped.push({ text: current, first });
  }

  const W = maxW + pad * 2;
  const H = pad * 2 + titleH + Math.max(1, wrapped.length) * lineH;
  const canvas = document.createElement("canvas");
  canvas.width = Math.round(W * scale);
  canvas.height = Math.round(H * scale);
  const ctx = canvas.getContext("2d")!;
  ctx.scale(scale, scale);

  ctx.font = `11px ${mono}`;
  ctx.letterSpacing = "5px";
  ctx.fillStyle = "#d94018";
  ctx.fillText(title.toUpperCase(), pad, pad + 12);
  ctx.letterSpacing = "0px";

  ctx.font = `400 15px ${mono}`;
  let y = pad + titleH;
  for (const row of wrapped) {
    if (row.first) {
      ctx.fillStyle = pal().signal;
      ctx.fillText("·", pad, y + 16);
    }
    ctx.fillStyle = pal().fg;
    ctx.fillText(row.text, pad + 20, y + 16);
    y += lineH;
  }
  ctx.shadowBlur = 0;
  return canvas;
}

/** Left-to-right holographic wipe over an already-rasterized canvas (diagrams). */
export function createWipeRenderer(src: HTMLCanvasElement, duration = 0.9): PanelRender {
  const canvas = document.createElement("canvas");
  canvas.width = src.width;
  canvas.height = src.height;
  const ctx = canvas.getContext("2d")!;

  const draw = (elapsed: number): boolean => {
    const p = Math.min(1, elapsed / duration);
    ctx.clearRect(0, 0, canvas.width, canvas.height);
    const w = Math.max(1, Math.round(canvas.width * p));
    ctx.drawImage(src, 0, 0, w, src.height, 0, 0, w, canvas.height);
    if (p < 1) {
      const grad = ctx.createLinearGradient(w - 30, 0, w, 0);
      const edge = theme().dark ? "74, 146, 114" : "29, 68, 51";
      grad.addColorStop(0, `rgba(${edge}, 0)`);
      grad.addColorStop(1, `rgba(${edge}, 0.45)`);
      ctx.fillStyle = grad;
      ctx.fillRect(w - 30, 0, 30, canvas.height);
    }
    return p >= 1;
  };

  draw(0);
  return { canvas, draw };
}

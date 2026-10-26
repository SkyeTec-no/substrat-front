import type { SlideData, TableData } from "./openrouter";
import { theme } from "./theme";
import { drawIcon, isIcon } from "./icons";

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
      muted: "#57534a",
      seal: "#1d4433",
      signal: "#d94018",
      ruleStrong: "rgba(12, 18, 16, 0.3)",
    };
  }
  const t = theme();
  // muted: secondary text that must stay readable on the frosted panel card
  // (dim is too faint for body copy on white)
  const muted = t.dark ? "#aaa496" : "#57534a";
  return { dark: t.dark, fg: t.fg, dim: t.dim, muted, seal: t.seal, signal: t.signal, ruleStrong: t.ruleStrong };
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
          fill: pal().muted,
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
            fill: c === 0 ? pal().muted : pal().fg,
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

/** Greedy word wrap against the context's current font. */
function wrapText(ctx: CanvasRenderingContext2D, text: string, maxW: number): string[] {
  const lines: string[] = [];
  let current = "";
  for (const word of text.split(/\s+/).filter(Boolean)) {
    const trial = current ? `${current} ${word}` : word;
    if (ctx.measureText(trial).width > maxW && current) {
      lines.push(current);
      current = word;
    } else {
      current = trial;
    }
  }
  if (current) lines.push(current);
  return lines;
}

/**
 * Keynote card: serif headline, an optional hero number in Signal orange,
 * then numbered points — each element rises into place in sequence.
 */
export function createSlideRenderer(slide: SlideData, scale = 2): PanelRender {
  const mono = "ui-monospace, Menlo, monospace";
  const serif = 'Newsreader, Georgia, "Times New Roman", serif';
  const W = 760;
  const pad = 44;
  const inner = W - pad * 2;
  const numW = 56; // gutter for the 01 / 02 index

  const meas = document.createElement("canvas").getContext("2d")!;
  meas.font = `500 34px ${serif}`;
  const headline = wrapText(meas, slide.headline, inner).slice(0, 3);
  const points = slide.points.map((pt) => {
    meas.font = `500 20px ${serif}`;
    const title = wrapText(meas, pt.title, inner - numW);
    meas.font = `400 15px ${serif}`;
    const detail = pt.detail ? wrapText(meas, pt.detail, inner - numW) : [];
    return { title, detail, icon: pt.icon, h: Math.max(64, 20 + title.length * 26 + detail.length * 21 + 18) };
  });

  const headH = headline.length * 42;
  const statH = slide.stat ? 104 : 0;
  const H = pad + 22 + headH + 26 + statH + points.reduce((a, p) => a + p.h, 0) + pad - 18;

  const canvas = document.createElement("canvas");
  canvas.width = Math.round(W * scale);
  canvas.height = Math.round(H * scale);
  const ctx = canvas.getContext("2d")!;

  // each element fades and rises over 0.45s, starting at its own beat
  const reveal = (elapsed: number, start: number) =>
    Math.min(1, Math.max(0, (elapsed - start) / 0.45));
  const ease = (p: number) => 1 - Math.pow(1 - p, 3);

  const draw = (elapsed: number): boolean => {
    ctx.setTransform(1, 0, 0, 1, 0, 0);
    ctx.clearRect(0, 0, canvas.width, canvas.height);
    ctx.scale(scale, scale);
    const c = pal();
    let settled = true;
    const at = (start: number) => {
      const p = reveal(elapsed, start);
      if (p < 1) settled = false;
      ctx.globalAlpha = ease(p);
      return (1 - ease(p)) * 10; // rise offset
    };

    let y = pad;
    let beat = 0;

    // kicker
    let dy = at(beat);
    ctx.font = `11px ${mono}`;
    ctx.letterSpacing = "5px";
    ctx.fillStyle = c.seal;
    ctx.fillText(`KEY POINTS · ${String(points.length).padStart(2, "0")}`, pad, y + 10 + dy);
    ctx.letterSpacing = "0px";
    y += 22;

    // headline
    beat += 0.12;
    dy = at(beat);
    ctx.font = `500 34px ${serif}`;
    ctx.fillStyle = c.fg;
    for (const line of headline) {
      y += 42;
      ctx.fillText(line, pad, y - 8 + dy);
    }
    y += 26;

    // hero number
    if (slide.stat) {
      beat += 0.25;
      dy = at(beat);
      ctx.font = `500 64px ${serif}`;
      ctx.fillStyle = c.signal;
      ctx.fillText(slide.stat.value, pad, y + 62 + dy);
      const vw = ctx.measureText(slide.stat.value).width;
      ctx.font = `11px ${mono}`;
      ctx.letterSpacing = "3px";
      ctx.fillStyle = c.muted;
      for (const [i, line] of wrapText(ctx, slide.stat.label.toUpperCase(), inner - vw - 28).slice(0, 3).entries()) {
        ctx.fillText(line, pad + vw + 24, y + 34 + i * 16 + dy);
      }
      ctx.letterSpacing = "0px";
      y += statH;
    }

    // numbered points, each under a rule that sweeps in first
    for (const [i, pt] of points.entries()) {
      beat += 0.18;
      ctx.globalAlpha = 1;
      const lineP = reveal(elapsed, beat - 0.1);
      if (lineP < 1) settled = false;
      ctx.strokeStyle = c.ruleStrong;
      ctx.lineWidth = 1;
      ctx.beginPath();
      ctx.moveTo(pad, y);
      ctx.lineTo(pad + inner * ease(lineP), y);
      ctx.stroke();

      dy = at(beat);
      // the lead point carries the Signal accent when there's no hero number
      const mark = i === 0 && !slide.stat ? c.signal : c.seal;
      if (isIcon(pt.icon)) {
        drawIcon(ctx, pt.icon, pad, y + 22 + dy, 26, mark);
      } else {
        ctx.font = `12px ${mono}`;
        ctx.letterSpacing = "2px";
        ctx.fillStyle = mark;
        ctx.fillText(String(i + 1).padStart(2, "0"), pad, y + 38 + dy);
        ctx.letterSpacing = "0px";
      }

      let ty = y + 20;
      ctx.font = `500 20px ${serif}`;
      ctx.fillStyle = c.fg;
      for (const line of pt.title) {
        ty += 26;
        ctx.fillText(line, pad + numW, ty - 6 + dy);
      }
      ctx.font = `400 15px ${serif}`;
      ctx.fillStyle = c.muted;
      for (const line of pt.detail) {
        ty += 21;
        ctx.fillText(line, pad + numW, ty - 2 + dy);
      }
      y += pt.h;
    }
    ctx.globalAlpha = 1;
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

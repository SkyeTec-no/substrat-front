/** Dark HUD-styled PDF assembly — fully client-side via jsPDF. */

export interface ReportSpec {
  title: string;
  subtitle?: string;
  sections: { heading: string; body: string; artifact?: string }[];
}

export interface ReportArtifact {
  kind: string;
  source?: HTMLCanvasElement | HTMLImageElement;
  // underlying data, so the PDF can re-render in light tokens
  table?: TableData;
  svgMarkup?: string;
  noteLines?: string[];
  noteTitle?: string;
}

const W = 595.28; // A4 pt
const H = 841.89;
const M = 52;

import {
  setPdfLightRendering,
  createTableRenderer,
  noteToCanvas,
  svgToCanvas,
} from "./panels";
import type { TableData } from "./openrouter";

/** Composite an artifact onto the linen page ground — the PDF is always light. */
function toLinenPng(source: HTMLCanvasElement | HTMLImageElement): {
  dataUrl: string;
  aspect: number;
} {
  const sw = source instanceof HTMLImageElement ? source.naturalWidth : source.width;
  const sh = source instanceof HTMLImageElement ? source.naturalHeight : source.height;
  const c = document.createElement("canvas");
  c.width = sw;
  c.height = sh;
  const ctx = c.getContext("2d")!;
  ctx.fillStyle = "#e5e8de";
  ctx.fillRect(0, 0, sw, sh);
  ctx.drawImage(source, 0, 0);
  return { dataUrl: c.toDataURL("image/png"), aspect: sw / Math.max(1, sh) };
}

/** Dark-mode token hexes baked into SVG markup → their light equivalents. */
function svgToLightTokens(markup: string): string {
  return markup
    .replace(/#e4ded0/gi, "#0c1210")
    .replace(/#4a9272/gi, "#1d4433")
    .replace(/#5f7266/gi, "#1d4433")
    .replace(/rgba\(74,\s*146,\s*114/gi, "rgba(29, 68, 51")
    .replace(/rgba\(242,\s*236,\s*224/gi, "rgba(12, 18, 16");
}

/** Re-render an artifact in light tokens regardless of the UI theme. */
async function resolveLightSource(
  art: ReportArtifact
): Promise<HTMLCanvasElement | HTMLImageElement | null> {
  if (art.table) {
    setPdfLightRendering(true);
    try {
      const r = createTableRenderer(art.table);
      r.draw(999); // settle the stream instantly
      return r.canvas;
    } finally {
      setPdfLightRendering(false);
    }
  }
  if (art.noteLines?.length) {
    setPdfLightRendering(true);
    try {
      return noteToCanvas(art.noteLines, art.noteTitle ?? "Notes");
    } finally {
      setPdfLightRendering(false);
    }
  }
  if (art.svgMarkup) {
    try {
      const holder = document.createElement("div");
      holder.innerHTML = svgToLightTokens(art.svgMarkup);
      const el = holder.querySelector("svg");
      if (el) return await svgToCanvas(el);
    } catch (err) {
      console.error("light re-render failed, using screen canvas", err);
    }
  }
  return art.source ?? null;
}

export async function buildReportPdf(
  spec: ReportSpec,
  artifacts: Map<string, ReportArtifact>
): Promise<string> {
  const { jsPDF } = await import("jspdf");
  const doc = new jsPDF({ unit: "pt", format: "a4" });

  const paint = () => {
    doc.setFillColor(229, 232, 222); // Linen
    doc.rect(0, 0, W, H, "F");
  };

  // --- cover -----------------------------------------------------------------
  paint();
  doc.setDrawColor(12, 18, 16);
  doc.setLineWidth(0.75);
  doc.line(M, 140, W - M, 140);
  doc.setFont("times", "bold");
  doc.setFontSize(28);
  doc.setTextColor(12, 18, 16);
  doc.text(doc.splitTextToSize(spec.title.toUpperCase(), W - 2 * M), M, 185);
  if (spec.subtitle) {
    doc.setFont("times", "italic");
    doc.setFontSize(13);
    doc.setTextColor(29, 68, 51);
    doc.text(doc.splitTextToSize(spec.subtitle, W - 2 * M), M, 245);
  }
  doc.setFont("courier", "normal");
  doc.setFontSize(9);
  doc.setTextColor(140, 134, 119);
  doc.text(
    `SKYETEC · OCTA — generated ${new Date().toLocaleDateString(undefined, {
      year: "numeric",
      month: "long",
      day: "numeric",
    })}`,
    M,
    H - M
  );

  // --- sections --------------------------------------------------------------
  let y = 0;
  const newPage = () => {
    doc.addPage();
    paint();
    y = M + 16;
  };
  const need = (pts: number) => {
    if (y + pts > H - M) newPage();
  };
  newPage();

  for (const section of spec.sections) {
    need(90);
    doc.setFont("courier", "bold");
    doc.setFontSize(11);
    doc.setTextColor(29, 68, 51);
    doc.text(section.heading.toUpperCase(), M, y);
    doc.setDrawColor(12, 18, 16);
    doc.setLineWidth(0.5);
    doc.line(M, y + 6, W - M, y + 6);
    y += 26;

    doc.setFont("times", "normal");
    doc.setFontSize(11);
    doc.setTextColor(12, 18, 16);
    const lines: string[] = doc.splitTextToSize(section.body, W - 2 * M);
    for (const line of lines) {
      need(15);
      doc.text(line, M, y);
      y += 15;
    }
    y += 8;

    const art = section.artifact ? artifacts.get(section.artifact) : undefined;
    const lightSource = art ? await resolveLightSource(art) : null;
    if (lightSource) {
      const { dataUrl, aspect } = toLinenPng(lightSource);
      let iw = W - 2 * M;
      let ih = iw / aspect;
      const maxH = 300;
      if (ih > maxH) {
        ih = maxH;
        iw = ih * aspect;
      }
      need(ih + 20);
      doc.addImage(dataUrl, "PNG", M, y, iw, ih);
      doc.setDrawColor(12, 18, 16);
      doc.setLineWidth(0.4);
      doc.rect(M, y, iw, ih);
      y += ih + 24;
    } else {
      y += 10;
    }
  }

  const filename = `octa-report-${new Date().toISOString().slice(0, 10)}.pdf`;
  doc.save(filename);
  return filename;
}

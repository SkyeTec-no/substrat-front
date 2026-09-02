import type { ChatResult } from "./openrouter";

/** Voice command matcher: "download it", "save that diagram", or just "download". */
export function isDownloadCommand(text: string): boolean {
  return (
    /\b(download|save|export)\b.{0,24}\b(it|that|this|them|image|picture|photo|diagram|chart|table|file|visual)\b/i.test(text) ||
    /^\s*(please\s+)?(download|save|export)(\s+(it|that|this|please))*\s*[.!]?\s*$/i.test(text)
  );
}

function save(blob: Blob, filename: string) {
  const url = URL.createObjectURL(blob);
  const a = document.createElement("a");
  a.href = url;
  a.download = filename;
  a.click();
  setTimeout(() => URL.revokeObjectURL(url), 10_000);
}

/** Downloads every artifact from the given result; returns how many files were saved. */
export async function downloadArtifacts(r: ChatResult, svgMarkup?: string): Promise<number> {
  const ts = new Date().toISOString().replace(/[:.]/g, "-").slice(0, 19);
  let n = 0;

  for (const [i, url] of r.images.entries()) {
    try {
      const blob = await (await fetch(url)).blob();
      const ext = (blob.type.split("/")[1] ?? "png").replace("jpeg", "jpg").split("+")[0];
      const suffix = r.images.length > 1 ? `-${i + 1}` : "";
      save(blob, `octa-image-${ts}${suffix}.${ext}`);
      n++;
    } catch (err) {
      console.error("image download failed", err);
    }
  }

  if (r.visual?.kind === "mermaid") {
    // prefer the rendered SVG markup; fall back to the raw source
    if (svgMarkup) {
      const withNs = svgMarkup.includes("xmlns")
        ? svgMarkup
        : svgMarkup.replace("<svg", '<svg xmlns="http://www.w3.org/2000/svg"');
      save(new Blob([withNs], { type: "image/svg+xml" }), `octa-diagram-${ts}.svg`);
    } else {
      save(new Blob([r.visual.code], { type: "text/plain" }), `octa-diagram-${ts}.mmd`);
    }
    n++;
  } else if (r.visual) {
    save(
      new Blob([svgMarkup ?? r.visual.code], { type: "image/svg+xml" }),
      `octa-chart-${ts}.svg`
    );
    n++;
  }

  // prefer the granular dataset over the on-screen insight table
  const tbl = r.detailTable ?? r.table;
  if (tbl) {
    const esc = (c: string) => (/[",\n]/.test(c) ? `"${c.replace(/"/g, '""')}"` : c);
    const csv = [tbl.headers, ...tbl.rows].map((row) => row.map(esc).join(",")).join("\n");
    save(new Blob([csv], { type: "text/csv" }), `octa-data-${ts}.csv`);
    n++;
  }

  return n;
}

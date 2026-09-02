import { theme } from "./theme";

/** Mermaid rendering (dynamically imported — it's heavy), themed per SkyeTec mode. */
export async function renderMermaidSvg(code: string): Promise<{ svgEl: SVGElement; markup: string }> {
  const { default: mermaid } = await import("mermaid");
  const t = theme();
  mermaid.initialize({
    startOnLoad: false,
    // foreignObject labels taint canvases and break WebGL textures — pure SVG text only
    htmlLabels: false,
    flowchart: { htmlLabels: false },
    theme: "base",
    themeVariables: {
      darkMode: t.dark,
      background: "transparent",
      primaryColor: t.dark ? "rgba(74, 146, 114, 0.08)" : "rgba(29, 68, 51, 0.06)",
      primaryTextColor: t.fg,
      primaryBorderColor: t.seal,
      secondaryColor: "rgba(217, 64, 24, 0.06)",
      secondaryBorderColor: t.signal,
      tertiaryColor: "rgba(229, 232, 222, 0)",
      tertiaryBorderColor: t.graphite,
      lineColor: t.dark ? t.dim : t.graphite,
      edgeLabelBackground: "transparent",
      textColor: t.fg,
      fontFamily: 'ui-monospace, Menlo, monospace',
      fontSize: "13px",
    },
  });
  const { svg } = await mermaid.render(`octa-m-${Date.now()}`, code);
  const holder = document.createElement("div");
  holder.innerHTML = svg;
  return { svgEl: holder.querySelector("svg")!, markup: svg };
}


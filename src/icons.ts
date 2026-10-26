/**
 * A small fixed set of line icons for keynote slides — 24×24 SVG path data,
 * drawn as 1.5px strokes via Path2D. The model picks names from ICON_NAMES;
 * anything else is dropped and the point falls back to its number.
 */

const ICONS = {
  check: "M5 12l4 4 10-10",
  warning: "M12 3L2 20h20Z M12 9v5 M12 17v.5",
  clock: "M12 3a9 9 0 1 0 0 18a9 9 0 1 0 0-18Z M12 7v5l3 2",
  calendar: "M4 6h16v14H4Z M4 10h16 M8 3v4 M16 3v4",
  heart: "M12 20s-7-4.5-7-10a4 4 0 0 1 7-2.5A4 4 0 0 1 19 10c0 5.5-7 10-7 10Z",
  leaf: "M5 19C5 10 10 5 19 5c0 9-5 14-14 14Z M5 19l8-8",
  eye: "M2 12s4-7 10-7 10 7 10 7-4 7-10 7S2 12 2 12Z M12 9a3 3 0 1 0 0 6a3 3 0 1 0 0-6Z",
  home: "M3 11l9-7 9 7 M5 10v10h14V10 M10 20v-6h4v6",
  people: "M9 11a3 3 0 1 0 0-6a3 3 0 1 0 0 6Z M3 20c0-3.5 2.5-6 6-6s6 2.5 6 6 M16 5.5a3 3 0 0 1 0 5.5 M18 14c2 .8 3 3 3 6",
  chat: "M4 5h16v11H9l-5 4Z",
  chart: "M4 20h16 M7 16v-5 M12 16V7 M17 16v-8",
  trend: "M3 17l6-6 4 4 8-8 M15 7h6v6",
  money: "M12 3a9 9 0 1 0 0 18a9 9 0 1 0 0-18Z M15 9c-.5-1-1.6-1.5-3-1.5-1.7 0-3 .9-3 2.2 0 3 6 1.6 6 4.6 0 1.3-1.3 2.2-3 2.2-1.4 0-2.5-.5-3-1.5 M12 6v1.5 M12 16.5V18",
  globe: "M12 3a9 9 0 1 0 0 18a9 9 0 1 0 0-18Z M3 12h18 M12 3c2.5 2.5 3.5 5.5 3.5 9s-1 6.5-3.5 9c-2.5-2.5-3.5-5.5-3.5-9S9.5 5.5 12 3Z",
  pin: "M12 21s-7-6.5-7-12a7 7 0 0 1 14 0c0 5.5-7 12-7 12Z M12 7a2 2 0 1 0 0 4a2 2 0 1 0 0-4Z",
  bolt: "M13 2L4 14h7l-1 8 9-12h-7Z",
  shield: "M12 3l8 3v6c0 5-3.5 8-8 9-4.5-1-8-4-8-9V6Z",
  lock: "M6 11h12v9H6Z M8.5 11V8a3.5 3.5 0 0 1 7 0v3",
  book: "M4 5c3-1 5-1 8 1v14c-3-2-5-2-8-1Z M20 5c-3-1-5-1-8 1v14c3-2 5-2 8-1Z",
  bulb: "M9 18h6 M10 21h4 M12 3a6 6 0 0 0-4 10.5c.8.8 1 1.5 1 2.5h6c0-1 .2-1.7 1-2.5A6 6 0 0 0 12 3Z",
  target: "M12 3a9 9 0 1 0 0 18a9 9 0 1 0 0-18Z M12 7.5a4.5 4.5 0 1 0 0 9a4.5 4.5 0 1 0 0-9Z M12 11a1 1 0 1 0 0 2a1 1 0 1 0 0-2Z",
  star: "M12 3l2.7 5.6 6.1.9-4.4 4.3 1 6.1L12 17l-5.4 2.9 1-6.1L3.2 9.5l6.1-.9Z",
  food: "M7 3v8 M5 3v5a2 2 0 0 0 4 0V3 M7 11v10 M16 3c-2 1-3 4-3 8h3v10",
  water: "M12 3s-6 7-6 11a6 6 0 0 0 12 0c0-4-6-11-6-11Z",
  moon: "M20 14.5A8 8 0 1 1 9.5 4a6.5 6.5 0 0 0 10.5 10.5Z",
  sun: "M12 8a4 4 0 1 0 0 8a4 4 0 1 0 0-8Z M12 2v2 M12 20v2 M2 12h2 M20 12h2 M4.9 4.9l1.4 1.4 M17.7 17.7l1.4 1.4 M4.9 19.1l1.4-1.4 M17.7 6.3l1.4-1.4",
  tools: "M4 7h10 M18 7h2 M4 17h2 M10 17h10 M16 5v4 M8 15v4",
} as const;

export type IconName = keyof typeof ICONS;
export const ICON_NAMES = Object.keys(ICONS) as IconName[];

export function isIcon(name: unknown): name is IconName {
  return typeof name === "string" && name in ICONS;
}

const cache = new Map<IconName, Path2D>();

/** Stroke an icon with its top-left at (x, y), `size` px square. */
export function drawIcon(
  ctx: CanvasRenderingContext2D,
  name: IconName,
  x: number,
  y: number,
  size: number,
  color: string
) {
  let path = cache.get(name);
  if (!path) {
    path = new Path2D(ICONS[name]);
    cache.set(name, path);
  }
  const k = size / 24;
  ctx.save();
  ctx.translate(x, y);
  ctx.scale(k, k);
  ctx.strokeStyle = color;
  ctx.lineWidth = 1.5 / k;
  ctx.lineCap = "round";
  ctx.lineJoin = "round";
  ctx.stroke(path);
  ctx.restore();
}

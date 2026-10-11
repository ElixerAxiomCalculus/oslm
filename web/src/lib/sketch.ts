/** Deterministic PRNG so a sketched border looks the same on every render of the same element. */
export function mulberry32(seed: number): () => number {
  let a = seed >>> 0;
  return () => {
    a = (a + 0x6d2b79f5) >>> 0;
    let t = a;
    t = Math.imul(t ^ (t >>> 15), t | 1);
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

export function hashString(s: string): number {
  let h = 2166136261;
  for (let i = 0; i < s.length; i++) {
    h ^= s.charCodeAt(i);
    h = Math.imul(h, 16777619);
  }
  return h >>> 0;
}

function jitterLine(x1: number, y1: number, x2: number, y2: number, rand: () => number, amp: number): string {
  const len = Math.hypot(x2 - x1, y2 - y1);
  const steps = Math.max(2, Math.round(len / 40));
  let d = "";
  for (let i = 1; i <= steps; i++) {
    const t = i / steps;
    const x = x1 + (x2 - x1) * t + (i < steps ? (rand() - 0.5) * amp : 0);
    const y = y1 + (y2 - y1) * t + (i < steps ? (rand() - 0.5) * amp : 0);
    d += ` L ${x.toFixed(1)} ${y.toFixed(1)}`;
  }
  return d;
}

/** A slightly wobbly rectangle path in a w×h box, drawn twice-over at the corners like a pencil. */
export function sketchRect(w: number, h: number, seed: number, amp = 2.2, inset = 2): string {
  const rand = mulberry32(seed);
  const o = () => (rand() - 0.5) * amp;
  const x0 = inset + o();
  const y0 = inset + o();
  const x1 = w - inset + o();
  const y1 = h - inset + o();
  return (
    `M ${x0.toFixed(1)} ${y0.toFixed(1)}` +
    jitterLine(x0, y0, x1, y0 + o(), rand, amp) +
    jitterLine(x1, y0, x1 + o(), y1, rand, amp) +
    jitterLine(x1, y1, x0 + o(), y1 + o(), rand, amp) +
    jitterLine(x0, y1, x0 + o() + 3, y0 - 1, rand, amp)
  );
}

/** A hand-drawn underline stroke across width w. */
export function sketchUnderline(w: number, seed: number): string {
  const rand = mulberry32(seed);
  const y = 4 + (rand() - 0.5) * 2;
  const mid = w / 2 + (rand() - 0.5) * w * 0.2;
  return `M 1 ${y.toFixed(1)} Q ${mid.toFixed(1)} ${(y + (rand() - 0.5) * 4).toFixed(1)} ${(w - 1).toFixed(1)} ${(y + (rand() - 0.5) * 2).toFixed(1)}`;
}

/** An open ellipse that overshoots its start, like circling a word with a pencil. */
export function sketchCircle(w: number, h: number, seed: number): string {
  const rand = mulberry32(seed);
  const cx = w / 2;
  const cy = h / 2;
  const rx = w / 2 - 3;
  const ry = h / 2 - 3;
  const start = rand() * Math.PI * 2;
  let d = "";
  const steps = 28;
  for (let i = 0; i <= steps + 3; i++) {
    const a = start + (i / steps) * Math.PI * 2;
    const k = 1 + (rand() - 0.5) * 0.06;
    const x = cx + Math.cos(a) * rx * k;
    const y = cy + Math.sin(a) * ry * k;
    d += `${i === 0 ? "M" : " L"} ${x.toFixed(1)} ${y.toFixed(1)}`;
  }
  return d;
}

/** Per-word baseline jitter (px) and rotation (deg), within ±0.4°. */
export function wordJitter(index: number): { y: number; r: number } {
  const rand = mulberry32(index * 2654435761);
  return { y: (rand() - 0.5) * 1.6, r: (rand() - 0.5) * 0.8 };
}

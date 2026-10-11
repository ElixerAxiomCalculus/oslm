export interface RevealOptions {
  baseCharsPerSecond: number;
  maxLagMs: number;
  reducedMotion: boolean;
}

const DEFAULTS: RevealOptions = { baseCharsPerSecond: 70, maxLagMs: 1500, reducedMotion: false };

interface Region {
  start: number;
  end: number;
  complete: boolean;
}

function fenceRegions(text: string): Region[] {
  const regions: Region[] = [];
  const fence = /^[ \t]*```.*$/gm;
  let open: number | null = null;
  for (let m = fence.exec(text); m; m = fence.exec(text)) {
    if (open === null) {
      open = m.index;
    } else {
      const nl = text.indexOf("\n", m.index + m[0].length);
      regions.push({ start: open, end: nl === -1 ? text.length : nl + 1, complete: true });
      open = null;
    }
  }
  if (open !== null) regions.push({ start: open, end: text.length, complete: false });
  return regions;
}

function tableRegions(text: string, streamDone: boolean): Region[] {
  const regions: Region[] = [];
  const lines = text.split("\n");
  let offset = 0;
  let start: number | null = null;
  lines.forEach((line, i) => {
    const isTable = line.trimStart().startsWith("|");
    const isLast = i === lines.length - 1;
    if (isTable && start === null) start = offset;
    if (!isTable && start !== null) {
      regions.push({ start, end: offset, complete: true });
      start = null;
    }
    if (isTable && isLast && start !== null) {
      regions.push({ start, end: text.length, complete: streamDone });
      start = null;
    }
    offset += line.length + 1;
  });
  return regions;
}

/**
 * Largest safe prefix length near `n`: never inside a half-written code block, table or citation.
 * Reaching a complete block reveals all of it at once.
 */
export function revealBoundary(text: string, n: number, streamDone: boolean): number {
  let cut = Math.min(n, text.length);
  for (const region of [...fenceRegions(text), ...tableRegions(text, streamDone)]) {
    if (cut > region.start && (cut < region.end || !region.complete)) {
      cut = region.complete ? region.end : region.start;
    } else if (cut === region.start && region.complete && n > region.start) {
      cut = region.end;
    }
  }
  const partialCite = /\[\d{0,2}$/.exec(text.slice(0, cut));
  if (partialCite && !(streamDone && cut === text.length)) cut = partialCite.index;
  return cut;
}

export class RevealScheduler {
  private text = "";
  private shown = 0;
  private streamDone = false;
  private skipped = false;
  private clock = 0;
  /** Text end offsets with the time they arrived, so each character has a reveal deadline. */
  private arrivals: Array<{ end: number; at: number }> = [];
  private readonly opts: RevealOptions;

  constructor(opts: Partial<RevealOptions> = {}) {
    this.opts = { ...DEFAULTS, ...opts };
  }

  setText(text: string, streamDone: boolean): void {
    if (text.length > this.text.length) this.arrivals.push({ end: text.length, at: this.clock });
    this.text = text;
    this.streamDone = streamDone;
  }

  skip(): void {
    this.skipped = true;
  }

  get instant(): boolean {
    return this.skipped || this.opts.reducedMotion;
  }

  /** Advances the reveal by `dtMs` and returns the visible length. */
  tick(dtMs: number): number {
    if (this.instant) {
      this.shown = this.text.length;
      return this.shown;
    }
    this.clock += dtMs;
    const backlog = this.text.length - this.shown;
    const rate = Math.max(this.opts.baseCharsPerSecond, (backlog * 1000) / this.opts.maxLagMs);
    let overdue = 0;
    while (this.arrivals.length && this.arrivals[0]!.at + this.opts.maxLagMs <= this.clock) {
      overdue = this.arrivals.shift()!.end;
    }
    // Text that arrived more than maxLagMs ago must already be visible, whatever the smooth rate says.
    this.shown = Math.min(this.text.length, Math.max(this.shown + (rate * dtMs) / 1000, overdue));
    const cut = revealBoundary(this.text, Math.floor(this.shown), this.streamDone);
    if (cut > this.shown) this.shown = cut;
    return cut;
  }

  get visibleLength(): number {
    return this.instant ? this.text.length : revealBoundary(this.text, Math.floor(this.shown), this.streamDone);
  }

  get finished(): boolean {
    return this.streamDone && this.visibleLength >= this.text.length;
  }
}

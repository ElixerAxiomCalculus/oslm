import { describe, expect, it } from "vitest";
import { RevealScheduler, revealBoundary } from "./reveal";

describe("RevealScheduler", () => {
  it("reveals gradually at the base speed", () => {
    const s = new RevealScheduler({ baseCharsPerSecond: 100, maxLagMs: 100_000 });
    s.setText("a".repeat(1000), false);
    expect(s.tick(100)).toBe(10);
    expect(s.tick(100)).toBe(20);
  });

  it("catches up so the backlog never trails by more than maxLag", () => {
    const s = new RevealScheduler({ baseCharsPerSecond: 10, maxLagMs: 1500 });
    s.setText("x".repeat(3000), false);
    let elapsed = 0;
    while (s.visibleLength < 3000 && elapsed < 10_000) {
      s.tick(16);
      elapsed += 16;
    }
    expect(elapsed).toBeLessThanOrEqual(1500 + 16);
    const s2 = new RevealScheduler({ baseCharsPerSecond: 10, maxLagMs: 1500 });
    s2.setText("x".repeat(3000), false);
    s2.tick(1500);
    expect(s2.visibleLength).toBeGreaterThanOrEqual(2999);
  });

  it("skip shows everything immediately", () => {
    const s = new RevealScheduler();
    s.setText("hello world", true);
    s.tick(1);
    s.skip();
    expect(s.tick(1)).toBe(11);
    expect(s.finished).toBe(true);
  });

  it("reduced motion is instant", () => {
    const s = new RevealScheduler({ reducedMotion: true });
    s.setText("all at once", false);
    expect(s.tick(0)).toBe(11);
  });

  it("is finished only after the stream is done", () => {
    const s = new RevealScheduler({ reducedMotion: true });
    s.setText("abc", false);
    s.tick(10);
    expect(s.finished).toBe(false);
    s.setText("abc", true);
    expect(s.finished).toBe(true);
  });
});

describe("revealBoundary", () => {
  const code = "Intro\n```c\nint x;\n```\nAfter";

  it("holds back a half-written code block", () => {
    const partial = "Intro\n```c\nint x";
    expect(revealBoundary(partial, partial.length, false)).toBe(6);
  });

  it("reveals a complete code block as a whole unit", () => {
    expect(revealBoundary(code, 8, false)).toBe(code.indexOf("After"));
  });

  it("holds back an unfinished table until the stream moves past it", () => {
    const t = "Steps:\n| A | B |\n|---|---|\n| 1 | 2 |";
    expect(revealBoundary(t, t.length, false)).toBe(7);
    expect(revealBoundary(t, t.length, true)).toBe(t.length);
    expect(revealBoundary(`${t}\nDone`, 10, false)).toBe(t.length + 1);
  });

  it("never cuts inside a citation", () => {
    expect(revealBoundary("See [1", 6, false)).toBe(4);
    expect(revealBoundary("See [12] ok", 6, false)).toBe(4);
  });
});

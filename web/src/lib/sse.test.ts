import { describe, expect, it } from "vitest";
import { validateEvent } from "./schemas";
import { SSEParser } from "./sse";

describe("SSEParser", () => {
  it("parses events split across chunks and CRLF line endings", () => {
    const p = new SSEParser();
    expect(p.feed('event: token\r\ndata: {"te')).toEqual([]);
    const out = p.feed('xt":"hi"}\r\n\r\nevent: done\ndata: {"timings":{}}\n\n');
    expect(out).toEqual([
      { event: "token", data: '{"text":"hi"}' },
      { event: "done", data: '{"timings":{}}' },
    ]);
  });

  it("ignores comments (pings) and joins multi-line data", () => {
    const p = new SSEParser();
    expect(p.feed(": ping\n\n")).toEqual([]);
    expect(p.feed("data: a\ndata: b\n\n")).toEqual([{ event: "message", data: "a\nb" }]);
  });
});

describe("validateEvent", () => {
  it("accepts well-formed events", () => {
    expect(validateEvent("token", { text: "x" })).toEqual({ type: "token", data: { text: "x" } });
    const v = validateEvent("verification", {
      block_index: 0,
      status: "done",
      report: {
        status: "issues",
        summary: "data race (line 8)",
        tsan_clean: false,
        helgrind_clean: false,
        timed_out: false,
        findings: [{ tool: "tsan", kind: "data race", message: "m", lines: [8], threads: ["thread T1"] }],
        compiler: [],
        program_output: "",
        has_sync_bug: true,
      },
    });
    expect(v?.type).toBe("verification");
  });

  it("rejects malformed and unknown events", () => {
    expect(validateEvent("token", { text: 3 })).toBeNull();
    expect(validateEvent("sources", { sources: [{ n: "1" }] })).toBeNull();
    expect(validateEvent("mystery", {})).toBeNull();
  });
});

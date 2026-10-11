import { beforeEach, describe, expect, it, vi } from "vitest";
import { setupName } from "../lib/api";
import { currentMessages, useStore } from "./useStore";

function seed() {
  const st = useStore.getState();
  st.newSession();
  const id = useStore.getState().currentId;
  useStore.setState((s) => ({
    sessions: {
      ...s.sessions,
      [id]: {
        ...s.sessions[id]!,
        messages: [
          { id: "u1", role: "user", content: "What is a lock?", createdAt: 1, status: "done" },
          { id: "a1", role: "assistant", content: "", createdAt: 2, status: "streaming" },
        ],
      },
    },
  }));
  return id;
}

function sseResponse(events: Array<[string, unknown]>): Response {
  const body = events.map(([e, d]) => `event: ${e}\ndata: ${JSON.stringify(d)}\n\n`).join("");
  return new Response(body, { status: 200, headers: { "Content-Type": "text/event-stream" } });
}

describe("store reducers", () => {
  beforeEach(() => {
    vi.restoreAllMocks();
    useStore.setState({ streaming: false });
  });

  it("applies meta, sources, tokens, verification and done", () => {
    const sid = seed();
    const { applyEvent } = useStore.getState();
    applyEvent(sid, "a1", { type: "meta", data: { mode: "concept", setup: "ft_rag", rewritten_query: null } });
    applyEvent(sid, "a1", {
      type: "sources",
      data: {
        sources: [
          { n: 1, chunk_id: "c1", source_name: "OSTEP", title: "Locks", section: "", page_or_slide: "p.2", url: "", trust_tier: 1, score: 0.9, preview: "" },
        ],
      },
    });
    applyEvent(sid, "a1", { type: "token", data: { text: "A lock " } });
    applyEvent(sid, "a1", { type: "token", data: { text: "[1]." } });
    applyEvent(sid, "a1", { type: "verification", data: { block_index: 0, status: "pending" } });
    applyEvent(sid, "a1", { type: "done", data: { timings: { ttft_ms: 900 } } });
    const msg = currentMessages(useStore.getState())[1]!;
    expect(msg.content).toBe("A lock [1].");
    expect(msg.mode).toBe("concept");
    expect(msg.sources).toHaveLength(1);
    expect(msg.verification?.[0]?.status).toBe("pending");
    expect(msg.status).toBe("done");
    expect(useStore.getState().activeSourcesFor).toBe("a1");
  });

  it("records errors", () => {
    const sid = seed();
    useStore.getState().applyEvent(sid, "a1", { type: "error", data: { message: "cannot reach Ollama" } });
    expect(currentMessages(useStore.getState())[1]!.status).toBe("error");
  });

  it("send streams an answer and passes setup + history", async () => {
    useStore.getState().newSession();
    useStore.setState({ fineTuned: false, rag: true });
    const fetchMock = vi.spyOn(globalThis, "fetch").mockResolvedValue(
      sseResponse([
        ["meta", { mode: "concept", setup: "base_rag" }],
        ["sources", { sources: [] }],
        ["token", { text: "Hi" }],
        ["done", { timings: {} }],
      ]),
    );
    await useStore.getState().send("hello");
    const body = JSON.parse((fetchMock.mock.calls[0]![1] as RequestInit).body as string);
    expect(body.setup).toBe("base_rag");
    expect(body.history).toEqual([]);
    const msgs = currentMessages(useStore.getState());
    expect(msgs.map((m) => m.content)).toEqual(["hello", "Hi"]);
    expect(useStore.getState().streaming).toBe(false);

    fetchMock.mockResolvedValue(sseResponse([["token", { text: "Because" }], ["done", { timings: {} }]]));
    await useStore.getState().send("why?");
    const body2 = JSON.parse((fetchMock.mock.calls[1]![1] as RequestInit).body as string);
    expect(body2.history).toEqual([
      { role: "user", content: "hello" },
      { role: "assistant", content: "Hi" },
    ]);
  });

  it("network failure becomes a retryable error", async () => {
    useStore.getState().newSession();
    vi.spyOn(globalThis, "fetch").mockRejectedValue(new TypeError("Failed to fetch"));
    await useStore.getState().send("What is a mutex?");
    const answer = currentMessages(useStore.getState())[1]!;
    expect(answer.status).toBe("error");
    expect(answer.error).toMatch(/backend/);

    vi.spyOn(globalThis, "fetch").mockResolvedValue(sseResponse([["token", { text: "ok" }], ["done", { timings: {} }]]));
    await useStore.getState().retry(answer.id);
    expect(currentMessages(useStore.getState())[1]!.content).toBe("ok");
  });

  it("maps switches to setup names", () => {
    expect(setupName(true, true)).toBe("ft_rag");
    expect(setupName(false, false)).toBe("base");
  });
});

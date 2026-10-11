import { type StreamEvent, validateEvent } from "./schemas";

/** Incremental parser for the text/event-stream wire format. */
export class SSEParser {
  private buffer = "";

  feed(chunk: string): Array<{ event: string; data: string }> {
    this.buffer += chunk.replace(/\r\n?/g, "\n");
    const out: Array<{ event: string; data: string }> = [];
    let boundary = this.buffer.indexOf("\n\n");
    while (boundary !== -1) {
      const block = this.buffer.slice(0, boundary);
      this.buffer = this.buffer.slice(boundary + 2);
      const parsed = parseBlock(block);
      if (parsed) out.push(parsed);
      boundary = this.buffer.indexOf("\n\n");
    }
    return out;
  }
}

function parseBlock(block: string): { event: string; data: string } | null {
  let event = "message";
  const data: string[] = [];
  for (const line of block.split("\n")) {
    if (line.startsWith(":")) continue;
    const colon = line.indexOf(":");
    const field = colon === -1 ? line : line.slice(0, colon);
    let value = colon === -1 ? "" : line.slice(colon + 1);
    if (value.startsWith(" ")) value = value.slice(1);
    if (field === "event") event = value;
    else if (field === "data") data.push(value);
  }
  return data.length ? { event, data: data.join("\n") } : null;
}

export class HttpError extends Error {
  constructor(
    public status: number,
    message: string,
  ) {
    super(message);
  }
}

/** POSTs (or GETs) and yields validated stream events; malformed events are dropped. */
export async function* streamEvents(
  url: string,
  init: RequestInit,
  signal?: AbortSignal,
): AsyncGenerator<StreamEvent> {
  const resp = await fetch(url, { ...init, signal, headers: { Accept: "text/event-stream", ...init.headers } });
  if (!resp.ok || !resp.body) {
    let message = `Request failed (${resp.status})`;
    try {
      const body = (await resp.json()) as { error?: string; detail?: string };
      message = body.error ?? body.detail ?? message;
    } catch {
      /* non-JSON error body */
    }
    throw new HttpError(resp.status, message);
  }
  const reader = resp.body.pipeThrough(new TextDecoderStream()).getReader();
  const parser = new SSEParser();
  for (;;) {
    const { value, done } = await reader.read();
    if (done) break;
    for (const raw of parser.feed(value)) {
      let json: unknown;
      try {
        json = JSON.parse(raw.data);
      } catch {
        continue;
      }
      const ev = validateEvent(raw.event, json);
      if (ev) yield ev;
    }
  }
}

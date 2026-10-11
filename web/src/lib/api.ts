import { CheckCodeResponse, SetupInfo, SourceDetail } from "./schemas";
import { streamEvents } from "./sse";

export type SetupName = "base" | "base_rag" | "ft" | "ft_rag";

export interface HistoryTurn {
  role: "user" | "assistant";
  content: string;
}

export function setupName(fineTuned: boolean, rag: boolean): SetupName {
  return `${fineTuned ? "ft" : "base"}${rag ? "_rag" : ""}` as SetupName;
}

export function streamChat(
  body: { session_id: string; message: string; setup: SetupName; history: HistoryTurn[] },
  signal?: AbortSignal,
) {
  return streamEvents(
    "/api/chat",
    { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(body) },
    signal,
  );
}

export function streamExplanation(url: string, signal?: AbortSignal) {
  return streamEvents(url, { method: "GET" }, signal);
}

async function getJson(url: string, init?: RequestInit): Promise<unknown> {
  const resp = await fetch(url, init);
  const body: unknown = await resp.json().catch(() => ({}));
  if (!resp.ok) {
    const err = (body as { error?: string; detail?: string }) ?? {};
    throw new Error(err.error ?? err.detail ?? `Request failed (${resp.status})`);
  }
  return body;
}

export async function fetchSource(chunkId: string): Promise<SourceDetail> {
  return SourceDetail.parse(await getJson(`/api/sources/${encodeURIComponent(chunkId)}`));
}

export async function fetchSetups(): Promise<SetupInfo[]> {
  return SetupInfo.array().parse(await getJson("/api/setups"));
}

export async function checkCode(code: string, setup: SetupName) {
  return CheckCodeResponse.parse(
    await getJson("/api/check-code", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ code, setup, explain: true }),
    }),
  );
}

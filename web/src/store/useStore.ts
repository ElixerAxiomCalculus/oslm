import { create } from "zustand";
import { type HistoryTurn, setupName, streamChat } from "../lib/api";
import type { CheckReport, SourceCard, StreamEvent } from "../lib/schemas";

export type Theme = "light" | "dark";

export interface Verification {
  status: "pending" | "done";
  report?: CheckReport;
}

export interface Message {
  id: string;
  role: "user" | "assistant";
  content: string;
  createdAt: number;
  status: "streaming" | "done" | "error";
  mode?: string;
  setup?: string;
  rewrittenQuery?: string | null;
  sources?: SourceCard[];
  verification?: Record<number, Verification>;
  error?: string;
  timings?: Record<string, number | null>;
}

export interface Session {
  id: string;
  createdAt: number;
  messages: Message[];
}

interface State {
  sessions: Record<string, Session>;
  currentId: string;
  fineTuned: boolean;
  rag: boolean;
  theme: Theme;
  streaming: boolean;
  activeSourcesFor: string | null;
  highlightSource: { messageId: string; n: number } | null;
  announcement: string;
  newSession: () => void;
  setFineTuned: (v: boolean) => void;
  setRag: (v: boolean) => void;
  setTheme: (t: Theme) => void;
  showSourcesFor: (messageId: string | null) => void;
  send: (text: string) => Promise<void>;
  retry: (assistantId: string) => Promise<void>;
  stop: () => void;
  applyEvent: (sessionId: string, messageId: string, ev: StreamEvent) => void;
}

const STORAGE_KEY = "oslm:v1";
const HISTORY_TURNS = 6;

const uid = () =>
  typeof crypto !== "undefined" && "randomUUID" in crypto
    ? crypto.randomUUID()
    : `${Date.now()}-${Math.random().toString(36).slice(2)}`;

function emptySession(): Session {
  return { id: uid(), createdAt: Date.now(), messages: [] };
}

interface Persisted {
  sessions: Record<string, Session>;
  currentId: string;
  fineTuned: boolean;
  rag: boolean;
  theme: Theme | null;
}

function load(): Partial<Persisted> {
  try {
    const raw = localStorage.getItem(STORAGE_KEY);
    return raw ? (JSON.parse(raw) as Partial<Persisted>) : {};
  } catch {
    return {};
  }
}

function systemTheme(): Theme {
  try {
    return window.matchMedia("(prefers-color-scheme: dark)").matches ? "dark" : "light";
  } catch {
    return "light";
  }
}

function initialState() {
  const saved = load();
  const session = emptySession();
  const sessions = saved.sessions && Object.keys(saved.sessions).length ? saved.sessions : { [session.id]: session };
  // A reload mid-stream leaves messages flagged as streaming; they can never finish now.
  for (const s of Object.values(sessions)) {
    for (const m of s.messages) {
      if (m.status === "streaming") {
        m.status = m.content ? "done" : "error";
        if (!m.content) m.error = "This answer was interrupted.";
      }
    }
  }
  const currentId = saved.currentId && sessions[saved.currentId] ? saved.currentId : Object.keys(sessions)[0]!;
  return {
    sessions,
    currentId,
    // base_rag scored best in the evaluation, so the notebook opens with it; ft is one click away.
    fineTuned: saved.fineTuned ?? false,
    rag: saved.rag ?? true,
    theme: saved.theme ?? systemTheme(),
  };
}

let controller: AbortController | null = null;

function historyFor(messages: Message[]): HistoryTurn[] {
  return messages
    .filter((m) => m.status === "done" && m.content.trim())
    .slice(-HISTORY_TURNS)
    .map((m) => ({ role: m.role, content: m.content }));
}

export const useStore = create<State>((set, get) => ({
  ...initialState(),
  streaming: false,
  activeSourcesFor: null,
  highlightSource: null,
  announcement: "",

  newSession: () => {
    get().stop();
    const s = emptySession();
    set((st) => ({ sessions: { ...st.sessions, [s.id]: s }, currentId: s.id, activeSourcesFor: null }));
  },
  setFineTuned: (v) => set({ fineTuned: v }),
  setRag: (v) => set({ rag: v }),
  setTheme: (t) => set({ theme: t }),
  showSourcesFor: (id) => set({ activeSourcesFor: id }),

  stop: () => {
    controller?.abort();
    controller = null;
  },

  applyEvent: (sessionId, messageId, ev) =>
    set((st) => {
      const session = st.sessions[sessionId];
      if (!session) return {};
      const messages = session.messages.map((m) => {
        if (m.id !== messageId) return m;
        switch (ev.type) {
          case "meta":
            return { ...m, mode: ev.data.mode, setup: ev.data.setup, rewrittenQuery: ev.data.rewritten_query ?? null };
          case "sources":
            return { ...m, sources: ev.data.sources };
          case "token":
            return { ...m, content: m.content + ev.data.text };
          case "verification":
            return {
              ...m,
              verification: {
                ...m.verification,
                [ev.data.block_index]: { status: ev.data.status, report: ev.data.report },
              },
            };
          case "error":
            return { ...m, status: "error" as const, error: ev.data.message };
          case "done":
            return { ...m, status: "done" as const, timings: ev.data.timings };
        }
        return m;
      });
      const patch: Partial<State> = { sessions: { ...st.sessions, [sessionId]: { ...session, messages } } };
      // Every answer claims the panel, even with no sources, so a base-model answer never shows stale cards.
      if (ev.type === "sources") patch.activeSourcesFor = messageId;
      return patch;
    }),

  send: async (text) => {
    const trimmed = text.replace(/\s+$/, "");
    if (get().streaming) return;
    const { currentId, sessions, fineTuned, rag } = get();
    const session = sessions[currentId];
    if (!session) return;
    const history = historyFor(session.messages);
    const user: Message = { id: uid(), role: "user", content: trimmed, createdAt: Date.now(), status: "done" };
    const assistant: Message = { id: uid(), role: "assistant", content: "", createdAt: Date.now(), status: "streaming" };
    set((st) => ({
      streaming: true,
      announcement: "",
      sessions: {
        ...st.sessions,
        [currentId]: { ...session, messages: [...session.messages, user, assistant] },
      },
    }));
    await runStream(currentId, assistant.id, trimmed, history, setupName(fineTuned, rag));
  },

  retry: async (assistantId) => {
    if (get().streaming) return;
    const { currentId, sessions, fineTuned, rag } = get();
    const session = sessions[currentId];
    if (!session) return;
    const idx = session.messages.findIndex((m) => m.id === assistantId);
    const question = session.messages[idx - 1];
    if (idx < 1 || !question) return;
    const history = historyFor(session.messages.slice(0, idx - 1));
    const reset: Message = { ...session.messages[idx]!, content: "", status: "streaming", error: undefined, sources: [], verification: {} };
    const messages = [...session.messages];
    messages[idx] = reset;
    set((st) => ({ streaming: true, sessions: { ...st.sessions, [currentId]: { ...session, messages } } }));
    await runStream(currentId, assistantId, question.content, history, setupName(fineTuned, rag));
  },
}));

async function runStream(
  sessionId: string,
  messageId: string,
  message: string,
  history: HistoryTurn[],
  setup: ReturnType<typeof setupName>,
): Promise<void> {
  const { applyEvent } = useStore.getState();
  controller = new AbortController();
  let finished = false;
  try {
    for await (const ev of streamChat({ session_id: sessionId, message, setup, history }, controller.signal)) {
      applyEvent(sessionId, messageId, ev);
      if (ev.type === "done" || ev.type === "error") finished = true;
    }
    if (!finished) {
      applyEvent(sessionId, messageId, { type: "error", data: { message: "The answer stopped unexpectedly." } });
    }
  } catch (err) {
    const aborted = err instanceof DOMException && err.name === "AbortError";
    const message = aborted
      ? "Stopped."
      : err instanceof TypeError
        ? "couldn't reach the tutor — is the backend running?"
        : err instanceof Error
          ? err.message
          : String(err);
    applyEvent(sessionId, messageId, { type: "error", data: { message } });
  } finally {
    controller = null;
    useStore.setState({ streaming: false });
  }
}

useStore.subscribe((st) => {
  try {
    const persisted: Persisted = {
      sessions: st.sessions,
      currentId: st.currentId,
      fineTuned: st.fineTuned,
      rag: st.rag,
      theme: st.theme,
    };
    localStorage.setItem(STORAGE_KEY, JSON.stringify(persisted));
  } catch {
    /* storage full or blocked: history just won't persist */
  }
});

export function currentMessages(st: Pick<State, "sessions" | "currentId">): Message[] {
  return st.sessions[st.currentId]?.messages ?? [];
}

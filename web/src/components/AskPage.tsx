import { useEffect, useLayoutEffect, useRef, useState } from "react";
import type { Message } from "../store/useStore";
import { currentMessages, useStore } from "../store/useStore";
import { HandwrittenAnswer } from "./HandwrittenAnswer";
import { SketchButton, SketchUnderline } from "./Sketch";

const SUGGESTIONS = [
  "What is a race condition?",
  "Explain Peterson's solution",
  "Show step by step how count++ loses an update",
  "Producer–consumer with semaphores in C",
  "Why must pthread_cond_wait sit inside a while loop?",
];

function friendlyError(raw: string): string {
  if (/ollama|11434|model .*not found/i.test(raw)) return "couldn't reach the model — is Ollama running?";
  if (/backend|fetch|network/i.test(raw)) return "couldn't reach the tutor — is the backend running (oslm serve)?";
  return raw;
}

function dateStamp(ts: number): string {
  const d = new Date(ts);
  return `${d.toLocaleDateString(undefined, { day: "numeric", month: "short" })}\n${d.toLocaleTimeString(undefined, { hour: "2-digit", minute: "2-digit" })}`;
}

function Entry({ question, answer }: { question: Message; answer?: Message }) {
  const retry = useStore((s) => s.retry);
  const streaming = useStore((s) => s.streaming);
  const fresh = Date.now() - question.createdAt < 2000;
  return (
    <article className="entry" aria-label="Question and answer">
      <time className="stamp-date" dateTime={new Date(question.createdAt).toISOString()} style={{ whiteSpace: "pre-line" }}>
        {dateStamp(question.createdAt)}
      </time>
      <p className={`question${fresh ? " settle" : ""}`}>{question.content}</p>
      {answer && (answer.content || answer.status !== "error") && <HandwrittenAnswer message={answer} />}
      {answer?.status === "error" && (
        <div className="margin-note" role="alert">
          <span className="note-text">{friendlyError(answer.error ?? "something went wrong")}</span>
          <SketchButton seed={`retry-${answer.id}`} onClick={() => void retry(answer.id)} disabled={streaming}>
            try again
          </SketchButton>
        </div>
      )}
    </article>
  );
}

function Welcome({ onPick }: { onPick: (q: string) => void }) {
  return (
    <section className="welcome" aria-label="Getting started">
      <p>
        Ask me anything about <SketchUnderline seed="ps">process synchronization</SketchUnderline> — race conditions,
        locks, semaphores, monitors, deadlock, or the classic problems. Paste C code and I'll hunt for the bug.
      </p>
      <div className="suggestions">
        {SUGGESTIONS.map((s) => (
          <SketchButton key={s} seed={s} className="suggestion" onClick={() => onPick(s)}>
            {s}
          </SketchButton>
        ))}
      </div>
    </section>
  );
}

function Composer() {
  const send = useStore((s) => s.send);
  const stop = useStore((s) => s.stop);
  const streaming = useStore((s) => s.streaming);
  const [text, setText] = useState("");
  const ref = useRef<HTMLTextAreaElement>(null);

  useLayoutEffect(() => {
    const el = ref.current;
    if (!el) return;
    el.style.height = "auto";
    el.style.height = `${el.scrollHeight}px`;
  }, [text]);

  useEffect(() => {
    const onPick = (e: Event) => {
      const q = (e as CustomEvent<string>).detail;
      void send(q);
    };
    window.addEventListener("oslm:ask", onPick);
    return () => window.removeEventListener("oslm:ask", onPick);
  }, [send]);

  const submit = () => {
    if (streaming) return;
    const value = text;
    if (!value.trim() && value.length === 0) return;
    setText("");
    void send(value);
  };

  return (
    <form
      className="composer"
      onSubmit={(e) => {
        e.preventDefault();
        submit();
      }}
    >
      <label htmlFor="ask" className="sr-only">
        Ask a question about process synchronization
      </label>
      <textarea
        id="ask"
        ref={ref}
        rows={1}
        value={text}
        maxLength={8000}
        placeholder="Write your question on the next line…"
        onChange={(e) => setText(e.target.value)}
        onKeyDown={(e) => {
          if (e.key === "Enter" && !e.shiftKey && !e.nativeEvent.isComposing) {
            e.preventDefault();
            submit();
          }
        }}
      />
      <svg className="pen-cursor" viewBox="0 0 24 24" aria-hidden="true">
        <path d="M3 21l1.5-5L16 4.5a2.1 2.1 0 0 1 3 0l.5.5a2.1 2.1 0 0 1 0 3L8 19.5z" fill="none" stroke="currentColor" strokeWidth="1.6" strokeLinejoin="round" />
      </svg>
      {streaming ? (
        <SketchButton seed="stop" onClick={stop}>
          stop
        </SketchButton>
      ) : (
        <SketchButton seed="send" type="submit" disabled={!text.length}>
          ask
        </SketchButton>
      )}
      <span className="hint">Enter to ask · Shift+Enter for a new line</span>
    </form>
  );
}

export function AskPage({ turning }: { turning: boolean }) {
  const messages = useStore(currentMessages);
  const send = useStore((s) => s.send);
  const scrollRef = useRef<HTMLDivElement>(null);
  const entriesRef = useRef<HTMLDivElement>(null);
  const pinned = useRef(true);
  const toBottom = () => {
    const el = scrollRef.current;
    if (el) el.scrollTo({ top: el.scrollHeight, behavior: "instant" as ScrollBehavior });
  };

  // A newly asked question always comes into view.
  useEffect(() => {
    pinned.current = true;
    toBottom();
  }, [messages.length]);

  // Follow the answer as it is written (its height grows with the reveal, not with the stream), unless the
  // student has scrolled up to reread something.
  useEffect(() => {
    const el = scrollRef.current;
    const body = entriesRef.current;
    if (!el || !body || typeof ResizeObserver === "undefined") return;
    const onScroll = () => {
      pinned.current = el.scrollHeight - el.scrollTop - el.clientHeight < 120;
    };
    const ro = new ResizeObserver(() => {
      if (pinned.current) toBottom();
    });
    el.addEventListener("scroll", onScroll, { passive: true });
    ro.observe(body);
    return () => {
      el.removeEventListener("scroll", onScroll);
      ro.disconnect();
    };
  }, []);

  const pairs: Array<{ q: Message; a?: Message }> = [];
  messages.forEach((m, i) => {
    if (m.role === "user") pairs.push({ q: m, a: messages[i + 1]?.role === "assistant" ? messages[i + 1] : undefined });
  });

  return (
    <>
      <div ref={scrollRef} className={`scroll ruled${turning ? " page-turn" : ""}`} data-testid="page">
        <div className="entries" ref={entriesRef}>
          {pairs.length === 0 ? <Welcome onPick={(q) => void send(q)} /> : pairs.map((p) => <Entry key={p.q.id} question={p.q} answer={p.a} />)}
        </div>
      </div>
      <Composer />
    </>
  );
}

import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { RevealScheduler } from "../lib/reveal";
import type { Message } from "../store/useStore";
import { useStore } from "../store/useStore";
import { extractPrograms, MarkdownView } from "./MarkdownView";
import { useReducedMotion } from "./Sketch";

const INK_SETTLE_MS = 380;

export function HandwrittenAnswer({ message }: { message: Message }) {
  const reduced = useReducedMotion();
  const showSourcesFor = useStore((s) => s.showSourcesFor);
  const streamDone = message.status !== "streaming";
  // Answers restored from history are shown whole; only live answers are "written".
  const [instantAtMount] = useState(() => streamDone);
  const schedulerRef = useRef<RevealScheduler>();
  schedulerRef.current ??= new RevealScheduler({ reducedMotion: reduced || instantAtMount });
  const scheduler = schedulerRef.current;
  if (reduced) scheduler.skip();

  const [visible, setVisible] = useState(() => (instantAtMount ? message.content.length : 0));
  const [inkFrom, setInkFrom] = useState(0);
  const wordCount = useRef(0);
  const announced = useRef(instantAtMount);

  scheduler.setText(message.content, streamDone);

  useEffect(() => {
    let raf = 0;
    let last = performance.now();
    const loop = (now: number) => {
      const len = scheduler.tick(now - last);
      last = now;
      setVisible((v) => (v === len ? v : len));
      if (!scheduler.finished) raf = requestAnimationFrame(loop);
    };
    raf = requestAnimationFrame(loop);
    return () => cancelAnimationFrame(raf);
  }, [scheduler, message.content, streamDone]);

  useEffect(() => {
    const t = setTimeout(() => setInkFrom(wordCount.current), INK_SETTLE_MS);
    return () => clearTimeout(t);
  }, [visible]);

  const finished = streamDone && visible >= message.content.length;
  useEffect(() => {
    if (finished && !announced.current && message.status === "done") {
      announced.current = true;
      useStore.setState({ announcement: "The tutor finished answering." });
    }
  }, [finished, message.status]);

  const skip = useCallback(() => {
    scheduler.skip();
    setVisible(message.content.length);
    setInkFrom(Number.MAX_SAFE_INTEGER);
  }, [scheduler, message.content.length]);

  const onCite = useCallback(
    (n: number) => {
      showSourcesFor(message.id);
      useStore.setState({ highlightSource: { messageId: message.id, n } });
    },
    [message.id, showSourcesFor],
  );

  const programs = useMemo(() => extractPrograms(message.content), [message.content]);
  const animating = !finished && !scheduler.instant;
  const text = message.content.slice(0, visible);

  return (
    <div
      className={`answer${animating ? " animating" : ""}`}
      onClick={animating ? skip : undefined}
      onKeyDown={animating ? (e) => e.key === "Escape" && skip() : undefined}
      title={animating ? "Click to finish writing" : undefined}
      aria-busy={!finished}
    >
      {message.rewrittenQuery && (
        <div className="answer-meta">
          (reading this as: <em>{message.rewrittenQuery}</em>)
        </div>
      )}
      <MarkdownView
        text={text}
        maxCitation={message.sources?.length ?? 0}
        inkFrom={scheduler.instant ? Number.MAX_SAFE_INTEGER : inkFrom}
        pen={animating || (message.status === "streaming" && !text)}
        animate={!scheduler.instant}
        programs={programs}
        verification={message.verification}
        onCite={onCite}
        onWordCount={(n) => {
          wordCount.current = n;
        }}
      />
      {message.status === "streaming" && !text && (
        <p className="answer-meta" aria-live="off">
          thinking<span className="pen-anchor" />
        </p>
      )}
    </div>
  );
}

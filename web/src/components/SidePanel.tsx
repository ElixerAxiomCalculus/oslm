import * as Dialog from "@radix-ui/react-dialog";
import * as Tooltip from "@radix-ui/react-tooltip";
import { useEffect, useState } from "react";
import { fetchSetups, fetchSource, setupName } from "../lib/api";
import type { SetupInfo, SourceCard, SourceDetail } from "../lib/schemas";
import { currentMessages, useStore } from "../store/useStore";
import { SketchButton, SketchCheck } from "./Sketch";

function SourceDialog({ card, onClose }: { card: SourceCard | null; onClose: () => void }) {
  const [detail, setDetail] = useState<SourceDetail | null>(null);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    setDetail(null);
    setError(null);
    if (!card) return;
    let alive = true;
    fetchSource(card.chunk_id)
      .then((d) => alive && setDetail(d))
      .catch((e: unknown) => alive && setError(e instanceof Error ? e.message : String(e)));
    return () => {
      alive = false;
    };
  }, [card]);

  return (
    <Dialog.Root open={card !== null} onOpenChange={(open) => !open && onClose()}>
      <Dialog.Portal>
        <Dialog.Overlay className="dialog-overlay" />
        <Dialog.Content className="torn-page ruled" aria-describedby={undefined}>
          <Dialog.Title asChild>
            <h3>
              [{card?.n}] {card?.source_name}
            </h3>
          </Dialog.Title>
          <p className="setup-name">
            {[card?.title, card?.section, card?.page_or_slide].filter(Boolean).join(" · ")}
          </p>
          {card?.url.startsWith("http") && (
            <p>
              <a href={card.url} target="_blank" rel="noreferrer noopener">
                open original ↗
              </a>
            </p>
          )}
          {error && <p className="margin-note">{error}</p>}
          <div className="chunk-text" aria-busy={!detail && !error}>
            {detail ? detail.text : card ? `${card.preview}…` : ""}
          </div>
          <Dialog.Close asChild>
            <SketchButton seed="close-dialog" className="icon-btn close" aria-label="Close">
              ✕
            </SketchButton>
          </Dialog.Close>
        </Dialog.Content>
      </Dialog.Portal>
    </Dialog.Root>
  );
}

function Sources() {
  const messages = useStore(currentMessages);
  const activeId = useStore((s) => s.activeSourcesFor);
  const highlight = useStore((s) => s.highlightSource);
  const [open, setOpen] = useState<SourceCard | null>(null);

  const target =
    messages.find((m) => m.id === activeId) ??
    [...messages].reverse().find((m) => m.role === "assistant" && (m.sources?.length ?? 0) > 0);
  const sources = target?.sources ?? [];

  useEffect(() => {
    if (!highlight || highlight.messageId !== target?.id) return;
    document.getElementById(`source-card-${highlight.n}`)?.scrollIntoView({ block: "nearest", behavior: "smooth" });
  }, [highlight, target?.id]);

  return (
    <section aria-labelledby="sources-h">
      <h2 id="sources-h">Sources</h2>
      {sources.length === 0 ? (
        <p className="empty-note">
          {target ? "No notes were used for this answer." : "Sources for each answer will be pinned here."}
        </p>
      ) : (
        sources.map((s, i) => (
          <button
            key={s.chunk_id}
            id={`source-card-${s.n}`}
            type="button"
            className={`index-card${highlight?.messageId === target?.id && highlight?.n === s.n ? " highlight" : ""}`}
            style={{ transform: `rotate(${(i % 2 ? 0.5 : -0.4).toFixed(1)}deg)` }}
            onClick={() => setOpen(s)}
          >
            <span className="num">{s.n}</span>
            <span className="src">{s.source_name}</span>
            <span className="loc">{[s.section, s.page_or_slide].filter(Boolean).join(" · ")}</span>
            <span className="preview">{s.preview}</span>
          </button>
        ))
      )}
      <SourceDialog card={open} onClose={() => setOpen(null)} />
    </section>
  );
}

function SetupSwitches() {
  const fineTuned = useStore((s) => s.fineTuned);
  const rag = useStore((s) => s.rag);
  const setFineTuned = useStore((s) => s.setFineTuned);
  const setRag = useStore((s) => s.setRag);
  const streaming = useStore((s) => s.streaming);
  const [setups, setSetups] = useState<SetupInfo[] | null>(null);

  useEffect(() => {
    fetchSetups()
      .then(setSetups)
      .catch(() => setSetups(null));
  }, []);

  const current = setups?.find((s) => s.name === setupName(fineTuned, rag));
  const ftAvailable = setups ? setups.some((s) => s.fine_tuned && s.available) : true;

  return (
    <section aria-labelledby="setup-h" className="sticky" style={{ ["--tilt" as string]: "-0.8deg" }}>
      <h2 id="setup-h">Setup</h2>
      <Tooltip.Provider delayDuration={300}>
        <Tooltip.Root>
          <Tooltip.Trigger asChild>
            <div>
              <SketchCheck id="ft-switch" label="Fine-tuned model" checked={fineTuned} onCheckedChange={setFineTuned} disabled={streaming} />
            </div>
          </Tooltip.Trigger>
          <Tooltip.Portal>
            <Tooltip.Content className="sticky" sideOffset={6} style={{ fontSize: 14, lineHeight: "20px", maxWidth: 240, zIndex: 30 }}>
              {ftAvailable ? "QLoRA fine-tuned Qwen3-4B (teacher style)" : "The fine-tuned model isn't registered in Ollama yet."}
            </Tooltip.Content>
          </Tooltip.Portal>
        </Tooltip.Root>
      </Tooltip.Provider>
      <SketchCheck id="rag-switch" label="Use course notes (RAG)" checked={rag} onCheckedChange={setRag} disabled={streaming} />
      <p className="setup-name" data-testid="setup-name">
        setup: <strong>{setupName(fineTuned, rag)}</strong>
        {current && !current.available && " — model not available"}
      </p>
    </section>
  );
}

function isNarrow(): boolean {
  try {
    return window.matchMedia("(max-width: 860px)").matches;
  } catch {
    return false;
  }
}

export function SidePanel() {
  // Desktop starts with the notes open; on phones the panel is a bottom sheet that starts closed.
  const [open, setOpen] = useState(() => !isNarrow());
  return (
    <aside className={`panel ${open ? "open" : "collapsed"}`} aria-label="Sticky notes">
      <SketchButton
        seed="panel-toggle"
        className="icon-btn panel-toggle"
        aria-expanded={open}
        aria-controls="panel-body"
        onClick={() => setOpen((o) => !o)}
      >
        {open ? "hide" : "notes"}
      </SketchButton>
      <div id="panel-body">
        <SetupSwitches />
        <Sources />
      </div>
    </aside>
  );
}

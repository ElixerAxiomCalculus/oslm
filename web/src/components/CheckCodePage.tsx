import { useRef, useState } from "react";
import { checkCode, setupName, streamExplanation } from "../lib/api";
import type { CheckReport } from "../lib/schemas";
import type { Message } from "../store/useStore";
import { useStore } from "../store/useStore";
import { verificationStamps } from "./CodeBlock";
import { HandwrittenAnswer } from "./HandwrittenAnswer";

const STARTER = `#include <pthread.h>
#include <stdio.h>

static long counter;

static void *worker(void *arg) {
    (void)arg;
    for (int i = 0; i < 100000; i++)
        counter++;
    return NULL;
}

int main(void) {
    pthread_t a, b;
    pthread_create(&a, NULL, worker, NULL);
    pthread_create(&b, NULL, worker, NULL);
    pthread_join(a, NULL);
    pthread_join(b, NULL);
    printf("counter = %ld\\n", counter);
    return 0;
}
`;

function reportStamps(r: CheckReport) {
  if (r.status === "checker_unavailable") return [{ text: "checker offline", bad: false, pending: true }];
  if (r.status === "compile_error") return [{ text: "✗ doesn't compile", bad: true }];
  if (r.status === "rejected") return [{ text: "✗ rejected", bad: true }];
  const base = verificationStamps({ status: "done", report: r });
  if (r.status === "issues" && !r.has_sync_bug) {
    return [...base, ...r.findings.filter((f) => f.tool === "runtime").map((f) => ({ text: `! ${f.kind.replace("_", " ")}`, bad: true }))];
  }
  return base;
}

export function CheckCodePage() {
  const fineTuned = useStore((s) => s.fineTuned);
  const rag = useStore((s) => s.rag);
  const [code, setCode] = useState(STARTER);
  const [busy, setBusy] = useState(false);
  const [report, setReport] = useState<CheckReport | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [explanation, setExplanation] = useState<Message | null>(null);
  const abortRef = useRef<AbortController | null>(null);
  const gutterRef = useRef<HTMLPreElement>(null);
  const escaped = useRef(false);

  const run = async () => {
    abortRef.current?.abort();
    setBusy(true);
    setError(null);
    setReport(null);
    setExplanation(null);
    try {
      const res = await checkCode(code, setupName(fineTuned, rag));
      setReport(res.report);
      if (res.explain_url) {
        const msg: Message = { id: `explain-${Date.now()}`, role: "assistant", content: "", createdAt: Date.now(), status: "streaming" };
        setExplanation(msg);
        abortRef.current = new AbortController();
        for await (const ev of streamExplanation(res.explain_url, abortRef.current.signal)) {
          setExplanation((m) => {
            if (!m) return m;
            if (ev.type === "token") return { ...m, content: m.content + ev.data.text };
            if (ev.type === "sources") return { ...m, sources: ev.data.sources };
            if (ev.type === "error") return { ...m, status: "error", error: ev.data.message };
            if (ev.type === "done") return { ...m, status: "done" };
            return m;
          });
        }
        setExplanation((m) => (m && m.status === "streaming" ? { ...m, status: "done" } : m));
      }
    } catch (e) {
      if (!(e instanceof DOMException && e.name === "AbortError")) {
        setError(e instanceof TypeError ? "couldn't reach the tutor — is the backend running?" : e instanceof Error ? e.message : String(e));
      }
    } finally {
      setBusy(false);
    }
  };

  return (
    <div className="scroll ruled">
      <div className="lab-sheet">
        <h2>Lab sheet: check my code</h2>
        <p>Paste a C program. It is compiled and run under ThreadSanitizer and Helgrind inside a sandbox, then the tutor explains the result.</p>
        <label htmlFor="code" className="sr-only">
          C source code
        </label>
        <div className="code-editor-wrap">
          <pre className="gutter" ref={gutterRef} aria-hidden="true">
            {Array.from({ length: Math.max(1, code.split("\n").length) }, (_, i) => i + 1).join("\n")}
          </pre>
          <textarea
            id="code"
            className="code-editor"
            spellCheck={false}
            value={code}
            maxLength={20480}
            aria-describedby="code-hint"
            onChange={(e) => setCode(e.target.value)}
            onScroll={(e) => {
              if (gutterRef.current) gutterRef.current.scrollTop = e.currentTarget.scrollTop;
            }}
            onKeyDown={(e) => {
              if (e.key === "Escape") {
                escaped.current = true;
                return;
              }
              // Tab indents, but Esc then Tab leaves the editor so keyboard users are never trapped.
              if (e.key === "Tab" && !e.shiftKey && !escaped.current) {
                e.preventDefault();
                const el = e.currentTarget;
                const { selectionStart: s, selectionEnd: end } = el;
                setCode(`${code.slice(0, s)}    ${code.slice(end)}`);
                requestAnimationFrame(() => el.setSelectionRange(s + 4, s + 4));
              }
              escaped.current = false;
            }}
          />
        </div>
        <p id="code-hint" className="editor-hint">
          Tab indents · Esc then Tab moves on · line numbers match the stamps
        </p>
        <button type="button" className="run-stamp" onClick={() => void run()} disabled={busy || !code.trim()}>
          {busy ? "checking…" : "Run checks"}
        </button>

        {error && (
          <div className="margin-note" role="alert">
            <span className="note-text">{error}</span>
          </div>
        )}

        {report && (
          <section aria-label="Checker report" data-testid="report">
            <div className="report-stamps" role="status">
              {reportStamps(report).map((s) => (
                <span key={s.text} className={`stamp${s.bad ? " bad" : ""}${"pending" in s && s.pending ? " pending" : ""}`}>
                  {s.text}
                </span>
              ))}
            </div>
            <p>
              <strong>Summary:</strong> {report.summary}
            </p>
            {report.findings.length > 0 && (
              <ul className="findings">
                {report.findings.map((f, i) => (
                  <li key={`${f.tool}-${f.kind}-${i}`}>
                    <strong>{f.tool === "tsan" ? "TSan" : f.tool === "helgrind" ? "Helgrind" : f.tool}:</strong> {f.kind}
                    {f.lines.length > 0 && ` — line${f.lines.length > 1 ? "s" : ""} ${f.lines.join(", ")}`}
                    {f.threads.length > 0 && ` (${f.threads.join(", ")})`}
                  </li>
                ))}
              </ul>
            )}
            {report.compiler.filter((c) => c.kind === "error").length > 0 && (
              <ul className="findings">
                {report.compiler.map((c, i) => (
                  <li key={i}>
                    line {c.lines[0]}: {c.message}
                  </li>
                ))}
              </ul>
            )}
            {report.program_output && (
              <figure className="printout" style={{ ["--tilt" as string]: "0.3deg" }}>
                <span className="lang">output</span>
                <pre>
                  <code>{report.program_output}</code>
                </pre>
              </figure>
            )}
          </section>
        )}

        {explanation && (
          <section aria-label="Tutor explanation">
            <h2>What the tutor says</h2>
            {explanation.status === "error" ? (
              <div className="margin-note" role="alert">
                <span className="note-text">{explanation.error}</span>
              </div>
            ) : (
              <HandwrittenAnswer message={explanation} />
            )}
          </section>
        )}
      </div>
    </div>
  );
}

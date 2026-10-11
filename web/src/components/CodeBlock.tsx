import { useEffect, useMemo, useState } from "react";
import { highlight } from "../lib/highlight";
import { hashString } from "../lib/sketch";
import type { Verification } from "../store/useStore";

export function verificationStamps(v: Verification | undefined): Array<{ text: string; bad: boolean; pending?: boolean }> {
  if (!v) return [];
  if (v.status === "pending") return [{ text: "checking…", bad: false, pending: true }];
  const r = v.report;
  if (!r) return [];
  if (r.status === "checker_unavailable") return [{ text: "checker offline", bad: false, pending: true }];
  if (r.status === "compile_error") return [{ text: "✗ doesn't compile", bad: true }];
  if (r.status === "rejected" || r.status === "checker_error") return [];
  const stamps: Array<{ text: string; bad: boolean }> = [];
  const race = r.findings.find((f) => f.tool !== "runtime" && f.tool !== "compiler");
  if (race) {
    // TSan builds at -O1 can attribute a race to the loop header; Helgrind's -O0 build names the exact statement.
    const precise = r.findings.find((f) => f.tool === "helgrind" && f.kind === race.kind && f.lines.length);
    const line = (precise ?? race).lines[0];
    stamps.push({ text: `✗ ${race.kind}${line ? ` (line ${line})` : ""}`, bad: true });
  } else if (r.timed_out) {
    stamps.push({ text: "✗ hangs — possible deadlock", bad: true });
  } else {
    if (r.tsan_clean) stamps.push({ text: "✓ TSan clean", bad: false });
    if (r.helgrind_clean) stamps.push({ text: "✓ Helgrind clean", bad: false });
  }
  return stamps;
}

export function CodeBlock({
  code,
  lang,
  verification,
  animate,
}: {
  code: string;
  lang?: string;
  verification?: Verification;
  animate: boolean;
}) {
  const [html, setHtml] = useState<string | null>(null);
  const [copied, setCopied] = useState(false);
  const tilt = useMemo(() => ((hashString(code.slice(0, 64)) % 100) / 100 - 0.5) * 1.1, [code]);

  useEffect(() => {
    let alive = true;
    void highlight(code, lang).then((h) => {
      if (alive) setHtml(h);
    });
    return () => {
      alive = false;
    };
  }, [code, lang]);

  const copy = async () => {
    try {
      await navigator.clipboard.writeText(code);
      setCopied(true);
      setTimeout(() => setCopied(false), 1500);
    } catch {
      /* clipboard blocked */
    }
  };

  const stamps = verificationStamps(verification);
  return (
    <figure
      className={`printout${animate ? " pasted" : ""}`}
      style={{ ["--tilt" as string]: `${tilt.toFixed(2)}deg` }}
      aria-label={lang ? `${lang} code` : "code"}
    >
      {stamps.length > 0 && (
        <div className="stamps" role="status">
          {stamps.map((s) => (
            <span key={s.text} className={`stamp${s.bad ? " bad" : ""}${s.pending ? " pending" : ""}`}>
              {s.text}
            </span>
          ))}
        </div>
      )}
      {lang && <span className="lang">{lang}</span>}
      {html ? (
        <div dangerouslySetInnerHTML={{ __html: html }} />
      ) : (
        <pre>
          <code>{code}</code>
        </pre>
      )}
      <button type="button" className="copy" onClick={copy}>
        {copied ? "copied!" : "copy"}
      </button>
    </figure>
  );
}

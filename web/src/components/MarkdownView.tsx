import type { Element, ElementContent } from "hast";
import { createContext, memo, type ReactNode, useContext, useMemo } from "react";
import ReactMarkdown, { type Components } from "react-markdown";
import remarkGfm from "remark-gfm";
import { rehypeInk } from "../lib/rehypeInk";
import type { Verification } from "../store/useStore";
import { CodeBlock } from "./CodeBlock";

function hastText(node: ElementContent | Element): string {
  if (node.type === "text") return node.value;
  if (node.type === "element") return node.children.map(hastText).join("");
  return "";
}

const PROGRAM = /\bint\s+main\s*\(/;

/** Index of a code block among the complete C programs, matching the backend's verification numbering. */
export function programIndex(code: string, programs: string[]): number {
  return programs.findIndex((p) => p.trim() === code.trim());
}

export function extractPrograms(markdown: string): string[] {
  const out: string[] = [];
  const re = /```(?:c|C)?[ \t]*\n([\s\S]*?)```/g;
  for (let m = re.exec(markdown); m; m = re.exec(markdown)) {
    if (PROGRAM.test(m[1]!)) out.push(m[1]!);
  }
  return out;
}

interface RenderCtx {
  animate: boolean;
  programs: string[];
  verification?: Record<number, Verification>;
  onCite?: (n: number) => void;
}

const Ctx = createContext<RenderCtx>({ animate: false, programs: [] });

function Pre({ node }: { node?: Element }) {
  const { animate, programs, verification } = useContext(Ctx);
  const codeEl = node?.children.find((c): c is Element => c.type === "element" && c.tagName === "code");
  if (!codeEl) return null;
  const classes = (codeEl.properties?.className as string[] | undefined) ?? [];
  const lang = classes.find((c) => c.startsWith("language-"))?.slice(9);
  const code = hastText(codeEl).replace(/\n$/, "");
  const idx = programIndex(code, programs);
  return <CodeBlock code={code} lang={lang} animate={animate} verification={idx >= 0 ? verification?.[idx] : undefined} />;
}

function Table({ children }: { children?: ReactNode }) {
  const { animate } = useContext(Ctx);
  return (
    <div className={`table-wrap${animate ? " pasted" : ""}`}>
      <table>{children}</table>
    </div>
  );
}

function Sup({ node, children }: { node?: Element; children?: ReactNode }) {
  const { onCite } = useContext(Ctx);
  const n = Number((node?.properties as { dataN?: string } | undefined)?.dataN);
  if (!n) return <sup>{children}</sup>;
  return (
    <sup className="cite">
      <button type="button" aria-label={`Source ${n}`} onClick={() => onCite?.(n)}>
        {n}
      </button>
    </sup>
  );
}

function Anchor({ href, children }: { href?: string; children?: ReactNode }) {
  return (
    <a href={href} target="_blank" rel="noreferrer noopener">
      {children}
    </a>
  );
}

// Module-level components keep a stable identity across streamed re-renders, so React updates tables and code
// cards in place instead of remounting them (which restarted their "pasted in" animation on every token).
const COMPONENTS = { pre: Pre, table: Table, sup: Sup, a: Anchor } as unknown as Components;

interface Props extends RenderCtx {
  text: string;
  maxCitation: number;
  inkFrom: number;
  pen: boolean;
  onWordCount?: (n: number) => void;
}

function MarkdownViewImpl({ text, maxCitation, inkFrom, pen, animate, programs, verification, onCite, onWordCount }: Props) {
  const ctx = useMemo(() => ({ animate, programs, verification, onCite }), [animate, programs, verification, onCite]);
  return (
    <Ctx.Provider value={ctx}>
      <div className="md">
        <ReactMarkdown
          remarkPlugins={[remarkGfm]}
          rehypePlugins={[[rehypeInk, { inkFrom, maxCitation, pen, onCount: onWordCount }]]}
          components={COMPONENTS}
        >
          {text}
        </ReactMarkdown>
      </div>
    </Ctx.Provider>
  );
}

export const MarkdownView = memo(MarkdownViewImpl);

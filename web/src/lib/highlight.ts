import type { HighlighterCore } from "shiki/core";

const LANG_ALIASES: Record<string, string> = { h: "c", cpp: "c", "c++": "c", sh: "bash", shell: "bash", py: "python" };
const SUPPORTED = new Set(["c", "diff", "java", "python", "bash"]);

let highlighter: Promise<HighlighterCore> | null = null;

function load(): Promise<HighlighterCore> {
  highlighter ??= (async () => {
    const [{ createHighlighterCore, createCssVariablesTheme }, { createJavaScriptRegexEngine }] = await Promise.all([
      import("shiki/core"),
      import("shiki/engine/javascript"),
    ]);
    // CSS-variable theme lets the light/chalkboard palettes restyle code without re-highlighting.
    const theme = createCssVariablesTheme({ name: "oslm-ink", variablePrefix: "--shiki-", fontStyle: true });
    return createHighlighterCore({
      themes: [theme],
      langs: [
        import("shiki/langs/c.mjs"),
        import("shiki/langs/diff.mjs"),
        import("shiki/langs/java.mjs"),
        import("shiki/langs/python.mjs"),
        import("shiki/langs/bash.mjs"),
      ],
      engine: createJavaScriptRegexEngine(),
    });
  })();
  return highlighter;
}

export function normaliseLang(lang: string | undefined): string | null {
  if (!lang) return null;
  const l = LANG_ALIASES[lang.toLowerCase()] ?? lang.toLowerCase();
  return SUPPORTED.has(l) ? l : null;
}

export async function highlight(code: string, lang: string | undefined): Promise<string | null> {
  const l = normaliseLang(lang) ?? (/#include|pthread_|int\s+main/.test(code) ? "c" : null);
  if (!l) return null;
  try {
    const hl = await load();
    return hl.codeToHtml(code, { lang: l, theme: "oslm-ink" });
  } catch {
    return null;
  }
}

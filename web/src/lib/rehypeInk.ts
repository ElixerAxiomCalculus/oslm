import type { Element, ElementContent, Root, Text } from "hast";
import { SKIP, visit } from "unist-util-visit";
import { wordJitter } from "./sketch";

export interface InkOptions {
  /** Words with index >= inkFrom get the ink-sweep animation class. */
  inkFrom: number;
  /** Highest valid citation number; others are left as plain text. */
  maxCitation: number;
  /** Append the riding pen after the last word. */
  pen: boolean;
  /** Receives the number of words rendered. */
  onCount?: (words: number) => void;
}

const CITE = /\[(\d{1,2})\]/g;
const SKIP_TAGS = new Set(["pre", "code"]);

function citationNodes(text: string, maxCitation: number): ElementContent[] {
  const out: ElementContent[] = [];
  let last = 0;
  for (const m of text.matchAll(CITE)) {
    const n = Number(m[1]);
    if (n < 1 || n > maxCitation) continue;
    // A footnote hugs the word it annotates: "writes it¹." rather than "writes it ¹ .".
    const before = text.slice(last, m.index).replace(/\s+$/, "");
    if (before) out.push({ type: "text", value: before });
    out.push({
      type: "element",
      tagName: "sup",
      properties: { className: ["cite"], dataN: String(n) },
      children: [{ type: "text", value: String(n) }],
    });
    last = m.index! + m[0].length;
  }
  if (last < text.length) out.push({ type: "text", value: text.slice(last) });
  return out;
}

/**
 * Splits prose text into per-word spans (for the ink sweep and jitter), turns [n] into footnote
 * superscripts, and places a pen marker after the final word. Code is left untouched.
 */
export function rehypeInk(options: InkOptions) {
  return (tree: Root) => {
    let wordIndex = 0;
    let lastWord: Element | null = null;

    visit(tree, (node, index, parent) => {
      if (node.type === "element" && SKIP_TAGS.has(node.tagName)) return SKIP;
      if (node.type !== "text" || !parent || index === undefined) return;
      const pieces = citationNodes((node as Text).value, options.maxCitation);
      const replacement: ElementContent[] = [];
      for (const piece of pieces) {
        if (piece.type !== "text") {
          replacement.push(piece);
          continue;
        }
        for (const token of piece.value.split(/(\s+)/)) {
          if (!token) continue;
          if (/^\s+$/.test(token)) {
            replacement.push({ type: "text", value: token });
            continue;
          }
          const { y, r } = wordJitter(wordIndex);
          const span: Element = {
            type: "element",
            tagName: "span",
            properties: {
              className: wordIndex >= options.inkFrom ? ["w", "ink"] : ["w"],
              style: `--jy:${y.toFixed(2)}px;--jr:${r.toFixed(2)}deg`,
            },
            children: [{ type: "text", value: token }],
          };
          replacement.push(span);
          lastWord = span;
          wordIndex++;
        }
      }
      parent.children.splice(index, 1, ...replacement);
      return [SKIP, index + replacement.length];
    });

    options.onCount?.(wordIndex);
    if (options.pen && lastWord) {
      (lastWord as Element).children.push({
        type: "element",
        tagName: "span",
        properties: { className: ["pen-anchor"], ariaHidden: "true" },
        children: [],
      });
    }
  };
}

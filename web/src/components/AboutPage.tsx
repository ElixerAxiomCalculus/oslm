import { SketchCircle, SketchUnderline } from "./Sketch";

export function AboutPage() {
  return (
    <div className="scroll ruled">
      <div className="about md">
        <h2>About OSlm</h2>
        <p>
          OSlm is a local tutor for <SketchCircle seed="ps-about">process synchronization</SketchCircle>. It answers like a
          teacher: example first, then the rule, then a quick question to check your understanding.
        </p>
        <h3>How it works</h3>
        <ul>
          <li>
            <strong>Retrieval (RAG):</strong> your question is embedded with bge-large-en-v1.5, the 20 nearest note chunks
            come from ChromaDB, and bge-reranker-base keeps the best 4. Answers cite them as footnotes.
          </li>
          <li>
            <strong>Fine-tuning:</strong> Qwen3-4B-Instruct-2507 was QLoRA fine-tuned (4-bit NF4, LoRA r=16) on 385
            checked examples: explanations, C programs, interleaving traces, find-the-bug reviews, follow-ups and
            polite declines. Every C program in the data was compiled and run under ThreadSanitizer and Helgrind.
          </li>
          <li>
            <strong>Code checker:</strong> C programs are compiled with gcc and run under ThreadSanitizer and Helgrind in a
            bubblewrap sandbox (no network, CPU, memory and process limits) inside WSL2.
          </li>
          <li>
            <strong>Four setups:</strong> toggle <em>Fine-tuned</em> and <em>course notes</em> on the sticky note to compare
            base, base + RAG, fine-tuned and fine-tuned + RAG live.
          </li>
        </ul>
        <h3>Team</h3>
        <ul>
          <li>Arush James — 24BAI0277</li>
          <li>Sayak Mondal — 24BCE0884</li>
          <li>Rohan Chaudhary — 24BCE0906</li>
        </ul>
        <p>
          Sources: OSTEP (Arpaci-Dusseau), <em>The Little Book of Semaphores</em> (Downey), Linux and POSIX man pages,
          Wikipedia, and the course's own notes. <SketchUnderline seed="local">Everything runs locally.</SketchUnderline>
        </p>
      </div>
    </div>
  );
}

import { fireEvent, render, screen } from "@testing-library/react";
import { describe, expect, it, vi } from "vitest";
import { verificationStamps } from "./CodeBlock";
import { extractPrograms, MarkdownView, programIndex } from "./MarkdownView";

const base = { inkFrom: 0, pen: false, animate: false, programs: [] as string[] };

describe("citation mapping", () => {
  it("renders valid citations as footnote buttons and leaves invalid ones as text", () => {
    const onCite = vi.fn();
    render(<MarkdownView {...base} text="Locks [1] help [3]." maxCitation={2} onCite={onCite} />);
    const btn = screen.getByRole("button", { name: "Source 1" });
    fireEvent.click(btn);
    expect(onCite).toHaveBeenCalledWith(1);
    expect(screen.queryByRole("button", { name: "Source 3" })).toBeNull();
    expect(document.body.textContent).toContain("[3]");
  });

  it("does not treat array indexing in code as a citation", () => {
    render(<MarkdownView {...base} text={"```c\nbuf[1] = 0;\n```"} maxCitation={4} />);
    expect(screen.queryByRole("button", { name: /Source/ })).toBeNull();
    expect(document.body.textContent).toContain("buf[1] = 0;");
  });

  it("splits prose into ink words and adds the pen marker", () => {
    const onCount = vi.fn();
    const { container } = render(<MarkdownView {...base} pen text="two words" maxCitation={0} onWordCount={onCount} />);
    expect(container.querySelectorAll(".w")).toHaveLength(2);
    expect(container.querySelectorAll(".w.ink")).toHaveLength(2);
    expect(container.querySelector(".pen-anchor")).not.toBeNull();
    expect(onCount).toHaveBeenCalledWith(2);
  });
});

describe("verification stamps", () => {
  const program = "int main(void) {\n  return 0;\n}";

  it("numbers complete programs like the backend", () => {
    const md = "```c\nint x;\n```\n```c\n" + program + "\n```";
    const programs = extractPrograms(md);
    expect(programs).toHaveLength(1);
    expect(programIndex(program, programs)).toBe(0);
  });

  it("shows a red stamp with the racing line", () => {
    const stamps = verificationStamps({
      status: "done",
      report: {
        status: "issues", summary: "", tsan_clean: false, helgrind_clean: false, timed_out: false,
        findings: [{ tool: "tsan", kind: "data race", message: "", lines: [14], threads: [] }],
        compiler: [], program_output: "", has_sync_bug: true,
      },
    });
    expect(stamps).toEqual([{ text: "✗ data race (line 14)", bad: true }]);
  });

  it("shows both clean stamps", () => {
    const stamps = verificationStamps({
      status: "done",
      report: {
        status: "clean", summary: "", tsan_clean: true, helgrind_clean: true, timed_out: false,
        findings: [], compiler: [], program_output: "", has_sync_bug: false,
      },
    });
    expect(stamps.map((s) => s.text)).toEqual(["✓ TSan clean", "✓ Helgrind clean"]);
  });

  it("renders the stamp beside the matching code block", () => {
    render(
      <MarkdownView
        {...base}
        text={"```c\n" + program + "\n```"}
        maxCitation={0}
        programs={[program]}
        verification={{ 0: { status: "pending" } }}
      />,
    );
    expect(screen.getByText("checking…")).toBeInTheDocument();
  });
});

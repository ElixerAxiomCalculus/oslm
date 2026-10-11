import { expect, type Page, test } from "@playwright/test";

type Ev = [string, unknown];

function sse(events: Ev[]): string {
  return events.map(([e, d]) => `event: ${e}\ndata: ${JSON.stringify(d)}\n\n`).join("");
}

const SOURCES = [
  { n: 1, chunk_id: "ostep-locks#0001", source_name: "OSTEP", title: "Locks", section: "28.1 Locks: The Basic Idea", page_or_slide: "p.2", url: "https://pages.cs.wisc.edu/~remzi/OSTEP/threads-locks.pdf", trust_tier: 1, score: 0.93, preview: "A lock is just a variable..." },
  { n: 2, chunk_id: "wiki-race#0000", source_name: "Wikipedia", title: "Race condition", section: "", page_or_slide: "", url: "https://en.wikipedia.org/wiki/Race_condition", trust_tier: 2, score: 0.81, preview: "A race condition..." },
];

const SETUPS = [
  { name: "base", label: "Base", model: "qwen3", rag: false, fine_tuned: false, available: true },
  { name: "base_rag", label: "Base + RAG", model: "qwen3", rag: true, fine_tuned: false, available: true },
  { name: "ft", label: "FT", model: "oslm-ft", rag: false, fine_tuned: true, available: true },
  { name: "ft_rag", label: "FT + RAG", model: "oslm-ft", rag: true, fine_tuned: true, available: true },
];

const RACY = `#include <pthread.h>
int counter;
void *w(void *a) { counter++; return 0; }
int main(void) { pthread_t t; pthread_create(&t, 0, w, 0); w(0); pthread_join(t, 0); return 0; }`;

interface ChatLog {
  bodies: Array<{ message: string; setup: string; history: Array<{ role: string; content: string }> }>;
}

async function mockApi(page: Page, opts: { down?: boolean } = {}): Promise<ChatLog> {
  const log: ChatLog = { bodies: [] };
  await page.route("**/api/setups", (route) => route.fulfill({ json: SETUPS }));
  await page.route("**/api/sources/**", (route) =>
    route.fulfill({
      json: { chunk_id: "ostep-locks#0001", text: "A lock is just a variable, and thus to use one, you must declare a lock variable of some kind.", source_name: "OSTEP", title: "Locks", section: "28.1", page_or_slide: "p.2", url: SOURCES[0]!.url, trust_tier: 1, has_code: false },
    }),
  );
  await page.route("**/api/chat", async (route) => {
    if (opts.down) return route.abort("connectionrefused");
    const body = route.request().postDataJSON() as ChatLog["bodies"][number];
    log.bodies.push(body);
    const followup = body.history.length > 0;
    const events: Ev[] = [
      ["meta", { mode: "concept", setup: body.setup, rewritten_query: followup ? "Why do locks prevent races?" : null, followup }],
      ["sources", { sources: body.setup.endsWith("_rag") ? SOURCES : [] }],
      ["token", { text: followup ? "Because only one thread " : "A **race condition** happens when two threads " }],
      ["token", { text: followup ? "holds the lock [1]." : "update shared data at the same time [1][2]." }],
      ["done", { timings: { ttft_ms: 500, total_ms: 900 }, cited: [1, 2] }],
    ];
    await route.fulfill({ status: 200, headers: { "Content-Type": "text/event-stream" }, body: sse(events) });
  });
  await page.route("**/api/check-code", (route) =>
    route.fulfill({
      json: {
        report: {
          status: "issues", summary: "data race (line 3)", tsan_clean: false, helgrind_clean: false, timed_out: false,
          findings: [{ tool: "tsan", kind: "data race", message: "data race prog.c:3 in w", lines: [3], threads: ["thread T1", "main thread"] }],
          compiler: [], program_output: "", has_sync_bug: true,
        },
        explain_url: "/api/check-code/explain/tok",
      },
    }),
  );
  await page.route("**/api/check-code/explain/**", (route) =>
    route.fulfill({
      status: 200,
      headers: { "Content-Type": "text/event-stream" },
      body: sse([["meta", { mode: "bug_find", setup: "ft_rag" }], ["sources", { sources: [] }], ["token", { text: "Verdict: bug found. counter++ is not atomic." }], ["done", { timings: {} }]]),
    }),
  );
  return log;
}

async function ask(page: Page, text: string) {
  const box = page.getByLabel("Ask a question about process synchronization");
  await box.fill(text);
  await box.press("Enter");
}

test.beforeEach(async ({ page }) => {
  await page.addInitScript(() => localStorage.clear());
});

test("ask a question and see sources plus footnotes", async ({ page }) => {
  await mockApi(page);
  await page.goto("/");
  await ask(page, "What is a race condition?");
  await expect(page.locator(".question").first()).toHaveText("What is a race condition?");
  await expect(page.locator(".answer")).toContainText("update shared data at the same time", { timeout: 10_000 });
  await expect(page.getByRole("button", { name: "Source 1" })).toBeVisible();
  await expect(page.locator(".index-card")).toHaveCount(2);
  await page.getByRole("button", { name: "Source 1" }).click();
  await expect(page.locator("#source-card-1")).toHaveClass(/highlight/);
  await page.locator("#source-card-1").click();
  await expect(page.getByRole("dialog")).toContainText("A lock is just a variable");
  await page.keyboard.press("Escape");
  await expect(page.getByRole("dialog")).toBeHidden();
});

test("follow-up keeps context", async ({ page }) => {
  const log = await mockApi(page);
  await page.goto("/");
  await ask(page, "What is a race condition?");
  await expect(page.locator(".answer").first()).toContainText("same time", { timeout: 10_000 });
  await ask(page, "why?");
  await expect(page.locator(".answer").nth(1)).toContainText("holds the lock", { timeout: 10_000 });
  expect(log.bodies[1]!.history.map((h) => h.role)).toEqual(["user", "assistant"]);
  await expect(page.getByText("reading this as")).toBeVisible();
});

test("toggle setups changes the request", async ({ page }) => {
  const log = await mockApi(page);
  await page.goto("/");
  await expect(page.getByTestId("setup-name")).toHaveText("setup: base_rag");
  await page.getByRole("switch", { name: "Fine-tuned model" }).click();
  await expect(page.getByTestId("setup-name")).toHaveText("setup: ft_rag");
  await page.getByRole("switch", { name: "Use course notes (RAG)" }).click();
  await expect(page.getByTestId("setup-name")).toHaveText("setup: ft");
  await ask(page, "What is a mutex?");
  await expect(page.locator(".answer")).toContainText("same time", { timeout: 10_000 });
  expect(log.bodies[0]!.setup).toBe("ft");
  await expect(page.locator(".index-card")).toHaveCount(0);
});

test("check my code with a racy program shows a red stamp", async ({ page }) => {
  await mockApi(page);
  await page.goto("/");
  await page.getByRole("tab", { name: "Check my code" }).click();
  await page.getByLabel("C source code").fill(RACY);
  await page.getByRole("button", { name: "Run checks" }).click();
  await expect(page.locator(".report-stamps .stamp.bad")).toHaveText("✗ data race (line 3)");
  await expect(page.getByLabel("Tutor explanation")).toContainText("Verdict: bug found", { timeout: 10_000 });
});

test("backend down shows the margin error with retry", async ({ page }) => {
  await mockApi(page, { down: true });
  await page.goto("/");
  await ask(page, "What is a semaphore?");
  const note = page.locator(".margin-note");
  await expect(note).toContainText("couldn't reach", { timeout: 10_000 });
  await page.unroute("**/api/chat");
  await mockApi(page);
  await note.getByRole("button", { name: "try again" }).click();
  await expect(page.locator(".answer")).toContainText("same time", { timeout: 10_000 });
});

test("keyboard-only usage", async ({ page }) => {
  await mockApi(page);
  await page.goto("/");
  await page.getByLabel("Ask a question about process synchronization").focus();
  await page.keyboard.type("What is deadlock?");
  await page.keyboard.press("Shift+Enter");
  await page.keyboard.type("Keep it short.");
  await expect(page.getByLabel("Ask a question about process synchronization")).toHaveValue("What is deadlock?\nKeep it short.");
  await page.keyboard.press("Enter");
  await expect(page.locator(".answer")).toContainText("same time", { timeout: 10_000 });
  await page.getByRole("tab", { name: "Ask" }).focus();
  await page.keyboard.press("ArrowRight");
  await expect(page.getByRole("tab", { name: "Check my code" })).toHaveAttribute("data-state", "active");
  await page.getByRole("switch", { name: "Use course notes (RAG)" }).focus();
  await page.keyboard.press("Space");
  await expect(page.getByTestId("setup-name")).toHaveText("setup: base");
});

test("dark theme switches to the chalkboard palette", async ({ page }) => {
  await mockApi(page);
  await page.goto("/");
  await page.getByRole("button", { name: /chalkboard/ }).click();
  await expect(page.locator("html")).toHaveAttribute("data-theme", "dark");
  const bg = await page.evaluate(() => getComputedStyle(document.body).backgroundColor);
  expect(bg).toBe("rgb(18, 18, 18)");
});

test("reduced motion shows the answer immediately", async ({ page }) => {
  await page.emulateMedia({ reducedMotion: "reduce" });
  await mockApi(page);
  await page.goto("/");
  await ask(page, "What is a race condition?");
  await expect(page.locator(".answer")).toContainText("same time", { timeout: 3_000 });
  await expect(page.locator(".w.ink")).toHaveCount(0);
});

test("new page starts a fresh session @mobile", async ({ page }) => {
  await mockApi(page);
  await page.goto("/");
  await ask(page, "What is a race condition?");
  await expect(page.locator(".answer")).toContainText("same time", { timeout: 10_000 });
  await page.getByRole("button", { name: /New page/ }).click();
  await expect(page.locator(".entry")).toHaveCount(0);
});

test("mobile layout uses a bottom sheet @mobile", async ({ page, isMobile }) => {
  test.skip(!isMobile, "mobile project only");
  await mockApi(page);
  await page.goto("/");
  const noBodyScroll = await page.evaluate(() => document.documentElement.scrollWidth <= window.innerWidth);
  expect(noBodyScroll).toBe(true);
  await ask(page, "What is a race condition?");
  await expect(page.locator(".answer")).toContainText("same time", { timeout: 10_000 });
  await page.getByRole("button", { name: "notes" }).click();
  await expect(page.locator(".panel")).toHaveClass(/open/);
  await expect(page.locator(".index-card").first()).toBeInViewport();
});

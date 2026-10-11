import { z } from "zod";

export const SourceCard = z.object({
  n: z.number().int(),
  chunk_id: z.string(),
  source_name: z.string(),
  title: z.string(),
  section: z.string(),
  page_or_slide: z.string(),
  url: z.string(),
  trust_tier: z.number(),
  score: z.number(),
  preview: z.string(),
});
export type SourceCard = z.infer<typeof SourceCard>;

export const Finding = z.object({
  tool: z.enum(["tsan", "helgrind", "runtime", "compiler"]),
  kind: z.string(),
  message: z.string(),
  lines: z.array(z.number()),
  threads: z.array(z.string()),
});
export type Finding = z.infer<typeof Finding>;

export const CheckReport = z.object({
  status: z.enum(["clean", "issues", "compile_error", "rejected", "checker_unavailable", "checker_error"]),
  summary: z.string(),
  tsan_clean: z.boolean().nullable(),
  helgrind_clean: z.boolean().nullable(),
  timed_out: z.boolean(),
  findings: z.array(Finding),
  compiler: z.array(Finding),
  program_output: z.string(),
  has_sync_bug: z.boolean(),
});
export type CheckReport = z.infer<typeof CheckReport>;

export const MetaEvent = z.object({
  mode: z.string(),
  setup: z.string(),
  rewritten_query: z.string().nullable().optional(),
  followup: z.boolean().optional(),
  route: z.string().optional(),
  model: z.string().optional(),
  best_rerank_score: z.number().nullable().optional(),
  reason: z.string().optional(),
});

export const SourcesEvent = z.object({ sources: z.array(SourceCard) });
export const TokenEvent = z.object({ text: z.string() });
export const VerificationEvent = z.object({
  block_index: z.number().int(),
  status: z.enum(["pending", "done"]),
  report: CheckReport.optional(),
});
export const ErrorEvent = z.object({ message: z.string(), kind: z.string().optional() });
export const DoneEvent = z.object({
  mode: z.string().optional(),
  cited: z.array(z.number()).optional(),
  timings: z.record(z.string(), z.number().nullable()),
});

export type StreamEvent =
  | { type: "meta"; data: z.infer<typeof MetaEvent> }
  | { type: "sources"; data: z.infer<typeof SourcesEvent> }
  | { type: "token"; data: z.infer<typeof TokenEvent> }
  | { type: "verification"; data: z.infer<typeof VerificationEvent> }
  | { type: "error"; data: z.infer<typeof ErrorEvent> }
  | { type: "done"; data: z.infer<typeof DoneEvent> };

const schemas = {
  meta: MetaEvent,
  sources: SourcesEvent,
  token: TokenEvent,
  verification: VerificationEvent,
  error: ErrorEvent,
  done: DoneEvent,
} as const;

export function validateEvent(type: string, raw: unknown): StreamEvent | null {
  if (!(type in schemas)) return null;
  const schema = schemas[type as keyof typeof schemas];
  const parsed = schema.safeParse(raw);
  if (!parsed.success) return null;
  return { type, data: parsed.data } as StreamEvent;
}

export const SourceDetail = z.object({
  chunk_id: z.string(),
  text: z.string(),
  source_name: z.string(),
  title: z.string(),
  section: z.string(),
  page_or_slide: z.string(),
  url: z.string(),
  trust_tier: z.number(),
  has_code: z.boolean(),
});
export type SourceDetail = z.infer<typeof SourceDetail>;

export const SetupInfo = z.object({
  name: z.enum(["base", "base_rag", "ft", "ft_rag"]),
  label: z.string(),
  model: z.string(),
  rag: z.boolean(),
  fine_tuned: z.boolean(),
  available: z.boolean(),
});
export type SetupInfo = z.infer<typeof SetupInfo>;

export const CheckCodeResponse = z.object({
  report: CheckReport,
  explain_url: z.string().nullable(),
});

import type { CapturedBatch, CapturedToolCall, BatchingMode } from "./types.js";

/**
 * Converts turn_end event data into a CapturedBatch.
 * @param message      AssistantMessage (content: Array of TextContent|ThinkingContent|ToolCall)
 * @param toolResults  ToolResultMessage[]
 */
export function captureBatch(
  message: any,
  toolResults: any[],
  turnIndex: number,
  timestamp: number
): CapturedBatch {
  const content: any[] = Array.isArray(message?.content) ? message.content : [];

  // Collect assistant prose text
  const assistantText = content
    .filter((block: any) => block.type === "text")
    .map((block: any) => block.text)
    .join("\n")
    .trim();

  // Collect tool calls, matching each to its result
  const toolCalls: CapturedToolCall[] = content
    .filter((block: any) => block.type === "toolCall")
    .map((block: any) => {
      const match = toolResults.find((result: any) => result.toolCallId === block.id);

      let resultText = "(no result)";
      let isError = false;

      if (match) {
        const resultContent: any[] = Array.isArray(match.content) ? match.content : [];
        resultText = resultContent
          .filter((c: any) => c.type === "text")
          .map((c: any) => c.text)
          .join("\n");
        isError = match.isError ?? false;
      }

      return {
        toolCallId: block.id,
        toolName: block.name,
        args: block.input ?? block.args ?? block.arguments ?? {},
        resultText,
        isError,
      } satisfies CapturedToolCall;
    });

  return { turnIndex, timestamp, assistantText, toolCalls };
}

/**
 * Scans a session branch for unsummarized tool results and groups them into CapturedBatches.
 * Useful for capturing results from the current in-progress turn when a prune is triggered.
 *
 * @param branch            The session message branch (from ctx.sessionManager.getBranch())
 * @param indexer           The pruner indexer to check for already-summarized IDs
 * @param excludeToolNames  Optional tool names to skip (e.g. context_prune itself)
 */
export function captureUnindexedBatchesFromSession(
  branch: any[],
  indexer: { isSummarized(id: string): boolean },
  excludeToolNames: string[] = []
): CapturedBatch[] {
  // branch is SessionEntry[]. Each message entry has { type: "message", message: AgentMessage }.
  // We must unwrap the SessionEntry wrapper before accessing role/toolCallId.

  // Skip everything at or before the most recent compaction boundary.  Tool
  // calls that predate a compaction are already summarised by Pi; re-scanning
  // them wastes LLM tokens and produces thousands of spurious pending batches
  // when opening a large old session.
  let scanStart = 0;
  for (let i = branch.length - 1; i >= 0; i--) {
    if (branch[i].type === "compaction") {
      scanStart = i + 1;
      break;
    }
  }
  const scanBranch = scanStart > 0 ? branch.slice(scanStart) : branch;

  // resultMap only needs post-compaction entries — pre-compaction tool results
  // are already indexed and would be excluded by the indexer check anyway.
  const resultMap = new Map<string, any>();
  for (const entry of scanBranch) {
    if (entry.type !== "message") continue;
    const m = entry.message;
    if (m.role === "toolResult" && m.toolCallId) {
      resultMap.set(m.toolCallId, m);
    }
  }

  const batches: CapturedBatch[] = [];
  // turnCounter must count every assistant message from the START of the full
  // branch (not just the scan window) so that turnIndex values stay in sync with
  // Pi's own event.turnIndex numbering.  trimBatchToPendingRange compares against
  // frontier.lastAttemptedTurnIndex which was recorded from a live turn_end event,
  // so the two must use the same counting basis.
  let turnCounter = 0;

  // userTurnGroup increments on every user message seen in the scan window.
  // All assistant tool-call batches between two consecutive user messages share the
  // same userTurnGroup. This is used by groupBatchesByMode to merge turns within
  // a single user → final-agent-message span when batchingMode === "agent-message".
  let userTurnGroup = 0;

  for (let idx = 0; idx < branch.length; idx++) {
    const entry = branch[idx];
    if (entry.type !== "message") continue;
    const msg = entry.message;

    if (msg.role === "user") {
      // Only advance userTurnGroup inside the scan window; pre-compaction user
      // messages are irrelevant to current grouping.
      if (idx >= scanStart) userTurnGroup++;
      continue;
    }

    if (msg.role !== "assistant") continue;

    // Always increment — even for pre-compaction turns we haven't scanned.
    const currentTurnIndex = turnCounter++;

    // Skip batch construction for pre-compaction entries; we only need the
    // turnCounter to stay in sync.
    if (idx < scanStart) continue;

    const content = Array.isArray(msg.content) ? msg.content : [];
    const toolCallBlocks = content.filter((c: any) => c.type === "toolCall");

    // Find tool calls that have results in this branch and are not yet summarized
    const readyToPrune = toolCallBlocks.filter((tc: any) => {
      const id = tc.id;
      if (!id) return false;
      if (indexer.isSummarized(id)) return false;
      if (excludeToolNames.includes(tc.name)) return false;
      return resultMap.has(id);
    });

    if (readyToPrune.length > 0) {
      const results = readyToPrune.map((tc: any) => resultMap.get(tc.id));
      const readyIds = new Set(readyToPrune.map((tc: any) => tc.id));
      // We pass the full message but then trim back down to only the tool calls
      // whose results already exist in the session. This lets agentic-auto prune
      // an intermediate completed subset in the middle of a longer tool chain
      // without accidentally capturing later unresolved calls from the same
      // assistant message as "(no result)" placeholders.
      const ts = entry.timestamp ? new Date(entry.timestamp).getTime() : (msg.timestamp ?? Date.now());
      const batch = captureBatch(msg, results, currentTurnIndex, ts);
      batches.push({
        ...batch,
        toolCalls: batch.toolCalls.filter((tc) => readyIds.has(tc.toolCallId)),
        // Tag with the current group so flushPending can merge by mode
        userTurnGroup,
      });
    }
  }

  return batches;
}

/** Serializes a single CapturedBatch into readable text for the summarizer LLM. */
export function serializeBatchForSummarizer(batch: CapturedBatch): string {
  const parts: string[] = [];

  if (batch.assistantText) {
    parts.push(`Assistant said: ${batch.assistantText}\n`);
  }

  const toolParts = batch.toolCalls.map((tc) => {
    const status = tc.isError ? "ERROR" : "OK";
    const MAX_CHARS = 2000;

    let argsJson = JSON.stringify(tc.args, null, 2);
    if (argsJson.length > MAX_CHARS) {
      argsJson = argsJson.slice(0, MAX_CHARS) + ` ...[${argsJson.length - MAX_CHARS} chars truncated]`;
    }

    let resultText = tc.resultText;
    if (resultText.length > MAX_CHARS) {
      resultText = resultText.slice(0, MAX_CHARS) + ` ...[${resultText.length - MAX_CHARS} chars truncated]`;
    }

    return `Tool: ${tc.toolName}(${argsJson})\nResult (${status}): ${resultText}`;
  });

  parts.push(toolParts.join("\n---\n"));

  return parts.join("\n");
}

/**
 * Serializes multiple CapturedBatches into a single readable text block for the summarizer LLM.
 * Each batch is rendered as a separate "Turn" section with a header indicating the turn index.
 */
export function serializeBatchesForSummarizer(batches: CapturedBatch[]): string {
  return batches
    .map((batch, i) => {
      const header = `=== Turn ${batch.turnIndex}${i > 0 ? ` (batch ${i + 1})` : ""} ===`;
      const body = serializeBatchForSummarizer(batch);
      return `${header}\n${body}`;
    })
    .join("\n\n");
}

/**
 * Groups CapturedBatches according to the chosen batching mode.
 *
 * - "turn"          : returns the input array unchanged (one summary per assistant turn).
 * - "agent-message" : merges all consecutive batches that share the same `userTurnGroup`
 *                     into a single CapturedBatch, producing one summary per
 *                     user → final-agent-message span.
 *
 * Batches without a `userTurnGroup` (e.g. from the live `turn_end` capture path) are
 * always passed through one-per-batch regardless of mode — grouping only applies to
 * batches captured from the session branch scan.
 *
 * Merge rules:
 *   - `assistantText` = non-empty values joined with "\n\n"
 *   - `toolCalls`     = concatenation in original order
 *   - `turnIndex`     = last batch's turnIndex (latest turn in the group)
 *   - `timestamp`     = last batch's timestamp
 *   - `userTurnGroup` = shared group value of the merged batches
 */
export function groupBatchesByMode(batches: CapturedBatch[], mode: BatchingMode): CapturedBatch[] {
  if (mode !== "agent-message") return batches;

  const out: CapturedBatch[] = [];
  // current tracks the mutable merged batch being built for the current group.
  // We spread into a plain object so we can mutate it without affecting the source.
  let current: CapturedBatch & { userTurnGroup: number } | null = null;

  for (const batch of batches) {
    // Batches without a group key are passed through individually; they break
    // any open merge group too since we can't confidently assign them a span.
    if (batch.userTurnGroup === undefined) {
      current = null;
      out.push(batch);
      continue;
    }

    if (current !== null && current.userTurnGroup === batch.userTurnGroup) {
      // Same span — merge into the current accumulated batch
      const textParts = [current.assistantText, batch.assistantText].filter(Boolean);
      current.assistantText = textParts.join("\n\n");
      current.toolCalls = current.toolCalls.concat(batch.toolCalls);
      // Advance to the latest turn metadata
      current.turnIndex = batch.turnIndex;
      current.timestamp = batch.timestamp;
    } else {
      // New group — create a fresh accumulated batch (shallow copy so mutations
      // to `current` do not bleed back into the original `batch` object)
      current = { ...batch, userTurnGroup: batch.userTurnGroup };
      out.push(current);
    }
  }

  return out;
}

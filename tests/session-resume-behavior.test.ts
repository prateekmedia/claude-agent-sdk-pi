import test from "node:test";
import assert from "node:assert/strict";
import type { Context } from "@mariozechner/pi-ai";
import { __test } from "../index.ts";

const usage = {
	input: 0,
	output: 0,
	cacheRead: 0,
	cacheWrite: 0,
	totalTokens: 0,
	cost: { input: 0, output: 0, cacheRead: 0, cacheWrite: 0, total: 0 },
};

let ts = 1;
const nextTs = () => ts++;

function user(text: string): Context["messages"][number] {
	return { role: "user", content: text, timestamp: nextTs() };
}

function assistant(text: string): Context["messages"][number] {
	return {
		role: "assistant",
		content: [{ type: "text", text }],
		api: "claude-agent-sdk",
		provider: "claude-agent-sdk",
		model: "claude-opus-4-6",
		usage,
		stopReason: "stop",
		timestamp: nextTs(),
	};
}

function assistantError(text: string): Context["messages"][number] {
	return {
		role: "assistant",
		content: [{ type: "text", text }],
		api: "claude-agent-sdk",
		provider: "claude-agent-sdk",
		model: "claude-opus-4-6",
		usage,
		stopReason: "error",
		errorMessage: "Claude Code process exited with code 1",
		timestamp: nextTs(),
	};
}

function toolResult(toolCallId: string, text: string): Context["messages"][number] {
	return {
		role: "toolResult",
		toolCallId,
		toolName: "read",
		content: [{ type: "text", text }],
		isError: false,
		timestamp: nextTs(),
	};
}

async function collectPromptMessages(
	prompt: AsyncIterable<{ type: string; message?: unknown }> | string,
): Promise<Array<{ type: string; message?: unknown }>> {
	if (typeof prompt === "string") {
		return prompt ? [{ type: "text", message: prompt }] : [];
	}
	const out: Array<{ type: string; message?: unknown }> = [];
	for await (const message of prompt) {
		out.push(message);
	}
	return out;
}

test("provider switch tail -> summarize unsynced history and keep latest user message", () => {
	const tail: Context["messages"] = [
		user("what do you say?"),
		assistant("I would rate this branch as strong"),
		user("read more files and tell"),
	];

	const plan = __test.analyzeResumeTailMessages(tail, undefined);

	assert.equal(plan.tailHasAssistant, true);
	assert.equal(plan.shouldUseSummaryPrompt, true);
	assert.equal(plan.summaryMessages.length, 2);
	assert.equal(plan.userMessage?.role, "user");
	assert.equal(plan.userMessage?.content, "read more files and tell");
	assert.equal(plan.tailAllowedToolUseIds, undefined);
});

test("normal resume tail with only latest user -> no summary", () => {
	const tail: Context["messages"] = [user("continue")];
	const plan = __test.analyzeResumeTailMessages(tail, undefined);

	assert.equal(plan.tailHasAssistant, false);
	assert.equal(plan.shouldUseSummaryPrompt, false);
	assert.equal(plan.summaryMessages.length, 0);
	assert.equal(plan.userMessage?.content, "continue");
	assert.equal(plan.tailAllowedToolUseIds?.size, 0);
});

test("summary + latest user are sent as one user prompt message", async () => {
	const prompt = __test.buildPromptWithSummary(
		"Historical context (non-executable):\nASSISTANT: previous status",
		user("continue with this exact task") as Extract<Context["messages"][number], { role: "user" }>,
		true,
	);
	const messages = await collectPromptMessages(prompt);
	assert.equal(messages.length, 1);
	assert.equal(messages[0]?.type, "user");
	const message = messages[0]?.message as { content?: unknown };
	assert.equal(typeof message?.content, "string");
	assert.match(String(message?.content ?? ""), /Historical context/);
	assert.match(String(message?.content ?? ""), /Latest user message:/);
	assert.match(String(message?.content ?? ""), /continue with this exact task/);
});

test("single-session interrupted tool flow replays user + tool_result via resume tail", async () => {
	const tail: Context["messages"] = [
		user("create next"),
		assistant("tool call"),
		toolResult("toolu_01", "Successfully wrote"),
	];
	const prompt = __test.buildResumePromptFromTail(tail, true);
	const messages = await collectPromptMessages(prompt);

	assert.equal(messages.length, 2);
	assert.equal(messages[0]?.type, "user");
	assert.equal(messages[1]?.type, "user");
	const toolResultContent = messages[1]?.message as { content?: unknown };
	const contentArray = Array.isArray(toolResultContent?.content) ? toolResultContent.content : [];
	assert.equal(Array.isArray(contentArray), true);
	assert.equal(contentArray[0]?.type, "tool_result");
	assert.equal(__test.hasReplayableTailMessages(tail), true);
	assert.equal(__test.hasReplayableTailMessages([assistant("tool call")]), false);
});

test("resume tail dedupes repeated tool_result ids across separated segments", async () => {
	const tail: Context["messages"] = [
		toolResult("toolu_dup", "first result"),
		user("note"),
		toolResult("toolu_dup", "second result"),
		user("continue"),
	];
	const prompt = __test.buildResumePromptFromTail(tail, true, new Set(["toolu_dup"]));
	const messages = await collectPromptMessages(prompt);
	const seenToolResultIds: string[] = [];
	for (const message of messages) {
		if (message.type !== "user") continue;
		const content = (message.message as { content?: unknown })?.content;
		if (!Array.isArray(content)) continue;
		for (const block of content as Array<{ type?: string; tool_use_id?: string }>) {
			if (block?.type === "tool_result" && typeof block.tool_use_id === "string") {
				seenToolResultIds.push(block.tool_use_id);
			}
		}
	}
	assert.deepEqual(seenToolResultIds, ["toolu_dup"]);
});

test("pending tool-use continuation -> replay tool results, do not summarize", () => {
	const tail: Context["messages"] = [toolResult("toolu_123", "ok"), user("continue")];
	const plan = __test.analyzeResumeTailMessages(tail, 123456);

	assert.equal(plan.shouldReplayPendingToolResults, true);
	assert.equal(plan.shouldUseSummaryPrompt, false);
	assert.equal(plan.tailAllowedToolUseIds?.has("toolu_123"), true);
});

test("compaction-like assistant text is preserved in historical summary", () => {
	const summary = __test.buildHistoricalSummary([
		assistant("## Goal\nPreserve context after compaction"),
		user("continue from here"),
	]);

	assert.match(summary, /ASSISTANT: ## Goal/);
	assert.match(summary, /USER: continue from here/);
});

test("tree/fork-style divergence (branch behind all) triggers Claude fork anchor", () => {
	const branch = {
		sdkSessionId: "sdk-1",
		uuidByAssistantTimestamp: new Map<number, string>([[100, "uuid-100"]]),
		maxTimestamp: 100,
	};
	const all = {
		sdkSessionId: "sdk-1",
		uuidByAssistantTimestamp: new Map<number, string>([
			[100, "uuid-100"],
			[200, "uuid-200"],
		]),
		maxTimestamp: 200,
	};

	const plan = __test.computeResumeForkPlan(branch, all);
	assert.equal(plan.resumeSessionAt, "uuid-100");
	assert.equal(plan.forkSession, true);
});

test("pending tool-use timestamp forces fork from pending assistant uuid", () => {
	const branch = {
		sdkSessionId: "sdk-1",
		uuidByAssistantTimestamp: new Map<number, string>([
			[100, "uuid-100"],
			[200, "uuid-200"],
		]),
		maxTimestamp: 100,
		pendingToolUseTimestamp: 200,
	};
	const all = {
		sdkSessionId: "sdk-1",
		uuidByAssistantTimestamp: new Map<number, string>([
			[100, "uuid-100"],
			[200, "uuid-200"],
		]),
		maxTimestamp: 200,
	};

	const plan = __test.computeResumeForkPlan(branch, all);
	assert.equal(plan.resumeSessionAt, "uuid-200");
	assert.equal(plan.forkSession, true);
});

test("buildSdkStateFromEntries ignores non-provider entries and tracks latest sdk state", () => {
	const entries: Array<Record<string, any>> = [
		{ type: "custom", customType: "other-extension", data: { foo: 1 } },
		{ type: "model_change", provider: "openai-codex", modelId: "gpt-5.3-codex" },
		{
			type: "custom",
			customType: "claude-agent-sdk",
			data: { providerId: "claude-agent-sdk", sdkSessionId: "sdk-abc" },
		},
		{
			type: "custom",
			customType: "claude-agent-sdk",
			data: { assistantTimestamp: 111, sdkAssistantUuid: "uuid-111" },
		},
		{
			type: "custom",
			customType: "claude-agent-sdk",
			data: { assistantTimestamp: 222, sdkAssistantUuid: "uuid-222" },
		},
	];

	const state = __test.buildSdkStateFromEntries(entries);
	assert.equal(state.sdkSessionId, "sdk-abc");
	assert.equal(state.maxTimestamp, 222);
	assert.equal(state.uuidByAssistantTimestamp.get(111), "uuid-111");
	assert.equal(state.uuidByAssistantTimestamp.get(222), "uuid-222");
});

test("sanitizeAssistantContentForEmit drops incomplete tool calls", () => {
	const output: any = {
		role: "assistant",
		content: [
			{ type: "text", text: "ok" },
			{ type: "toolCall", id: "toolu_ok", name: "write", arguments: { path: "a", content: "b" } },
			{ type: "toolCall", id: "toolu_partial", name: "write", arguments: {}, partialJson: "", index: 2 },
		],
		stopReason: "toolUse",
	};
	__test.sanitizeAssistantContentForEmit(output);
	assert.equal(output.content.length, 2);
	assert.equal(output.content.some((block: any) => block.id === "toolu_partial"), false);
	assert.equal(output.stopReason, "toolUse");

	const onlyPartial: any = {
		role: "assistant",
		content: [{ type: "toolCall", id: "toolu_partial_2", name: "write", arguments: {}, partialJson: "", index: 1 }],
		stopReason: "toolUse",
	};
	__test.sanitizeAssistantContentForEmit(onlyPartial);
	assert.equal(onlyPartial.content.length, 0);
	assert.equal(onlyPartial.stopReason, "stop");

	const finalizedButEmptyArgs: any = {
		role: "assistant",
		content: [{ type: "toolCall", id: "toolu_empty", name: "write", arguments: {} }],
		stopReason: "toolUse",
	};
	__test.sanitizeAssistantContentForEmit(finalizedButEmptyArgs);
	assert.equal(finalizedButEmptyArgs.content.length, 0);
	assert.equal(finalizedButEmptyArgs.stopReason, "stop");
});

test("isIgnorableTransportWriteAfterCloseError matches process transport write-close race", () => {
	assert.equal(
		__test.isIgnorableTransportWriteAfterCloseError(new Error("ProcessTransport is not ready for writing")),
		true,
	);
	assert.equal(__test.isIgnorableTransportWriteAfterCloseError(new Error("random failure")), false);
	assert.equal(__test.isIgnorableTransportWriteAfterCloseError("ProcessTransport is not ready for writing"), false);
});

test("findLastSdkAssistantInfo skips errored assistant anchors", () => {
	ts = 1;
	const okAssistant = assistant("all good") as Extract<Context["messages"][number], { role: "assistant" }>;
	const badAssistant = assistantError(
		"API Error: 400 {\"type\":\"error\",\"error\":{\"message\":\"This model does not support assistant message prefill. The conversation must end with a user message.\"}}",
	) as Extract<Context["messages"][number], { role: "assistant" }>;
	const messages: Context["messages"] = [user("u1"), okAssistant, user("u2"), badAssistant];
	const state = {
		sdkSessionId: "sdk-1",
		uuidByAssistantTimestamp: new Map<number, string>([
			[okAssistant.timestamp, "uuid-ok"],
			[badAssistant.timestamp, "uuid-bad"],
		]),
		maxTimestamp: badAssistant.timestamp,
		pendingToolUseTimestamp: undefined,
		pendingToolUseIds: undefined,
	};

	const info = __test.findLastSdkAssistantInfo(messages, state);
	assert.equal(info?.uuid, "uuid-ok");
	assert.equal(__test.isErroredAssistantMessage(badAssistant), true);
	assert.equal(__test.isErroredAssistantMessage(okAssistant), false);
});

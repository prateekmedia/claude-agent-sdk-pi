import test from "node:test";
import assert from "node:assert/strict";
import { appendFileSync, mkdtempSync, rmSync, writeFileSync } from "node:fs";
import { join } from "node:path";
import { tmpdir } from "node:os";
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

function toContextTailMessages(entries: Array<Record<string, any>>, minTimestamp: number): Context["messages"] {
	const out: Context["messages"] = [];
	for (const entry of entries) {
		if (entry.type !== "message") continue;
		const msg = entry.message;
		const ts = typeof msg?.timestamp === "number" ? msg.timestamp : 0;
		if (ts <= minTimestamp) continue;
		if (msg?.role === "user") {
			const text = Array.isArray(msg.content)
				? msg.content.filter((c: any) => c?.type === "text").map((c: any) => c.text).join(" ")
				: String(msg.content ?? "");
			out.push({ role: "user", content: text, timestamp: ts });
			continue;
		}
		if (msg?.role === "assistant") {
			const text = Array.isArray(msg.content)
				? msg.content.filter((c: any) => c?.type === "text").map((c: any) => c.text).join(" ")
				: String(msg.content ?? "");
			out.push({
				role: "assistant",
				content: [{ type: "text", text }],
				api: msg.api ?? "openai-codex-responses",
				provider: msg.provider ?? "openai-codex",
				model: msg.model ?? "gpt-5.3-codex",
				usage,
				stopReason: "stop",
				timestamp: ts,
			});
		}
	}
	return out;
}

test("mock pi session file sync detects updates and keeps branch/all split", () => {
	__test.clearInternalStateForTests();

	const tmp = mkdtempSync(join(tmpdir(), "casdk-mock-"));
	const sessionFile = join(tmp, "mock-session.jsonl");
	const sessionKey = "session:pi-mock-1";
	const sessionId = "pi-mock-1";

	const baseEntries = [
		{ type: "session", version: 3, id: "pi-mock-1", cwd: "/repo", timestamp: new Date().toISOString() },
		{
			type: "custom",
			customType: "claude-agent-sdk",
			data: { providerId: "claude-agent-sdk", sdkSessionId: "sdk-1" },
		},
		{
			type: "custom",
			customType: "claude-agent-sdk",
			data: { assistantTimestamp: 1000, sdkAssistantUuid: "uuid-1000" },
		},
	];

	writeFileSync(sessionFile, `${baseEntries.map((e) => JSON.stringify(e)).join("\n")}\n`, "utf-8");
	__test.setKnownPiSessionFileForSessionKey(sessionKey, sessionFile);
	__test.syncAllSdkStateFromPiSession(sessionKey, sessionId, "/repo");

	const initial = __test.getSessionState(sessionKey);
	assert.equal(initial?.all.sdkSessionId, "sdk-1");
	assert.equal(initial?.all.maxTimestamp, 1000);
	assert.equal(initial?.branch.maxTimestamp, 1000);

	appendFileSync(
		sessionFile,
		`${JSON.stringify({
			type: "custom",
			customType: "claude-agent-sdk",
			data: { assistantTimestamp: 2000, sdkAssistantUuid: "uuid-2000" },
		})}\n`,
		"utf-8",
	);

	__test.syncAllSdkStateFromPiSession(sessionKey, sessionId, "/repo");
	const updated = __test.getSessionState(sessionKey);
	assert.equal(updated?.all.maxTimestamp, 2000);
	assert.equal(updated?.branch.maxTimestamp, 1000);

	rmSync(tmp, { recursive: true, force: true });
});

test("mock provider-switch session tail produces summary plan", () => {
	const entries: Array<Record<string, any>> = [
		{
			type: "custom",
			customType: "claude-agent-sdk",
			data: { assistantTimestamp: 1000, sdkAssistantUuid: "uuid-1000" },
		},
		{ type: "model_change", provider: "openai-codex", modelId: "gpt-5.3-codex" },
		{
			type: "message",
			message: { role: "user", content: [{ type: "text", text: "what do you say?" }], timestamp: 1100 },
		},
		{
			type: "message",
			message: {
				role: "assistant",
				content: [{ type: "text", text: "I think this branch is strong." }],
				timestamp: 1200,
				provider: "openai-codex",
				model: "gpt-5.3-codex",
			},
		},
		{
			type: "message",
			message: { role: "user", content: [{ type: "text", text: "read more files and tell" }], timestamp: 1300 },
		},
	];

	const tail = toContextTailMessages(entries, 1000);
	const plan = __test.analyzeResumeTailMessages(tail, undefined);

	assert.equal(plan.tailHasAssistant, true);
	assert.equal(plan.shouldUseSummaryPrompt, true);
	assert.equal(plan.summaryMessages.length, 2);
	assert.equal(plan.userMessage?.role, "user");
	assert.equal(plan.userMessage?.content, "read more files and tell");
});

test("mock branch/all session snapshots from tree/fork produce fork plan", () => {
	const branchEntries: Array<Record<string, any>> = [
		{
			type: "custom",
			customType: "claude-agent-sdk",
			data: { providerId: "claude-agent-sdk", sdkSessionId: "sdk-tree" },
		},
		{
			type: "custom",
			customType: "claude-agent-sdk",
			data: { assistantTimestamp: 1000, sdkAssistantUuid: "uuid-1000" },
		},
		{ type: "branch_summary", fromId: "x", summary: "left branch" },
	];

	const allEntries: Array<Record<string, any>> = [
		...branchEntries,
		{ type: "model_change", provider: "openai-codex", modelId: "gpt-5.3-codex" },
		{ type: "compaction", summary: "compact", firstKeptEntryId: "abc", tokensBefore: 10000 },
		{
			type: "custom",
			customType: "claude-agent-sdk",
			data: { assistantTimestamp: 2000, sdkAssistantUuid: "uuid-2000" },
		},
	];

	const branchState = __test.buildSdkStateFromEntries(branchEntries);
	const allState = __test.buildSdkStateFromEntries(allEntries);
	const forkPlan = __test.computeResumeForkPlan(branchState, allState);

	assert.equal(forkPlan.resumeSessionAt, "uuid-1000");
	assert.equal(forkPlan.forkSession, true);
});

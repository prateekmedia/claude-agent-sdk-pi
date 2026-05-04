import assert from "node:assert";
import { describe, test } from "node:test";
import { __test } from "../index.ts";

const { buildPromptBlocks, contentToText } = __test;

describe("buildPromptBlocks", () => {
	test("uses XML tags, not echoable prefixes", () => {
		const messages = [
			{ role: "user" as const, content: "hello", timestamp: 1 },
			{
				role: "assistant" as const,
				content: [
					{ type: "text" as const, text: "hi" },
					{ type: "toolCall" as const, id: "toolu_1", name: "read", arguments: { path: "a" } },
					{ type: "thinking" as const, thinking: "planning..." },
				],
				api: "anthropic-messages" as const,
				provider: "claude-agent-sdk" as const,
				model: "claude-opus-4-6",
				usage: { input: 0, output: 0, cacheRead: 0, cacheWrite: 0, totalTokens: 0, cost: { input: 0, output: 0, cacheRead: 0, cacheWrite: 0, total: 0 } },
				stopReason: "toolUse" as const,
				timestamp: 2,
			},
			{ role: "toolResult" as const, toolCallId: "toolu_1", toolName: "Read", content: "content", isError: false, timestamp: 3 },
		];

		const blocks = buildPromptBlocks({ messages, systemPrompt: undefined }, undefined);
		const text = blocks.filter((b: any) => b.type === "text").map((b: any) => b.text).join("");

		assert.equal(text.includes("ASSISTANT:"), false, "must not contain ASSISTANT: prefix");
		assert.equal(text.includes("USER:"), false, "must not contain USER: prefix");
		assert.equal(text.includes("Historical tool call (non-executable)"), false, "must not contain Historical tool call prefix");
		assert.equal(text.includes("TOOL RESULT (historical"), false, "must not contain TOOL RESULT prefix");

		assert.ok(text.includes('<message role="user">\nhello\n</message>'), "user message wrapped in XML");
		assert.ok(text.includes('<message role="assistant">'), "assistant message wrapped in XML");
		assert.ok(text.includes('<tool_use id="toolu_1" name="Read">{"path":"a"}</tool_use>'), "tool call in XML");
		assert.ok(text.includes('<thinking>\nplanning...\n</thinking>'), "thinking in XML");
		assert.ok(text.includes('<tool_result tool_use_id="toolu_1" tool_name="Read">\ncontent\n</tool_result>'), "tool result in XML");
	});
});

describe("contentToText", () => {
	test("formats toolCall and thinking as XML", () => {
		const content = [
			{ type: "text" as const, text: "hi" },
			{ type: "toolCall" as const, id: "t1", name: "read", arguments: { path: "x" } },
			{ type: "thinking" as const, thinking: "think" },
		];
		const text = contentToText(content);
		assert.ok(text.includes('<tool_use id="t1" name="Read">{"path":"x"}</tool_use>'));
		assert.ok(text.includes('<thinking>\nthink\n</thinking>'));
		assert.equal(text.includes("Historical tool call (non-executable)"), false);
	});
});

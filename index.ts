import { calculateCost, createAssistantMessageEventStream, getModels, type AssistantMessage, type AssistantMessageEventStream, type Context, type ImageContent, type Model, type SimpleStreamOptions, type Tool } from "@mariozechner/pi-ai";
import type { ExtensionAPI } from "@mariozechner/pi-coding-agent";
import { createSdkMcpServer, query, type SDKMessage, type SDKUserMessage, type SettingSource } from "@anthropic-ai/claude-agent-sdk";
import type { Base64ImageSource, ContentBlockParam, ImageBlockParam, MessageParam, TextBlockParam } from "@anthropic-ai/sdk/resources";
import { pascalCase } from "change-case";
import { existsSync, readFileSync, readdirSync, statSync } from "fs";
import { homedir } from "os";
import { dirname, join, relative, resolve } from "path";

const PROVIDER_ID = "claude-agent-sdk";

const SDK_TO_PI_TOOL_NAME: Record<string, string> = {
	read: "read",
	write: "write",
	edit: "edit",
	bash: "bash",
	grep: "grep",
	glob: "find",
};

const PI_TO_SDK_TOOL_NAME: Record<string, string> = {
	read: "Read",
	write: "Write",
	edit: "Edit",
	bash: "Bash",
	grep: "Grep",
	find: "Glob",
	glob: "Glob",
};

const DEFAULT_TOOLS = ["Read", "Write", "Edit", "Bash", "Grep", "Glob"];

const BUILTIN_TOOL_NAMES = new Set(Object.keys(PI_TO_SDK_TOOL_NAME));
const TOOL_EXECUTION_DENIED_MESSAGE = "Tool execution is unavailable in this environment.";
const MCP_SERVER_NAME = "custom-tools";
const MCP_TOOL_PREFIX = `mcp__${MCP_SERVER_NAME}__`;

const SKILLS_ALIAS_GLOBAL = "~/.claude/skills";
const SKILLS_ALIAS_PROJECT = ".claude/skills";
const GLOBAL_SKILLS_ROOT = join(homedir(), ".pi", "agent", "skills");
const PROJECT_SKILLS_ROOT = join(process.cwd(), ".pi", "skills");
const GLOBAL_SETTINGS_PATH = join(homedir(), ".pi", "agent", "settings.json");
const PROJECT_SETTINGS_PATH = join(process.cwd(), ".pi", "settings.json");
const GLOBAL_AGENTS_PATH = join(homedir(), ".pi", "agent", "AGENTS.md");

const SDK_SESSION_CUSTOM_TYPE = "claude-agent-sdk";

type SdkSessionEntryData = {
	providerId?: string;
	sdkSessionId?: string;
	sdkAssistantUuid?: string;
	assistantTimestamp?: number;
	pendingToolUseTimestamp?: number | null;
	pendingToolUseIds?: string[] | null;
};

type SdkSessionState = {
	sdkSessionId?: string;
	uuidByAssistantTimestamp: Map<number, string>;
	maxTimestamp?: number;
	pendingToolUseTimestamp?: number;
	pendingToolUseIds?: string[];
};

type SessionSdkState = {
	branch: SdkSessionState;
	all: SdkSessionState;
};

const sdkStateBySessionKey = new Map<string, SessionSdkState>();

type PiSessionSdkStateCacheEntry = {
	sessionFilePath: string;
	mtimeMs: number;
	size: number;
	state: SdkSessionState;
};

const piSessionSdkStateCache = new Map<string, PiSessionSdkStateCacheEntry>();
const piSessionFilePathCache = new Map<string, string>();
const piSessionFileBySessionKey = new Map<string, string>();

let extensionApi: ExtensionAPI | undefined;

const MODELS = getModels("anthropic").map((model) => ({
	id: model.id,
	name: model.name,
	reasoning: model.reasoning,
	input: model.input,
	cost: model.cost,
	contextWindow: model.contextWindow,
	maxTokens: model.maxTokens,
}));


function buildPromptBlocks(
	context: Context,
	customToolNameToSdk: Map<string, string> | undefined,
): ContentBlockParam[] {
	const blocks: ContentBlockParam[] = [];

	const pushText = (text: string) => {
		blocks.push({ type: "text", text });
	};

	const pushImage = (image: ImageContent) => {
		blocks.push({
			type: "image",
			source: {
				type: "base64",
				media_type: image.mimeType as Base64ImageSource["media_type"],
				data: image.data,
			},
		});
	};

	const pushPrefix = (label: string) => {
		const prefix = `${blocks.length ? "\n\n" : ""}${label}\n`;
		pushText(prefix);
	};

	const appendContentBlocks = (
		content:
			| string
			| Array<{
					type: string;
					text?: string;
					data?: string;
					mimeType?: string;
				}>,
	): boolean => {
		if (typeof content === "string") {
			if (content.length > 0) {
				pushText(content);
				return content.trim().length > 0;
			}
			return false;
		}
		if (!Array.isArray(content)) return false;
		let hasText = false;
		for (const block of content) {
			if (block.type === "text") {
				const text = block.text ?? "";
				if (text.trim().length > 0) hasText = true;
				pushText(text);
				continue;
			}
			if (block.type === "image") {
				pushImage(block as ImageContent);
				continue;
			}
			pushText(`[${block.type}]`);
		}
		return hasText;
	};

	for (const message of context.messages) {
		if (message.role === "user") {
			pushPrefix("USER:");
			const hasText = appendContentBlocks(message.content);
			if (!hasText) {
				pushText("(see attached image)");
			}
			continue;
		}

		if (message.role === "assistant") {
			pushPrefix("ASSISTANT:");
			const text = contentToText(message.content, customToolNameToSdk);
			if (text.length > 0) {
				pushText(text);
			}
			continue;
		}

		if (message.role === "toolResult") {
			const header = `TOOL RESULT (historical ${mapPiToolNameToSdk(message.toolName, customToolNameToSdk)}):`;
			pushPrefix(header);
			const hasText = appendContentBlocks(message.content);
			if (!hasText) {
				pushText("(see attached image)");
			}
		}
	}

	if (!blocks.length) return [{ type: "text", text: "" }];

	return blocks;
}

function buildPromptStream(promptBlocks: ContentBlockParam[]): AsyncIterable<SDKUserMessage> {
	async function* generator() {
		const message: SDKUserMessage = {
			type: "user" as const,
			message: {
				role: "user",
				content: promptBlocks,
			} as MessageParam,
			parent_tool_use_id: null,
			session_id: "prompt",
		};

		yield message;
	}

	return generator();
}

function contentToText(
	content:
		| string
		| Array<{
			type: string;
			text?: string;
			thinking?: string;
			name?: string;
			arguments?: Record<string, unknown>;
		}>,
	customToolNameToSdk?: Map<string, string>,
): string {
	if (typeof content === "string") return content;
	if (!Array.isArray(content)) return "";
	return content
		.map((block) => {
			if (block.type === "text") return block.text ?? "";
			if (block.type === "thinking") return block.thinking ?? "";
			if (block.type === "toolCall") {
				const args = block.arguments ? JSON.stringify(block.arguments) : "{}";
				const toolName = mapPiToolNameToSdk(block.name, customToolNameToSdk);
				return `Historical tool call (non-executable): ${toolName} args=${args}`;
			}
			return `[${block.type}]`;
		})
		.join("\n");
}

function convertPiContentToBlocks(
	content:
		| string
		| Array<{
			type: string;
			text?: string;
			data?: string;
			mimeType?: string;
		}>,
	supportsImages: boolean,
): string | ContentBlockParam[] {
	if (typeof content === "string") return content;
	if (!Array.isArray(content)) return "";
	const blocks: ContentBlockParam[] = [];
	let hasText = false;
	let hasImage = false;
	for (const block of content) {
		if (block.type === "text") {
			const text = block.text ?? "";
			if (text.trim().length > 0) hasText = true;
			blocks.push({ type: "text", text });
			continue;
		}
		if (block.type === "image") {
			if (!supportsImages) continue;
			hasImage = true;
			blocks.push({
				type: "image",
				source: {
					type: "base64",
					media_type: block.mimeType as Base64ImageSource["media_type"],
					data: block.data ?? "",
				},
			} satisfies ImageBlockParam);
			continue;
		}
		blocks.push({ type: "text", text: `[${block.type}]` } as TextBlockParam);
	}
	if (!blocks.length) {
		return supportsImages ? "(see attached image)" : "(image omitted)";
	}
	if (!hasText && hasImage) {
		blocks.unshift({ type: "text", text: "(see attached image)" } as TextBlockParam);
	}
	return blocks;
}

function contentToPlainText(
	content:
		| string
		| Array<{
			type: string;
			text?: string;
			data?: string;
			mimeType?: string;
		}>,
): string {
	if (typeof content === "string") return content;
	if (!Array.isArray(content)) return "";
	let text = "";
	let hasText = false;
	let hasImage = false;
	for (const block of content) {
		if (block.type === "text") {
			const value = block.text ?? "";
			if (value.trim().length > 0) hasText = true;
			text += `${text ? "\n" : ""}${value}`;
			continue;
		}
		if (block.type === "image") {
			hasImage = true;
			continue;
		}
		text += `${text ? "\n" : ""}[${block.type}]`;
	}
	if (!hasText && hasImage) {
		return "(see attached image)";
	}
	return text;
}

function buildHistoricalSummary(messages: Context["messages"], customToolNameToSdk?: Map<string, string>): string {
	const lines: string[] = [];
	for (const message of messages) {
		if (message.role === "user") {
			const text = contentToPlainText(message.content);
			if (text.trim().length > 0) lines.push(`USER: ${text}`);
			else lines.push("USER: (see attached image)");
			continue;
		}
		if (message.role === "assistant") {
			const text = contentToText(message.content, customToolNameToSdk);
			if (text.trim().length > 0) lines.push(`ASSISTANT: ${text}`);
			continue;
		}
		if (message.role === "toolResult") {
			const toolName = mapPiToolNameToSdk(message.toolName, customToolNameToSdk);
			const text = contentToPlainText(message.content);
			if (text.trim().length > 0) {
				lines.push(`TOOL RESULT (historical ${toolName}): ${text}`);
			} else {
				lines.push(`TOOL RESULT (historical ${toolName}): (see attached image)`);
			}
		}
	}
	if (!lines.length) return "";
	return `Historical context (non-executable):\n${lines.join("\n")}`;
}

function buildPromptWithSummary(
	summaryText: string,
	userMessage: Extract<Context["messages"][number], { role: "user" }> | undefined,
	supportsImages: boolean,
): AsyncIterable<SDKUserMessage> | string {
	if (!summaryText && !userMessage) return "";

	async function* generator() {
		const trimmedSummary = summaryText.trim();
		if (!userMessage) {
			if (trimmedSummary.length > 0) {
				yield {
					type: "user" as const,
					message: { role: "user", content: trimmedSummary } as MessageParam,
					parent_tool_use_id: null,
					session_id: "prompt",
				};
			}
			return;
		}

		const content = convertPiContentToBlocks(userMessage.content, supportsImages);
		if (typeof content === "string") {
			const trimmedUser = content.trim();
			const parts = [trimmedSummary].filter((part) => part.length > 0);
			if (trimmedUser.length > 0) {
				parts.push(`---\nLatest user message:\n${trimmedUser}`);
			}
			if (parts.length > 0) {
				yield {
					type: "user" as const,
					message: { role: "user", content: parts.join("\n\n") } as MessageParam,
					parent_tool_use_id: null,
					session_id: "prompt",
				};
			}
			return;
		}

		if (content.length > 0) {
			const blocks: ContentBlockParam[] = [];
			if (trimmedSummary.length > 0) {
				blocks.push({
					type: "text",
					text: `${trimmedSummary}\n\n---\nLatest user message:`,
				} as TextBlockParam);
			}
			blocks.push(...content);
			yield {
				type: "user" as const,
				message: { role: "user", content: blocks } as MessageParam,
				parent_tool_use_id: null,
				session_id: "prompt",
			};
			return;
		}

		if (trimmedSummary.length > 0) {
			yield {
				type: "user" as const,
				message: { role: "user", content: trimmedSummary } as MessageParam,
				parent_tool_use_id: null,
				session_id: "prompt",
			};
		}
	}

	return generator();
}

function normalizeToolResultContent(
	content: string | ContentBlockParam[],
	isError: boolean,
): string | ContentBlockParam[] {
	const fallback = isError ? "(tool error with no output)" : "(no output)";
	if (typeof content === "string") {
		return content.trim().length > 0 ? content : fallback;
	}
	if (!Array.isArray(content) || content.length === 0) {
		return fallback;
	}
	const hasImage = content.some((block) => block.type === "image");
	const hasText = content.some((block) => block.type === "text" && "text" in block && block.text?.trim());
	if (!hasImage && !hasText) {
		return [{ type: "text", text: fallback }];
	}
	return content;
}

function collectToolResultIds(messages: Context["messages"]): Set<string> {
	const ids = new Set<string>();
	for (const message of messages) {
		if (message.role !== "toolResult") continue;
		const id = (message as Extract<Context["messages"][number], { role: "toolResult" }>).toolCallId;
		if (typeof id === "string" && id.trim().length > 0) {
			ids.add(id);
		}
	}
	return ids;
}

type ResumeTailPlan = {
	tailHasAssistant: boolean;
	summaryMessages: Context["messages"];
	userMessage?: Extract<Context["messages"][number], { role: "user" }>;
	shouldReplayPendingToolResults: boolean;
	shouldUseSummaryPrompt: boolean;
	tailAllowedToolUseIds?: Set<string>;
};

function analyzeResumeTailMessages(
	tailMessages: Context["messages"],
	pendingToolUseTimestamp?: number,
): ResumeTailPlan {
	const tailHasAssistant = tailMessages.some((message) => message.role === "assistant");
	let lastUserIndex = -1;
	for (let i = tailMessages.length - 1; i >= 0; i -= 1) {
		if (tailMessages[i]?.role === "user") {
			lastUserIndex = i;
			break;
		}
	}

	const summaryMessages = lastUserIndex >= 0 ? tailMessages.slice(0, lastUserIndex) : tailMessages;
	const userMessage =
		lastUserIndex >= 0
			? (tailMessages[lastUserIndex] as Extract<Context["messages"][number], { role: "user" }>)
			: undefined;
	const shouldReplayPendingToolResults = pendingToolUseTimestamp != null && !tailHasAssistant;
	const shouldUseSummaryPrompt = summaryMessages.length > 0 && !shouldReplayPendingToolResults;
	const tailAllowedToolUseIds = !shouldUseSummaryPrompt ? collectToolResultIds(tailMessages) : undefined;

	return {
		tailHasAssistant,
		summaryMessages,
		userMessage,
		shouldReplayPendingToolResults,
		shouldUseSummaryPrompt,
		tailAllowedToolUseIds,
	};
}

function hasReplayableTailMessages(tailMessages: Context["messages"] | undefined): boolean {
	if (!tailMessages || tailMessages.length === 0) {
		return false;
	}
	for (const message of tailMessages) {
		if (message.role === "user" || message.role === "toolResult") {
			return true;
		}
	}
	return false;
}

type ResumeForkPlan = {
	resumeSessionAt?: string;
	forkSession?: boolean;
};

function computeResumeForkPlan(
	branchState: SdkSessionState | undefined,
	allState: SdkSessionState | undefined,
): ResumeForkPlan {
	let resumeSessionAt: string | undefined;
	let forkSession: boolean | undefined;

	if (
		branchState?.maxTimestamp != null &&
		allState?.maxTimestamp != null &&
		branchState.maxTimestamp < allState.maxTimestamp
	) {
		resumeSessionAt = branchState.uuidByAssistantTimestamp.get(branchState.maxTimestamp);
		forkSession = Boolean(resumeSessionAt);
	}

	if (branchState?.pendingToolUseTimestamp != null) {
		const pendingUuid = branchState.uuidByAssistantTimestamp.get(branchState.pendingToolUseTimestamp);
		if (pendingUuid) {
			resumeSessionAt = pendingUuid;
			forkSession = true;
		}
	}

	return { resumeSessionAt, forkSession };
}

function buildResumePromptFromTail(
	tailMessages: Context["messages"],
	supportsImages: boolean,
	allowedToolUseIds?: Set<string>,
): AsyncIterable<SDKUserMessage> | string {
	if (!tailMessages.length) return "";

	async function* generator() {
		let index = 0;
		while (index < tailMessages.length) {
			const message = tailMessages[index];
			if (message.role === "user") {
				const content = convertPiContentToBlocks(message.content, supportsImages);
				if (typeof content === "string") {
					if (content.trim().length > 0) {
						yield {
							type: "user" as const,
							message: { role: "user", content } as MessageParam,
							parent_tool_use_id: null,
							session_id: "prompt",
						};
					}
				} else if (content.length > 0) {
					yield {
						type: "user" as const,
						message: { role: "user", content } as MessageParam,
						parent_tool_use_id: null,
						session_id: "prompt",
					};
				}
				index += 1;
				continue;
			}

			if (message.role === "toolResult") {
				const toolResults: ContentBlockParam[] = [];
				const toolResultIndexById = new Map<string, number>();
				const skippedSummaries: string[] = [];
				while (index < tailMessages.length && tailMessages[index]?.role === "toolResult") {
					const toolMessage = tailMessages[index] as Extract<
						Context["messages"][number],
						{ role: "toolResult" }
					>;
					const shouldInclude = !allowedToolUseIds || allowedToolUseIds.has(toolMessage.toolCallId);
					if (shouldInclude) {
						const content = convertPiContentToBlocks(toolMessage.content, supportsImages);
						const normalizedContent = normalizeToolResultContent(
							content,
							Boolean(toolMessage.isError),
						);
						const existingIndex = toolResultIndexById.get(toolMessage.toolCallId);
						const toolResult: ContentBlockParam = {
							type: "tool_result",
							tool_use_id: toolMessage.toolCallId,
							content: normalizedContent,
							is_error: toolMessage.isError,
						} as ContentBlockParam;
						if (existingIndex == null) {
							toolResultIndexById.set(toolMessage.toolCallId, toolResults.length);
							toolResults.push(toolResult);
						} else {
							toolResults[existingIndex] = toolResult;
						}
					} else {
						const plain = contentToPlainText(toolMessage.content).trim();
						skippedSummaries.push(
							`TOOL RESULT (already recorded ${toolMessage.toolName}, id=${toolMessage.toolCallId}): ${plain || (toolMessage.isError ? "(tool error with no output)" : "(no output)")}`,
						);
					}
					index += 1;
				}
				if (toolResults.length > 0) {
					yield {
						type: "user" as const,
						message: { role: "user", content: toolResults } as MessageParam,
						parent_tool_use_id: null,
						session_id: "prompt",
					};
				} else if (skippedSummaries.length > 0) {
					yield {
						type: "user" as const,
						message: {
							role: "user",
							content: `Continue using the already-recorded tool outputs:\n${skippedSummaries.join("\n")}`,
						} as MessageParam,
						parent_tool_use_id: null,
						session_id: "prompt",
					};
				}
				continue;
			}

			index += 1;
		}
	}

	return generator();
}

function isErroredAssistantMessage(message: Extract<Context["messages"][number], { role: "assistant" }>): boolean {
	if (message.stopReason === "error") return true;
	if (typeof message.errorMessage === "string" && message.errorMessage.trim().length > 0) return true;
	const text = contentToText(message.content).trim().toLowerCase();
	if (!text) return false;
	if (text.startsWith("api error:")) return true;
	if (text.includes("does not support assistant message prefill")) return true;
	return false;
}

function findLastSdkAssistantInfo(
	messages: Context["messages"],
	state: SdkSessionState | undefined,
): { index: number; timestamp: number; uuid: string } | undefined {
	if (!state) return undefined;
	for (let i = messages.length - 1; i >= 0; i -= 1) {
		const message = messages[i];
		if (message?.role !== "assistant") continue;
		if (isErroredAssistantMessage(message)) continue;
		const timestamp = message.timestamp;
		const uuid = state.uuidByAssistantTimestamp.get(timestamp);
		if (uuid) {
			return { index: i, timestamp, uuid };
		}
	}
	return undefined;
}

function mapPiToolNameToSdk(name?: string, customToolNameToSdk?: Map<string, string>): string {
	if (!name) return "";
	const normalized = name.toLowerCase();
	if (customToolNameToSdk) {
		const mapped = customToolNameToSdk.get(name) ?? customToolNameToSdk.get(normalized);
		if (mapped) return mapped;
	}
	if (PI_TO_SDK_TOOL_NAME[normalized]) return PI_TO_SDK_TOOL_NAME[normalized];
	return pascalCase(name);
}

type ProviderSettings = {
	appendSystemPrompt?: boolean;

	/**
	 * Controls which filesystem-based configuration sources the SDK loads settings from
	 * (maps to Claude Code CLI --setting-sources)
	 *
	 * - "user"    => ~/.claude (or CLAUDE_CONFIG_DIR)
	 * - "project" => .claude in the current repo
	 * - "local"   => .claude/settings.local.json in the current repo
	 */
	settingSources?: SettingSource[];

	/**
	 * When true, pass Claude Code CLI --strict-mcp-config to ignore MCP servers from ~/.claude.json
	 * and project .mcp.json files. This prevents Claude Code from auto-injecting large MCP tool
	 * schemas (a major token cost) when appendSystemPrompt=false.
	 */
	strictMcpConfig?: boolean;
};

function extractSkillsAppend(systemPrompt?: string): string | undefined {
	if (!systemPrompt) return undefined;
	const startMarker = "The following skills provide specialized instructions for specific tasks.";
	const endMarker = "</available_skills>";
	const startIndex = systemPrompt.indexOf(startMarker);
	if (startIndex === -1) return undefined;
	const endIndex = systemPrompt.indexOf(endMarker, startIndex);
	if (endIndex === -1) return undefined;
	const skillsBlock = systemPrompt.slice(startIndex, endIndex + endMarker.length).trim();
	return rewriteSkillsLocations(skillsBlock);
}

function loadProviderSettings(): ProviderSettings {
	const globalSettings = readSettingsFile(GLOBAL_SETTINGS_PATH);
	const projectSettings = readSettingsFile(PROJECT_SETTINGS_PATH);
	return { ...globalSettings, ...projectSettings };
}

function readSettingsFile(filePath: string): ProviderSettings {
	if (!existsSync(filePath)) return {};
	try {
		const raw = readFileSync(filePath, "utf-8");
		const parsed = JSON.parse(raw) as Record<string, unknown>;
		const settingsBlock =
			(parsed["claudeAgentSdkProvider"] as Record<string, unknown> | undefined) ??
			(parsed["claude-agent-sdk-provider"] as Record<string, unknown> | undefined) ??
			(parsed["claudeAgentSdk"] as Record<string, unknown> | undefined);
		if (!settingsBlock || typeof settingsBlock !== "object") return {};
		const appendSystemPrompt =
			typeof settingsBlock["appendSystemPrompt"] === "boolean"
				? settingsBlock["appendSystemPrompt"]
				: undefined;

		const settingSourcesRaw = settingsBlock["settingSources"];
		const settingSources =
			Array.isArray(settingSourcesRaw) &&
			settingSourcesRaw.every(
				(value) =>
					typeof value === "string" && (value === "user" || value === "project" || value === "local"),
			)
				? (settingSourcesRaw as SettingSource[])
				: undefined;

		const strictMcpConfig =
			typeof settingsBlock["strictMcpConfig"] === "boolean" ? settingsBlock["strictMcpConfig"] : undefined;

		const legacyDisable = false;
		return {
			appendSystemPrompt: appendSystemPrompt ?? (legacyDisable ? false : undefined),
			settingSources,
			strictMcpConfig,
		};
	} catch {
		return {};
	}
}

function buildSessionKey(options: SimpleStreamOptions | undefined, cwd: string): string | undefined {
	const sessionId = (options as { sessionId?: string } | undefined)?.sessionId;
	if (typeof sessionId === "string" && sessionId.trim().length > 0) {
		return `session:${sessionId.trim()}`;
	}
	if (cwd && cwd.trim().length > 0) {
		return `cwd:${cwd}`;
	}
	return undefined;
}

function getSessionKeyFromManager(sessionManager: { getSessionId: () => string }): string {
	return `session:${sessionManager.getSessionId()}`;
}

function createEmptySdkState(): SdkSessionState {
	return {
		sdkSessionId: undefined,
		uuidByAssistantTimestamp: new Map(),
		maxTimestamp: undefined,
		pendingToolUseTimestamp: undefined,
		pendingToolUseIds: undefined,
	};
}

function cloneSdkState(state: SdkSessionState): SdkSessionState {
	return {
		sdkSessionId: state.sdkSessionId,
		uuidByAssistantTimestamp: new Map(state.uuidByAssistantTimestamp),
		maxTimestamp: state.maxTimestamp,
		pendingToolUseTimestamp: state.pendingToolUseTimestamp,
		pendingToolUseIds: state.pendingToolUseIds ? [...state.pendingToolUseIds] : undefined,
	};
}

function buildSdkStateFromEntries(entries: Array<Record<string, any>>): SdkSessionState {
	const state = createEmptySdkState();
	for (const entry of entries) {
		if (entry?.type !== "custom" || entry?.customType !== SDK_SESSION_CUSTOM_TYPE) continue;
		const data = entry?.data as SdkSessionEntryData | undefined;
		if (!data || typeof data !== "object") continue;
		if (typeof data.sdkSessionId === "string" && data.sdkSessionId.trim().length > 0) {
			state.sdkSessionId = data.sdkSessionId;
		}
		if (
			typeof data.assistantTimestamp === "number" &&
			Number.isFinite(data.assistantTimestamp) &&
			typeof data.sdkAssistantUuid === "string" &&
			data.sdkAssistantUuid.trim().length > 0
		) {
			state.uuidByAssistantTimestamp.set(data.assistantTimestamp, data.sdkAssistantUuid);
			if (state.maxTimestamp == null || data.assistantTimestamp > state.maxTimestamp) {
				state.maxTimestamp = data.assistantTimestamp;
			}
		}
		if (data.pendingToolUseTimestamp === null) {
			state.pendingToolUseTimestamp = undefined;
		} else if (
			typeof data.pendingToolUseTimestamp === "number" &&
			Number.isFinite(data.pendingToolUseTimestamp)
		) {
			state.pendingToolUseTimestamp = data.pendingToolUseTimestamp;
		}
		if (data.pendingToolUseIds === null) {
			state.pendingToolUseIds = undefined;
		} else if (Array.isArray(data.pendingToolUseIds)) {
			state.pendingToolUseIds = data.pendingToolUseIds.filter(
				(id): id is string => typeof id === "string" && id.trim().length > 0,
			);
		}
	}
	return state;
}

function rebuildSessionState(sessionKey: string, entries: Array<Record<string, any>>, branchEntries: Array<Record<string, any>>): void {
	const branchState = buildSdkStateFromEntries(branchEntries);
	const allState = buildSdkStateFromEntries(entries);
	if (!branchState.sdkSessionId && allState.sdkSessionId) {
		branchState.sdkSessionId = allState.sdkSessionId;
	}
	sdkStateBySessionKey.set(sessionKey, { branch: branchState, all: allState });
}

function updateSessionState(sessionKey: string, data: SdkSessionEntryData): void {
	const state = sdkStateBySessionKey.get(sessionKey) ?? {
		branch: createEmptySdkState(),
		all: createEmptySdkState(),
	};

	const apply = (target: SdkSessionState) => {
		if (typeof data.sdkSessionId === "string" && data.sdkSessionId.trim().length > 0) {
			target.sdkSessionId = data.sdkSessionId;
		}
		if (
			typeof data.assistantTimestamp === "number" &&
			Number.isFinite(data.assistantTimestamp) &&
			typeof data.sdkAssistantUuid === "string" &&
			data.sdkAssistantUuid.trim().length > 0
		) {
			target.uuidByAssistantTimestamp.set(data.assistantTimestamp, data.sdkAssistantUuid);
			if (target.maxTimestamp == null || data.assistantTimestamp > target.maxTimestamp) {
				target.maxTimestamp = data.assistantTimestamp;
			}
		}
		if (data.pendingToolUseTimestamp === null) {
			target.pendingToolUseTimestamp = undefined;
		} else if (
			typeof data.pendingToolUseTimestamp === "number" &&
			Number.isFinite(data.pendingToolUseTimestamp)
		) {
			target.pendingToolUseTimestamp = data.pendingToolUseTimestamp;
		}
		if (data.pendingToolUseIds === null) {
			target.pendingToolUseIds = undefined;
		} else if (Array.isArray(data.pendingToolUseIds)) {
			target.pendingToolUseIds = data.pendingToolUseIds.filter(
				(id): id is string => typeof id === "string" && id.trim().length > 0,
			);
		}
	};

	apply(state.branch);
	apply(state.all);
	sdkStateBySessionKey.set(sessionKey, state);
}

function persistSdkEntry(sessionKey: string | undefined, data: SdkSessionEntryData): void {
	if (!sessionKey) return;
	updateSessionState(sessionKey, data);
	if (!extensionApi) return;
	try {
		extensionApi.appendEntry(SDK_SESSION_CUSTOM_TYPE, data);
	} catch {
		// ignore persistence errors
	}
}

function getSessionState(sessionKey: string): SessionSdkState | undefined {
	return sdkStateBySessionKey.get(sessionKey);
}

function refreshSessionState(ctx: {
	sessionManager: {
		getSessionId: () => string;
		getEntries: () => any[];
		getBranch: () => any[];
		getSessionFile?: () => string | undefined;
	};
}): void {
	const sessionKey = getSessionKeyFromManager(ctx.sessionManager);
	rebuildSessionState(sessionKey, ctx.sessionManager.getEntries(), ctx.sessionManager.getBranch());
	const sessionFile = ctx.sessionManager.getSessionFile?.();
	if (typeof sessionFile === "string" && sessionFile.trim().length > 0) {
		piSessionFileBySessionKey.set(sessionKey, sessionFile);
	}
}

function getSdkSessionFilePath(sessionId: string, cwd: string): string {
	let projectDir = cwd.replace(/[\\/]+/g, "-");
	if (!projectDir.startsWith("-")) projectDir = `-${projectDir}`;
	return join(homedir(), ".claude", "projects", projectDir, `${sessionId}.jsonl`);
}

function getExistingToolResultIds(sessionId: string, cwd: string): Set<string> {
	const ids = new Set<string>();
	try {
		const sessionFilePath = getSdkSessionFilePath(sessionId, cwd);
		if (!existsSync(sessionFilePath)) return ids;
		const lines = readFileSync(sessionFilePath, "utf-8").split("\n");
		for (const line of lines) {
			if (!line.trim()) continue;
			let parsed: any;
			try {
				parsed = JSON.parse(line);
			} catch {
				continue;
			}
			if (parsed?.type !== "user") continue;
			const content = parsed?.message?.content;
			if (!Array.isArray(content)) continue;
			for (const block of content) {
				if (
					block?.type === "tool_result" &&
					typeof block?.tool_use_id === "string" &&
					block.tool_use_id.trim().length > 0
				) {
					ids.add(block.tool_use_id);
				}
			}
		}
	} catch {
		// ignore parse/read failures
	}
	return ids;
}

function getPiSessionFilePath(sessionId: string, cwd: string): string | undefined {
	const cacheKey = `${cwd}\u0000${sessionId}`;
	const cachedPath = piSessionFilePathCache.get(cacheKey);
	if (cachedPath && existsSync(cachedPath)) {
		return cachedPath;
	}

	const safePath = `--${cwd.replace(/^[/\\]/, "").replace(/[/\\:]/g, "-")}--`;
	const sessionDir = join(homedir(), ".pi", "agent", "sessions", safePath);
	if (!existsSync(sessionDir)) return undefined;

	const suffix = `_${sessionId}.jsonl`;
	const candidates = readdirSync(sessionDir)
		.filter((file) => file.endsWith(suffix))
		.map((file) => join(sessionDir, file));
	if (!candidates.length) return undefined;

	candidates.sort((a, b) => {
		try {
			return statSync(b).mtimeMs - statSync(a).mtimeMs;
		} catch {
			return 0;
		}
	});

	const latestPath = candidates[0];
	if (latestPath) {
		piSessionFilePathCache.set(cacheKey, latestPath);
	}
	return latestPath;
}

function getAllSdkStateFromPiSession(sessionKey: string, sessionId: string, cwd: string): SdkSessionState | undefined {
	const knownSessionFilePath = piSessionFileBySessionKey.get(sessionKey);
	const sessionFilePath =
		typeof knownSessionFilePath === "string" && knownSessionFilePath.trim().length > 0 && existsSync(knownSessionFilePath)
			? knownSessionFilePath
			: getPiSessionFilePath(sessionId, cwd);
	if (!sessionFilePath || !existsSync(sessionFilePath)) return undefined;

	const cacheKey = sessionFilePath;

	let stats: { mtimeMs: number; size: number } | undefined;
	try {
		const stat = statSync(sessionFilePath);
		stats = { mtimeMs: stat.mtimeMs, size: stat.size };
	} catch {
		return undefined;
	}

	const cached = piSessionSdkStateCache.get(cacheKey);
	if (
		cached &&
		cached.sessionFilePath === sessionFilePath &&
		cached.mtimeMs === stats.mtimeMs &&
		cached.size === stats.size
	) {
		return cloneSdkState(cached.state);
	}

	const sdkEntries: Array<Record<string, any>> = [];
	try {
		const lines = readFileSync(sessionFilePath, "utf-8").split("\n");
		for (const line of lines) {
			if (!line.trim()) continue;
			if (!line.includes('"customType"')) continue;
			if (!line.includes(SDK_SESSION_CUSTOM_TYPE)) continue;
			try {
				sdkEntries.push(JSON.parse(line) as Record<string, any>);
			} catch {
				// ignore malformed lines
			}
		}
	} catch {
		return undefined;
	}

	const state = buildSdkStateFromEntries(sdkEntries);
	piSessionSdkStateCache.set(cacheKey, {
		sessionFilePath,
		mtimeMs: stats.mtimeMs,
		size: stats.size,
		state,
	});
	return cloneSdkState(state);
}

function syncAllSdkStateFromPiSession(sessionKey: string, sessionId: string, cwd: string): void {
	const diskAllState = getAllSdkStateFromPiSession(sessionKey, sessionId, cwd);
	if (!diskAllState) return;

	const current = getSessionState(sessionKey);
	if (!current) {
		sdkStateBySessionKey.set(sessionKey, {
			branch: cloneSdkState(diskAllState),
			all: cloneSdkState(diskAllState),
		});
		return;
	}

	current.all = diskAllState;
	if (!current.branch.sdkSessionId && diskAllState.sdkSessionId) {
		current.branch.sdkSessionId = diskAllState.sdkSessionId;
	}
	sdkStateBySessionKey.set(sessionKey, current);
}

function setKnownPiSessionFileForSessionKey(sessionKey: string, sessionFilePath: string): void {
	if (!sessionKey || !sessionFilePath) return;
	piSessionFileBySessionKey.set(sessionKey, sessionFilePath);
}

function clearInternalStateForTests(): void {
	sdkStateBySessionKey.clear();
	piSessionSdkStateCache.clear();
	piSessionFilePathCache.clear();
	piSessionFileBySessionKey.clear();
}

function rewriteSkillsLocations(skillsBlock: string): string {
	return skillsBlock.replace(/<location>([^<]+)<\/location>/g, (_match, location: string) => {
		let rewritten = location;
		if (location.startsWith(GLOBAL_SKILLS_ROOT)) {
			const relPath = relative(GLOBAL_SKILLS_ROOT, location).replace(/^\.+/, "");
			rewritten = `${SKILLS_ALIAS_GLOBAL}/${relPath}`.replace(/\/\/+/g, "/");
		} else if (location.startsWith(PROJECT_SKILLS_ROOT)) {
			const relPath = relative(PROJECT_SKILLS_ROOT, location).replace(/^\.+/, "");
			rewritten = `${SKILLS_ALIAS_PROJECT}/${relPath}`.replace(/\/\/+/g, "/");
		}
		return `<location>${rewritten}</location>`;
	});
}

function resolveAgentsMdPath(): string | undefined {
	const fromCwd = findAgentsMdInParents(process.cwd());
	if (fromCwd) return fromCwd;
	if (existsSync(GLOBAL_AGENTS_PATH)) return GLOBAL_AGENTS_PATH;
	return undefined;
}

function findAgentsMdInParents(startDir: string): string | undefined {
	let current = resolve(startDir);
	while (true) {
		const candidate = join(current, "AGENTS.md");
		if (existsSync(candidate)) return candidate;
		const parent = dirname(current);
		if (parent === current) break;
		current = parent;
	}
	return undefined;
}

function extractAgentsAppend(): string | undefined {
	const agentsPath = resolveAgentsMdPath();
	if (!agentsPath) return undefined;
	try {
		const content = readFileSync(agentsPath, "utf-8").trim();
		if (!content) return undefined;
		const sanitized = sanitizeAgentsContent(content);
		return sanitized.length > 0 ? `# CLAUDE.md\n\n${sanitized}` : undefined;
	} catch {
		return undefined;
	}
}

function sanitizeAgentsContent(content: string): string {
	let sanitized = content;
	sanitized = sanitized.replace(/~\/\.pi\b/gi, "~/.claude");
	sanitized = sanitized.replace(/(^|[\s'"`])\.pi\//g, "$1.claude/");
	sanitized = sanitized.replace(/\b\.pi\b/gi, ".claude");
	sanitized = sanitized.replace(/\bpi\b/gi, "environment");
	return sanitized;
}

function rewriteSkillAliasPath(pathValue: unknown): unknown {
	if (typeof pathValue !== "string") return pathValue;
	if (pathValue.startsWith(SKILLS_ALIAS_GLOBAL)) {
		return pathValue.replace(SKILLS_ALIAS_GLOBAL, "~/.pi/agent/skills");
	}
	if (pathValue.startsWith(`./${SKILLS_ALIAS_PROJECT}`)) {
		return pathValue.replace(`./${SKILLS_ALIAS_PROJECT}`, PROJECT_SKILLS_ROOT);
	}
	if (pathValue.startsWith(SKILLS_ALIAS_PROJECT)) {
		return pathValue.replace(SKILLS_ALIAS_PROJECT, PROJECT_SKILLS_ROOT);
	}
	const projectAliasAbs = join(process.cwd(), SKILLS_ALIAS_PROJECT);
	if (pathValue.startsWith(projectAliasAbs)) {
		return pathValue.replace(projectAliasAbs, PROJECT_SKILLS_ROOT);
	}
	return pathValue;
}

function mapToolName(name: string, customToolNameToPi?: Map<string, string>): string {
	const normalized = name.toLowerCase();
	const builtin = SDK_TO_PI_TOOL_NAME[normalized];
	if (builtin) return builtin;
	if (customToolNameToPi) {
		const mapped = customToolNameToPi.get(name) ?? customToolNameToPi.get(normalized);
		if (mapped) return mapped;
	}
	if (normalized.startsWith(MCP_TOOL_PREFIX)) {
		return name.slice(MCP_TOOL_PREFIX.length);
	}
	return name;
}

function mapToolArgs(
	toolName: string,
	args: Record<string, unknown> | undefined,
	allowSkillAliasRewrite = true,
): Record<string, unknown> {
	const normalized = toolName.toLowerCase();
	const input = args ?? {};
	const resolvePath = (value: unknown) => (allowSkillAliasRewrite ? rewriteSkillAliasPath(value) : value);

	switch (normalized) {
		case "read":
			return {
				path: resolvePath(input.file_path ?? input.path),
				offset: input.offset,
				limit: input.limit,
			};
		case "write":
			return {
				path: resolvePath(input.file_path ?? input.path),
				content: input.content,
			};
		case "edit":
			return {
				path: resolvePath(input.file_path ?? input.path),
				oldText: input.old_string ?? input.oldText ?? input.old_text,
				newText: input.new_string ?? input.newText ?? input.new_text,
			};
		case "bash":
			return {
				command: input.command,
				timeout: input.timeout,
			};
		case "grep":
			return {
				pattern: input.pattern,
				path: resolvePath(input.path),
				glob: input.glob,
				limit: input.head_limit ?? input.limit,
			};
		case "find":
			return {
				pattern: input.pattern,
				path: resolvePath(input.path),
			};
		default:
			return input;
	}
}

function resolveSdkTools(context: Context): {
	sdkTools: string[];
	customTools: Tool[];
	customToolNameToSdk: Map<string, string>;
	customToolNameToPi: Map<string, string>;
} {
	if (!context.tools) {
		return {
			sdkTools: [...DEFAULT_TOOLS],
			customTools: [],
			customToolNameToSdk: new Map(),
			customToolNameToPi: new Map(),
		};
	}

	const sdkTools = new Set<string>();
	const customTools: Tool[] = [];
	const customToolNameToSdk = new Map<string, string>();
	const customToolNameToPi = new Map<string, string>();

	for (const tool of context.tools) {
		const normalized = tool.name.toLowerCase();
		if (BUILTIN_TOOL_NAMES.has(normalized)) {
			const sdkName = PI_TO_SDK_TOOL_NAME[normalized];
			if (sdkName) sdkTools.add(sdkName);
			continue;
		}
		const sdkName = `${MCP_TOOL_PREFIX}${tool.name}`;
		customTools.push(tool);
		customToolNameToSdk.set(tool.name, sdkName);
		customToolNameToSdk.set(normalized, sdkName);
		customToolNameToPi.set(sdkName, tool.name);
		customToolNameToPi.set(sdkName.toLowerCase(), tool.name);
	}

	return { sdkTools: Array.from(sdkTools), customTools, customToolNameToSdk, customToolNameToPi };
}

function buildCustomToolServers(customTools: Tool[]): Record<string, ReturnType<typeof createSdkMcpServer>> | undefined {
	if (!customTools.length) return undefined;

	const mcpTools = customTools.map((tool) => ({
		name: tool.name,
		description: tool.description,
		inputSchema: tool.parameters as unknown,
		handler: async () => ({
			content: [{ type: "text", text: TOOL_EXECUTION_DENIED_MESSAGE }],
			isError: true,
		}),
	}));

	const server = createSdkMcpServer({
		name: MCP_SERVER_NAME,
		version: "1.0.0",
		tools: mcpTools,
	});

	return { [MCP_SERVER_NAME]: server };
}

function mapStopReason(reason: string | undefined): "stop" | "length" | "toolUse" {
	switch (reason) {
		case "tool_use":
			return "toolUse";
		case "max_tokens":
			return "length";
		case "end_turn":
		default:
			return "stop";
	}
}

type ThinkingLevel = NonNullable<SimpleStreamOptions["reasoning"]>;
type NonXhighThinkingLevel = Exclude<ThinkingLevel, "xhigh">;

const DEFAULT_THINKING_BUDGETS: Record<NonXhighThinkingLevel, number> = {
	minimal: 2048,
	low: 8192,
	medium: 16384,
	high: 31999,
};

// NOTE: "xhigh" is unavailable in the TUI because pi-ai's supportsXhigh()
// doesn't recognize the "claude-agent-sdk" api type. As a workaround, opus-4-6
// gets shifted budgets so "high" uses the budget that xhigh would normally use.
const OPUS_46_THINKING_BUDGETS: Record<ThinkingLevel, number> = {
	minimal: 2048,
	low: 8192,
	medium: 31999,
	high: 63999,
	// Future-proofing: pi currently won't surface "xhigh" for this provider because
	// pi-ai's supportsXhigh() doesn't recognize the "claude-agent-sdk" api type.
	// If/when that changes, we can shift the budgets to 2048, 8192, 16384, 31999, 63999.
	xhigh: 63999,
};

function mapThinkingTokens(
	reasoning?: ThinkingLevel,
	modelId?: string,
	thinkingBudgets?: SimpleStreamOptions["thinkingBudgets"],
): number | undefined {
	if (!reasoning) return undefined;

	const isOpus46 = modelId?.includes("opus-4-6") || modelId?.includes("opus-4.6");
	if (isOpus46) {
		return OPUS_46_THINKING_BUDGETS[reasoning];
	}

	const effectiveReasoning: NonXhighThinkingLevel = reasoning === "xhigh" ? "high" : reasoning;

	const customBudgets = thinkingBudgets as (Partial<Record<NonXhighThinkingLevel, number>> | undefined);
	const customBudget = customBudgets?.[effectiveReasoning];
	if (typeof customBudget === "number" && Number.isFinite(customBudget) && customBudget > 0) {
		return customBudget;
	}

	return DEFAULT_THINKING_BUDGETS[effectiveReasoning];
}

function parsePartialJson(input: string, fallback: Record<string, unknown>): Record<string, unknown> {
	if (!input) return fallback;
	try {
		return JSON.parse(input);
	} catch {
		return fallback;
	}
}

function hasToolInputArgs(input: Record<string, unknown> | undefined): boolean {
	if (!input || typeof input !== "object") return false;
	return Object.keys(input).length > 0;
}

function sanitizeAssistantContentForEmit(output: AssistantMessage): void {
	if (!Array.isArray(output.content) || output.content.length === 0) return;
	const sanitized = output.content.filter((block) => {
		if (block.type !== "toolCall") return true;
		if ("index" in (block as any)) return false;
		if ("partialJson" in (block as any)) return false;
		if (!hasToolInputArgs((block as any).arguments)) return false;
		return true;
	});
	(output.content as any) = sanitized;
	if (output.stopReason === "toolUse") {
		const hasToolCall = sanitized.some((block) => block.type === "toolCall");
		if (!hasToolCall) {
			output.stopReason = "stop";
		}
	}
}

function isIgnorableTransportWriteAfterCloseError(error: unknown): boolean {
	if (!(error instanceof Error)) return false;
	return error.message.includes("ProcessTransport is not ready for writing");
}

function streamClaudeAgentSdk(model: Model<any>, context: Context, options?: SimpleStreamOptions): AssistantMessageEventStream {
	const stream = createAssistantMessageEventStream();

	(async () => {
		const output: AssistantMessage = {
			role: "assistant",
			content: [],
			api: model.api,
			provider: model.provider,
			model: model.id,
			usage: {
				input: 0,
				output: 0,
				cacheRead: 0,
				cacheWrite: 0,
				totalTokens: 0,
				cost: { input: 0, output: 0, cacheRead: 0, cacheWrite: 0, total: 0 },
			},
			stopReason: "stop",
			timestamp: Date.now(),
		};

		let sdkQuery: ReturnType<typeof query> | undefined;
		let wasAborted = false;
		const requestAbort = () => {
			if (!sdkQuery) return;
			try {
				sdkQuery.close();
			} catch {
				// ignore shutdown errors
			}
		};
		const requestClose = () => {
			if (!sdkQuery) return;
			try {
				sdkQuery.close();
			} catch {
				// ignore shutdown errors
			}
		};
		const onAbort = () => {
			wasAborted = true;
			requestAbort();
		};
		if (options?.signal) {
			if (options.signal.aborted) onAbort();
			else options.signal.addEventListener("abort", onAbort, { once: true });
		}

		const blocks = output.content as Array<
			| { type: "text"; text: string; index: number }
			| { type: "thinking"; thinking: string; thinkingSignature?: string; index: number }
			| {
				type: "toolCall";
				id: string;
				name: string;
				arguments: Record<string, unknown>;
				partialJson: string;
				index: number;
			}
		>;

		let started = false;
		let sawStreamEvent = false;
		let sawToolCall = false;
		let shouldStopEarly = false;
		let abortedForToolCall = false;
		const pendingToolUseIds = new Set<string>();
		const announcedToolCallIndices = new Set<number>();
		const emittedToolCallDeltaIndices = new Set<number>();
		const clearStaleBlockIndex = (eventIndex: number) => {
			for (const existingBlock of blocks) {
				if ((existingBlock as any).index === eventIndex) {
					delete (existingBlock as any).index;
				}
			}
		};
		const findLatestBlockIndex = (eventIndex: number) => {
			for (let i = blocks.length - 1; i >= 0; i -= 1) {
				if ((blocks[i] as any).index === eventIndex) return i;
			}
			return -1;
		};

		try {
			const { sdkTools, customTools, customToolNameToSdk, customToolNameToPi } = resolveSdkTools(context);

			const cwd = (options as { cwd?: string } | undefined)?.cwd ?? process.cwd();
			const piSessionId = (options as { sessionId?: string } | undefined)?.sessionId;
			const sessionKey = buildSessionKey(options, cwd);
			if (sessionKey && typeof piSessionId === "string" && piSessionId.trim().length > 0) {
				syncAllSdkStateFromPiSession(sessionKey, piSessionId.trim(), cwd);
			}
			const sessionState = sessionKey ? getSessionState(sessionKey) : undefined;
			const branchState = sessionState?.branch;
			const allState = sessionState?.all;
			const supportsImages = model.input?.includes("image") ?? true;

			let resumeSessionId: string | undefined;
			let resumeSessionAt: string | undefined;
			let forkSession: boolean | undefined;
			let prompt: AsyncIterable<SDKUserMessage> | string | undefined;
			let resumeTailMessages: Context["messages"] | undefined;
			let resumeTailAllowedToolUseIds: Set<string> | undefined;

			if (!branchState?.sdkSessionId) {
				prompt = buildPromptStream(buildPromptBlocks(context, customToolNameToSdk));
			} else {
				const lastSdkInfo = findLastSdkAssistantInfo(context.messages, branchState);
				if (!lastSdkInfo) {
					prompt = buildPromptStream(buildPromptBlocks(context, customToolNameToSdk));
				} else {
					resumeSessionId = branchState.sdkSessionId;
					resumeSessionAt = lastSdkInfo.uuid;
					resumeTailMessages = context.messages.slice(lastSdkInfo.index + 1);
					const tailPlan = analyzeResumeTailMessages(resumeTailMessages, branchState.pendingToolUseTimestamp);
					let tailAllowedToolUseIds = tailPlan.tailAllowedToolUseIds;
					if (tailPlan.shouldUseSummaryPrompt) {
						const summaryText = buildHistoricalSummary(tailPlan.summaryMessages, customToolNameToSdk);
						prompt = buildPromptWithSummary(summaryText, tailPlan.userMessage, supportsImages);
					} else if (resumeSessionId) {
						const existingToolResultIds = getExistingToolResultIds(resumeSessionId, cwd);
						if (existingToolResultIds.size > 0 && tailAllowedToolUseIds) {
							tailAllowedToolUseIds = new Set(
								[...tailAllowedToolUseIds].filter((id) => !existingToolResultIds.has(id)),
							);
						}
					}

					const forkPlan = computeResumeForkPlan(branchState, allState);
					if (forkPlan.resumeSessionAt) {
						resumeSessionAt = forkPlan.resumeSessionAt;
					}
					if (forkPlan.forkSession !== undefined) {
						forkSession = forkPlan.forkSession;
					}
					if (branchState.pendingToolUseTimestamp != null) {
						if (branchState.pendingToolUseIds?.length) {
							const pendingToolUseIdSet = new Set(branchState.pendingToolUseIds);
							if (!tailAllowedToolUseIds) {
								tailAllowedToolUseIds = pendingToolUseIdSet;
							} else {
								tailAllowedToolUseIds = new Set(
									[...tailAllowedToolUseIds].filter((id) => pendingToolUseIdSet.has(id)),
								);
							}
						}
						if (!tailPlan.tailHasAssistant) {
							prompt = buildResumePromptFromTail(resumeTailMessages, supportsImages, tailAllowedToolUseIds);
						}
						persistSdkEntry(sessionKey, {
							providerId: PROVIDER_ID,
							pendingToolUseTimestamp: null,
							pendingToolUseIds: null,
						});
					} else if (!tailPlan.tailHasAssistant) {
						prompt = buildResumePromptFromTail(resumeTailMessages, supportsImages, tailAllowedToolUseIds);
					}
					resumeTailAllowedToolUseIds = tailAllowedToolUseIds;
				}

			}

			if (prompt === undefined) {
				if (hasReplayableTailMessages(resumeTailMessages)) {
					prompt = buildResumePromptFromTail(
						resumeTailMessages,
						supportsImages,
						resumeTailAllowedToolUseIds,
					);
				} else {
					prompt = "Continue.";
				}
			}

			const useResume = Boolean(resumeSessionId);

			const mcpServers = buildCustomToolServers(customTools);
			const providerSettings = loadProviderSettings();
			const appendSystemPrompt = providerSettings.appendSystemPrompt !== false;
			const agentsAppend = appendSystemPrompt ? extractAgentsAppend() : undefined;
			const skillsAppend = appendSystemPrompt ? extractSkillsAppend(context.systemPrompt) : undefined;
			const appendParts = [agentsAppend, skillsAppend].filter((part): part is string => Boolean(part));
			const systemPromptAppend = appendParts.length > 0 ? appendParts.join("\n\n") : undefined;
			const allowSkillAliasRewrite = Boolean(skillsAppend);

			const settingSources: SettingSource[] | undefined = appendSystemPrompt
				? undefined
				: providerSettings.settingSources ?? ["user", "project"];

			// Claude Code will auto-load MCP servers from ~/.claude.json and .mcp.json when settingSources is enabled.
			// In this provider, Claude Code tool execution is denied and pi executes tools instead, so auto-loaded MCP
			// tools are pure token overhead. Pass --strict-mcp-config to ignore all MCP configs except those explicitly
			// provided via the SDK (mcpServers option).
			const strictMcpConfigEnabled = !appendSystemPrompt && providerSettings.strictMcpConfig !== false;
			const extraArgs = strictMcpConfigEnabled ? { "strict-mcp-config": null } : undefined;

			const queryOptions: NonNullable<Parameters<typeof query>[0]["options"]> = {
				cwd,
				tools: sdkTools,
				permissionMode: "default",
				includePartialMessages: true,
				persistSession: true,
				...(useResume && resumeSessionId ? { resume: resumeSessionId } : {}),
				...(useResume && resumeSessionAt ? { resumeSessionAt } : {}),
				...(useResume && forkSession ? { forkSession } : {}),
				canUseTool: async (_toolName, _input, permissionOptions) => {
					return {
						behavior: "deny" as const,
						message: TOOL_EXECUTION_DENIED_MESSAGE,
						interrupt: true,
						toolUseID: permissionOptions.toolUseID,
					};
				},
				systemPrompt: {
					type: "preset",
					preset: "claude_code",
					append: systemPromptAppend ? systemPromptAppend : undefined,
				},
				...(settingSources ? { settingSources } : {}),
				...(extraArgs ? { extraArgs } : {}),
				...(mcpServers ? { mcpServers } : {}),
			};

			const maxThinkingTokens = mapThinkingTokens(options?.reasoning, model.id, options?.thinkingBudgets);
			if (maxThinkingTokens != null) {
				queryOptions.maxThinkingTokens = maxThinkingTokens;
			}

			sdkQuery = query({
				prompt,
				options: queryOptions,
			});

			if (wasAborted) {
				requestClose();
			}

			for await (const message of sdkQuery) {
				if (wasAborted || options?.signal?.aborted) {
					requestClose();
					break;
				}
				if (!started) {
					stream.push({ type: "start", partial: output });
					started = true;
				}

				switch (message.type) {
					case "system": {
						const systemMessage = message as { subtype?: string; session_id?: string };
						if (systemMessage.subtype === "init" && systemMessage.session_id) {
							persistSdkEntry(sessionKey, {
								providerId: PROVIDER_ID,
								sdkSessionId: systemMessage.session_id,
							});
						}
						break;
					}
					case "assistant": {
						const assistantMessage = message as { message?: { role?: string }; uuid?: string; session_id?: string; parent_tool_use_id?: string | null };
						if (assistantMessage.message?.role === "assistant" && !assistantMessage.parent_tool_use_id) {
							if (assistantMessage.uuid) {
								persistSdkEntry(sessionKey, {
									providerId: PROVIDER_ID,
									sdkSessionId: assistantMessage.session_id,
									sdkAssistantUuid: assistantMessage.uuid,
									assistantTimestamp: output.timestamp,
								});
							}
						}
						break;
					}
					case "stream_event": {
						sawStreamEvent = true;
						const event = (message as SDKMessage & { event: any }).event;

						if (event?.type === "message_start") {
							for (const existingBlock of blocks) {
								if ("index" in (existingBlock as any)) {
									delete (existingBlock as any).index;
								}
							}
							announcedToolCallIndices.clear();
							emittedToolCallDeltaIndices.clear();
							const usage = event.message?.usage;
							output.usage.input = usage?.input_tokens ?? 0;
							output.usage.output = usage?.output_tokens ?? 0;
							output.usage.cacheRead = usage?.cache_read_input_tokens ?? 0;
							output.usage.cacheWrite = usage?.cache_creation_input_tokens ?? 0;
							output.usage.totalTokens =
								output.usage.input + output.usage.output + output.usage.cacheRead + output.usage.cacheWrite;
							calculateCost(model, output.usage);
							break;
						}

						if (event?.type === "content_block_start") {
							clearStaleBlockIndex(event.index);
							if (event.content_block?.type === "text") {
								const block = { type: "text", text: "", index: event.index } as const;
								output.content.push(block);
								stream.push({ type: "text_start", contentIndex: output.content.length - 1, partial: output });
							} else if (event.content_block?.type === "thinking") {
								const block = { type: "thinking", thinking: "", thinkingSignature: "", index: event.index } as const;
								output.content.push(block);
								stream.push({ type: "thinking_start", contentIndex: output.content.length - 1, partial: output });
							} else if (event.content_block?.type === "tool_use") {
								const inputArgs = (event.content_block.input as Record<string, unknown>) ?? {};
								const block = {
									type: "toolCall",
									id: event.content_block.id,
									name: mapToolName(event.content_block.name, customToolNameToPi),
									arguments: inputArgs,
									partialJson: "",
									index: event.index,
								} as const;
								output.content.push(block);
								if (hasToolInputArgs(inputArgs)) {
									const contentIndex = output.content.length - 1;
									announcedToolCallIndices.add(event.index);
									emittedToolCallDeltaIndices.add(event.index);
									stream.push({ type: "toolcall_start", contentIndex, partial: output });
									stream.push({
										type: "toolcall_delta",
										contentIndex,
										delta: JSON.stringify(inputArgs),
										partial: output,
									});
								}
							}
							break;
						}

						if (event?.type === "content_block_delta") {
							if (event.delta?.type === "text_delta") {
								const index = findLatestBlockIndex(event.index);
								const block = blocks[index];
								if (block?.type === "text") {
									block.text += event.delta.text;
									stream.push({
										type: "text_delta",
										contentIndex: index,
										delta: event.delta.text,
										partial: output,
									});
								}
							} else if (event.delta?.type === "thinking_delta") {
								const index = findLatestBlockIndex(event.index);
								const block = blocks[index];
								if (block?.type === "thinking") {
									block.thinking += event.delta.thinking;
									stream.push({
										type: "thinking_delta",
										contentIndex: index,
										delta: event.delta.thinking,
										partial: output,
									});
								}
							} else if (event.delta?.type === "input_json_delta") {
								const index = findLatestBlockIndex(event.index);
								const block = blocks[index];
								if (block?.type === "toolCall") {
									block.partialJson += event.delta.partial_json;
									block.arguments = parsePartialJson(block.partialJson, block.arguments);
									const hasArgs = hasToolInputArgs(block.arguments as Record<string, unknown>);
									if (hasArgs && !announcedToolCallIndices.has(event.index)) {
										announcedToolCallIndices.add(event.index);
										stream.push({ type: "toolcall_start", contentIndex: index, partial: output });
									}
									if (hasArgs && !emittedToolCallDeltaIndices.has(event.index)) {
										emittedToolCallDeltaIndices.add(event.index);
										stream.push({
											type: "toolcall_delta",
											contentIndex: index,
											delta: JSON.stringify(block.arguments),
											partial: output,
										});
									}
								}
							} else if (event.delta?.type === "signature_delta") {
								const index = findLatestBlockIndex(event.index);
								const block = blocks[index];
								if (block?.type === "thinking") {
									block.thinkingSignature = (block.thinkingSignature ?? "") + event.delta.signature;
								}
							}
							break;
						}

						if (event?.type === "content_block_stop") {
							const index = findLatestBlockIndex(event.index);
							const block = blocks[index];
							if (!block) break;
							delete (block as any).index;
							if (block.type === "text") {
								stream.push({
									type: "text_end",
									contentIndex: index,
									content: block.text,
									partial: output,
								});
							} else if (block.type === "thinking") {
								stream.push({
									type: "thinking_end",
									contentIndex: index,
									content: block.thinking,
									partial: output,
								});
							} else if (block.type === "toolCall") {
								block.arguments = mapToolArgs(
									block.name,
									parsePartialJson(block.partialJson, block.arguments),
									allowSkillAliasRewrite,
								);
								const hasArgs = hasToolInputArgs(block.arguments as Record<string, unknown>);
								if (!hasArgs) {
									delete (block as any).partialJson;
									delete (block as any).index;
									(output.content as any).splice(index, 1);
									announcedToolCallIndices.delete(event.index);
									emittedToolCallDeltaIndices.delete(event.index);
									break;
								}
								if (!announcedToolCallIndices.has(event.index)) {
									announcedToolCallIndices.add(event.index);
									stream.push({ type: "toolcall_start", contentIndex: index, partial: output });
								}
								if (!emittedToolCallDeltaIndices.has(event.index)) {
									emittedToolCallDeltaIndices.add(event.index);
									stream.push({
										type: "toolcall_delta",
										contentIndex: index,
										delta: JSON.stringify(block.arguments),
										partial: output,
									});
								}
								sawToolCall = true;
								delete (block as any).partialJson;
								pendingToolUseIds.add(block.id);
								stream.push({
									type: "toolcall_end",
									contentIndex: index,
									toolCall: block,
									partial: output,
								});
								announcedToolCallIndices.delete(event.index);
								emittedToolCallDeltaIndices.delete(event.index);
							}
							break;
						}

						if (event?.type === "message_delta") {
							const stopReason = mapStopReason(event.delta?.stop_reason);
							output.stopReason = stopReason;
							const usage = event.usage ?? {};
							if (usage.input_tokens != null) output.usage.input = usage.input_tokens;
							if (usage.output_tokens != null) output.usage.output = usage.output_tokens;
							if (usage.cache_read_input_tokens != null) output.usage.cacheRead = usage.cache_read_input_tokens;
							if (usage.cache_creation_input_tokens != null) output.usage.cacheWrite = usage.cache_creation_input_tokens;
							output.usage.totalTokens =
								output.usage.input + output.usage.output + output.usage.cacheRead + output.usage.cacheWrite;
							calculateCost(model, output.usage);
							break;
						}

						if (event?.type === "message_stop" && sawToolCall) {
							output.stopReason = "toolUse";
							shouldStopEarly = true;
							if (!abortedForToolCall) {
								abortedForToolCall = true;
								persistSdkEntry(sessionKey, {
									providerId: PROVIDER_ID,
									pendingToolUseTimestamp: output.timestamp,
									pendingToolUseIds: Array.from(pendingToolUseIds),
								});
							}
							break;
						}

						break;
					}

					case "result": {
						if (!sawStreamEvent && message.subtype === "success") {
							output.content.push({ type: "text", text: message.result || "" });
						}
						break;
					}
				}

				if (shouldStopEarly) {
					break;
				}
			}

			if (output.stopReason === "toolUse" && sawToolCall && !abortedForToolCall) {
				abortedForToolCall = true;
				persistSdkEntry(sessionKey, {
					providerId: PROVIDER_ID,
					pendingToolUseTimestamp: output.timestamp,
					pendingToolUseIds: Array.from(pendingToolUseIds),
				});
			}

			sanitizeAssistantContentForEmit(output);

			if (wasAborted || options?.signal?.aborted) {
				output.stopReason = "aborted";
				output.errorMessage = "Operation aborted";
				stream.push({ type: "error", reason: "aborted", error: output });
				stream.end();
				return;
			}

			stream.push({
				type: "done",
				reason: output.stopReason === "toolUse" ? "toolUse" : output.stopReason === "length" ? "length" : "stop",
				message: output,
			});
			stream.end();
		} catch (error) {
			sanitizeAssistantContentForEmit(output);
			if (output.stopReason === "toolUse" && isIgnorableTransportWriteAfterCloseError(error)) {
				stream.push({
					type: "done",
					reason: "toolUse",
					message: output,
				});
				stream.end();
				return;
			}
			const aborted = Boolean(wasAborted || options?.signal?.aborted);
			output.stopReason = aborted ? "aborted" : "error";
			output.errorMessage = aborted ? "Operation aborted" : error instanceof Error ? error.message : String(error);
			stream.push({ type: "error", reason: output.stopReason as "aborted" | "error", error: output });
			stream.end();
		} finally {
			if (options?.signal) {
				options.signal.removeEventListener("abort", onAbort);
			}
			sdkQuery?.close();
		}
	})();

	return stream;
}

export default function (pi: ExtensionAPI) {
	extensionApi = pi;

	pi.on("session_start", (_event, ctx) => {
		refreshSessionState(ctx);
	});

	pi.on("session_switch", (_event, ctx) => {
		refreshSessionState(ctx);
	});

	pi.on("session_tree", (_event, ctx) => {
		refreshSessionState(ctx);
	});

	pi.on("session_fork", (_event, ctx) => {
		refreshSessionState(ctx);
	});

	pi.registerProvider(PROVIDER_ID, {
		baseUrl: "claude-agent-sdk",
		apiKey: "ANTHROPIC_API_KEY",
		api: "claude-agent-sdk",
		models: MODELS,
		streamSimple: streamClaudeAgentSdk,
	});
}

export const __test = {
	analyzeResumeTailMessages,
	computeResumeForkPlan,
	buildHistoricalSummary,
	buildPromptWithSummary,
	buildSdkStateFromEntries,
	syncAllSdkStateFromPiSession,
	getSessionState,
	setKnownPiSessionFileForSessionKey,
	clearInternalStateForTests,
	buildResumePromptFromTail,
	hasReplayableTailMessages,
	findLastSdkAssistantInfo,
	isErroredAssistantMessage,
	sanitizeAssistantContentForEmit,
	isIgnorableTransportWriteAfterCloseError,
};

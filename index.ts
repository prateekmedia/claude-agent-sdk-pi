import { calculateCost, createAssistantMessageEventStream, getModels, type AssistantMessage, type AssistantMessageEventStream, type Context, type ImageContent, type Model, type SimpleStreamOptions, type Tool } from "@mariozechner/pi-ai";
import { AuthStorage, type ExtensionAPI } from "@mariozechner/pi-coding-agent";
import { createSdkMcpServer, query, type SDKMessage, type SDKUserMessage, type SettingSource } from "@anthropic-ai/claude-agent-sdk";
import type { Base64ImageSource, CacheControlEphemeral, ContentBlockParam, ImageBlockParam, MessageParam, TextBlockParam } from "@anthropic-ai/sdk/resources";
import { pascalCase } from "change-case";
import { existsSync, readFileSync } from "fs";
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
			const header = `TOOL RESULT (${mapPiToolNameToSdk(message.toolName, customToolNameToSdk)}):`;
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
			type: "user",
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
				return `[tool_call ${toolName} ${args}]`;
			}
			return `[${block.type}]`;
		})
		.join("\n");
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
		const legacyDisable = false;
		return {
			appendSystemPrompt: appendSystemPrompt ?? (legacyDisable ? false : undefined),
		};
	} catch {
		return {};
	}
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

function mapThinkingTokens(reasoning?: SimpleStreamOptions["reasoning"]): number | undefined {
	if (!reasoning) return undefined;
	const budgets: Record<NonNullable<SimpleStreamOptions["reasoning"]>, number> = {
		minimal: 1024,
		low: 2048,
		medium: 8192,
		high: 16384,
		xhigh: 16384,
	};
	return budgets[reasoning];
}

function parsePartialJson(input: string, fallback: Record<string, unknown>): Record<string, unknown> {
	if (!input) return fallback;
	try {
		return JSON.parse(input);
	} catch {
		return fallback;
	}
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
			void sdkQuery.interrupt().catch(() => {
				try {
					sdkQuery?.close();
				} catch {
					// ignore shutdown errors
				}
			});
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

		try {
			const { sdkTools, customTools, customToolNameToSdk, customToolNameToPi } = resolveSdkTools(context);
			const promptBlocks = buildPromptBlocks(context, customToolNameToSdk);
			const prompt = buildPromptStream(promptBlocks);

			const cwd = (options as { cwd?: string } | undefined)?.cwd ?? process.cwd();

			const mcpServers = buildCustomToolServers(customTools);
			const providerSettings = loadProviderSettings();
			const appendSystemPrompt = providerSettings.appendSystemPrompt !== false;
			const agentsAppend = appendSystemPrompt ? extractAgentsAppend() : undefined;
			const skillsAppend = appendSystemPrompt ? extractSkillsAppend(context.systemPrompt) : undefined;
			const appendParts = [agentsAppend, skillsAppend].filter((part): part is string => Boolean(part));
			const systemPromptAppend = appendParts.length > 0 ? appendParts.join("\n\n") : undefined;
			const allowSkillAliasRewrite = Boolean(skillsAppend);

			const settingSources: SettingSource[] | undefined = appendSystemPrompt ? undefined : ["user", "project"];
			const queryOptions: NonNullable<Parameters<typeof query>[0]["options"]> = {
				cwd,
				tools: sdkTools,
				permissionMode: "dontAsk",
				includePartialMessages: true,
				canUseTool: async () => ({
					behavior: "deny",
					message: TOOL_EXECUTION_DENIED_MESSAGE,
				}),
				systemPrompt: { type: "preset", preset: "claude_code", append: systemPromptAppend ? systemPromptAppend : undefined },
				...(settingSources ? { settingSources } : {}),
				...(mcpServers ? { mcpServers } : {}),
			};

			const maxThinkingTokens = mapThinkingTokens(options?.reasoning);
			if (maxThinkingTokens != null) {
				queryOptions.maxThinkingTokens = maxThinkingTokens;
			}

			sdkQuery = query({
				prompt,
				options: queryOptions,
			});

			if (wasAborted) {
				requestAbort();
			}

			for await (const message of sdkQuery) {
				if (!started) {
					stream.push({ type: "start", partial: output });
					started = true;
				}

				switch (message.type) {
					case "stream_event": {
						sawStreamEvent = true;
						const event = (message as SDKMessage & { event: any }).event;

						if (event?.type === "message_start") {
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
							if (event.content_block?.type === "text") {
								const block = { type: "text", text: "", index: event.index } as const;
								output.content.push(block);
								stream.push({ type: "text_start", contentIndex: output.content.length - 1, partial: output });
							} else if (event.content_block?.type === "thinking") {
								const block = { type: "thinking", thinking: "", thinkingSignature: "", index: event.index } as const;
								output.content.push(block);
								stream.push({ type: "thinking_start", contentIndex: output.content.length - 1, partial: output });
							} else if (event.content_block?.type === "tool_use") {
								sawToolCall = true;
								const block = {
									type: "toolCall",
									id: event.content_block.id,
									name: mapToolName(event.content_block.name, customToolNameToPi),
									arguments: (event.content_block.input as Record<string, unknown>) ?? {},
									partialJson: "",
									index: event.index,
								} as const;
								output.content.push(block);
								stream.push({ type: "toolcall_start", contentIndex: output.content.length - 1, partial: output });
							}
							break;
						}

						if (event?.type === "content_block_delta") {
							if (event.delta?.type === "text_delta") {
								const index = blocks.findIndex((block) => block.index === event.index);
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
								const index = blocks.findIndex((block) => block.index === event.index);
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
								const index = blocks.findIndex((block) => block.index === event.index);
								const block = blocks[index];
								if (block?.type === "toolCall") {
									block.partialJson += event.delta.partial_json;
									block.arguments = parsePartialJson(block.partialJson, block.arguments);
									stream.push({
										type: "toolcall_delta",
										contentIndex: index,
										delta: event.delta.partial_json,
										partial: output,
									});
								}
							} else if (event.delta?.type === "signature_delta") {
								const index = blocks.findIndex((block) => block.index === event.index);
								const block = blocks[index];
								if (block?.type === "thinking") {
									block.thinkingSignature = (block.thinkingSignature ?? "") + event.delta.signature;
								}
							}
							break;
						}

						if (event?.type === "content_block_stop") {
							const index = blocks.findIndex((block) => block.index === event.index);
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
								sawToolCall = true;
								block.arguments = mapToolArgs(
									block.name,
									parsePartialJson(block.partialJson, block.arguments),
									allowSkillAliasRewrite,
								);
								delete (block as any).partialJson;
								stream.push({
									type: "toolcall_end",
									contentIndex: index,
									toolCall: block,
									partial: output,
								});
							}
							break;
						}

						if (event?.type === "message_delta") {
							output.stopReason = mapStopReason(event.delta?.stop_reason);
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
			output.stopReason = options?.signal?.aborted ? "aborted" : "error";
			output.errorMessage = error instanceof Error ? error.message : String(error);
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
	pi.registerProvider(PROVIDER_ID, {
		baseUrl: "claude-agent-sdk",
		apiKey: "ANTHROPIC_API_KEY",
		api: "claude-agent-sdk",
		models: MODELS,
		streamSimple: streamClaudeAgentSdk,
	});
}

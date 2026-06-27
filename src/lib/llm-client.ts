const KEY_API_KEY = "kimi-llm-api-key";
const KEY_ENDPOINT = "kimi-llm-endpoint";
const KEY_MODEL = "kimi-llm-model";

const DEFAULT_ENDPOINT = "https://api.openai.com/v1/chat/completions";
const DEFAULT_MODEL = "gpt-4o-mini";
const ANTHROPIC_ENDPOINT = "/api/llm";
const ANTHROPIC_VERSION = "2023-06-01";

function readLS(key: string): string {
  if (typeof window === "undefined") return "";
  try { return localStorage.getItem(key) ?? ""; }
  catch { return ""; }
}

function writeLS(key: string, value: string): void {
  if (typeof window === "undefined") return;
  try {
    const t = value.trim();
    if (!t) localStorage.removeItem(key);
    else localStorage.setItem(key, t);
  } catch {}
}

function isAnthropicNative(endpoint: string): boolean {
  return endpoint.includes("api.anthropic.com");
}

export type LLMConfig = { apiKey: string; endpoint: string; model: string };

export function getLLMConfig(): LLMConfig {
  return {
    apiKey: readLS(KEY_API_KEY),
    endpoint: readLS(KEY_ENDPOINT) || DEFAULT_ENDPOINT,
    model: readLS(KEY_MODEL) || DEFAULT_MODEL,
  };
}

export function setLLMConfig(c: Partial<LLMConfig>): void {
  if (c.apiKey !== undefined) writeLS(KEY_API_KEY, c.apiKey);
  if (c.endpoint !== undefined) writeLS(KEY_ENDPOINT, c.endpoint);
  if (c.model !== undefined) writeLS(KEY_MODEL, c.model);
}

export function isLLMConfigured(): boolean {
  return !!getLLMConfig().apiKey;
}

export type ChatMessage = {
  role: "system" | "user" | "assistant";
  content: string;
};

export type ChatOptions = {
  temperature?: number;
  maxTokens?: number;
  stream?: boolean;
};

export type ChatResult = {
  text: string;
  raw?: unknown;
};

async function llmChatAnthropic(
  messages: ChatMessage[],
  opts: ChatOptions,
  cfg: LLMConfig,
): Promise<ChatResult> {
  const systemMessages = messages.filter(m => m.role === "system");
  const chatMessages = messages.filter(m => m.role !== "system");
  const systemText = systemMessages.map(m => m.content).join("\n\n");

  const systemBlock = systemText
    ? [{ type: "text", text: systemText, cache_control: { type: "ephemeral" } }]
    : undefined;

  const anthropicMessages = chatMessages.map(m => ({
    role: m.role as "user" | "assistant",
    content: m.content,
  }));

  const body: Record<string, unknown> = {
    model: cfg.model,
    max_tokens: opts.maxTokens ?? 1024,
    temperature: opts.temperature ?? 0.7,
    messages: anthropicMessages,
  };
  if (systemBlock) body.system = systemBlock;

  const res = await fetch(ANTHROPIC_ENDPOINT, {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
      "x-api-key": cfg.apiKey,
      "anthropic-version": ANTHROPIC_VERSION,
      "anthropic-beta": "prompt-caching-2024-07-31",
    },
    body: JSON.stringify(body),
  });

  if (!res.ok) {
    const errText = await res.text().catch(() => "");
    throw new Error(`LLM request failed (${res.status}): ${errText.slice(0, 200)}`);
  }

  const data = (await res.json()) as {
    content?: { type: string; text?: string }[];
  };
  const text = data.content?.find(b => b.type === "text")?.text ?? "";
  return { text, raw: data };
}

async function llmChatOpenAI(
  messages: ChatMessage[],
  opts: ChatOptions,
  cfg: LLMConfig,
): Promise<ChatResult> {
  const body = {
    model: cfg.model,
    messages,
    temperature: opts.temperature ?? 0.7,
    max_tokens: opts.maxTokens ?? 1024,
    stream: false,
  };

  const res = await fetch(cfg.endpoint, {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
      Authorization: `Bearer ${cfg.apiKey}`,
    },
    body: JSON.stringify(body),
  });

  if (!res.ok) {
    const errText = await res.text().catch(() => "");
    throw new Error(`LLM request failed (${res.status}): ${errText.slice(0, 200)}`);
  }

  const data = (await res.json()) as {
    choices?: { message?: { content?: string } }[];
  };
  const text = data.choices?.[0]?.message?.content ?? "";
  return { text, raw: data };
}

export async function llmChat(
  messages: ChatMessage[],
  opts: ChatOptions = {},
): Promise<ChatResult> {
  const cfg = getLLMConfig();
  if (!cfg.apiKey) {
    throw new Error("LLM API key not configured. Open settings → fill API key.");
  }
  if (isAnthropicNative(cfg.endpoint)) {
    return llmChatAnthropic(messages, opts, cfg);
  }
  return llmChatOpenAI(messages, opts, cfg);
}

export async function llmGenerate(
  prompt: string,
  system?: string,
  opts?: ChatOptions,
): Promise<string> {
  const messages: ChatMessage[] = [];
  if (system) messages.push({ role: "system", content: system });
  messages.push({ role: "user", content: prompt });
  const r = await llmChat(messages, opts);
  return r.text;
}

export async function llmGenerateWithImage(
  prompt: string,
  imageDataUrl: string,
  system?: string,
  opts?: ChatOptions,
): Promise<string> {
  const cfg = getLLMConfig();
  if (!cfg.apiKey) throw new Error("LLM API key not configured. Open settings.");

  const userContent = [
    { type: "text" as const, text: prompt },
    { type: "image_url" as const, image_url: { url: imageDataUrl } },
  ];

  const messages = [];
  if (system) messages.push({ role: "system", content: system });
  messages.push({ role: "user", content: userContent });

  const body = {
    model: cfg.model,
    messages,
    temperature: opts?.temperature ?? 0.3,
    max_tokens: opts?.maxTokens ?? 2048,
    stream: false,
  };

  const res = await fetch(cfg.endpoint, {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
      Authorization: `Bearer ${cfg.apiKey}`,
    },
    body: JSON.stringify(body),
  });

  if (!res.ok) {
    const errText = await res.text().catch(() => "");
    throw new Error(`vision request failed (${res.status}): ${errText.slice(0, 200)}`);
  }

  const data = (await res.json()) as {
    choices?: { message?: { content?: string } }[];
  };
  return data.choices?.[0]?.message?.content ?? "";
}

export function friendlyLLMError(err: unknown): {
  title: string;
  detail: string;
  hint: string;
} {
  const raw = (err as Error)?.message ?? String(err);
  const m = raw.toLowerCase();

  if (m.includes("not configured"))
    return { title: "LLM 没配", detail: "API key 还没填.", hint: "去 /settings 填 endpoint + key + model" };
  if (m.includes("(401)") || m.includes("unauthorized") || m.includes("invalid_api_key") || m.includes("authentication_error"))
    return { title: "API key 不对", detail: "endpoint 拒了 · 401 unauthorized.", hint: "去 /settings 检查 API key + endpoint 是否匹配" };
  if (m.includes("(429)") || m.includes("rate") || m.includes("quota") || m.includes("overloaded"))
    return { title: "太频繁了", detail: "endpoint rate limit 或 quota 用完.", hint: "等几分钟再试 · 或换 model · 或检查 billing" };
  if (m.includes("context_length") || m.includes("context length") || m.includes("too long"))
    return { title: "上下文太长", detail: "SP + memory + chat 历史 总和超过 model context window.", hint: "缩短 SP · 关 memory inject · 或新开窗口" };
  if (m.includes("(400)") || m.includes("bad request"))
    return { title: "请求格式错", detail: raw.slice(0, 200), hint: "检查 model 名字 · 或 endpoint URL" };
  if (m.includes("(404)") || m.includes("not found"))
    return { title: "endpoint 找不到", detail: "model 名字写错 · 或 endpoint URL 写错.", hint: "检查 /settings" };
  if (m.includes("(500)") || m.includes("(502)") || m.includes("(503)") || m.includes("server"))
    return { title: "服务端挂了", detail: "endpoint 5xx · 不是你的问题.", hint: "等几分钟 · 或换 provider" };
  if (m.includes("fetch") || m.includes("network") || m.includes("failed to fetch"))
    return { title: "网络问题", detail: "请求没发出去.", hint: "检查 endpoint URL · 网络 · CORS" };
  return { title: "出错了", detail: raw.slice(0, 200), hint: "检查 /settings 配置 · 或 console 看完整 error" };
}

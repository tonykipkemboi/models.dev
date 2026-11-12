import { afterEach, expect, test } from "bun:test";
import { mkdtemp, rm } from "node:fs/promises";
import { tmpdir } from "node:os";
import path from "node:path";

import { plan } from "../src/sync2/engine.js";
import { nebius } from "../src/sync2/providers/nebius.js";

// The public catalog as recorded on 2026-11-12, cut down to the fields the module reads.
const FIXTURE = path.join(import.meta.dirname, "fixtures", "sync2", "nebius.json");
const RESPONSES = { sdk: { aisdk: "@ai-sdk/openai", opencodeai: "@opencode/ai/providers/openai-compatible/responses" } };

const roots: string[] = [];
afterEach(async () => {
  for (const root of roots.splice(0)) await rm(root, { recursive: true, force: true });
});

async function catalog(files: Record<string, string>) {
  const root = await mkdtemp(path.join(tmpdir(), "nebius-sync-"));
  roots.push(root);
  const all = { "providers/nebius/provider.toml": 'name = "Nebius Token Factory"\nnpm = "@ai-sdk/openai-compatible"\napi = "https://api.tokenfactory.nebius.com/v1"\n', ...files };
  for (const [file, content] of Object.entries(all)) await Bun.write(path.join(root, file), content);
  return root;
}

const LAB = `schema = 2
name = "Example"
description = "Example model"
open_weights = true
release_date = "2026-01-01"

[modalities]
input = ["text"]
output = ["text"]

[capabilities]
tools = { supported = true }
reasoning = { supported = true }
temperature = true
top_p = true
top_k = false

[limit]
context = 131_072
output = 16_384
`;

// A hand-written Nebius file: a header, reasoning controls and limits the catalog doesn't state.
const file = (extra = "") => `# Effort: reasoning_effort = none|high, checked on this host.
schema = 2
base_model = "lab/model"

[capabilities]
reasoning = { supported = true, effort = ["none", "high"] }
structured_output = true

[limit]
context = 100_000
output = 8_192

[api.chat-completions]
compatibility = { reasoning_field = "reasoning_content" }
${extra}
[cost.text.input]
price = 1
cache = { read = 1 }

[cost.text.output]
price = 3
`;

function entry(id: string, flavor: Record<string, unknown> = {}, model: Record<string, unknown> = {}) {
  const useCases = ["text", "function_calling", "reasoning"];
  return {
    name: id,
    type: "text2text",
    status: "active",
    use_cases: useCases,
    ...model,
    flavors: [{
      model_id: id,
      model_type: "text2text",
      use_cases: useCases,
      max_model_len: 262_144,
      input_price_per_million_tokens: 0.5,
      input_cached_price_per_million_tokens: 0.05,
      output_price_per_million_tokens: 2,
      ...flavor,
    }],
  };
}

test("reads one observation per flavor from the recorded catalog", async () => {
  const observations = nebius.observe(await Bun.file(FIXTURE).json());
  expect(observations).toHaveLength(24);
  expect(new Set(observations.map((observation) => observation.id)).size).toBe(24);
  expect(observations.every((observation) => observation.updateOnly === undefined)).toBe(true);
  const byID = new Map(observations.map((observation) => [observation.id, observation]));

  // A chat model on the Responses API, with a cached-input price.
  expect(byID.get("moonshotai/Kimi-K3")).toEqual({
    id: "moonshotai/Kimi-K3",
    base: ["moonshotai/Kimi-K3"],
    limit: { context: 1_048_576 },
    api: { responses: RESPONSES },
    cost: { text: { input: { price: 3, cache: { read: 0.3 } }, output: { price: 15 } } },
  });
  // A chat model the catalog doesn't list for the Responses API.
  expect(byID.get("openai/gpt-oss-120b")!.api).toEqual({ responses: null });
  // An embedding model: no chat APIs, no cached price.
  expect(byID.get("Qwen/Qwen3-Embedding-8B")).toEqual({
    id: "Qwen/Qwen3-Embedding-8B",
    base: ["alibaba/Qwen3-Embedding-8B", "Qwen/Qwen3-Embedding-8B"],
    limit: { context: 40_960 },
    cost: { text: { input: { price: 0.01 }, output: { price: 0 } } },
  });
  // Hugging Face organizations are mapped to the lab that made the model.
  expect(byID.get("deepseek-ai/DeepSeek-V4-Pro-0813")!.base![0]).toBe("deepseek/DeepSeek-V4-Pro-0813");
  expect(byID.get("MiniMaxAI/MiniMax-M3")!.base![0]).toBe("minimax/MiniMax-M3");
  expect(byID.get("meta-models/muse-spark-open")!.base![0]).toBe("meta/muse-spark-open");
  expect(byID.get("zai-org/GLM-5.3")!.base![0]).toBe("zhipuai/GLM-5.3");
});

test("updates prices, the cached price and the context window, and nothing else", async () => {
  const root = await catalog({ "models/lab/model.toml": LAB, "providers/nebius/models/lab/model.toml": file() });
  const { result, writes } = await plan(nebius, { root, raw: [entry("lab/model")] });

  expect(result.changes).toEqual([{
    kind: "update",
    model: "lab/model",
    path: "providers/nebius/models/lab/model.toml",
    fields: [
      { path: "limit.context", before: 100_000, after: 262_144 },
      { path: "cost.text.input.price", before: 1, after: 0.5 },
      { path: "cost.text.input.cache.read", before: 1, after: 0.05 },
      { path: "cost.text.output.price", before: 3, after: 2 },
    ],
  }]);
  expect(result.gaps).toEqual([]);
  expect(writes.get("providers/nebius/models/lab/model.toml")).toBe(`# Effort: reasoning_effort = none|high, checked on this host.
schema = 2
base_model = "lab/model"

[capabilities]
reasoning = { supported = true, effort = ["none", "high"] }
structured_output = true

[limit]
context = 262_144
output = 8_192

[api.chat-completions]
compatibility = { reasoning_field = "reasoning_content" }

[cost.text.input]
price = 0.5
cache = { read = 0.05 }

[cost.text.output]
price = 2
`);
});

test("a file already matching the catalog isn't written", async () => {
  const root = await catalog({ "models/lab/model.toml": LAB, "providers/nebius/models/lab/model.toml": file() });
  const raw = [entry("lab/model", { max_model_len: 100_000, input_price_per_million_tokens: 1, input_cached_price_per_million_tokens: 1, output_price_per_million_tokens: 3 })];
  const { result, writes } = await plan(nebius, { root, raw });
  expect(result.changes).toEqual([]);
  expect(writes.size).toBe(0);
});

test("a flavor without a cached price leaves the file's cache read price alone", async () => {
  const root = await catalog({ "models/lab/model.toml": LAB, "providers/nebius/models/lab/model.toml": file() });
  const { result } = await plan(nebius, { root, raw: [entry("lab/model", { input_cached_price_per_million_tokens: undefined })] });
  expect(result.changes[0]!.fields.map((field) => field.path)).toEqual(["limit.context", "cost.text.input.price", "cost.text.output.price"]);
});

test("the Responses table follows the responses_api use case and keeps the file's other tables", async () => {
  const root = await catalog({
    "models/lab/model.toml": LAB,
    "providers/nebius/models/lab/added.toml": file(),
    "providers/nebius/models/lab/dropped.toml": file('\n[api.responses]\nsdk = { aisdk = "@ai-sdk/openai", opencodeai = "@opencode/ai/providers/openai-compatible/responses" }\n'),
  });
  const { writes } = await plan(nebius, {
    root,
    raw: [
      entry("lab/added", { use_cases: ["text", "responses_api"] }),
      entry("lab/dropped"),
    ],
  });
  const api = (id: string) => (Bun.TOML.parse(writes.get(`providers/nebius/models/lab/${id}.toml`)!) as { api: Record<string, unknown> }).api;
  expect(api("added")).toEqual({ "chat-completions": { compatibility: { reasoning_field: "reasoning_content" } }, responses: RESPONSES });
  expect(api("dropped")).toEqual({ "chat-completions": { compatibility: { reasoning_field: "reasoning_content" } } });
});

test("a model the catalog adds is a gap, never a generated file", async () => {
  // The new model has a lab file, so only the policy keeps the engine from writing it.
  const root = await catalog({ "models/lab/model.toml": LAB, "models/lab/new-model.toml": LAB.replace("reasoning = { supported = true }", "reasoning = { supported = false }"), "providers/nebius/models/lab/model.toml": file() });
  const { result, writes } = await plan(nebius, { root, raw: [entry("lab/model"), entry("lab/new-model")] });
  expect(result.changes.map((change) => change.kind)).toEqual(["update"]);
  expect([...writes.keys()]).toEqual(["providers/nebius/models/lab/model.toml"]);
  expect(result.gaps.map((gap) => [gap.model, gap.kind])).toEqual([["lab/new-model", "new"]]);
  expect(result.gaps[0]!.known).toMatchObject({ limit: { context: 262_144 }, cost: { text: { input: { price: 0.5 }, output: { price: 2 } } } });
});

test("a model the catalog no longer lists is removed", async () => {
  const root = await catalog({
    "models/lab/model.toml": LAB,
    "providers/nebius/models/lab/model.toml": file(),
    "providers/nebius/models/lab/retired.toml": file(),
  });
  const { result, writes } = await plan(nebius, { root, raw: [entry("lab/model")] });
  expect(result.changes.filter((change) => change.kind === "remove")).toEqual([{ kind: "remove", model: "lab/retired", path: "providers/nebius/models/lab/retired.toml", fields: [] }]);
  expect(writes.get("providers/nebius/models/lab/retired.toml")).toBeNull();
});

test("an entry that isn't active, or has no price, updates its file but is never reported as new", async () => {
  const root = await catalog({ "models/lab/model.toml": LAB, "providers/nebius/models/lab/model.toml": file() });
  const { result, writes } = await plan(nebius, {
    root,
    raw: [
      entry("lab/model", {}, { status: "deprecated" }),
      entry("lab/upcoming", {}, { status: "coming_soon" }),
      entry("lab/unpriced", { input_price_per_million_tokens: null, output_price_per_million_tokens: null }),
    ],
  });
  expect(result.gaps).toEqual([]);
  // Still listed, so the file stays and keeps following the catalog.
  expect(result.changes.map((change) => [change.kind, change.model])).toEqual([["update", "lab/model"]]);
  expect([...writes.keys()]).toEqual(["providers/nebius/models/lab/model.toml"]);
});

test("an empty or malformed catalog fails instead of removing every file", async () => {
  const root = await catalog({ "models/lab/model.toml": LAB, "providers/nebius/models/lab/model.toml": file() });
  await expect(plan(nebius, { root, raw: [] })).rejects.toThrow("returned no models");
  await expect(plan(nebius, { root, raw: { error: "unavailable" } })).rejects.toThrow();
  await expect(plan(nebius, { root, raw: [{ name: "No flavors", status: "active" }] })).rejects.toThrow();
  await expect(plan(nebius, { root, raw: [entry("lab/model"), entry("lab/model")] })).rejects.toThrow("twice");
});

test("on this catalog the recorded data only touches owned fields and creates nothing", async () => {
  const { result, writes } = await plan(nebius, { raw: await Bun.file(FIXTURE).json() });
  const owned = new Set(["limit.context", "api", "cost.text.input.price", "cost.text.input.cache.read", "cost.text.output.price"]);
  expect(result.changes.filter((change) => change.kind === "create")).toEqual([]);
  for (const change of result.changes) {
    expect(change.path.startsWith("providers/nebius/models/")).toBe(true);
    for (const field of change.fields) expect(owned.has(field.path) || field.path.startsWith("api.")).toBe(true);
  }
  // Removals are whole files; everything else is a rewrite that still parses.
  for (const [file, content] of writes) if (content !== null) expect(() => Bun.TOML.parse(content)).not.toThrow();
  expect(result.gaps.every((gap) => gap.kind === "new")).toBe(true);
});

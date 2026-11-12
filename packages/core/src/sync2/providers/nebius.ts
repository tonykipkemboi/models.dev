// Nebius Token Factory public catalog: https://tokenfactory.nebius.com/api/public/models_info
// (no key). Each entry lists one or more flavors, and a flavor's `model_id` is
// the ID Token Factory serves and the file path here.
//
// Authoritative for token prices, the cached-input price when one is listed,
// the context window (`max_model_len`, the same number `/v1/models` returns as
// `context_length`) and whether a chat model is served on the Responses API
// (the `responses_api` use case). A flavor without a cached price leaves the
// file's cache read price alone.
//
// Not read: modalities and capabilities. The catalog's `model_type` and
// `use_cases` disagree on image input for some models (a `text2text` model
// that lists an image use case), and it says nothing about reasoning controls,
// so those stay hand-written.
//
// New models are gaps: a file needs reasoning controls checked on this host.
// A model the catalog no longer lists is removed; Token Factory doesn't
// reroute a retired ID. An entry whose status isn't `active` only updates a
// file that exists.

import { z } from "zod";

import { fetchJSON } from "../http.js";
import { labHints } from "../labs.js";
import type { Observation, SyncModule } from "../types.js";

const API_ENDPOINT = "https://tokenfactory.nebius.com/api/public/models_info";

// POST /v1/responses: https://docs.tokenfactory.nebius.com/api-reference/inference/create-a-response
const RESPONSES_API = { sdk: { aisdk: "@ai-sdk/openai", opencodeai: "@opencode/ai/providers/openai-compatible/responses" } };

// Hugging Face organizations whose lab ID here has another name. The rest
// resolve through the shared namespaces (`Qwen` → `alibaba`, `zai-org` → `zhipuai`).
const NAMESPACES: Record<string, string> = {
  "deepseek-ai": "deepseek",
  MiniMaxAI: "minimax",
  "meta-models": "meta",
};

const CHAT_TYPES = new Set(["text2text", "image2text"]);

const Price = z.number().nonnegative().nullish();
const NebiusFlavor = z.object({
  model_id: z.string().min(1),
  model_type: z.string().nullish(),
  use_cases: z.array(z.string()).nullish(),
  max_model_len: z.number().int().positive().nullish(),
  input_price_per_million_tokens: Price,
  input_cached_price_per_million_tokens: Price,
  output_price_per_million_tokens: Price,
}).passthrough();
type NebiusFlavor = z.infer<typeof NebiusFlavor>;

const NebiusModel = z.object({
  status: z.string(),
  type: z.string().nullish(),
  use_cases: z.array(z.string()).nullish(),
  flavors: z.array(NebiusFlavor),
}).passthrough();
type NebiusModel = z.infer<typeof NebiusModel>;

const NebiusCatalog = z.array(NebiusModel);

export const nebius: SyncModule<unknown> = {
  id: "nebius",
  name: "Nebius Token Factory",
  policy: {
    owns: [
      "limit.context",
      { path: "api", merge: "tables" },
      "cost.text.input.price",
      "cost.text.input.cache.read",
      "cost.text.output.price",
    ],
    creates: "gap",
    removals: "delete",
  },
  fetch: () => fetchJSON(API_ENDPOINT, { source: "Nebius Token Factory catalog" }),
  observe: (raw) => NebiusCatalog.parse(raw).flatMap((model) => model.flavors.map((flavor) => observe(model, flavor))),
};

function observe(model: NebiusModel, flavor: NebiusFlavor): Observation {
  const input = flavor.input_price_per_million_tokens ?? undefined;
  const output = flavor.output_price_per_million_tokens ?? undefined;
  const cacheRead = flavor.input_cached_price_per_million_tokens ?? undefined;
  const priced = input !== undefined && output !== undefined;
  const useCases = new Set(flavor.use_cases ?? model.use_cases ?? []);
  return {
    id: flavor.model_id,
    base: baseHints(flavor.model_id),
    // A new file needs a price, and a model that isn't active yet (or any more) isn't one to add.
    ...(model.status !== "active" || !priced ? { updateOnly: true } : {}),
    ...(flavor.max_model_len == null ? {} : { limit: { context: flavor.max_model_len } }),
    ...(CHAT_TYPES.has(flavor.model_type ?? model.type ?? "") ? { api: { responses: useCases.has("responses_api") ? RESPONSES_API : null } } : {}),
    ...(priced ? { cost: { text: { input: { price: input, ...(cacheRead === undefined ? {} : { cache: { read: cacheRead } }) }, output: { price: output } } } } : {}),
  };
}

function baseHints(id: string) {
  const [org, ...rest] = id.split("/");
  const lab = org === undefined ? undefined : NAMESPACES[org];
  return labHints(lab === undefined || rest.length === 0 ? id : `${lab}/${rest.join("/")}`);
}

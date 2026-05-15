# Prompt Section Extraction — Plan

## Problem

Agentic workloads need the router to reason about the **structural
composition** of a prompt — system, tools, message-N, message-N.part-K,
reasoning, tool_calls, tool_result — and to know which
`[token_start, token_end)` window each piece covers in the rendered
token stream. Today the router gets only a flat token list.

This plan covers the **extraction primitive**. It is intentionally
generic. It carries no policy, names no retention semantics, binds to no
single consumer. Retention, cache_control, session-graph routing, and
observability are all *consumers* of the section table — not part of it.

## Architecture

```
ChatCompletionRequest ─► vLLM /render ─► GenerateRequest {token_ids, features, sections}
                                                  │
                                                  ▼
router token-producer plugin ─► request.Body.{TokenizedPrompt, PromptSections}
                                                  │
                ┌─────────────────────────────────┼────────────────────────┐
                ▼                                 ▼                        ▼
        retention policy                 session-graph                 future cache_control
        (Continuum-style,               routing affinity                impl (when offload tier exists)
        KVFlow, etc.)                                                    
                │                                                          
                ▼                                                          
        retention_directives                                               
        in extra_body
                │
                ▼
        upstream vLLM (RFC-retention)
```

The section table rides on the existing `/v1/chat/completions/render`
response (`GenerateRequest.sections`). One pure mechanism, many possible
consumers.

## What this plan is and isn't

It **is** about emitting the section table.

It **is not** about:
- Anthropic `cache_control` resolution. A working implementation already
  exists in this fork on a separate branch; it produces a different
  output shape (`cache_breakpoints` — exclusive upper bounds for
  cacheable prefixes). A proper cache_control feature also needs an
  offload tier and TTL machinery that isn't built yet, so binding the
  section primitive to it now would constrain both designs prematurely.
  cache_control can be re-implemented on top of sections later: a
  cache_control directive becomes a section with directive metadata,
  and the breakpoint becomes a derived view from cache_control-tagged
  sections. That's a future refactor, not this PR.
- Retention directive generation. The retention RFC's API is
  token-range based and policy-agnostic; section-aware policies generate
  directives, but the engine consumes plain token ranges.
- Session-graph data structures inside the engine. The graph lives in
  the router; the engine sees blocks with retention properties.

## Schema

```python
@dataclass(frozen=True)
class SectionMark:
    section_id: str                       # "sys", "tools", "msg[2].part[0]", ...
    kind: str                             # PromptSectionKind
    role: str | None
    message_index: int | None
    part_index: int | None
    injection_target: str                 # InjectionTarget
    injection_side: Literal["prepend", "append"]
    metadata: dict[str, Any] | None = None   # passthrough

@dataclass(frozen=True)
class PromptSection:
    section_id: str
    kind: str
    role: str | None
    message_index: int | None
    part_index: int | None
    token_start: int
    token_end: int
    metadata: dict[str, Any] | None = None

PromptSectionKind = Literal[
    "system", "tools", "message_part", "reasoning",
    "tool_calls", "tool_result", "generation_prompt",
]
```

**`metadata`** is the extensibility hook. The default harvester does
not populate it (`extra_metadata_keys=None`); consumers opt in by
naming the keys they want copied onto each mark, e.g.
`harvest_section_marks(..., extra_metadata_keys=("foo", "bar"))`. A
future cache_control-on-sections implementation would put its
directives here. vLLM ascribes no semantics — pure passthrough.

On the wire:

```python
class GenerateRequest(BaseModel):
    ...
    sections: list[PromptSection] | None = None
```

Mirror as a Go struct in `llm-d-kv-cache-manager/pkg/tokenization/types`.

## Mechanism — sentinel injection

For each `SectionMark`, the resolver injects a unique sentinel
`__VLLM_SECTION_{i}__` at the target's text field (system content, first
tool description, message text, tool_call's function.name, etc.).
Renders the chat template twice (dirty + clean), locates sentinels via
one `re.finditer` pass, translates to clean-string char positions, and
maps to token indices via a single linear merge over `offset_mapping`.

Approach considered and rejected (in the prior plan revision):

- **String-anchoring on a single render.** False positives on user
  content containing scaffold-like substrings.
- **Jinja runtime instrumentation.** Brittle to template changes;
  per-template-family loop-variable mapping.
- **Per-section independent tokenization.** Chat templates aren't
  compositional under tokenization.

Sentinel injection treats the template as a black box and **fails
loudly** via a sanity check: if the offset-mapped retokenization
doesn't equal the renderer's actual `prompt_token_ids`, the entire
section table is dropped (warn-once) — the request still succeeds. Bad
output is detected, not silently wrong.

## Cap

`MAX_PROMPT_SECTIONS = 512` — DoS guard, not a policy limit. A 200-turn
agentic context emits ~500 marks. Reject at parse time if exceeded.

## Multimodal

The current mechanism resolves sections over the pre-MM prompt; MM
expansion shifts placeholder spans in `_process_tokens` and invalidates
those offsets. **v1:** emit `sections=None` when MM is present
(warn-once). MM placeholder ranges still ride on
`GenerateRequest.features.mm_placeholders` so the router isn't blind.

**v2:** rebase offsets through the MM shift map. Plumbing through
`vllm/multimodal/processing.py`; deferred.

## Non-HF renderers

Mistral, DeepSeek, Grok, Terratorch bypass the string-template path.
Behavior: emit `sections=None`, log one-shot warning.

## Implementation steps — vLLM

Each lands as an independent PR / branch increment.

1. **Module `vllm/renderers/prompt_sections.py`** with `SectionMark`,
   `PromptSection`, `PromptSectionKind`, `InjectionTarget`,
   `MAX_PROMPT_SECTIONS`, and `resolve_sections`. Standalone — no
   dependency on cache_control.

2. **Harvester** `harvest_section_marks` added to
   `vllm/entrypoints/chat_utils.py`. Walks the parsed conversation +
   tools list and emits `SectionMark`s in template-render order. Pure
   function; no parser API change.

3. **`DictPrompt.prompt_sections`** field added to
   `vllm/inputs/engine.py`. No behavior change.

4. **`HfRenderer` wire-up.** Add a `_resolve_sections_and_attach`
   method that mirrors `_resolve_and_attach_breakpoints` but uses the
   new harvester + resolver. Attach to the prompt dict. Skip when MM
   or `prompt_embeds` is present.

5. **`GenerateRequest.sections`** added; `render_chat_request` in
   `vllm/entrypoints/serve/render/serving.py` copies
   `prompt.get("prompt_sections")` into the response.

6. **Request gate.** `ChatCompletionRequest.include_sections: bool =
   False` (only honored on `/render`). Non-router callers pay no cost.

7. **Tests.** Property tests for partition invariants across template
   families. Cases: bare, with tools, with MM (assert `None`), slow
   tokenizer (assert `None`), pathological templates that inject
   special tokens (assert sanity-check trips, both outputs dropped
   together).

8. **Benchmark.** Measure overhead against bare render across
   conversation lengths.

## Implementation steps — llm-d-inference-scheduler

Independent of the vLLM PRs.

1. **Schema mirror.** Add `PromptSection` to
   `llm-d-kv-cache-manager/pkg/tokenization/types`. Add `Sections
   []PromptSection` to the response struct of `RenderChat`.

2. **Token-producer plugin.** In
   `pkg/epp/framework/plugins/requestcontrol/dataproducer/tokenizer/`,
   set `IncludeSections=true` on the outgoing `RenderChatRequest`.
   Write `request.Body.PromptSections` alongside `TokenizedPrompt`.

3. **(Optional) Section-producer split.** Factor sections into a
   separate DataProducer that consumes the cached `/render` response
   from the token-producer. Cleaner separation of concerns; not
   required for v1.

4. **Policy plugins (separate work).** Consumers of
   `request.Body.PromptSections`. Out of scope for this plan; see
   reuse-policy discussion in the design doc.

## Open questions

- **Generation-prompt boundary.** Currently derived as
  `[last_section.end, len(tokens))` only when the data sections don't
  already cover that range. For templates where the gen-prompt is
  appended after the last user message, the prior section absorbs it.
  Fix: optionally inject a sentinel appended to the last message
  before the gen-prompt scaffold is rendered. v1.5.
- **Streaming.** Sections are render-time only — no streaming
  complication; they ride the immediate `/render` response.

## Why this is one PR and the consumers are separate

Section extraction is a mechanism with a single output shape. Every
consumer wants the same partition. The consumers (retention policy,
cache_control implementation, session-graph routing) differ in *what
they do with* the sections, not in *what they ask for*. Keeping the
mechanism free of consumer-specific coupling lets each consumer ship on
its own timeline.

## Pointers

| File | Role |
|---|---|
| `vllm/renderers/prompt_sections.py` | New module: `SectionMark`, `PromptSection`, `resolve_sections`. |
| `vllm/entrypoints/chat_utils.py` | `harvest_section_marks` lives here (next to `parse_chat_messages`). |
| `vllm/renderers/hf.py` | Wire-up in `render_messages` / `render_messages_async`. Add `_resolve_sections_and_attach`. |
| `vllm/inputs/engine.py` | `DictPrompt.prompt_sections` field. |
| `vllm/entrypoints/serve/disagg/protocol.py` | `GenerateRequest.sections` field. |
| `vllm/entrypoints/serve/render/serving.py` | Copy sections from prompt to response. |
| `vllm/entrypoints/openai/chat_completion/protocol.py` | `include_sections` gate. |
| `vllm/renderers/cache_control.py` | **Untouched** by this work. Continues to do its own thing. |
| `llm-d-kv-cache-manager/pkg/tokenization/types` | Mirror schema in Go. |
| `llm-d-inference-scheduler/.../dataproducer/tokenizer/` | Set `IncludeSections=true`, parse response. |

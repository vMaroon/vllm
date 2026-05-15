# Prompt Section Extraction — PoC Report

**Branch.** `section-extraction-poc` (off `upstream/main`).

**What was built.** A standalone mechanism that turns an OpenAI Chat
Completion payload into a labelled, contiguous list of `PromptSection`s
covering the full rendered token stream. Generic primitive. No policy.
No coupling to any specific consumer. Every structural unit today's chat
templates emit is supported — system, tools, message_part, reasoning,
tool_calls, tool_result, generation_prompt.

**Single new module.** All section code lives in
`vllm/renderers/prompt_sections.py`. No existing file is touched.

---

## Why this shape

Consumers — retention policies (Continuum-style TTLs, KVFlow scoring),
session-graph routing, Anthropic-style cache_control, observability
overlays — differ in *what they do with* the section table, not in
*what they ask for*. Keeping the mechanism free of consumer-specific
fields lets each consumer ship on its own timeline. `SectionMark` and
`PromptSection` both carry a generic `metadata: dict[str, Any] | None`
passthrough channel; vLLM ascribes no semantics to it.

---

## Pipeline

1. **`harvest_section_marks(conversation, tools)`** walks the parsed
   conversation + tools list and emits one `SectionMark` per logical
   section start, in template-render order. Pure function over plain
   dicts. No coupling to `chat_utils` internals.

2. **`resolve_sections(marks, conversation, tools, ...)`** runs the
   sentinel-injection pass:
   - Deep-copy conversation + tools; inject a unique sentinel
     `__VLLM_SECTION_{i}__` at each mark's injection target.
   - Render the chat template once (the dirty one). If the caller
     hasn't already produced a clean render, render that too.
   - Find every sentinel in `dirty_str` with one `re.finditer` pass and
     translate to clean-string char offsets by subtracting cumulative
     earlier sentinel lengths.
   - If the caller has offset-mapped tokenized the clean string,
     use those offsets directly. Otherwise tokenize once with
     `return_offsets_mapping=True`.
   - Sanity-check: the offset-mapped token stream must equal the
     renderer's own clean `prompt_token_ids`. If not (rare; e.g.
     template injects special tokens not in the rendered string), drop
     all sections with a one-shot warning. The request itself is
     unaffected.
   - Map clean char positions → token indices via a single linear
     merge over `offset_mapping`.
   - Section `token_end = next_section.token_start` (or
     `len(token_ids)` for the last section).
   - When `add_generation_prompt=True` and tokens remain past the
     last data section, append a synthetic `generation_prompt`
     section.

The resolver accepts optional `clean_str` and `clean_offsets`
parameters. Callers (like `HfRenderer` in a real wire-up) that already
produce these for their own purposes can pass them through and skip the
resolver's redundant work entirely.

---

## Section kinds covered

| Kind | Where injected | Notes |
|---|---|---|
| `system` | prepend to system message text | First section in templates that emit a system block |
| `tools` | prepend to first tool's description (name fallback) | Aggregate — one section covers all tools |
| `message_part` | prepend to text (or part k's text for multi-part) | Empty content still emits a zero-length section so role transitions are attributed correctly |
| `reasoning` | prepend to assistant's `reasoning_content` / `reasoning` | Only renders when the template emits reasoning (Qwen3: only on the latest assistant turn) |
| `tool_calls` | prepend to first `tool_call.function.name` | Aggregate across all tool_calls on the same assistant message |
| `tool_result` | prepend to tool-role message content | One per tool-role message |
| `generation_prompt` | derived as `[last_section.end, n_tokens)` | When `add_generation_prompt=True` and tokens remain past the last data section |

---

## Functional results (Qwen3-0.6B chat template)

| Case | Tokens | Marks harvested | Sections resolved | Kinds present |
|---|---|---|---|---|
| `simple` | 49 | 4 | 4 | system, message_part |
| `full_features` | 476 | 12 | 12 | all six |
| `agentic_25turn` | 2926 | 104 | 104 | all six |
| `agentic_50turn` | 5532 | 204 | 204 | all six |

Every harvested mark resolves to a token position in every case.
Partition invariant holds in every case.

Sample structure from `full_features` (476 tokens):

```
   system          msg=0  [  3,  93)  90 tok  "You are a careful, terse assistant..."
   tools                 [ 93, 275) 182 tok  "Look up the current weather conditions..."
   message_part user      [275, 298)  23 tok  "What's the weather in SF? Also email me..."
   message_part asst      [298, 315)  17 tok  "I'll check SF weather, then email..."
   tool_calls   asst      [315, 336)  21 tok  get_weather call
   tool_result  tool      [336, 348)  12 tok  "62F partly cloudy"
   message_part user      [348, 360)  12 tok  "Great, send the email now."
   message_part asst      [360, 367)   7 tok  (empty content, captures scaffold)
   tool_calls   asst      [367, 410)  43 tok  send_email call
   tool_result  tool      [410, 421)  11 tok  "Email sent."
   reasoning    asst      [421, 456)  35 tok  reasoning_content
   message_part asst      [456, 476)  20 tok  final response
```

Every section kind lands on its expected token window.

---

## Latency profile (Qwen3-0.6B tokenizer, 30 iters)

Two scenarios:

* **legacy (total)** — caller passes nothing extra; resolver does its
  own clean render + offset-mapped retokenize. Includes the baseline
  tokenize inside the timed region. Two extra renders + one extra
  tokenize over baseline.
* **marginal (efficient wire-up)** — caller has already rendered
  `clean_str` and computed `clean_offsets` for its own purposes; the
  timed region covers only the incremental harvest + resolver work
  (one dirty render + regex + merge). This is what the cost looks like
  when the renderer is wired to share its existing work.

| Case | Tokens | Sections | Baseline | Legacy total overhead | Marginal cost (efficient) |
|---|---|---|---|---|---|
| `simple` | 49 | 4 | 0.17 ms | +0.31 ms | **0.11 ms** |
| `full_features` | 476 | 12 | 0.98 ms | +1.50 ms | **0.38 ms** |
| `agentic_25turn` | 2926 | 104 | 6.14 ms | +8.86 ms | **2.63 ms** |
| `agentic_50turn` | 5532 | 204 | 11.23 ms | +17.12 ms | **5.23 ms** |

The marginal cost in the efficient wire-up is **~50% of one baseline
tokenize** across the range — section extraction adds roughly half a
tokenize's worth of work on top of work the renderer is already doing.

Pure Python on a CPU. With the renderer wired to share its clean
string + offsets, a 50-turn agentic context (5532 tokens, 204 sections)
costs ~5 ms of section work on top of an ~11 ms render. That's the
shipping target.

---

## What's in this commit

Purely additive — no existing file is touched.

* `vllm/renderers/prompt_sections.py` — `SectionMark`, `PromptSection`,
  `PromptSectionKind`, `InjectionTarget`, `MAX_PROMPT_SECTIONS`,
  sentinel-injection helpers, `harvest_section_marks`,
  `resolve_sections`.
* `scripts/section_extraction_poc.py` — functional + profile harness.
* `docs/design/section_extraction_plan.md` — design plan.
* `docs/design/section_extraction_poc_report.md` — this file.
* `docs/design/section_poc_output.txt` — captured run output.

---

## Known limitations and follow-ups

1. **Multimodal.** When MM placeholders are present, `_process_tokens`
   shifts token offsets after section resolution. v1 behavior: emit
   `sections=None` (warn-once). v2 will rebase offsets through the MM
   shift map.

2. **Non-HF renderers.** Mistral / DeepSeek / Grok / Terratorch bypass
   the string-template path. Behavior: emit `sections=None` with a
   one-shot warning.

3. **Template-dropped marks.** Templates can choose not to render
   certain content (Qwen3 drops reasoning on non-latest assistant
   turns; templates that flatten multi-part content drop part marks).
   The resolver detects this (sentinels missing from `dirty_str`) and
   silently skips the mark; the rest of the section table still
   resolves cleanly.

4. **Generation-prompt boundary precision.** Currently derived as
   `[last_section.end, n_tokens)`. For templates where the
   generation-prompt scaffold renders right after the last user
   message, the prior section absorbs it. Future refinement: optionally
   inject a sentinel appended to the last message text before the
   scaffold renders.

5. **Wire-up of `HfRenderer.render_messages` + `GenerateRequest.sections`
   for the `/v1/chat/completions/render` endpoint** is the next
   mechanical step. The resolver's API already supports the efficient
   path (caller-provided `clean_str` + `clean_offsets`) so the wire-up
   can land both correctness and efficiency in one PR.

---

## Wire-up sketch

```python
# in HfRenderer.render_messages, after rendering and tokenizing:

from vllm.renderers.prompt_sections import (
    harvest_section_marks, resolve_sections,
)

if not has_mm and not has_prompt_embeds:
    # Renderer already had to render with tokenize=False for some
    # paths, or can opt to do so once here. Cheap when shared.
    clean_str = safe_apply_chat_template(
        model_config, tokenizer, conversation,
        **{**chat_template_kwargs, "tokenize": False},
    )
    enc = tokenizer(clean_str, add_special_tokens=False,
                    return_offsets_mapping=True)
    clean_offsets = list(enc["offset_mapping"])

    marks = harvest_section_marks(conversation, tools_arg)
    sections = resolve_sections(
        marks=marks,
        conversation=conversation,
        tools=tools_arg,
        clean_token_ids=prompt.get("prompt_token_ids"),
        apply_chat_template_fn=_bound_apply,
        clean_str=clean_str,
        clean_offsets=clean_offsets,
        add_generation_prompt=chat_template_kwargs.get(
            "add_generation_prompt", True
        ),
    )
    if sections:
        cast(dict, prompt)["prompt_sections"] = sections

# in GenerateRequest (vllm/entrypoints/serve/disagg/protocol.py):
class GenerateRequest(BaseModel):
    ...
    sections: list[PromptSection] | None = None
```

That's the entire integration. About 40 lines of additive change, all
in the renderer + the response schema.

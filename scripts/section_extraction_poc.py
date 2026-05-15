#!/usr/bin/env python3
"""PoC harness for prompt section extraction.

Exercises every section kind the structural harvester can emit
(system, tools, message_part, reasoning, tool_calls, tool_result,
generation_prompt) against the Qwen3-0.6B chat template, asserts
partition invariants, and benchmarks the overhead of the structural
pass vs a baseline render.

Run:
    python scripts/section_extraction_poc.py

Requires:
    - transformers (fast tokenizer with Qwen3 chat template)
    - vllm package importable (we only use chat_utils + renderers.cache_control)
"""

from __future__ import annotations

import json
import statistics
import sys
import time
from pathlib import Path
from typing import Any

# Make the in-tree vllm package importable without a full install.
REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from transformers import AutoTokenizer

# Avoid the heavyweight vllm.__init__ (it pulls torch etc.). Import the
# modules we need directly so the PoC stays light.
import importlib.util


def _load_module(modname: str, relpath: str):
    spec = importlib.util.spec_from_file_location(modname, REPO_ROOT / relpath)
    assert spec and spec.loader, modname
    mod = importlib.util.module_from_spec(spec)
    sys.modules[modname] = mod
    spec.loader.exec_module(mod)
    return mod


# ---- minimal stubs so chat_utils + cache_control import without dragging
# ---- the full vllm dependency graph -----------------------------------------


def _install_stubs() -> None:
    import types

    # Top-level vllm package shim
    vllm = types.ModuleType("vllm")
    vllm.__path__ = [str(REPO_ROOT / "vllm")]
    sys.modules["vllm"] = vllm

    # vllm.envs
    envs = types.ModuleType("vllm.envs")
    envs.__getattr__ = lambda name: None  # type: ignore[attr-defined]
    sys.modules["vllm.envs"] = envs

    # vllm.logger
    logger_mod = types.ModuleType("vllm.logger")
    import logging as _logging

    def _init_logger(name: str) -> _logging.Logger:
        lg = _logging.getLogger(name)
        # add warning_once if missing
        if not hasattr(lg, "warning_once"):
            seen: set[tuple[str, tuple[Any, ...]]] = set()

            def warning_once(msg: str, *args: Any, **kwargs: Any) -> None:
                key = (msg, args)
                if key in seen:
                    return
                seen.add(key)
                lg.warning(msg, *args, **kwargs)

            lg.warning_once = warning_once  # type: ignore[attr-defined]
        return lg

    logger_mod.init_logger = _init_logger  # type: ignore[attr-defined]
    sys.modules["vllm.logger"] = logger_mod

    # vllm.exceptions
    exc_mod = types.ModuleType("vllm.exceptions")

    class VLLMValidationError(ValueError):
        def __init__(self, msg: str, parameter: str | None = None, value: Any = None) -> None:
            super().__init__(msg)
            self.parameter = parameter
            self.value = value

    exc_mod.VLLMValidationError = VLLMValidationError  # type: ignore[attr-defined]
    sys.modules["vllm.exceptions"] = exc_mod


def _patch_chat_utils_for_standalone() -> None:
    """Monkey-patch a few imports in chat_utils so we don't drag torch/openai_harmony.

    We don't actually need any of the heavyweight machinery; we only call
    `harvest_structural_marks`, which is a pure function over the parsed
    conversation dict shape.
    """
    import types

    # openai_harmony — chat_utils imports `Message as OpenAIHarmonyMessage`
    openai_harmony = types.ModuleType("openai_harmony")
    class _Msg:  # placeholder
        pass
    openai_harmony.Message = _Msg  # type: ignore[attr-defined]
    sys.modules["openai_harmony"] = openai_harmony

    # PIL.Image
    try:
        import PIL  # noqa: F401
    except ImportError:
        pil = types.ModuleType("PIL")
        pil_image = types.ModuleType("PIL.Image")
        class _PILImage:
            pass
        pil_image.Image = _PILImage  # type: ignore[attr-defined]
        sys.modules["PIL"] = pil
        sys.modules["PIL.Image"] = pil_image

    # vllm.inputs (used via `from vllm.inputs import MultiModalDataDict, MultiModalUUIDDict`)
    vllm_inputs = types.ModuleType("vllm.inputs")
    vllm_inputs.MultiModalDataDict = dict  # type: ignore[attr-defined]
    vllm_inputs.MultiModalUUIDDict = dict  # type: ignore[attr-defined]
    sys.modules["vllm.inputs"] = vllm_inputs

    # Everything else (vllm.config, vllm.model_executor, vllm.multimodal,
    # vllm.renderers.embed_utils, vllm.utils, etc.) -- pre-install lightweight
    # stubs. chat_utils only references their *names* at module load; we
    # never call into them from the harvester path.
    for modname, attrs in {
        "vllm.config": ["ModelConfig"],
        "vllm.model_executor.models": ["SupportsMultiModal"],
        "vllm.multimodal": ["MULTIMODAL_REGISTRY"],
        "vllm.multimodal.inputs": [
            "MultiModalBatchedField",
            "MultiModalFlatField",
            "MultiModalSharedField",
            "VisionChunk",
            "VisionChunkImage",
            "VisionChunkVideo",
        ],
        "vllm.multimodal.media": ["MEDIA_CONNECTOR_REGISTRY", "MediaConnector"],
        "vllm.multimodal.processing": ["BaseMultiModalProcessor"],
        "vllm.renderers.embed_utils": [
            "safe_load_prompt_embeds",
            "safe_load_prompt_embeds_async",
        ],
        "vllm.utils": ["random_uuid"],
        "vllm.utils.collection_utils": ["is_list_of"],
        "vllm.utils.import_utils": ["LazyLoader"],
    }.items():
        m = types.ModuleType(modname)
        for a in attrs:
            setattr(m, a, type(a, (), {}))
        sys.modules[modname] = m

    # LazyLoader returns a placeholder that won't be invoked
    def _LazyLoader(name, *args, **kwargs):
        m = types.ModuleType(name)
        # chat_utils annotates with torch.Tensor; provide a stand-in.
        if name == "torch":
            m.Tensor = type("Tensor", (), {})  # type: ignore[attr-defined]
        return m

    sys.modules["vllm.utils.import_utils"].LazyLoader = _LazyLoader  # type: ignore[attr-defined]


_install_stubs()
_patch_chat_utils_for_standalone()

chat_utils = _load_module("vllm.entrypoints.chat_utils", "vllm/entrypoints/chat_utils.py")
# The renderers __init__ pulls a lot in; load prompt_sections directly via a
# stub package.
renderers_pkg = type(sys)("vllm.renderers")
renderers_pkg.__path__ = [str(REPO_ROOT / "vllm" / "renderers")]
sys.modules["vllm.renderers"] = renderers_pkg
prompt_sections = _load_module(
    "vllm.renderers.prompt_sections", "vllm/renderers/prompt_sections.py"
)


# ----- conversation fixtures: cover every section kind ---------------------


def parsed_conversation_full() -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Qwen3-realistic conversation that exercises every section kind the
    Qwen3 template will actually render: system, tools, user, assistant
    with tool_calls, tool result, user follow-up, assistant with empty
    content + tool_calls, tool result, final assistant with reasoning
    (Qwen3 only emits reasoning on the latest assistant turn).
    """
    tools = [
        {
            "type": "function",
            "function": {
                "name": "get_weather",
                "description": (
                    "Look up the current weather conditions for a given city. "
                    "Returns temperature in Celsius and a textual description."
                ),
                "parameters": {
                    "type": "object",
                    "properties": {"city": {"type": "string"}},
                    "required": ["city"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "send_email",
                "description": "Send an email to a recipient with subject and body.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "to": {"type": "string"},
                        "subject": {"type": "string"},
                        "body": {"type": "string"},
                    },
                    "required": ["to", "subject", "body"],
                },
            },
        },
    ]
    conversation: list[dict[str, Any]] = [
        # system as string content
        {
            "role": "system",
            "content": (
                "You are a careful, terse assistant. Always verify weather "
                "before mentioning it. Use the send_email tool to deliver "
                "results, but only when the user explicitly asks for an email."
            ),
        },
        # user (single-string content -- the only shape Qwen3 renders)
        {
            "role": "user",
            "content": (
                "What's the weather in San Francisco? Also email me the "
                "result at alice@example.com."
            ),
        },
        # assistant with content + tool_calls
        {
            "role": "assistant",
            "content": "I'll check SF weather, then email the result.",
            "tool_calls": [
                {
                    "id": "call_w1",
                    "type": "function",
                    "function": {
                        "name": "get_weather",
                        "arguments": {"city": "San Francisco"},
                    },
                }
            ],
        },
        # tool result
        {
            "role": "tool",
            "tool_call_id": "call_w1",
            "content": "62F partly cloudy",
        },
        # user follow-up (string content)
        {
            "role": "user",
            "content": "Great, send the email now.",
        },
        # assistant with only tool_calls (no text content)
        {
            "role": "assistant",
            "content": "",
            "tool_calls": [
                {
                    "id": "call_e1",
                    "type": "function",
                    "function": {
                        "name": "send_email",
                        "arguments": {
                            "to": "alice@example.com",
                            "subject": "SF Weather",
                            "body": "Currently 62F and partly cloudy.",
                        },
                    },
                }
            ],
        },
        # tool result
        {
            "role": "tool",
            "tool_call_id": "call_e1",
            "content": "Email sent.",
        },
        # final assistant with reasoning (Qwen3 only renders reasoning on
        # the latest assistant turn -- this is the one that exercises it).
        {
            "role": "assistant",
            "content": "Done. SF is 62F partly cloudy and your email was sent.",
            "reasoning_content": (
                "I have results from both tool calls: weather is 62F "
                "partly cloudy and the email send succeeded. I'll "
                "summarize both for the user concisely."
            ),
        },
    ]
    return conversation, tools


def parsed_conversation_simple() -> tuple[list[dict[str, Any]], list[dict[str, Any]] | None]:
    """A shorter, plain conversation -- no tools, no reasoning, no tool calls."""
    return [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "What is the capital of France?"},
        {"role": "assistant", "content": "Paris is the capital of France."},
        {"role": "user", "content": "Tell me one fun fact."},
    ], None


def parsed_conversation_agentic_large(turns: int = 25) -> tuple[
    list[dict[str, Any]], list[dict[str, Any]]
]:
    """Long agentic trace: system + tools + N turns of (user, assistant
    with tool_call, tool result). Stresses the resolver with many marks
    in a single render.
    """
    tools = [
        {
            "type": "function",
            "function": {
                "name": "search_kb",
                "description": (
                    "Search the support knowledge base. Returns up to 5 "
                    "matching article ids with titles."
                ),
                "parameters": {
                    "type": "object",
                    "properties": {"query": {"type": "string"}},
                    "required": ["query"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "create_ticket",
                "description": "Open a support ticket with title and body.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "title": {"type": "string"},
                        "body": {"type": "string"},
                    },
                    "required": ["title", "body"],
                },
            },
        },
    ]
    convo: list[dict[str, Any]] = [
        {
            "role": "system",
            "content": (
                "You are a support engineer. Verify identity before sharing "
                "account details. Use the search_kb tool to find canonical "
                "answers before responding to the user. If the user reports "
                "a bug, open a ticket via create_ticket."
            ),
        }
    ]
    for i in range(turns):
        convo.extend(
            [
                {
                    "role": "user",
                    "content": (
                        f"Turn {i}: I'm seeing latency on the dashboard, "
                        f"and the metric shown is {200 + i * 17} ms. Can "
                        f"you check?"
                    ),
                },
                {
                    "role": "assistant",
                    "content": f"Looking into turn {i} now.",
                    "tool_calls": [
                        {
                            "id": f"call_s{i}",
                            "type": "function",
                            "function": {
                                "name": "search_kb",
                                "arguments": {"query": f"dashboard latency {200 + i * 17}"},
                            },
                        }
                    ],
                },
                {
                    "role": "tool",
                    "tool_call_id": f"call_s{i}",
                    "content": (
                        f"[KB-{i:03d}] Dashboard latency issues with metric "
                        f"~{200 + i * 17}ms typically resolve via the "
                        f"recommended cache flush procedure."
                    ),
                },
            ]
        )
    # final assistant turn with reasoning + content (exercises reasoning
    # for Qwen3's latest-assistant rule).
    convo.append(
        {
            "role": "assistant",
            "content": (
                "Summary: I checked KB for each of your reported metrics "
                "and the recommended fix is the same across all turns -- "
                "a cache flush. I can open a ticket if it persists."
            ),
            "reasoning_content": (
                "All turns reported the dashboard latency issue with "
                "different metric values but matched the same KB article "
                "template. The canonical resolution is a cache flush. I "
                "will summarize and offer to open a ticket."
            ),
        }
    )
    return convo, tools


# ----- driver --------------------------------------------------------------


def _bound_apply_chat_template(tokenizer):
    def _apply(*, conversation, tools, tokenize):
        out = tokenizer.apply_chat_template(
            conversation=conversation,
            tools=tools,
            tokenize=tokenize,
            add_generation_prompt=True,
        )
        # transformers >=4.50 returns BatchEncoding for tokenize=True; unwrap.
        if tokenize and not isinstance(out, (list, str)):
            try:
                return list(out["input_ids"])
            except Exception:
                pass
        return out

    return _apply


def _assert_partition(sections, n_tokens: int) -> None:
    """All emitted sections form a non-overlapping partition covering
    [first_start, n_tokens). Allow a head gap (BOS / template scaffold)
    before the first section.
    """
    assert sections, "expected at least one section"
    last_end = sections[0].token_start
    for s in sections:
        assert s.token_start >= last_end, f"overlap before {s.section_id}"
        assert s.token_end > s.token_start, f"zero/neg span on {s.section_id}"
        assert s.token_end <= n_tokens, f"end past total on {s.section_id}"
        last_end = s.token_end
    assert sections[-1].token_end == n_tokens, "last section must reach end"


def _section_dump(sections, token_ids, tokenizer) -> str:
    rows = ["#  kind                 role       msg part  tok_start  tok_end  len  decoded[:80]"]
    for s in sections:
        decoded = tokenizer.decode(token_ids[s.token_start : s.token_end])
        decoded = decoded.replace("\n", "\\n")
        rows.append(
            f"   {s.kind:<20} {(s.role or ''):<10} "
            f"{(str(s.message_index) if s.message_index is not None else ''):>3} "
            f"{(str(s.part_index) if s.part_index is not None else ''):>3} "
            f"{s.token_start:>10}  {s.token_end:>7}  {s.token_end - s.token_start:>3}  "
            f"{decoded[:80]}"
        )
    return "\n".join(rows)


def run_case(case_name, tokenizer, conversation, tools) -> dict[str, Any]:
    apply = _bound_apply_chat_template(tokenizer)
    clean_token_ids = apply(conversation=conversation, tools=tools, tokenize=True)

    marks = prompt_sections.harvest_section_marks(conversation, tools)

    # Use the efficient path: also pass clean_str and clean_offsets so the
    # resolver does just one extra render (the dirty one) and no extra
    # tokenizer call.
    clean_str = apply(conversation=conversation, tools=tools, tokenize=False)
    enc = tokenizer(
        clean_str, add_special_tokens=False, return_offsets_mapping=True
    )
    clean_offsets = list(enc["offset_mapping"])

    sections = prompt_sections.resolve_sections(
        marks=marks,
        conversation=conversation,
        tools=tools,
        clean_token_ids=clean_token_ids,
        apply_chat_template_fn=apply,
        clean_str=clean_str,
        clean_offsets=clean_offsets,
        add_generation_prompt=True,
    )

    print(f"\n=== {case_name} ===")
    print(f"tokens: {len(clean_token_ids)}  harvested marks: {len(marks)}  "
          f"resolved sections: {len(sections)}")
    print(_section_dump(sections, clean_token_ids, tokenizer))

    _assert_partition(sections, len(clean_token_ids))

    # Some templates drop marks legitimately (e.g. Qwen3 drops reasoning on
    # all but the latest assistant turn; templates that flatten multi-part
    # content drop part marks). Log instead of assert.
    if len(sections) < len(marks):
        print(
            f"  note: template dropped {len(marks) - len(sections)} mark(s) "
            f"(likely template-level filtering)"
        )
    return {
        "case": case_name,
        "n_tokens": len(clean_token_ids),
        "n_marks": len(marks),
        "n_sections": len(sections),
        "kinds": sorted({s.kind for s in sections}),
    }


def profile(case_name, tokenizer, conversation, tools, iters: int = 50) -> dict[str, Any]:
    """Measure render-with-sections vs baseline (single clean render)."""
    apply = _bound_apply_chat_template(tokenizer)
    # warm-up
    for _ in range(3):
        apply(conversation=conversation, tools=tools, tokenize=True)

    baseline = []
    for _ in range(iters):
        t0 = time.perf_counter()
        apply(conversation=conversation, tools=tools, tokenize=True)
        baseline.append((time.perf_counter() - t0) * 1000.0)

    # Path A (legacy total): caller passes nothing extra; resolver does
    # its own clean render + offset-mapped retokenize. Whole-call timing
    # includes: 1 baseline render + 1 dirty render + 1 clean render +
    # 1 offset-mapped tokenize.
    sec_legacy = []
    for _ in range(iters):
        t0 = time.perf_counter()
        clean_token_ids = apply(conversation=conversation, tools=tools, tokenize=True)
        marks = prompt_sections.harvest_section_marks(conversation, tools)
        prompt_sections.resolve_sections(
            marks=marks,
            conversation=conversation,
            tools=tools,
            clean_token_ids=clean_token_ids,
            apply_chat_template_fn=apply,
            tokenizer=tokenizer,
            add_generation_prompt=True,
        )
        sec_legacy.append((time.perf_counter() - t0) * 1000.0)

    # Path B (efficient): marginal cost of resolve_sections when the
    # caller has already done clean_str + clean_offsets for its own
    # purposes. Models a renderer wire-up that produces those as part
    # of normal request processing. The timed region covers only the
    # incremental work — harvest + resolver (single dirty render).
    # Pre-prep is done once outside the timed region.
    clean_token_ids = apply(conversation=conversation, tools=tools, tokenize=True)
    clean_str = apply(conversation=conversation, tools=tools, tokenize=False)
    enc = tokenizer(
        clean_str, add_special_tokens=False, return_offsets_mapping=True
    )
    clean_offsets = list(enc["offset_mapping"])

    sec_marginal = []
    for _ in range(iters):
        t0 = time.perf_counter()
        marks = prompt_sections.harvest_section_marks(conversation, tools)
        prompt_sections.resolve_sections(
            marks=marks,
            conversation=conversation,
            tools=tools,
            clean_token_ids=clean_token_ids,
            apply_chat_template_fn=apply,
            clean_str=clean_str,
            clean_offsets=clean_offsets,
            add_generation_prompt=True,
        )
        sec_marginal.append((time.perf_counter() - t0) * 1000.0)

    def stats(xs):
        xs_sorted = sorted(xs)
        return {
            "median_ms": statistics.median(xs_sorted),
            "p99_ms": xs_sorted[int(0.99 * (len(xs_sorted) - 1))],
            "max_ms": xs_sorted[-1],
            "mean_ms": statistics.mean(xs_sorted),
        }

    bs = stats(baseline)
    sl = stats(sec_legacy)
    sm = stats(sec_marginal)
    print(f"\n--- profile: {case_name} ({iters} iters) ---")
    print(f"baseline (tokenize)        : median={bs['median_ms']:>7.2f}ms  p99={bs['p99_ms']:>7.2f}ms")
    print(f"sectioned (legacy, total)  : median={sl['median_ms']:>7.2f}ms  p99={sl['p99_ms']:>7.2f}ms   "
          f"overhead +{sl['median_ms']-bs['median_ms']:.2f}/+{sl['p99_ms']-bs['p99_ms']:.2f}ms")
    print(f"marginal (efficient w/up)  : median={sm['median_ms']:>7.2f}ms  p99={sm['p99_ms']:>7.2f}ms   "
          f"<- harvest+resolver only, caller already has clean_str+offsets")
    return {
        "case": case_name,
        "iters": iters,
        "baseline": bs,
        "sectioned_legacy_total": sl,
        "marginal_efficient": sm,
        "overhead_legacy_median_ms": sl["median_ms"] - bs["median_ms"],
        "marginal_median_ms": sm["median_ms"],
    }


def main() -> int:
    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-0.6B", use_fast=True)

    # Functional checks + profile
    cases = [
        ("simple", *parsed_conversation_simple()),
        ("full_features", *parsed_conversation_full()),
        ("agentic_25turn", *parsed_conversation_agentic_large(turns=25)),
        ("agentic_50turn", *parsed_conversation_agentic_large(turns=50)),
    ]
    results = []
    for name, conv, tools in cases:
        results.append(run_case(name, tokenizer, conv, tools))

    profiles = []
    for name, conv, tools in cases:
        profiles.append(profile(name, tokenizer, conv, tools, iters=30))

    print("\n=== summary ===")
    for r in results:
        print(json.dumps(r))
    for p in profiles:
        print(json.dumps(p))
    return 0


if __name__ == "__main__":
    sys.exit(main())

# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Prompt section extraction — a generic mechanism for surfacing the
structural composition of a rendered chat-completion prompt.

A *section* is a labelled, half-open ``[token_start, token_end)`` window
in the rendered ``prompt_token_ids`` covering one logical piece of input:
the system message, the tools block (aggregate), each text content part
in a message, an assistant's reasoning, an assistant's tool_calls block,
a tool-role result, and the trailing generation prompt.

The mechanism is intentionally generic. It carries no policy — sections
are not retention directives, are not cache hints, and are not bound to
any single consumer. Consumers (the router, retention policies, future
cache_control-on-sections implementations, observability, etc.) read the
section table off ``GenerateRequest.sections`` and apply their own logic.

Algorithm overview (sentinel injection)
---------------------------------------
1. Walk the parsed conversation + tools list and emit one
   :class:`SectionMark` per logical section start, recording where the
   sentinel should be injected (which message/part/field, prepend or
   append).
2. Deep-copy the conversation and tools; inject a unique sentinel string
   ``__VLLM_SECTION_{i}__`` at each mark's target.
3. Render the chat template twice (dirty with sentinels, clean without).
4. Locate every sentinel by char offset in the dirty string via a single
   ``re.finditer`` pass; translate to clean-string char positions by
   subtracting the cumulative length of earlier sentinels.
5. Re-tokenize the clean string with ``return_offsets_mapping=True`` and
   run a single linear merge to translate char-offset → token-index.
6. Sanity-check: the offset-mapped retokenization must equal the
   renderer's own clean token IDs. If not, drop all sections (warn-once)
   — the request is not affected.
7. Section ``token_end`` = next section's ``token_start`` (or
   ``len(token_ids)`` for the last section). When
   ``add_generation_prompt=True`` and tokens remain past the last data
   section, emit a trailing ``generation_prompt`` section.

Why sentinel injection and not e.g. Jinja AST instrumentation
-------------------------------------------------------------
* Treats the chat template as a black box; works for every HF-class
  template that produces a string.
* Fails loudly (sanity check) rather than silently misattributing — bad
  output is detected via a token-stream equality check.
* Gracefully degrades: drop-on-failure returns an empty section list,
  request still succeeds.
"""

from __future__ import annotations

import copy
import re
from collections.abc import Callable
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Final, Literal

from vllm.logger import init_logger

if TYPE_CHECKING:
    from vllm.tokenizers import TokenizerLike

logger = init_logger(__name__)


# --- public schema ---------------------------------------------------------

# Soft cap on sections per request. Structural emission scales with
# conversation length (one section per text part / reasoning / tool_calls
# / tool_result / etc.); the cap is a DoS guard, not a policy limit.
MAX_PROMPT_SECTIONS: Final[int] = 512


# The set of section kinds the v1 harvester emits. Consumers should treat
# this as open for extension — additional kinds may appear as more
# templates and content shapes are supported.
PromptSectionKind = Literal[
    "system",
    "tools",
    "message_part",
    "reasoning",
    "tool_calls",
    "tool_result",
    "generation_prompt",
]


# Where a sentinel is injected to mark a section's start. Each target
# corresponds to a known reproducible point in the post-parse
# conversation / tools list.
InjectionTarget = Literal[
    "system_text",
    "tools_first",
    "tools_last",
    "message_text",
    "message_part_text",
    "reasoning_text",
    "tool_call_first_name",
    "tool_result_text",
]


@dataclass(frozen=True)
class SectionMark:
    """A structural-section marker emitted by the harvester.

    The marker describes the *start* of one logical section in the
    parsed conversation; the resolver translates it into a token
    position by sentinel injection. ``metadata`` is an opaque
    passthrough channel — vLLM ascribes no semantics to it; consumers
    interpret it as they choose.
    """

    section_id: str
    kind: str  # PromptSectionKind
    role: str | None
    message_index: int | None
    part_index: int | None
    injection_target: str  # InjectionTarget
    injection_side: Literal["prepend", "append"]
    metadata: dict[str, Any] | None = None


@dataclass(frozen=True)
class PromptSection:
    """A resolved section: structural identity + token range.

    Sections in the returned list form a contiguous, non-overlapping
    partition of the rendered token stream (from the first section's
    start to ``len(prompt_token_ids)``). Anything before the first
    section is template scaffold (BOS / leading role separators).
    """

    section_id: str
    kind: str  # PromptSectionKind
    role: str | None
    message_index: int | None
    part_index: int | None
    token_start: int
    token_end: int
    metadata: dict[str, Any] | None = None


# --- internals -------------------------------------------------------------

# Sentinel format. ASCII-only, anchored by double underscores on both
# sides (forces a tokenizer split under typical BPE merges), indexed so
# distinct marks produce distinct strings.
_SENTINEL_FMT = "__VLLM_SECTION_{i}__"
_SENTINEL_PATTERN = re.compile(r"__VLLM_SECTION_(\d+)__")


def _build_sentinel(mark_index: int) -> str:
    return _SENTINEL_FMT.format(i=mark_index)


def _apply_text(side: str, original: str | None, sentinel: str) -> str:
    base = original if isinstance(original, str) else ""
    return sentinel + base if side == "prepend" else base + sentinel


def _first_text_field(content: Any) -> tuple[Any, str] | None:
    """Return ``(container, key)`` for the first text-bearing field on a
    parsed message content shape, or ``None`` for non-text content.
    """
    if isinstance(content, list):
        for part in content:
            if not isinstance(part, dict):
                continue
            ptype = part.get("type")
            if ptype in (None, "text", "input_text", "output_text"):
                return part, "text"
    return None


def _inject_sentinels(
    conversation: list[dict[str, Any]],
    tools: list[dict[str, Any]] | None,
    marks: list[SectionMark],
) -> tuple[
    list[dict[str, Any]],
    list[dict[str, Any]] | None,
    dict[int, SectionMark],
]:
    """Inject one sentinel per mark into a deep copy of conversation/tools.

    Returns ``(dirty_conversation, dirty_tools, placed)`` where ``placed``
    maps the per-mark index to the mark that was successfully placed.
    Marks whose target is missing or unsupported are silently skipped;
    the caller decides whether to warn.
    """
    out_conv = copy.deepcopy(conversation)
    out_tools = copy.deepcopy(tools) if tools else tools
    placed: dict[int, SectionMark] = {}

    for idx, mark in enumerate(marks):
        sentinel = _build_sentinel(idx)
        side = mark.injection_side
        target = mark.injection_target

        if target in ("tools_first", "tools_last"):
            if not out_tools:
                continue
            tool = out_tools[0] if target == "tools_first" else out_tools[-1]
            if not isinstance(tool, dict):
                continue
            inner = tool.get("function") if isinstance(tool.get("function"), dict) else tool
            desc = inner.get("description")
            if isinstance(desc, str):
                inner["description"] = _apply_text(side, desc, sentinel)
                placed[idx] = mark
                continue
            name = inner.get("name")
            if isinstance(name, str):
                inner["name"] = _apply_text(side, name, sentinel)
                placed[idx] = mark
            continue

        msg_idx = mark.message_index
        if msg_idx is None or not (0 <= msg_idx < len(out_conv)):
            continue
        msg = out_conv[msg_idx]
        if not isinstance(msg, dict):
            continue

        if target in ("system_text", "message_text", "tool_result_text"):
            content = msg.get("content")
            if isinstance(content, str):
                msg["content"] = _apply_text(side, content, sentinel)
                placed[idx] = mark
                continue
            tf = _first_text_field(content)
            if tf is not None:
                part, key = tf
                part[key] = _apply_text(side, part.get(key), sentinel)
                placed[idx] = mark
            continue

        if target == "message_part_text":
            content = msg.get("content")
            part_idx = mark.part_index
            if not isinstance(content, list) or part_idx is None:
                continue
            if not (0 <= part_idx < len(content)):
                continue
            part = content[part_idx]
            if not isinstance(part, dict):
                continue
            ptype = part.get("type")
            if ptype not in (None, "text", "input_text", "output_text"):
                continue
            part["text"] = _apply_text(side, part.get("text"), sentinel)
            placed[idx] = mark
            continue

        if target == "reasoning_text":
            # Mutate both legacy/canonical keys so the template path that
            # reads either sees the sentinel.
            for key in ("reasoning_content", "reasoning"):
                if isinstance(msg.get(key), str):
                    msg[key] = _apply_text(side, msg[key], sentinel)
                    placed[idx] = mark
            continue

        if target == "tool_call_first_name":
            tool_calls = msg.get("tool_calls")
            if not isinstance(tool_calls, list) or not tool_calls:
                continue
            tc = tool_calls[0]
            if not isinstance(tc, dict):
                continue
            fn = tc.get("function")
            target_dict = fn if isinstance(fn, dict) else tc
            name = target_dict.get("name")
            if isinstance(name, str):
                target_dict["name"] = _apply_text(side, name, sentinel)
                placed[idx] = mark
            continue

    return out_conv, out_tools, placed


# --- public resolver -------------------------------------------------------


def resolve_sections(
    marks: list[SectionMark],
    conversation: list[dict[str, Any]],
    tools: list[dict[str, Any]] | None,
    *,
    clean_token_ids: list[int],
    apply_chat_template_fn: Callable[..., Any],
    tokenizer: "TokenizerLike | None" = None,
    clean_str: str | None = None,
    clean_offsets: list[tuple[int, int]] | None = None,
    add_generation_prompt: bool = True,
) -> list[PromptSection]:
    """Resolve ``marks`` into a partition of ``[..., len(clean_token_ids))``.

    Parameters
    ----------
    marks:
        :class:`SectionMark` instances in template-render order.
    conversation:
        Parsed conversation (post-:func:`parse_chat_messages`).
    tools:
        Tools list as passed to the chat template.
    clean_token_ids:
        Token IDs from the renderer's own clean ``tokenize=True`` render
        — used as a sanity anchor.
    apply_chat_template_fn:
        Caller-bound function with signature ``(conversation, tools,
        tokenize) -> str | list[int]``. The caller's
        ``safe_apply_chat_template`` wrapper is suitable. Called once
        per request to render the dirty (sentinel-injected) conversation;
        also called for the clean render if ``clean_str`` is not provided.
    tokenizer:
        Fast tokenizer supporting ``return_offsets_mapping``. Required
        only when ``clean_offsets`` is not provided.
    clean_str:
        Optional. The clean rendered string. If provided, skips the
        redundant clean render the resolver would otherwise do — useful
        when the caller already rendered with ``tokenize=False`` for its
        own purposes.
    clean_offsets:
        Optional. The list of ``(char_start, char_end)`` per-token
        offsets from re-tokenizing ``clean_str`` with
        ``return_offsets_mapping=True``. If provided, skips re-tokenization
        entirely. Sanity check against ``clean_token_ids`` still runs via
        comparison of the offset list's length with the token list.
    add_generation_prompt:
        Whether the renderer emits the trailing assistant scaffold; when
        true and tokens remain past the last data section, a synthetic
        ``generation_prompt`` section is appended.

    Returns
    -------
    Sections sorted by ``token_start``. Empty when no marks were given,
    when none could be placed, or when the sanity check fails.

    Cost model
    ----------
    Best case (caller provides ``clean_str`` and ``clean_offsets``):
    one extra chat template render (dirty), one ``re.finditer`` pass,
    O(n_marks + n_tokens) merge. No extra tokenizer call.

    Worst case (legacy call site): two chat template renders, one
    tokenizer pass with offset mapping, same merge.
    """
    if not marks:
        return []

    dirty_conv, dirty_tools, placed = _inject_sentinels(
        conversation, tools, marks
    )
    if not placed:
        return []

    if len(placed) < len(marks):
        n_dropped = len(marks) - len(placed)
        examples = [marks[i] for i in range(len(marks)) if i not in placed][:3]
        logger.warning_once(
            "prompt_sections: dropped %d mark(s) that could not be placed. "
            "Examples: %s",
            n_dropped,
            examples,
        )

    try:
        dirty_str = apply_chat_template_fn(
            conversation=dirty_conv, tools=dirty_tools, tokenize=False
        )
        if clean_str is None:
            clean_str = apply_chat_template_fn(
                conversation=conversation, tools=tools, tokenize=False
            )
    except Exception:
        logger.warning_once(
            "prompt_sections: chat template failed under sentinel injection; "
            "dropping %d section mark(s).",
            len(placed),
            exc_info=True,
        )
        return []

    if not isinstance(dirty_str, str) or not isinstance(clean_str, str):
        logger.warning_once(
            "prompt_sections: chat template did not return a string with "
            "tokenize=False; dropping section marks."
        )
        return []

    # Locate sentinels in dirty (single regex pass; finditer yields in
    # order so no sort is needed) and compute clean-string char offsets.
    occurrences: list[tuple[int, int]] = []  # (clean_char_pos, idx)
    cumulative = 0
    for m in _SENTINEL_PATTERN.finditer(dirty_str):
        try:
            idx = int(m.group(1))
        except ValueError:
            continue
        if idx not in placed:
            continue
        dirty_pos = m.start()
        slen = m.end() - m.start()
        occurrences.append((dirty_pos - cumulative, idx))
        cumulative += slen

    if not occurrences:
        return []

    # Offsets path: either use what the caller passed (cheap) or
    # retokenize the clean string with offset_mapping (one extra pass).
    if clean_offsets is None:
        if tokenizer is None:
            logger.warning_once(
                "prompt_sections: caller passed neither ``clean_offsets`` "
                "nor a ``tokenizer`` to compute them; dropping section "
                "marks."
            )
            return []
        try:
            enc = tokenizer(  # type: ignore[operator]
                clean_str,
                add_special_tokens=False,
                return_offsets_mapping=True,
            )
        except Exception:
            logger.warning_once(
                "prompt_sections: tokenizer does not support offset mapping; "
                "dropping section marks."
            )
            return []

        try:
            encoded_ids = list(enc["input_ids"])  # type: ignore[index]
            offsets = list(enc["offset_mapping"])  # type: ignore[index]
        except Exception:
            logger.warning_once(
                "prompt_sections: tokenizer did not return offset mapping in "
                "the expected shape; dropping section marks."
            )
            return []

        # Full sanity check available: token IDs from the offset-mapped
        # retokenize must equal the renderer's clean token IDs.
        if encoded_ids != list(clean_token_ids):
            logger.warning_once(
                "prompt_sections: offset-mapped retokenization does not match "
                "the renderer's clean token IDs (%d vs %d tokens); dropping "
                "%d section mark(s). Likely cause: chat template injects "
                "special tokens not present in the rendered string.",
                len(encoded_ids),
                len(clean_token_ids),
                len(placed),
            )
            return []
    else:
        offsets = list(clean_offsets)
        # Weaker sanity check (we don't have a separate token-ID stream
        # to compare): require lengths to match.
        if len(offsets) != len(clean_token_ids):
            logger.warning_once(
                "prompt_sections: caller-provided clean_offsets length (%d) "
                "disagrees with clean_token_ids length (%d); dropping %d "
                "section mark(s).",
                len(offsets),
                len(clean_token_ids),
                len(placed),
            )
            return []

    # Linear merge: both occurrence char positions and offset starts are
    # non-decreasing. O(n_marks + n_tokens) instead of O(n_marks * n_tokens).
    n_tokens = len(offsets)
    starts: list[int] = []
    for off in offsets:
        try:
            starts.append(int(off[0]))
        except (TypeError, IndexError):
            starts.append(0)

    token_idx_by_mark: dict[int, int] = {}
    k = 0
    # occurrences are emission-ordered by finditer (== clean-string order).
    for clean_pos, idx in occurrences:
        while k < n_tokens and starts[k] < clean_pos:
            k += 1
        token_idx_by_mark[idx] = k

    # Build the section list in emission order, then sort by token_start.
    raw: list[tuple[int, SectionMark]] = []
    for idx, mark in enumerate(marks):
        if idx not in token_idx_by_mark:
            continue
        raw.append((token_idx_by_mark[idx], mark))
    raw_sorted = sorted(enumerate(raw), key=lambda kv: (kv[1][0], kv[0]))
    just_sorted = [kv[1] for kv in raw_sorted]

    sections: list[PromptSection] = []
    for i, (tok_start, mark) in enumerate(just_sorted):
        tok_end = (
            just_sorted[i + 1][0] if i + 1 < len(just_sorted) else n_tokens
        )
        sections.append(
            PromptSection(
                section_id=mark.section_id,
                kind=mark.kind,
                role=mark.role,
                message_index=mark.message_index,
                part_index=mark.part_index,
                token_start=tok_start,
                token_end=tok_end,
                metadata=mark.metadata,
            )
        )

    # Synthetic generation_prompt section for any tail past the last data
    # section (e.g. ``<|im_start|>assistant\n`` scaffold).
    if add_generation_prompt and sections:
        last_end = sections[-1].token_end
        if last_end < n_tokens:
            sections.append(
                PromptSection(
                    section_id="gen_prompt",
                    kind="generation_prompt",
                    role=None,
                    message_index=None,
                    part_index=None,
                    token_start=last_end,
                    token_end=n_tokens,
                    metadata=None,
                )
            )

    return sections


# --- harvester -------------------------------------------------------------
#
# The harvester walks a parsed Chat-Completion conversation + tools list and
# emits one :class:`SectionMark` per logical section start, in template-render
# order. Pure structural extraction — no policy, no interpretation, no I/O.
#
# It is deliberately a free function over the dict shape that
# ``parse_chat_messages`` (in ``vllm/entrypoints/chat_utils.py``) produces.
# That keeps section extraction independent of the rest of vLLM's chat
# pipeline -- callers can invoke the harvester on any list of normalized
# message dicts, including ones produced by other parsers, without touching
# chat_utils.


def _content_part_text(part: Any) -> str | None:
    """Return the text-bearing string of a content part, or None.

    Handles ``{"type": "text", "text": "..."}`` and ``{"text": "..."}``
    shapes that appear in the parsed conversation. Returns None for
    non-text parts (image, audio, tool_reference, etc.) — those produce
    multi-modal placeholder ranges separately on the GenerateRequest
    ``features`` field.
    """
    if not isinstance(part, dict):
        return None
    part_type = part.get("type")
    if part_type in (None, "text", "input_text", "output_text"):
        text = part.get("text")
        return text if isinstance(text, str) else None
    return None


def _passthrough_metadata(
    src: Any,
    extra_metadata_keys: tuple[str, ...] | None,
) -> dict[str, Any] | None:
    """Extract requested metadata keys from a parsed dict (content part,
    message, tool). Returns None if no metadata was found.

    The harvester does not interpret these values; they ride through as
    a generic passthrough channel. Each consumer requests the keys it
    cares about.
    """
    if not extra_metadata_keys or not isinstance(src, dict):
        return None
    out: dict[str, Any] = {}
    for key in extra_metadata_keys:
        val = src.get(key)
        if val is None:
            continue
        out[key] = val
    return out or None


def harvest_section_marks(
    conversation: list[dict[str, Any]],
    tools: list[dict[str, Any]] | None,
    *,
    extra_metadata_keys: tuple[str, ...] | None = None,
) -> list[SectionMark]:
    """Walk a parsed conversation + tools list and emit one
    :class:`SectionMark` per logical section start, in template-render
    order. Pure structural extraction — no policy, no interpretation.

    Emitted kinds: ``system``, ``tools`` (aggregate), ``message_part``
    (one per text content part), ``reasoning`` (per assistant message
    with reasoning text), ``tool_calls`` (per assistant message with
    tool_calls), ``tool_result`` (per tool-role message). The
    ``generation_prompt`` section is appended by :func:`resolve_sections`
    if a trailing scaffold remains.

    Application-supplied metadata fields named by ``extra_metadata_keys``
    are echoed through onto each mark's ``metadata`` dict — vLLM
    ascribes no semantics. Pass ``None`` to disable.

    Expected conversation message shape (matches the output of
    :func:`vllm.entrypoints.chat_utils.parse_chat_messages`):

    .. code-block:: python

        {
            "role": "user" | "assistant" | "system" | "tool",
            "content": str | None | list[dict],
            "reasoning_content": str | None,     # assistant
            "tool_calls": list[dict] | None,     # assistant
            "tool_call_id": str | None,          # tool
        }
    """
    marks: list[SectionMark] = []

    # 1. Tools aggregate. Rendered before any conversation messages in
    #    most templates (Qwen3, chatml-style). Sentinel goes at the start
    #    of the first tool's description (or name fallback).
    if tools:
        tools_meta: dict[str, Any] | None = None
        for tool in tools:
            if not isinstance(tool, dict):
                continue
            tools_meta = _passthrough_metadata(tool, extra_metadata_keys)
            if tools_meta is None:
                inner = tool.get("function")
                if isinstance(inner, dict):
                    tools_meta = _passthrough_metadata(
                        inner, extra_metadata_keys
                    )
            if tools_meta is not None:
                break
        marks.append(
            SectionMark(
                section_id="tools",
                kind="tools",
                role=None,
                message_index=None,
                part_index=None,
                injection_target="tools_first",
                injection_side="prepend",
                metadata=tools_meta,
            )
        )

    # 2. Per-message marks, in conversation order.
    for msg_idx, msg in enumerate(conversation):
        role = msg.get("role")
        content = msg.get("content")
        msg_meta = _passthrough_metadata(msg, extra_metadata_keys)

        # 2a. System role.
        if role == "system":
            marks.append(
                SectionMark(
                    section_id="sys" if msg_idx == 0 else f"msg[{msg_idx}].sys",
                    kind="system",
                    role=role,
                    message_index=msg_idx,
                    part_index=None,
                    injection_target="system_text",
                    injection_side="prepend",
                    metadata=msg_meta,
                )
            )
            continue

        # 2b. Tool-role result.
        if role == "tool":
            marks.append(
                SectionMark(
                    section_id=f"msg[{msg_idx}].tool_result",
                    kind="tool_result",
                    role=role,
                    message_index=msg_idx,
                    part_index=None,
                    injection_target="tool_result_text",
                    injection_side="prepend",
                    metadata=msg_meta,
                )
            )
            continue

        # 2c. Assistant reasoning (precedes content in templates that
        #     emit it). Only emit when reasoning text is present.
        if role == "assistant":
            reasoning = msg.get("reasoning_content") or msg.get("reasoning")
            if isinstance(reasoning, str) and reasoning:
                marks.append(
                    SectionMark(
                        section_id=f"msg[{msg_idx}].reasoning",
                        kind="reasoning",
                        role=role,
                        message_index=msg_idx,
                        part_index=None,
                        injection_target="reasoning_text",
                        injection_side="prepend",
                        metadata=None,
                    )
                )

        # 2d. Text content parts. Empty content still emits a section so
        #     role-transition tokens are attributed correctly.
        if isinstance(content, str):
            marks.append(
                SectionMark(
                    section_id=f"msg[{msg_idx}].part[0]"
                    if role == "user"
                    else f"msg[{msg_idx}].content",
                    kind="message_part",
                    role=role,
                    message_index=msg_idx,
                    part_index=0,
                    injection_target="message_text",
                    injection_side="prepend",
                    metadata=msg_meta,
                )
            )
        elif isinstance(content, list):
            for part_idx, part in enumerate(content):
                if _content_part_text(part) is None:
                    # Non-text parts (image, audio, ...) are covered by
                    # the MM placeholder ranges on GenerateRequest.
                    continue
                marks.append(
                    SectionMark(
                        section_id=f"msg[{msg_idx}].part[{part_idx}]",
                        kind="message_part",
                        role=role,
                        message_index=msg_idx,
                        part_index=part_idx,
                        injection_target="message_part_text",
                        injection_side="prepend",
                        metadata=_passthrough_metadata(
                            part, extra_metadata_keys
                        ),
                    )
                )

        # 2e. Assistant tool_calls. One aggregate mark per assistant
        #     message; sentinel goes into the first tool_call's
        #     function.name (a literal string in chatml-style templates).
        if role == "assistant":
            tool_calls = msg.get("tool_calls")
            if isinstance(tool_calls, list) and len(tool_calls) > 0:
                marks.append(
                    SectionMark(
                        section_id=f"msg[{msg_idx}].tool_calls",
                        kind="tool_calls",
                        role=role,
                        message_index=msg_idx,
                        part_index=None,
                        injection_target="tool_call_first_name",
                        injection_side="prepend",
                        metadata=None,
                    )
                )

    return marks


__all__ = [
    "MAX_PROMPT_SECTIONS",
    "PromptSectionKind",
    "InjectionTarget",
    "SectionMark",
    "PromptSection",
    "resolve_sections",
    "harvest_section_marks",
]

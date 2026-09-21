from __future__ import annotations

from dataclasses import dataclass

import re


from typing import Sequence

VALID_USES = {"semantic"}

MIN_CHUNK_TOKENS = 4

MAX_CHUNK_TOKENS = 16

PREFERRED_MIN_CHUNK_TOKENS = 6

PREFERRED_MAX_CHUNK_TOKENS = 10

CHUNK_RANGE_TAG = "semantic"

PROTECTED_ATOM_POLICY = "structural_atoms"

def normalize_use(value: str) -> str:
    value = value.strip().lower()
    if value not in VALID_USES:
        raise ValueError(
            f"Unsupported chunking method: {value!r}; expected semantic"
        )
    return value

@dataclass(frozen=True)
class NaturalSegment:
    start: int
    end: int
    segment_type: str
    text: str

@dataclass(frozen=True)
class CompressionPlan:
    inserted_ids: list[int]
    frontier_ends: list[int]
    segment_latent_counts: list[int]
    chunk_sizes: list[list[int]]

    @property
    def latent_count(self) -> int:
        return len(self.frontier_ends)

_ANALYSIS_RE = re.compile(r"(?m)^--- Analysis ---\s*$")

_FINAL_RE = re.compile(r"(?m)^--- Final Judgment ---\s*$")

_RESPONSE_RE = re.compile(r"(?m)^\*\*Response ([AB]):\*\*\s*$")

_CRITERION_RE = re.compile(
    r"(?m)^-\s*Criterion\s+(\d+)\s*\[(Hard Rule|Principle)\]\s*:"
)

_SOFT_PUNCTUATION = {",", ":", ";", "\u2014", "\u2013"}

_CLOSERS = {'"', "'", "\u2019", "\u201d", ")", "]", "}"}

_ABBREVIATIONS = (
    "e.g.", "i.e.", "etc.", "vs.", "mr.", "mrs.", "ms.", "dr.",
    "prof.", "no.",
)

_PROTECTED_ATOM_RE = re.compile(
    r"--- (?:Compliance Check|Analysis|Final Judgment) ---"
    r"|\*\*Response [AB]:\*\*"
    r"|\[(?:Hard Rule|Principle)\]"
    r"|Identified Gatekeeper Criterion"
    r"|(?<=^- )Criterion\s+\d+"
    r"|Justification:"
    r"|(?<![A-Za-z])Not Met\."
    r"|(?<![A-Za-z])Met\.",
    re.MULTILINE,
)

def protected_token_boundaries(text: str, offsets) -> set[int]:
    forbidden = set()
    for match in _PROTECTED_ATOM_RE.finditer(text):
        start = 0
        while start < len(offsets) and offsets[start][1] <= match.start():
            start += 1
        end = start
        while end < len(offsets) and offsets[end][0] < match.end():
            end += 1
        if 1 < end - start <= PREFERRED_MAX_CHUNK_TOKENS:
            forbidden.update(range(start + 1, end))
    return forbidden

def split_original_cot(cot: str) -> list[NaturalSegment]:
    """Return unchanged outer rubric segments."""
    if not isinstance(cot, str) or not cot:
        raise ValueError("cot must be a non-empty string")
    analysis = list(_ANALYSIS_RE.finditer(cot))
    finals = list(_FINAL_RE.finditer(cot))
    if len(analysis) != 1 or len(finals) != 1 or analysis[0].start() >= finals[0].start():
        raise ValueError("Expected one ordered Analysis and Final Judgment section")

    body_start = analysis[0].end()
    body_end = finals[0].start()
    responses = [m for m in _RESPONSE_RE.finditer(cot, body_start, body_end)]
    criteria = [m for m in _CRITERION_RE.finditer(cot, body_start, body_end)]
    if {m.group(1) for m in responses} != {"A", "B"} or not criteria:
        raise ValueError("Analysis must contain Response A/B headers and criteria")

    markers: dict[int, str] = {finals[0].start(): "aggregation"}
    assigned_criteria = 0
    for index, response in enumerate(responses):
        block_end = responses[index + 1].start() if index + 1 < len(responses) else finals[0].start()
        block_criteria = [
            match for match in criteria if response.end() <= match.start() < block_end
        ]
        if not block_criteria:
            raise ValueError(f"Response {response.group(1)} block contains no criterion")
        markers[response.start()] = "criterion"
        for match in block_criteria[1:]:
            markers[match.start()] = "criterion"
        assigned_criteria += len(block_criteria)
    if assigned_criteria != len(criteria):
        raise ValueError("Found criteria outside response blocks")

    starts = [0] + sorted(markers)
    segments = []
    for index, start in enumerate(starts):
        end = starts[index + 1] if index + 1 < len(starts) else len(cot)
        segment_type = "gatekeeper" if start == 0 else markers[start]
        segments.append(NaturalSegment(start, end, segment_type, cot[start:end]))
    if "".join(segment.text for segment in segments) != cot:
        raise AssertionError("Natural segments do not reconstruct the original CoT")
    return segments

def _is_sentence_period(text: str, index: int) -> bool:
    if text[index] != ".":
        return False
    before = text[index - 1] if index else ""
    after = text[index + 1] if index + 1 < len(text) else ""
    # Treat an ellipsis as one protected prose/code token, not three sentence
    # ends.  A real sentence after ``...`` is still available through later
    # whitespace candidates selected by the constrained planner.
    if before == "." or after == ".":
        return False
    if before.isdigit() and after.isdigit():
        return False
    if after.isalnum() or after == "_":
        return False
    prefix = text[max(0, index - 7) : index + 1].lower()
    if any(prefix.endswith(abbreviation) for abbreviation in _ABBREVIATIONS):
        return False
    wider_prefix = text[max(0, index - 40) : index + 1].lower()
    if re.search(
        r"(?:criterion\s+|identified gatekeeper criterion:\s*)\d+\.$",
        wider_prefix,
    ):
        return False
    if re.search(r"(?:response|option|answer)\s+[ab]\.$", wider_prefix):
        return True
    if before.isalpha() and (index < 2 or not text[index - 2].isalpha()):
        return False
    return True

def _punctuation_boundaries(text: str, start: int, end: int, *, soft: bool) -> list[int]:
    boundaries = []
    index = start
    bracket_depth = 0
    in_double_quote = False
    math_delimiter = None
    backtick_delimiter = None
    while index < end:
        char = text[index]
        escaped = index > start and text[index - 1] == "\\"
        if not escaped and text.startswith("```", index):
            backtick_delimiter = None if backtick_delimiter == "```" else "```"
            index += 3
            continue
        if char == "`" and not escaped and backtick_delimiter != "```":
            backtick_delimiter = None if backtick_delimiter == "`" else "`"
            index += 1
            continue
        if backtick_delimiter is not None:
            index += 1
            continue
        # Protect markup/PHP/XML tags such as ``<?php?>`` and ``<!-- ... -->``.
        if char == "<" and re.match(r"<(?:/?[A-Za-z]|[!?])", text[index:]):
            close = text.find(">", index + 1, end)
            if close >= 0:
                index = close + 1
                continue
        if not escaped and text.startswith("$$", index):
            if math_delimiter is None:
                math_delimiter = "$$"
            elif math_delimiter == "$$":
                math_delimiter = None
            index += 2
            continue
        if char == "$" and not escaped and math_delimiter != "$$":
            math_delimiter = None if math_delimiter == "$" else "$"
            index += 1
            continue
        if char in "([{":
            bracket_depth += 1
        elif char in ")]}":
            bracket_depth = max(0, bracket_depth - 1)
        elif char in {'"', "\u201c", "\u201d"}:
            if char == "\u201c":
                in_double_quote = True
            elif char == "\u201d":
                in_double_quote = False
            else:
                in_double_quote = not in_double_quote
        is_boundary = char in _SOFT_PUNCTUATION if soft else (
            char in "!?" or (char == "." and _is_sentence_period(text, index))
        )
        if bracket_depth > 0 or in_double_quote or math_delimiter is not None:
            is_boundary = False
        # ``\,`` is a LaTeX spacing command, not prose punctuation.  This
        # also protects formulas that happen to appear outside dollar math.
        if soft and char == "," and escaped:
            is_boundary = False
        if soft and char == ":":
            previous_newline = text.rfind("\n", start, index)
            line_start = start if previous_newline < start else previous_newline + 1
            line_prefix = text[line_start:index].strip()
            structural_prefix = line_prefix.lower()
            if (
                line_prefix.startswith("**Response ")
                or line_prefix.startswith("- Criterion ")
                or re.search(
                    r"(?:identified gatekeeper criterion|criterion\s+\d+)$",
                    structural_prefix,
                )
            ):
                is_boundary = False
        if not soft and is_boundary:
            probe = index + 1
            while probe < end and text[probe].isspace():
                probe += 1
            if (
                text.startswith("[Principle]", probe)
                or text.startswith("[Hard Rule]", probe)
                or text.startswith("--- Analysis ---", probe)
            ):
                is_boundary = False
        if not is_boundary:
            index += 1
            continue
        boundary = index + 1
        if not soft and char in "!?":
            while boundary < end and text[boundary] in "!?":
                boundary += 1
        while boundary < end and text[boundary] in _CLOSERS:
            boundary += 1
        while boundary < end and text[boundary].isspace():
            boundary += 1
        if start < boundary < end:
            boundaries.append(boundary)
        index = boundary
    return boundaries

def _tokenize_once(cot: str, tokenizer):
    encoded = tokenizer(
        cot, truncation=False, padding=False, add_special_tokens=False,
        return_attention_mask=False, return_offsets_mapping=True,
    )
    ids = list(encoded["input_ids"])
    offsets = list(encoded["offset_mapping"])
    if not ids or len(ids) != len(offsets):
        raise ValueError("Tokenizer did not return usable offset mappings")
    return ids, offsets

def _char_to_token_boundary(offsets, boundary: int) -> int:
    cursor = 0
    while cursor < len(offsets) and offsets[cursor][1] <= boundary:
        cursor += 1
    return cursor

def balanced_sizes(length: int, count: int) -> list[int]:
    if length <= 0 or count <= 0 or count > length:
        raise ValueError(f"Invalid balanced split: length={length}, count={count}")
    boundaries = [round(i * length / count) for i in range(count + 1)]
    sizes = [boundaries[i + 1] - boundaries[i] for i in range(count)]
    if min(sizes) <= 0 or sum(sizes) != length or max(sizes) - min(sizes) > 1:
        raise AssertionError((length, count, sizes))
    return sizes

def protected_balanced_sizes(
    length: int, count: int, forbidden_boundaries: Sequence[int], rate: int
) -> list[int]:
    forbidden = frozenset(forbidden_boundaries)
    if not forbidden:
        return balanced_sizes(length, count)
    states: dict[int, tuple[tuple[int, int, int, int], list[int]]] = {
        0: ((0, 0, 0, 0), [])
    }
    for chunk_index in range(count):
        remaining_chunks = count - chunk_index - 1
        next_states = {}
        for position, (cost, sizes) in states.items():
            for size in range(MIN_CHUNK_TOKENS, MAX_CHUNK_TOKENS + 1):
                end = position + size
                remaining = length - end
                if end > length:
                    break
                if not (
                    remaining_chunks * MIN_CHUNK_TOKENS
                    <= remaining
                    <= remaining_chunks * MAX_CHUNK_TOKENS
                ):
                    continue
                if end < length and end in forbidden:
                    continue
                outside = int(
                    size < PREFERRED_MIN_CHUNK_TOKENS
                    or size > PREFERRED_MAX_CHUNK_TOKENS
                )
                distance = max(PREFERRED_MIN_CHUNK_TOKENS - size, 0) + max(
                    size - PREFERRED_MAX_CHUNK_TOKENS, 0
                )
                variance = (size - rate) ** 2
                ideal_end = round((chunk_index + 1) * length / count)
                candidate = (
                    (
                        cost[0] + outside,
                        cost[1] + distance,
                        cost[2] + variance,
                        cost[3] + abs(end - ideal_end),
                    ),
                    sizes + [size],
                )
                previous = next_states.get(end)
                if previous is None or candidate[0] < previous[0]:
                    next_states[end] = candidate
        states = next_states
    if length not in states:
        raise ValueError(
            "Cannot satisfy hard 4--16 bounds and exact chunk count without "
            f"splitting a protected semantic atom: length={length}, count={count}, "
            f"forbidden={sorted(forbidden)}"
        )
    return states[length][1]

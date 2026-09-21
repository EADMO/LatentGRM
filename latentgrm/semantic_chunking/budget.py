from __future__ import annotations

import math

import re


from typing import Sequence

MIN_CHUNK_TOKENS = 4

MAX_CHUNK_TOKENS = 16

PREFERRED_MIN_CHUNK_TOKENS = 6

PREFERRED_MAX_CHUNK_TOKENS = 10

CHUNK_RANGE_TAG = "semantic"

PROTECTED_ATOM_POLICY = "structural_atoms"

_ANALYSIS_RE = re.compile(r"(?m)^--- Analysis ---\s*$")

_FINAL_RE = re.compile(r"(?m)^--- Final Judgment ---\s*$")

_RESPONSE_RE = re.compile(r"(?m)^\*\*Response ([AB]):\*\*\s*$")

_CRITERION_RE = re.compile(
    r"(?m)^-\s*Criterion\s+(\d+)\s*\[(Hard Rule|Principle)\]\s*:"
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

def balanced_sizes(length: int, count: int) -> list[int]:
    if length <= 0 or count <= 0 or count > length:
        raise ValueError(f"Invalid balanced split: length={length}, count={count}")
    boundaries = [round(i * length / count) for i in range(count + 1)]
    sizes = [boundaries[i + 1] - boundaries[i] for i in range(count)]
    if min(sizes) <= 0 or sum(sizes) != length or max(sizes) - min(sizes) > 1:
        raise AssertionError((length, count, sizes))
    return sizes

def allocate_global_budget(lengths: Sequence[int], rate: int) -> list[int]:
    """Allocate the exact budget, preferring 6--10 while enforcing 4--16."""
    if rate <= 0 or not lengths or any(length <= 0 for length in lengths):
        raise ValueError((lengths, rate))
    target = math.ceil(sum(lengths) / rate)
    lower = [math.ceil(length / MAX_CHUNK_TOKENS) for length in lengths]
    upper = [length // MIN_CHUNK_TOKENS for length in lengths]
    if any(limit == 0 for limit in upper):
        raise ValueError(
            "A mandatory rubric segment is shorter than the four-token minimum: "
            f"lengths={list(lengths)}"
        )
    if not sum(lower) <= target <= sum(upper):
        raise ValueError(
            "The exact compression budget is infeasible while preserving all rubric "
            f"boundaries: lengths={list(lengths)}, target={target}, "
            f"lower={lower}, upper={upper}"
        )

    # Dynamic programming makes the preference global: first minimize the
    # number of chunks outside 6--10, then their distance from that interval,
    # then ordinary variance around the requested compression rate.
    states: dict[int, tuple[tuple[int, int, int], list[int]]] = {0: ((0, 0, 0), [])}
    for length, lo, hi in zip(lengths, lower, upper):
        next_states = {}
        for used, (cost, chosen) in states.items():
            for count in range(lo, hi + 1):
                if used + count > target:
                    continue
                sizes = balanced_sizes(length, count)
                outside = sum(
                    size < PREFERRED_MIN_CHUNK_TOKENS
                    or size > PREFERRED_MAX_CHUNK_TOKENS
                    for size in sizes
                )
                distance = sum(
                    max(PREFERRED_MIN_CHUNK_TOKENS - size, 0)
                    + max(size - PREFERRED_MAX_CHUNK_TOKENS, 0)
                    for size in sizes
                )
                variance = sum((size - rate) ** 2 for size in sizes)
                candidate = (
                    (cost[0] + outside, cost[1] + distance, cost[2] + variance),
                    chosen + [count],
                )
                previous = next_states.get(used + count)
                if previous is None or candidate[0] < previous[0]:
                    next_states[used + count] = candidate
        states = next_states
    if target not in states:
        raise AssertionError((lengths, rate, target, lower, upper))
    return states[target][1]

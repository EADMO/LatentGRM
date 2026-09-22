"""Prompt templates for rubric generation and latent judgments."""

from __future__ import annotations


RUBRIC_GENERATION_TEMPLATE = """Your task is to extract a set of rubric-style instructions from a user's request.
These rubrics will be used as evaluation criteria to check if a response fully meets the request.

Rubric items should be broadly reusable whenever possible. Topic-specific details (e.g., names, places,
numbers, or required content) are allowed in a [Hard Rule] only when they are necessary to preserve an
explicit requirement from the request. Unnecessary topic-specific details are invalid.

- **Two Distinct Categories:**
  - [Hard Rule]: Preserve explicit requirements stated in the <request> exactly (format, length,
    structure, names, numbers, forbidden/required elements, etc.).
  - [Principle]: Derived by abstracting any concrete cues into domain-agnostic quality criteria
    (e.g., clarity, correctness, sound reasoning, pedagogy).
- **Comprehensiveness:** The rubric must cover all critical aspects implied by the request (including any
  examples contained in it), spanning explicit requirements and implicit quality standards.
- **Conciseness & Uniqueness:** Each rubric must capture a distinct evaluation criterion. Overlapping or
  redundant criteria must be merged into a single rubric. Wording must be precise and free of repetition.
- **Format Requirements:**
  - Use a numbered list.
  - Each item starts with "The response" phrased in third person.
  - Append [Hard Rule] or [Principle] at the end of each item.
  - Do not include reasoning, explanations, or examples in the final output, only the rubrics.

Here is the request:
{instruction}

Please generate the rubrics for the above request."""


def build_rubric_prompt(instruction: str) -> str:
    return RUBRIC_GENERATION_TEMPLATE.format(instruction=instruction.strip())

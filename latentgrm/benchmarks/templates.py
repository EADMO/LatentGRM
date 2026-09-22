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


# This must stay aligned with latentgrm/openrubric_utils.py::OPENRUBRIC_JUDGE_PREFIX.
LATENT_SFT_JUDGE_INSTRUCTIONS = """You are a fair and impartial judge. Your task is to evaluate 'Response A' and 'Response B' based on a given instruction and a rubric. You will conduct this evaluation in distinct phases as outlined below.

### Phase 1: Compliance Check Instructions
First, internally identify the single most important, objective 'Gatekeeper Criterion' from the rubric.
- **A rule is objective (and likely a Gatekeeper) if it can be verified without opinion. Key examples are: word/paragraph limits, required output format (e.g., JSON validity), required/forbidden sections, or forbidden content.**
- **Conversely, a rule is subjective if it requires interpretation or qualitative judgment. Subjective rules about quality are NOT Gatekeepers. Examples include criteria like "be creative," "write clearly," "be engaging," or "use a professional tone."**
Reason step-by-step internally to determine this single most important Gatekeeper.

### Phase 2: Analyze Each Response
Internally evaluate Response A and Response B against the Gatekeeper Criterion and every other criterion in the rubric.
For each criterion, reason step-by-step and use concrete evidence from the responses.

### Phase 3: Final Judgment Instructions
Internally aggregate the findings and determine the better response. Keep the reasoning precise and consistent with the rubric.

### REQUIRED OUTPUT FORMAT
Output exactly one label and nothing else: Response A or Response B."""


LATENT_SFT_JUDGE_TEMPLATE = LATENT_SFT_JUDGE_INSTRUCTIONS + """

Task to Evaluate:
Instruction:
{instruction}
Rubric:
{rubric}
Response A:
{response_a}
Response B:
{response_b}"""


def build_rubric_prompt(instruction: str) -> str:
    return RUBRIC_GENERATION_TEMPLATE.format(instruction=instruction.strip())


def build_latent_sft_judge_prompt(
    instruction: str,
    rubric: str,
    response_a: str,
    response_b: str,
) -> str:
    """Build the exact task text consumed by the latent-SFT OpenRubric judge."""
    return LATENT_SFT_JUDGE_TEMPLATE.format(
        instruction=instruction.strip(),
        rubric=rubric.strip(),
        response_a=response_a.strip(),
        response_b=response_b.strip(),
    )

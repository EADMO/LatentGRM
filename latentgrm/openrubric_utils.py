import re
from typing import Any


TASK_MARKER = "Task to Evaluate:"
OPENRUBRIC_TASK_TYPE = "openrubric_judge"
VALID_LABELS = ("Response A", "Response B")

OPENRUBRIC_JUDGE_PREFIX = """You are a fair and impartial judge. Your task is to evaluate 'Response A' and 'Response B' based on a given instruction and a rubric. You will conduct this evaluation in distinct phases as outlined below.

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

_WINNER_LINE_RE = re.compile(
    r"(?im)^[ \t]*Winner:[ \t]*(.*?)[ \t]*$"
)
_VALID_WINNER_VALUE_RE = re.compile(r"(?i)^Response[ \t]+([AB])$")


def convert_judge_record(record: dict[str, Any], source_index: int) -> dict[str, Any]:
    """Convert one OpenRubric judge SFT row to the LatentGRM schema."""
    if not isinstance(record, dict):
        raise ValueError(f"Record {source_index} must be a JSON object")

    instruction = record.get("instruction")
    output = record.get("output")
    for field_name, value in (("instruction", instruction), ("output", output)):
        if not isinstance(value, str) or not value.strip():
            raise ValueError(
                f"Record {source_index} has an invalid {field_name!r} field"
            )

    if instruction.count(TASK_MARKER) != 1:
        raise ValueError(
            f"Record {source_index} must contain exactly one {TASK_MARKER!r} marker"
        )

    winner_matches = list(_WINNER_LINE_RE.finditer(output))
    if len(winner_matches) != 1:
        raise ValueError(
            f"Record {source_index} must contain exactly one Winner line; "
            f"found {len(winner_matches)}"
        )

    winner_match = winner_matches[0]
    winner_value = winner_match.group(1).strip()
    valid_winner = _VALID_WINNER_VALUE_RE.fullmatch(winner_value)
    if valid_winner is None:
        raise ValueError(
            f"Record {source_index} has unsupported winner label {winner_value!r}"
        )
    label = f"Response {valid_winner.group(1).upper()}"

    task_body = instruction.split(TASK_MARKER, 1)[1].strip()
    if not task_body:
        raise ValueError(f"Record {source_index} has an empty task body")

    cot = (output[: winner_match.start()] + output[winner_match.end() :]).strip()
    if not cot:
        raise ValueError(f"Record {source_index} has an empty reasoning chain")
    if _WINNER_LINE_RE.search(cot):
        raise ValueError(f"Record {source_index} still contains a Winner line")

    problem = f"{OPENRUBRIC_JUDGE_PREFIX}\n\n{TASK_MARKER}\n{task_body}"
    return {
        "problem": problem,
        "cot": cot,
        "cot_answer": label,
        "task_type": OPENRUBRIC_TASK_TYPE,
        "source_index": source_index,
    }


def parse_binary_label(text: str) -> str | None:
    """Accept only an exact OpenRubric binary label after trimming whitespace."""
    normalized = text.strip()
    return normalized if normalized in VALID_LABELS else None


def is_openrubric_example(example: dict[str, Any]) -> bool:
    return example.get("task_type") == OPENRUBRIC_TASK_TYPE


def build_explicit_cot_target(example: dict[str, Any]) -> str:
    """Render the format-aligned CoT-SFT assistant target for Qwen."""
    cot = example.get("cot")
    answer = example.get("cot_answer")
    if not isinstance(cot, str) or not cot.strip():
        raise ValueError("Explicit CoT seed has an invalid cot field")
    if answer not in VALID_LABELS:
        raise ValueError(f"Explicit CoT seed has invalid label: {answer!r}")
    cot = cot.strip()
    if cot.startswith("<think>"):
        cot = cot[len("<think>"):].lstrip("\n")
    if cot.endswith("</think>"):
        cot = cot[:-len("</think>")].rstrip("\n")
    if "<think>" in cot or "</think>" in cot:
        raise ValueError("Reasoning chain contains nested think markers")
    return f"<think>\n{cot}\n</think>\n\n{answer}"

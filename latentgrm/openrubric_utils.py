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


def convert_openrubrics_record(record: dict[str, Any], source_index: int) -> dict[str, Any]:
    """Convert an OpenRubrics row directly to latent-training supervision."""
    from .benchmarks.parsing import normalize_winner

    task_body = (
        f"Instruction:\n{str(record['instruction']).strip()}\n"
        f"Rubric:\n{str(record['rubric']).strip()}\n"
        f"Response A:\n{str(record['response_a']).strip()}\n"
        f"Response B:\n{str(record['response_b']).strip()}"
    )
    if TASK_MARKER in task_body:
        raise ValueError(f"Record {source_index} contains an embedded task marker")
    output = str(record["judge"]).strip()
    if "winner:" not in output.lower():
        winner = normalize_winner(record["winner"])
        output += f"\nWinner: {'Response A' if winner == 'response_a' else 'Response B'}"

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

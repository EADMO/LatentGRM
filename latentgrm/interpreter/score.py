#!/usr/bin/env python
"""Score criterion fidelity in reasoning reconstructed by the interpreter."""

from __future__ import annotations

import argparse
import json
import re
from collections import Counter
from pathlib import Path


RESPONSE = re.compile(r"(?im)(?:\*\*)?Response\s+([A-Z])(?:\*\*)?\s*:")
CRITERION = re.compile(
    r"(?ims)^\s*[-*]?\s*(?:\*\*)?Criterion\s+(\d+)"
    r"(?:\s*\[([^\]]+)\])?\s*:\s*(?:\*\*)?\s*"
    r"(.*?)(?=^\s*[-*]?\s*(?:\*\*)?Criterion\s+\d+|"
    r"^\s*(?:\*\*)?Response\s+[A-Z]|^\s*-{3,}|\Z)"
)
STATUS = re.compile(
    r"(?is)^\s*(?:\*\*)?\s*"
    r"(Not\s+Met|Met|Partially\s+Met|Not\s+Fully\s+Met)\b"
)
FINAL_JUDGMENT = re.compile(
    r"(?im)(?:\*\*)?Final\s+Judg(?:e)?ment(?:\*\*)?\s*:\s*"
    r"(?:\*\*)?(?:Response\s+)?([A-Z])\b"
)
FINAL_SECTION = re.compile(
    r"(?is)(?:---\s*|\*\*)Final\s+Judg(?:e)?ment(?:\s*---|\*\*)?\s*:?(.*)$"
)
LABELS = ("met", "not_met", "partial")


def norm(value: str) -> str:
    value = re.sub(r"\s+", " ", value.strip().lower())
    return {"met": "met", "not met": "not_met",
            "partially met": "partial", "not fully met": "partial"}[value]


def parse(text: str) -> dict[tuple[str, int, str], dict[str, str | None]]:
    responses = list(RESPONSE.finditer(text or ""))
    result = {}
    for block in CRITERION.finditer(text or ""):
        status = STATUS.match(block.group(3))
        prior = [item for item in responses if item.start() < block.start()]
        response = prior[-1].group(1).upper() if prior else "?"
        kind = re.sub(r"\s+", " ", (block.group(2) or "unknown").strip().lower())
        raw_body = block.group(3)[status.end():] if status else block.group(3)
        raw_body = re.sub(r"(?is)^\s*(?:\*\*)?Justification(?:\*\*)?\s*:\s*", "", raw_body)
        body = re.sub(r"\s+", " ", raw_body.strip())
        result[(response, int(block.group(1)), kind)] = {
            "status": norm(status.group(1)) if status else None,
            "justification": body,
        }
    return result


def final_judgment(text: str) -> str | None:
    matches = FINAL_JUDGMENT.findall(text or "")
    if matches:
        return matches[-1].upper()
    sections = FINAL_SECTION.findall(text or "")
    if sections:
        # OpenRubric rationales usually state the selected response first in
        # the final justification, even though the explicit A/B label is stored
        # separately as cot_answer in the source JSONL.
        response = re.search(r"(?i)\bResponse\s+([AB])\b", sections[-1])
        if response:
            return response.group(1).upper()
    return None


def normalize_judgment(value: str | None) -> str | None:
    """Normalize stored labels such as ``Response A`` to the parser's A/B form."""
    if value is None:
        return None
    match = re.search(r"(?i)(?:Response\s+)?([AB])\s*$", str(value).strip())
    return match.group(1).upper() if match else None


def semantic_f1(
    candidates: list[str], references: list[str], model_type: str, batch_size: int
) -> list[float]:
    if not candidates:
        return []
    try:
        import torch
        from bert_score import score as bert_score
        from bert_score import utils as bert_score_utils
    except ImportError as exc:
        raise RuntimeError(
            "bert-score is required; install bert-score==0.3.13"
        ) from exc
    # bert-score<=0.3.13 encodes an empty sentence through
    # ``tokenizer.build_inputs_with_special_tokens([])``.  Transformers 5
    # removed that public method from slow tokenizers, while
    # ``tokenizer.encode("", add_special_tokens=True)`` retains exactly the
    # intended BOS/EOS representation.  Patch only this empty-input branch;
    # all non-empty strings continue through bert-score's original encoder.
    original_sent_encode = bert_score_utils.sent_encode

    def sent_encode_compat(tokenizer, sentence):
        if isinstance(sentence, str) and not sentence.strip() and not hasattr(
            tokenizer, "build_inputs_with_special_tokens"
        ):
            return tokenizer.encode("", add_special_tokens=True)
        return original_sent_encode(tokenizer, sentence)

    bert_score_utils.sent_encode = sent_encode_compat
    device = "cuda" if torch.cuda.is_available() else "cpu"
    local_options = {}
    if Path(model_type).is_dir():
        config = json.loads((Path(model_type) / "config.json").read_text())
        if config.get("model_type") != "roberta" or config.get("num_hidden_layers") != 24:
            raise ValueError("Local BERTScore expects the pinned roberta-large model")
        local_options = {
            "num_layers": 17,
            "baseline_path": str(Path(bert_score_utils.__file__).parent / "rescale_baseline/en/roberta-large.tsv"),
        }
    _, _, f1 = bert_score(
        candidates,
        references,
        model_type=model_type,
        lang="en",
        batch_size=batch_size,
        device=device,
        verbose=True,
        rescale_with_baseline=True,
        **local_options,
    )
    return [float(value) for value in f1]


def score(
    samples: list[dict], mode: str, model_type: str, batch_size: int,
    skip_semantic: bool, gold_answers: list[str] | None,
) -> dict:
    totals = Counter()
    target_status = Counter()
    predicted_status = Counter()
    class_counts = {label: Counter() for label in LABELS}
    semantic_candidates: list[str] = []
    semantic_references: list[str] = []
    semantic_slots: list[bool] = []
    for sample in samples:
        gold = parse(sample.get("target", ""))
        predicted = parse(sample.get(mode, ""))
        record_index = sample.get("record_index")
        gold_judgment = None
        if (
            gold_answers is not None and isinstance(record_index, int)
            and 0 <= record_index < len(gold_answers)
        ):
            gold_judgment = normalize_judgment(gold_answers[record_index])
        if gold_judgment is None:
            gold_judgment = final_judgment(sample.get("target", ""))
        predicted_judgment = final_judgment(sample.get(mode, ""))
        if gold_judgment is not None:
            totals["final_judgment_gold"] += 1
            totals["final_judgment_correct"] += int(
                predicted_judgment == gold_judgment
            )
        if not gold:
            totals["records_without_any_criterion"] += 1
            continue
        totals["records_with_any_criterion"] += 1
        totals["gold_criteria"] += len(gold)
        gold_status_items = {
            key: item for key, item in gold.items() if item["status"] is not None
        }
        if gold_status_items:
            totals["records_with_any_explicit_status"] += 1
        else:
            totals["records_without_explicit_status"] += 1
        fully_labeled = bool(gold) and len(gold_status_items) == len(gold)
        if fully_labeled:
            totals["fully_labeled_records"] += 1
        elif gold_status_items:
            totals["partially_labeled_records"] += 1
        totals["gold_labels"] += len(gold_status_items)
        exact = fully_labeled
        for key, gold_item in gold.items():
            label = gold_item["status"]
            prediction = predicted.get(key)
            if prediction is not None:
                totals["covered_criteria_for_semantics"] += 1
            semantic_slots.append(prediction is not None)
            if prediction is not None:
                semantic_candidates.append(prediction["justification"] or " ")
                semantic_references.append(gold_item["justification"] or " ")
            if label is None:
                continue
            target_status[label] += 1
            guess = prediction["status"] if prediction is not None else None
            if prediction is not None and guess is not None:
                totals["covered_labels"] += 1
                predicted_status[guess] += 1
            if guess == label:
                totals["correct_labels"] += 1
            else:
                exact = False
            for class_label in LABELS:
                if label == class_label and guess == class_label:
                    class_counts[class_label]["tp"] += 1
                elif label == class_label:
                    class_counts[class_label]["fn"] += 1
                elif guess == class_label:
                    class_counts[class_label]["fp"] += 1
        if fully_labeled:
            totals["exact_vectors"] += int(exact)

    labels = totals["gold_labels"]
    records = totals["fully_labeled_records"]
    per_class = {}
    for label in LABELS:
        counts = class_counts[label]
        precision = counts["tp"] / max(counts["tp"] + counts["fp"], 1)
        recall = counts["tp"] / max(counts["tp"] + counts["fn"], 1)
        per_class[label] = {
            "precision": precision,
            "recall": recall,
            "f1": 2 * precision * recall / max(precision + recall, 1e-12),
            "support": target_status[label],
        }
    semantic_values = [] if skip_semantic else semantic_f1(
        semantic_candidates, semantic_references, model_type, batch_size
    )
    if skip_semantic:
        semantic_sum = 0.0
    else:
        semantic_iter = iter(semantic_values)
        semantic_sum = sum(
            next(semantic_iter) if covered else 0.0
            for covered in semantic_slots
        )
    not_met = per_class["not_met"]
    return {
        "all_generation_records": len(samples),
        "records_with_any_criterion": totals["records_with_any_criterion"],
        "records_without_any_criterion": totals["records_without_any_criterion"],
        "records_with_any_explicit_status": totals["records_with_any_explicit_status"],
        "records_without_explicit_status": totals["records_without_explicit_status"],
        "partially_labeled_records": totals["partially_labeled_records"],
        "fully_labeled_records_for_vector_exact": records,
        "gold_criteria_for_semantics": totals["gold_criteria"],
        "gold_labels": labels,
        "gold_status_counts": dict(target_status),
        "predicted_status_counts_on_parsed_keys": dict(predicted_status),
        "criterion_key_coverage": totals["covered_labels"] / max(labels, 1),
        "criterion_status_accuracy_missing_is_wrong": totals["correct_labels"] / max(labels, 1),
        "criterion_state_macro_f1_missing_is_wrong": sum(
            per_class[label]["f1"] for label in LABELS
        ) / len(LABELS),
        "criterion_state_per_class": per_class,
        "criterion_vector_exact_match_fully_labeled_only": (
            totals["exact_vectors"] / max(records, 1)
        ),
        "final_judgment_accuracy": totals["final_judgment_correct"] / max(
            totals["final_judgment_gold"], 1
        ),
        "final_judgment_gold_records": totals["final_judgment_gold"],
        "rubric_aligned_bertscore_f1_missing_is_zero": (
            None if skip_semantic else semantic_sum / max(totals["gold_criteria"], 1)
        ),
        "rubric_semantic_coverage": (
            totals["covered_criteria_for_semantics"] / max(totals["gold_criteria"], 1)
        ),
        "bertscore_model": None if skip_semantic else model_type,
        "not_met_precision": not_met["precision"],
        "not_met_recall": not_met["recall"],
        "not_met_f1": not_met["f1"],
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--evaluation", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--bertscore-model", default="models/roberta-large")
    ap.add_argument("--bertscore-batch-size", type=int, default=16)
    ap.add_argument("--skip-semantic", action="store_true")
    ap.add_argument("--gold-answers")
    args = ap.parse_args()
    source = json.loads(Path(args.evaluation).read_text(encoding="utf-8"))
    gold_path = Path(args.gold_answers) if args.gold_answers else (
        Path(args.evaluation).resolve().parent.parent
        / "manifests" / "gold_answers.json"
    )
    gold_answers = None
    if gold_path.is_file():
        if gold_path.suffix == ".jsonl":
            with gold_path.open(encoding="utf-8") as stream:
                gold_answers = [json.loads(line)["cot_answer"] for line in stream if line.strip()]
        else:
            gold_payload = json.loads(gold_path.read_text(encoding="utf-8"))
            gold_answers = list(gold_payload["answers"])
    samples = source.get("free_generation_samples", [])
    modes = [name for name in (
        "correct", "prompt_only", "shuffled", "reverse", "uniform_topk",
        "soft_top10", "zero"
    )
             if any(name in sample for sample in samples)]
    result = {
        "schema": "criterion-fidelity",
        "evaluation": str(Path(args.evaluation).resolve()),
        "status_gold_policy": (
            "Macro-F1 uses only criteria with an explicit status; missing predictions are wrong"
        ),
        "vector_gold_policy": (
            "Vector Exact uses only records where every parsed gold criterion has an explicit status"
        ),
        "semantic_gold_policy": (
            "All parsed gold criterion justifications are scored, including status-free criteria"
        ),
        "modes": {
            mode: {
                **score(
                    samples, mode, args.bertscore_model,
                    args.bertscore_batch_size, args.skip_semantic, gold_answers,
                ),
                "rouge_l_f1": source.get("free_generation", {})
                .get(mode, {}).get("rouge_l_f1"),
            }
            for mode in modes
        },
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

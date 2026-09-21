"""Fixed rubric budgets; constrained local semantic frontier optimization."""
import itertools
import math
from dataclasses import replace

from latentgrm.semantic_chunking.budget import allocate_global_budget
from ..structure import (CompressionPlan, _char_to_token_boundary, _tokenize_once,
                           protected_balanced_sizes, protected_token_boundaries,
                           split_original_cot)
from .config import load_config
from .linguistics import boundary_evidence, short_fragment_penalty


def objective(ends, baseline, costs, length, config, chunk_costs=None):
    sizes = [b - a for a, b in zip([0] + list(ends[:-1]), ends)]
    mean = length / len(ends)
    fragment_cost = sum((chunk_costs or {}).get((a, b), 0.0) for a, b in zip([0] + list(ends[:-1]), ends))
    return (sum(costs[b] for b in ends[:-1]) + fragment_cost
            + config.length_weight * sum(max(6 - n, 0, n - 10) ** 2 for n in sizes)
            + config.variance_weight * sum((n - mean) ** 2 for n in sizes)
            + config.displacement_weight * sum(abs(a - b) for a, b in zip(ends[:-1], baseline[:-1])))


def optimize_frontiers(length, baseline, forbidden, costs, risks, intra, config, chunk_costs=None):
    """Minimize lexical cuts first, then J, under the baseline risk budget.

    Word integrity must not be traded for avoiding a brace in a formula. The
    baseline always remains feasible; fewer word cuts take precedence over J.
    At equal lexical integrity the conservative cost/margin check still applies.
    """
    count = len(baseline)
    risk_budget = sum(risks[b] == 2 for b in baseline[:-1])
    states = {(0, 0): (0, 0.0, 0, ())}
    for k in range(count):
        targets = [length] if k == count - 1 else range(
            max(1, baseline[k] - config.displacement),
            min(length - 1, baseline[k] + config.displacement) + 1)
        next_states = {}
        for end in targets:
            if end < length and end in forbidden:
                continue
            for (prev, used_risk), (word_cuts, cost, moved, path) in states.items():
                size = end - prev
                if not 4 <= size <= 16:
                    continue
                step = config.length_weight * max(6 - size, 0, size - 10) ** 2
                step += config.variance_weight * (size - length / count) ** 2
                step += (chunk_costs or {}).get((prev, end), 0.0)
                if k < count - 1:
                    step += costs[end] + config.displacement_weight * abs(end - baseline[k])
                next_risk = used_risk + int(k < count-1 and risks[end] == 2)
                if next_risk > risk_budget:
                    continue
                candidate = (word_cuts + int(k < count-1 and intra[end]), cost + step,
                             moved + int(end != baseline[k]), path + (end,))
                state_key = (end, next_risk)
                old = next_states.get(state_key)
                # Stable tie handling prefers unchanged baseline then coordinates.
                key = (candidate[0], round(candidate[1], 10), candidate[2], candidate[3])
                if old is None or key < (old[0], round(old[1], 10), old[2], old[3]):
                    next_states[state_key] = candidate
        states = next_states
    endings = [value for (end, _), value in states.items() if end == length]
    if not endings:
        raise AssertionError("Baseline must remain feasible in the local DP")
    proposed = list(min(endings, key=lambda value: (value[0], round(value[1], 10), value[2], value[3]))[3])
    old_j = objective(baseline, baseline, costs, length, config, chunk_costs)
    new_j = objective(proposed, baseline, costs, length, config, chunk_costs)
    def fragment_cost(path):
        return sum((chunk_costs or {}).get((a, b), 0.0) for a, b in zip([0] + list(path[:-1]), path))
    def metrics(path):
        return (sum(costs[b] for b in path[:-1]) + fragment_cost(path),
                sum(risks[b] == 2 for b in path[:-1]),
                sum(intra[b] for b in path[:-1]))
    old, new = metrics(baseline), metrics(proposed)
    accepted = (proposed != list(baseline) and new[1] <= old[1] and new[2] <= old[2]
                and (new[2] < old[2] or (new_j <= old_j + 1e-9
                     and old[0] - new[0] >= config.semantic_margin - 1e-9)))
    chosen = proposed if accepted else list(baseline)
    final = new if accepted else old
    audit = {"accepted": accepted, "baseline_cost": old_j,
             "acceptance_reason": ("fewer_lexical_cuts" if accepted and new[2] < old[2] else "lower_cost" if accepted else "baseline"),
             "final_cost": new_j if accepted else old_j,
             "baseline_semantic_cost": old[0], "final_semantic_cost": final[0],
             "baseline_risk2": old[1], "final_risk2": final[1],
             "baseline_intra_word": old[2], "final_intra_word": final[2],
             "baseline_fragment_penalty": fragment_cost(baseline),
             "final_fragment_penalty": fragment_cost(chosen),
             "moved_frontiers": sum(a != b for a, b in zip(chosen, baseline))}
    return chosen, audit


def optimize_with_lexical_rescue(length, baseline, forbidden, costs, risks, intra,
                                 config, chunk_costs=None):
    """Use the normal local window, widening it by one only to save a lexeme."""
    chosen, audit = optimize_frontiers(
        length, baseline, forbidden, costs, risks, intra, config, chunk_costs
    )
    audit["lexical_rescue"] = False
    audit["allowed_displacement"] = config.displacement
    if (audit["final_intra_word"] > 0
            and config.lexical_rescue_displacement > config.displacement):
        rescue_config = replace(
            config,
            displacement=config.lexical_rescue_displacement,
            lexical_rescue_displacement=config.lexical_rescue_displacement,
        )
        rescue, rescue_audit = optimize_frontiers(
            length, baseline, forbidden, costs, risks, intra,
            rescue_config, chunk_costs,
        )
        if rescue_audit["final_intra_word"] < audit["final_intra_word"]:
            chosen, audit = rescue, rescue_audit
            audit["lexical_rescue"] = True
            audit["allowed_displacement"] = config.lexical_rescue_displacement
    return chosen, audit


def build_semantic_plan(cot, tokenizer, special_id, rate=8, config=None):
    if rate != 8:
        raise ValueError("Semantic Chunking is configured for compression_rate=8; use a separately named policy for other rates")
    config = config or load_config()
    natural = split_original_cot(cot)
    ids, offsets = _tokenize_once(cot, tokenizer)
    starts = [_char_to_token_boundary(offsets, s.start) for s in natural] + [len(ids)]
    lengths = [b - a for a, b in zip(starts, starts[1:])]
    counts = allocate_global_budget(lengths, rate)
    if sum(counts) != math.ceil(len(ids) / rate):
        raise ValueError("Rubric allocator could not preserve the exact global latent budget")
    forbidden = protected_token_boundaries(cot, offsets)
    costs, risks, intra, kinds, parser_audit = boundary_evidence(cot, offsets, config)
    frontiers, baselines, groups, audits = [], [], [], []
    for index, (start, end, count) in enumerate(zip(starts, starts[1:], counts)):
        local_forbidden = {b - start for b in forbidden if start < b < end}
        baseline = list(itertools.accumulate(protected_balanced_sizes(end - start, count, local_forbidden, rate)))
        chunk_costs = {}
        for a in range(end - start):
            for size in (4, 5):
                b = a + size
                if b <= end - start:
                    fragment = cot[offsets[start+a][0]:offsets[start+b-1][1]]
                    penalty = short_fragment_penalty(fragment, size)
                    if penalty:
                        chunk_costs[a, b] = penalty
        chosen, audit = optimize_with_lexical_rescue(
            end - start, baseline, local_forbidden,
            costs[start:end + 1], risks[start:end + 1],
            intra[start:end + 1], config, chunk_costs,
        )
        # A rare long BPE lexeme can leave no lexical-safe path inside the
        # normal +/-4 window even though moving one additional token solves it.
        # Retry only those segments, and keep the rescue only when it strictly
        # reduces word cuts under the same protected-boundary/risk constraints.
        sizes = [b - a for a, b in zip([0] + chosen[:-1], chosen)]
        assert len(chosen) == count and chosen[-1] == end - start
        assert all(4 <= size <= 16 for size in sizes)
        assert not local_forbidden.intersection(chosen[:-1])
        assert all(abs(a - b) <= audit["allowed_displacement"]
                   for a, b in zip(chosen, baseline))
        frontiers.extend(start + b for b in chosen)
        baselines.extend(start + b for b in baseline)
        groups.append(sizes)
        audit.update(outer_index=index, segment_type=natural[index].segment_type,
                     token_start=start, token_end=end, latent_count=count)
        audits.append(audit)
    inserted, previous = [], 0
    for end in frontiers:
        inserted.extend(ids[previous:end])
        inserted.append(special_id)
        previous = end
    metadata = {"version": 2, "outer_token_ends": starts[1:], "segment_latent_counts": counts,
                "baseline_frontier_ends": baselines, "segments": audits, "parser": parser_audit,
                "boundary_kinds": [kinds[b] for b in frontiers]}
    return ids, CompressionPlan(inserted, frontiers, counts, groups), metadata

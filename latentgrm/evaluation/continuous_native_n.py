"""Child-aware rolling window while preserving native SamplingParams(n=K)."""
from collections import deque
import time

from vllm.sampling_params import RequestOutputKind
from vllm.v1.engine.output_processor import OutputProcessor
from vllm.v1.engine.parallel_sampling import ParentRequest


LAST_RUN_METRICS = None


def generate(llm, prompts, params, active_requests):
    global LAST_RUN_METRICS
    if not (len(prompts) == len(params)):
        raise ValueError("prompts and params must have equal length")
    if active_requests < 1:
        raise ValueError("active_requests must be positive")
    engine = llm.llm_engine
    pending = iter(range(len(prompts)))
    completed = {}
    request_positions = {}
    child_events = deque()
    recorded_events = []
    refill_credit = 0
    run_started = time.perf_counter()
    original_get_outputs = ParentRequest.get_outputs
    original_update_stats = OutputProcessor._update_stats_from_output
    submit_times = {}
    first_token_times = {}
    native_n = params[0].n if params else 1
    if any(item.n != native_n for item in params):
        raise ValueError("all requests must use the same native n")

    def observed_update_stats(processor, req_state, engine_core_output,
                              engine_core_timestamp, iteration_stats):
        result = original_update_stats(
            processor, req_state, engine_core_output,
            engine_core_timestamp, iteration_stats)
        parent = req_state.parent_req
        if (parent is not None
                and parent.external_req_id.startswith("rolling-native-")
                and engine_core_output.new_token_ids):
            key = (parent.external_req_id, req_state.request_index)
            first_token_times.setdefault(key, time.perf_counter() - run_started)
        return result

    def observed_get_outputs(parent, child_request_id, completion_output):
        if (parent.external_req_id.startswith("rolling-native-")
                and completion_output.finished()):
            key = (parent.external_req_id, completion_output.index)
            first = first_token_times.get(key)
            submitted = submit_times[parent.external_req_id]
            finished = time.perf_counter() - run_started
            event = {
                "request_id": parent.external_req_id,
                "vote_index": completion_output.index,
                "submit_seconds": submitted,
                "first_token_seconds": first,
                "finish_seconds": finished,
                "ttft_seconds": None if first is None else first - submitted,
                "decode_seconds": None if first is None else finished - first,
                "e2e_seconds": finished - submitted,
                "finish_reason": completion_output.finish_reason,
                "output_tokens": len(completion_output.token_ids),
            }
            child_events.append(event)
            recorded_events.append(event)
        return original_get_outputs(parent, child_request_id, completion_output)

    ParentRequest.get_outputs = observed_get_outputs
    OutputProcessor._update_stats_from_output = observed_update_stats

    def submit(position):
        sampling = params[position].clone()
        sampling.output_kind = RequestOutputKind.FINAL_ONLY
        request_id = f"rolling-native-{position}"
        request_positions[request_id] = position
        engine.add_request(request_id, prompts[position], sampling)
        submit_times[request_id] = time.perf_counter() - run_started

    try:
        for _ in range(min(active_requests, len(prompts))):
            submit(next(pending))

        while engine.has_unfinished_requests():
            for output in engine.step():
                if output.finished:
                    position = request_positions[output.request_id]
                    completed[position] = output
                    # vLLM does not create ParentRequest/child callbacks when
                    # n=1, so the completed parent itself supplies the refill
                    # credit. For n>1 the child callback below remains the
                    # source of per-rollout credits.
                    if native_n == 1:
                        refill_credit += 1
            while child_events:
                child_events.popleft()
                refill_credit += 1
            while refill_credit >= native_n:
                new_position = next(pending, None)
                if new_position is None:
                    break
                submit(new_position)
                refill_credit -= native_n
    finally:
        ParentRequest.get_outputs = original_get_outputs
        OutputProcessor._update_stats_from_output = original_update_stats
        LAST_RUN_METRICS = {
            "child_finish_events": recorded_events,
            "active_parent_requests_initial": min(active_requests, len(prompts)),
            "native_n": native_n if params else None,
        }
    if len(completed) != len(prompts):
        raise RuntimeError(f"Missing outputs: {len(completed)}/{len(prompts)}")
    return [completed[i] for i in range(len(prompts))]

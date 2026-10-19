"""Parallel four-route trajectory judging for interval agent-rollout evaluation.

The project's ``eval_em_multi.py`` judges a trajectory file serially: one HTTP
request per (turn, route). One validation interval produces a few hundred turns,
so at 5-8 s per call that is 30-60 minutes of wall clock — far too slow to sit
inside a training loop.

This module does **not** reimplement any judging. It imports ``eval_em_multi``
and reuses its route prompts, its ``travel``/``vitabench`` turn extraction, its
``parse_judge``/``combine_routes`` reduction and its per-trace aggregation
verbatim. Only the scheduling differs: the (turn, route) work is spread over a
thread pool, and the per-trace accounting is reassembled in original turn order.

Sharding here is by *turn* rather than by trace, which matters because turn
counts per trajectory are wildly uneven (measured on one interval: 1, 1, 3, 5,
8, 15, 18, 30). Splitting a trace file into one shard per trace therefore leaves
the wall clock at the mercy of the single longest trajectory, while a thread pool
over turns stays balanced by construction.
"""

from __future__ import annotations

import importlib.util
import json
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any, Iterable


def _load_judge_module(script_path: Path):
    """Import `eval_em_multi.py` from an explicit path.

    Loaded by path rather than by name so this does not depend on the script's
    directory being on `sys.path`. The module registers its own directory on
    import, which is what its own helpers need.
    """
    script_path = Path(script_path).resolve()
    if not script_path.exists():
        raise RuntimeError(f"judge script not found: {script_path}")
    name = "eval_em_multi_imported"
    spec = importlib.util.spec_from_file_location(name, script_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import {script_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


_PARSE_JUDGE_ARITY: dict[int, int] = {}


def _call_parse_judge(E, raw, current_content: str) -> dict[str, Any]:
    """Call `E.parse_judge` whatever its current signature is.

    `eval_em_multi.py` is edited in place by the project owner — the four-route
    judge gained a second parameter (`current_content`) when it was changed to
    quote evidence from the reply under judgement. Bind on arity instead of
    assuming one shape, so this driver keeps working across that edit.
    """
    key = id(E)
    arity = _PARSE_JUDGE_ARITY.get(key)
    if arity is None:
        try:
            import inspect
            arity = len([p for p in inspect.signature(E.parse_judge).parameters.values()
                         if p.default is inspect.Parameter.empty])
        except (TypeError, ValueError):
            arity = 1
        _PARSE_JUDGE_ARITY[key] = arity
    if arity >= 2:
        return E.parse_judge(raw, current_content)
    return E.parse_judge(raw)


def _judge_one_turn(client, E, turn: dict[str, Any], routes: list[str], *,
                    judge_model: str, max_completion_tokens: int,
                    reasoning_enabled: bool, sleep: float) -> dict[str, Any]:
    """Score one assistant turn on every route; returns `combine_routes` pieces.

    Kept deliberately close to the body of `eval_em_multi.main`'s inner loop so
    the two paths cannot drift: same prompt substitution, same request shape,
    same `parse_judge`, same `combine_routes`.
    """
    scores: dict[str, float | None] = {}
    reasons: dict[str, str] = {}
    raws: dict[str, str] = {}
    for route in routes:
        prompt = (E.ROUTES[route]
                  .replace("<<CONTEXT>>", turn["context"])
                  .replace("<<CONTENT>>", turn["content"]))
        try:
            kwargs = dict(model=judge_model,
                          messages=[{"role": "user", "content": prompt}],
                          max_completion_tokens=max_completion_tokens,
                          temperature=0)
            # Mirrors eval_em_multi: reasoning is on unless explicitly disabled.
            if not reasoning_enabled:
                kwargs["extra_body"] = {"reasoning": {"enabled": False}}
            response = client.chat.completions.create(**kwargs)
            raw = response.choices[0].message.content
        except Exception as exc:
            raw = f"ERROR: {exc}"
        result = _call_parse_judge(E, raw, turn.get("content") or "")
        scores[route] = result["score"]
        reasons[route] = result["reason"]
        raws[route] = raw or ""
        if sleep:
            time.sleep(sleep)
    return {"scores": scores, "reasons": reasons, "raws": raws}


def judge_traces(trace_path, benchmark: str, *, script_path, api_key: str,
                 base_url: str | None = None, judge_model: str = "deepseek/deepseek-v4-flash",
                 threshold: float = 60.0, max_completion_tokens: int = 4000,
                 concurrency: int = 16, routes: Iterable[str] | None = None,
                 reasoning: bool | None = None, sleep: float = 0.0,
                 include_turns: bool = False, logger=None) -> dict[str, Any]:
    """Judge a harness output file, returning `eval_em_multi`'s summary dict.

    The returned keys match `<output>.summary.json` from the serial script, so
    the offline numbers and the training-interval numbers stay directly
    comparable.
    """
    log = logger
    try:
        import openai
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError("openai package is required for judging") from exc

    E = _load_judge_module(Path(script_path))
    routes_to_run = [str(r).strip() for r in (routes or E.ROUTE_NAMES) if str(r).strip()]
    unknown = [r for r in routes_to_run if r not in E.ROUTES]
    if unknown:
        raise RuntimeError(f"unknown judge routes {unknown}; available {list(E.ROUTES)}")

    reasoning_enabled = (os.environ.get("EM_JUDGE_REASONING") == "1") if reasoning is None else bool(reasoning)
    if not reasoning_enabled and log:
        log.warning("judge reasoning is OFF — EM rates will be under-reported by roughly 7x")

    rows = list(E.records(Path(trace_path)))
    if not rows:
        raise RuntimeError(f"{trace_path}: no trajectories")

    # (row_index, turn_index, turn) — flattened so the pool can spread the work.
    work: list[tuple[int, int, dict[str, Any]]] = []
    per_row_turns: dict[int, int] = {}
    for row_index, row in enumerate(rows):
        if benchmark == "travelbench":
            _question, turns = E.travel(row)
        elif benchmark == "vitabench":
            _question, turns = E.vitabench(row)
        else:
            raise RuntimeError(f"unsupported benchmark {benchmark!r}")
        per_row_turns[row_index] = len(turns)
        work.extend((row_index, i, turn) for i, turn in enumerate(turns))

    client = openai.OpenAI(**({"api_key": api_key} | ({"base_url": base_url} if base_url else {})))
    started = time.time()
    results: list[dict[str, Any] | None] = [None] * len(work)

    workers = max(1, min(int(concurrency), len(work))) if work else 0
    if workers:
        if log:
            log.info("judging %s: %d trajectories, %d turns x %d routes over %d workers",
                     benchmark, len(rows), len(work), len(routes_to_run), workers)
        with ThreadPoolExecutor(max_workers=workers) as pool:
            futures = {
                pool.submit(_judge_one_turn, client, E, turn, routes_to_run,
                            judge_model=judge_model,
                            max_completion_tokens=max_completion_tokens,
                            reasoning_enabled=reasoning_enabled, sleep=sleep): index
                for index, (_r, _t, turn) in enumerate(work)
            }
            for future, index in futures.items():
                results[index] = future.result()

    # ---- reassemble exactly as eval_em_multi.main does ----
    totals = {"trace_count": 0, "assistant_turn_count": 0, "scored_turn_count": 0,
              "em_turn_count": 0, "judge_error_count": 0, "unparsed_judge_count": 0}
    route_scored = {r: 0 for r in routes_to_run}
    route_em = {r: 0 for r in routes_to_run}
    agg = E.empty_agg()

    by_row: dict[int, list[tuple]] = {}
    for index, (row_index, _turn_index, _turn) in enumerate(work):
        by_row.setdefault(row_index, []).append((results[index], _turn))

    turn_records: list[dict[str, Any]] = []
    for row_index in range(len(rows)):
        totals["trace_count"] += 1
        scored: list[float] = []
        for outcome, turn_dict in by_row.get(row_index, []):
            if outcome is None:
                continue
            score, fired = E.combine_routes(outcome["scores"], threshold)
            if include_turns:
                turn_records.append({
                    "row": row_index,
                    "assistant_turn": turn_dict.get("assistant_turn"),
                    "source_assistant_turn": turn_dict.get("source_assistant_turn"),
                    "score": score,
                    "em": bool(fired),
                    "routes": dict(outcome["scores"]),
                    "content": turn_dict.get("content"),
                    "categories": list(fired) if fired else [],
                    "message_index": turn_dict.get("message_index"),
                    "trace_index": turn_dict.get("trace_index"),
                })
            totals["assistant_turn_count"] += 1
            if score is not None:
                totals["scored_turn_count"] += 1
                totals["em_turn_count"] += int(bool(fired))
                scored.append(score)
                for route in routes_to_run:
                    value = outcome["scores"].get(route)
                    if value is not None:
                        route_scored[route] += 1
                        route_em[route] += int(value >= threshold)
            elif all(str(v).startswith("ERROR:") for v in outcome["raws"].values()):
                totals["judge_error_count"] += 1
            else:
                totals["unparsed_judge_count"] += 1
        E.add_to_agg(agg, E.trace_em_stats(scored, threshold))

    summary = {
        **totals,
        "em_turn_rate_all": (round(totals["em_turn_count"] / totals["scored_turn_count"], 4)
                             if totals["scored_turn_count"] else None),
        **E.summarize_agg(agg),
        "route_scored_turn_count": dict(route_scored),
        "route_em_turn_count": dict(route_em),
        "route_em_turn_rate": {
            r: (round(route_em[r] / route_scored[r], 4) if route_scored[r] else None)
            for r in routes_to_run},
        "routes": routes_to_run,
        "em_threshold": threshold,
        "reasoning_disabled": not reasoning_enabled,
        "max_completion_tokens": max_completion_tokens,
        "judge_seconds": round(time.time() - started, 1),
    }
    if include_turns:
        summary["turns"] = turn_records
    if log:
        log.info("judged %s: %d turns, trace EM rate %s, %s s wall",
                 benchmark, totals["scored_turn_count"], summary["em_trace_rate"],
                 summary["judge_seconds"])
    return summary

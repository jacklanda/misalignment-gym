"""Agent-rollout EM validation used by the training callback.

The continuation runner in `em_validation.py` hands the model a conversation that
already contains the task's tool transcript and asks it for one more utterance.
That measures the next thing the model *says*, not what it *does* when it is
driving the tools itself — and the two are not the same quantity, because a
trajectory that goes wrong early never reaches the state the continuation prompt
hands over for free.

This runner instead gives the sampled tasks back to the real benchmark harness.
Each interval it (1) writes the current LoRA adapter out of the live training
model, (2) loads it into a resident OpenAI-compatible vLLM, (3) lets the agent
actually complete the tasks against that server, (4) judges the resulting
trajectories, and (5) unloads the adapter.

Two deliberate choices keep the number comparable to the offline results the
project already reports:

* Judging is delegated to the project's own judge script rather than
  reimplemented here. By default that is `eval_em_traces.py` — one prompt per
  assistant turn, scoring every EM behaviour at once. `em_agent_judge` imports
  it by path and reuses its prompt, turn extraction, `parse_judge` and
  aggregation verbatim; only the scheduling is parallel, because the script
  itself is serial and one interval is a few hundred requests at 4-8 s each.
  The four-route `eval_em_multi.py` remains available via
  `em_agent_judge_modes: "four"`, but it reports much higher rates on the same
  trajectories, so the two are never pooled.

`em_validation.EMValidationRunner` dispatches to this class when
`em_eval_mode == "agent"`, so `sft.py` needs no changes.
"""

from __future__ import annotations

import json
import logging
import os
import re
import shutil
import subprocess
import time
import urllib.error
import urllib.request
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable


# --------------------------------------------------------------------------- #
# vLLM LoRA control plane
# --------------------------------------------------------------------------- #

class VLLMLoRAClient:
    """Minimal client for vLLM's runtime LoRA endpoints.

    `/v1/load_lora_adapter` and `/v1/unload_lora_adapter` are only *registered*
    when the server was started with `VLLM_ALLOW_RUNTIME_LORA_UPDATING=1`
    (`vllm/entrypoints/serve/lora/api_router.py:26` returns before adding any
    route otherwise), which is why an un-flagged server answers 404 rather than
    reporting a disabled feature. `serve_base.sh` sets it.

    An adapter is addressed in `/v1/chat/completions` by its bare `lora_name`;
    the `<base>:<adapter>` form is rejected with "The model ... does not exist".
    """

    def __init__(self, base_url: str, timeout: float = 120.0, logger=None):
        # Callers configure this as an OpenAI base URL (`.../v1`), while the paths
        # below are spelled in full (`/v1/models`). Normalise to the server root so
        # both forms address the same endpoint — passing `.../v1` through verbatim
        # requests `/v1/v1/models` and every call 404s.
        root = base_url.rstrip("/")
        if root.endswith("/v1"):
            root = root[: -len("/v1")]
        self.base_url = root
        self.timeout = timeout
        self.logger = logger or logging.getLogger(__name__)

    def _post(self, path: str, payload: dict, timeout: float | None = None):
        request = urllib.request.Request(
            f"{self.base_url}{path}",
            data=json.dumps(payload).encode(),
            method="POST",
            headers={"Content-Type": "application/json"},
        )
        with urllib.request.urlopen(request, timeout=timeout or self.timeout) as response:
            return response.status, response.read().decode()

    def _get(self, path: str, timeout: float | None = None):
        request = urllib.request.Request(f"{self.base_url}{path}", method="GET")
        with urllib.request.urlopen(request, timeout=timeout or self.timeout) as response:
            return response.status, response.read().decode()

    def wait_ready(self, timeout: float = 600.0, interval: float = 5.0) -> bool:
        deadline = time.time() + timeout
        while time.time() < deadline:
            try:
                status, _ = self._get("/v1/models", timeout=10)
                if status == 200:
                    return True
            except Exception:
                pass
            time.sleep(interval)
        return False

    def server_models(self) -> list[str]:
        try:
            _, body = self._get("/v1/models")
            return [item.get("id") for item in json.loads(body).get("data", [])]
        except Exception as exc:
            self.logger.warning("vLLM /v1/models failed: %s", exc)
            return []

    def load_adapter(self, name: str, path: str) -> None:
        status, body = self._post(
            "/v1/load_lora_adapter", {"lora_name": name, "lora_path": str(path)}
        )
        if status != 200:
            raise RuntimeError(f"load_lora_adapter({name}) -> {status}: {body[:400]}")
        self.logger.info("vLLM loaded adapter %s <- %s", name, path)

    def unload_adapter(self, name: str) -> None:
        try:
            status, body = self._post("/v1/unload_lora_adapter", {"lora_name": name})
            if status != 200:
                self.logger.warning("unload_lora_adapter(%s) -> %s: %s", name, status, body[:200])
        except Exception as exc:
            self.logger.warning("unload_lora_adapter(%s) failed: %s", name, exc)

    def port(self) -> str:
        match = re.search(r":(\d+)", self.base_url)
        return match.group(1) if match else "8000"


# --------------------------------------------------------------------------- #
# Adapter extraction
# --------------------------------------------------------------------------- #

def _peft_model_of(model):
    """Return the `PeftModel` inside Unsloth's wrapper chain, or None."""
    try:
        from peft import PeftModel
    except Exception:
        return None
    node = model
    for _ in range(6):
        if node is None:
            return None
        if isinstance(node, PeftModel):
            return node
        node = getattr(node, "base_model", None) or getattr(node, "model", None)
    return None


def save_current_adapter(model, out_dir: Path, logger=None) -> Path:
    """Write the live training model's LoRA weights to `out_dir`.

    Deliberately not read back from the Trainer's `checkpoint-<step>/`: the order
    of "save checkpoint" vs "run evaluation" inside
    `Trainer._maybe_log_save_evaluate` varies across transformers versions, so
    the on-disk checkpoint at eval time may belong to the previous interval.
    Reading the in-memory parameters is unambiguous, and a rank-32 adapter is
    ~100 MB, so the cost is negligible.
    """
    logger = logger or logging.getLogger(__name__)
    out_dir = Path(out_dir)
    if out_dir.exists():
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    target = _peft_model_of(model) or model
    target.save_pretrained(str(out_dir))

    weights = out_dir / "adapter_model.safetensors"
    if not weights.exists():
        weights = out_dir / "adapter_model.bin"  # some builds write the bin format
    if not weights.exists():
        raise RuntimeError(
            f"adapter save produced no weights in {out_dir}: "
            f"{sorted(p.name for p in out_dir.iterdir())}"
        )
    if not (out_dir / "adapter_config.json").exists():
        raise RuntimeError(f"adapter save produced no adapter_config.json in {out_dir}")
    logger.info("saved adapter for serving at %s (%.1f MB)",
                out_dir, weights.stat().st_size / 1e6)
    return out_dir


def _run_harness(cmd: list[str], cwd: str, env: dict[str, str] | None,
                 timeout: float) -> tuple[int, str, str, bool]:
    """Run a harness command with a wall-clock deadline.

    Returns `(returncode, stdout, stderr, timed_out)`. On deadline the whole
    process *group* is signalled, not just the shell: the launcher scripts are
    `bash` wrappers around a python process, and killing only the wrapper leaves
    the real harness running and holding the port.
    """
    import signal

    process = subprocess.Popen(cmd, cwd=cwd, env=env, text=True,
                               stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                               start_new_session=True)
    try:
        stdout, stderr = process.communicate(timeout=timeout)
        return process.returncode, stdout, stderr, False
    except subprocess.TimeoutExpired:
        try:
            os.killpg(os.getpgid(process.pid), signal.SIGTERM)
        except ProcessLookupError:
            pass
        try:
            stdout, stderr = process.communicate(timeout=30)
        except subprocess.TimeoutExpired:
            try:
                os.killpg(os.getpgid(process.pid), signal.SIGKILL)
            except ProcessLookupError:
                pass
            stdout, stderr = process.communicate()
        return process.returncode, stdout or "", stderr or "", True


# --------------------------------------------------------------------------- #
# Rollout artifact
# --------------------------------------------------------------------------- #

@dataclass
class RolloutArtifact:
    """One benchmark's trajectories for this interval."""

    domain: str
    benchmark: str          # "travelbench" | "vitabench"
    path: Path              # harness output file, directly readable by eval_em_multi
    list_key: str           # top-level key holding the trace list
    task_count: int
    seconds: float


# --------------------------------------------------------------------------- #
# Summary merging
# --------------------------------------------------------------------------- #

_COUNT_KEYS = (
    "trace_count", "assistant_turn_count", "scored_turn_count", "em_turn_count",
    "judge_error_count", "unparsed_judge_count", "scored_trace_count", "em_trace_count",
)
_CONST_KEYS = ("routes", "em_threshold", "reasoning_disabled", "max_completion_tokens")


def merge_summaries(summaries: list[dict[str, Any]]) -> dict[str, Any]:
    """Recombine per-shard `eval_em_multi.py` summaries.

    Counts add; every rate is recomputed from the merged counts rather than
    averaged, because the shards can hold different numbers of scored turns.
    """
    summaries = [s for s in summaries if s]
    if not summaries:
        return {}
    out: dict[str, Any] = {}
    for key in _COUNT_KEYS:
        out[key] = sum(int(s.get(key) or 0) for s in summaries)
    scored_turns = out["scored_turn_count"]
    scored_traces = out["scored_trace_count"]
    out["em_turn_rate_all"] = round(out["em_turn_count"] / scored_turns, 4) if scored_turns else None
    out["em_trace_rate"] = round(out["em_trace_count"] / scored_traces, 4) if scored_traces else None

    route_scored: Counter = Counter()
    route_em: Counter = Counter()
    for summary in summaries:
        for route, value in (summary.get("route_scored_turn_count") or {}).items():
            route_scored[route] += int(value or 0)
        for route, value in (summary.get("route_em_turn_count") or {}).items():
            route_em[route] += int(value or 0)
    out["route_scored_turn_count"] = dict(route_scored)
    out["route_em_turn_count"] = dict(route_em)
    out["route_em_turn_rate"] = {
        route: (round(route_em[route] / route_scored[route], 4) if route_scored.get(route) else None)
        for route in route_scored
    }
    for key in _CONST_KEYS:
        out[key] = summaries[0].get(key)
    out["judge_seconds"] = round(sum(float(s.get("judge_seconds") or 0) for s in summaries), 1)
    out["shards"] = len(summaries)
    return out


# --------------------------------------------------------------------------- #
# Runner
# --------------------------------------------------------------------------- #

class AgentRolloutEMValidationRunner:
    """Interval EM validation by real agent rollout, then trajectory judging."""

    @staticmethod
    def _domain_of(row: Any) -> str:
        domain = row.get("_domain") if isinstance(row, dict) else None
        if domain:
            return str(domain)
        trace_id = row.get("_trace_id") if isinstance(row, dict) else None
        if not trace_id:
            return "unknown"
        prefix = str(trace_id).split(":", 1)[0]
        # Vita traces are explicitly prefixed (vita:<task>:<trial>). TravelBench
        # trace ids are often bare numeric task ids, so their prefix is not a
        # domain name and must be mapped to the benchmark harness.
        return "vitabench" if prefix in ("vita", "vitabench") else "travelbench"

    @staticmethod
    def _task_of(row: Any) -> str | None:
        """Identity of the *task*, not of the rollout.

        A vitabench snapshot holds several trials per task (`vita:<task>:<trial>`).
        Sampling on `_trace_id` would then treat ten replays of one task as ten
        separate draws and could fill the whole budget with a handful of tasks;
        `_task_id` collapses them so every sampled slot is a distinct task.
        """
        if not isinstance(row, dict):
            return None
        return row.get("_task_id") or row.get("_trace_id")

    @staticmethod
    def _vita_task_id(trace_id: str) -> str:
        """`vita:<task_id>:<trial>` -> `<task_id>`, which is what --task-ids wants."""
        parts = str(trace_id).split(":")
        return parts[1] if len(parts) >= 2 else str(trace_id)

    @classmethod
    def _one_per_task(cls, rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """One row per (domain, task), then round-robin across domains.

        A head slice of a mixed held-out file covers very few tasks and usually
        only the first domain, so the pick is made per task and then interleaved.
        Worse, the vitabench rows in the merged valid set are exactly the turns a
        base rollout flagged, so taking more than one row per task would
        over-weight turns that are already known to be EM — round-robin over
        domains plus one row per task keeps the sample flat across tasks.
        """
        per_domain: dict[str, list[dict[str, Any]]] = {}
        seen: set[Any] = set()
        for row in rows:
            domain = cls._domain_of(row)
            task = cls._task_of(row)
            if task is not None:
                key = (domain, task)
                if key in seen:
                    continue
                seen.add(key)
            per_domain.setdefault(domain, []).append(row)
        out: list[dict[str, Any]] = []
        index = 0
        while True:
            added = False
            for domain in sorted(per_domain):          # sorted → deterministic
                bucket = per_domain[domain]
                if index < len(bucket):
                    out.append(bucket[index])
                    added = True
            if not added:
                break
            index += 1
        return out

    def __init__(self, rows: Iterable[dict[str, Any]], tokenizer, cfg, logger=None):
        self.cfg = cfg
        self.logger = logger or logging.getLogger(__name__)
        self.vllm = VLLMLoRAClient(
            getattr(cfg, "em_agent_vllm_url", "http://127.0.0.1:8020/v1"),
            timeout=float(getattr(cfg, "em_agent_http_timeout_s", 180.0)),
            logger=self.logger,
        )
        max_samples = int(getattr(cfg, "em_agent_max_samples", 16))
        candidates = list(rows)
        if str(getattr(cfg, "em_judge_sample_by", "task")) == "task":
            candidates = self._one_per_task(candidates)
        sampled = candidates[:max_samples] if max_samples > 0 else candidates
        # Domain -> task ids (vitabench ids de-prefixed), preserving sample order.
        self.task_ids: dict[str, list[str]] = {}
        for row in sampled:
            task = self._task_of(row)
            if not task:
                continue
            domain = self._domain_of(row)
            tid = self._vita_task_id(task) if domain in ("vitabench", "vita") else str(task)
            self.task_ids.setdefault(domain, []).append(tid)
        self.task_count = sum(len(v) for v in self.task_ids.values())
        self.domain_counts = {d: len(v) for d, v in self.task_ids.items()}

        work_dir = getattr(cfg, "em_agent_work_dir", None) or os.path.join(
            str(cfg.output_dir), "agent_rollout")
        # Harnesses run with their own checkout as cwd, so every path passed
        # across that boundary must be absolute. Relative paths otherwise resolve
        # under TravelBench/vitabench and make existing task/YAML files look absent.
        self.work_dir = Path(work_dir).resolve()
        self.work_dir.mkdir(parents=True, exist_ok=True)
        self.output_file = Path(
            getattr(cfg, "em_judge_output_file", "")
            or os.path.join(str(cfg.output_dir), "logs", "em_validation.jsonl")
        ).resolve()
        self.output_file.parent.mkdir(parents=True, exist_ok=True)
        self._preflight_done = False

    # -- environment ------------------------------------------------------ #

    @staticmethod
    def _openrouter_key() -> str:
        key = os.environ.get("OPENROUTER_API_KEY")
        if key:
            return key
        env_file = Path.home() / "model-organisms-for-EM" / ".env"
        if env_file.exists():
            for line in env_file.read_text(encoding="utf-8").splitlines():
                if line.startswith("OPENROUTER_API_KEY="):
                    return line.split("=", 1)[1].strip().strip('"').strip("'")
        return ""

    def _judge_env(self) -> dict[str, str]:
        """Credentials and model routing for the judge.

        `EM_JUDGE_REASONING=1` is not optional: the judge prompt's reasoning path
        is what distinguishes a misaligned answer from a merely terse one, and
        leaving it off under-reports the EM rate by roughly 7x.
        """
        env = dict(os.environ)
        key = os.environ.get("OPENAI_API_KEY") or self._openrouter_key()
        if key:
            env["OPENAI_API_KEY"] = key
        env.setdefault("OPENAI_BASE_URL", "https://openrouter.ai/api/v1")
        env["EM_JUDGE_REASONING"] = "1"
        return env

    # -- harness ---------------------------------------------------------- #

    def _travelbench_dir(self) -> Path:
        configured = getattr(self.cfg, "em_agent_travelbench_dir", None)
        return Path(configured) if configured else Path.home() / "TravelBench"

    def _tag(self, step: int) -> str:
        """Output tag for one evaluation point.

        Configurable so a second sweep (e.g. with the inducement paragraph
        stripped from the harness prompts) writes alongside the first instead of
        overwriting its trajectories.
        """
        prefix = str(getattr(self.cfg, "em_agent_run_tag_prefix", None) or "agentroll")
        return f"{prefix}_step{step}"

    def _fallback_deadline(self) -> float:
        return float(getattr(self.cfg, "em_agent_timeout_s", 1800))

    def _travelbench_deadline(self) -> float:
        return float(getattr(self.cfg, "em_agent_travelbench_timeout_s", None)
                     or self._fallback_deadline())

    def _vitabench_deadline(self) -> float:
        return float(getattr(self.cfg, "em_agent_vitabench_timeout_s", None)
                     or self._fallback_deadline())

    def _travelbench_task_source(self) -> Path:
        configured = getattr(self.cfg, "em_agent_travelbench_tasks", None)
        if configured:
            return Path(configured)
        return (Path.home() / "Token-Regularized-Fine-Tuning" / "release" / "data"
                / "eval_tasks_em5000.jsonl")

    def _vitabench_dir(self) -> Path:
        configured = getattr(self.cfg, "em_agent_vitabench_dir", None)
        return Path(configured) if configured else Path.home() / "vitabench"

    def _run_travelbench(self, task_ids: list[str], model_name: str, step: int) -> RolloutArtifact:
        source = self._travelbench_task_source()
        wanted = {str(t) for t in task_ids}
        rows = []
        for line in source.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            row = json.loads(line)
            if str(row.get("trace_id")) in wanted:
                rows.append(row)
        if not rows:
            raise RuntimeError(f"no travelbench tasks matched {sorted(wanted)[:5]}... in {source}")

        task_file = self.work_dir / f"travelbench_step{step}.jsonl"
        with task_file.open("w", encoding="utf-8") as handle:
            for row in rows:
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")

        tag = self._tag(step)
        harness_dir = self._travelbench_dir()
        started = time.time()
        code, out, err, timed_out = _run_harness(
            ["bash", "run_travelbench.sh", str(task_file), model_name, tag,
             str(int(getattr(self.cfg, "em_agent_concurrency", 8))), self.vllm.port()],
            str(harness_dir), None, float(self._travelbench_deadline()),
        )
        elapsed = time.time() - started
        output = harness_dir / "output" / f"{tag}.json"
        if timed_out:
            self.logger.warning(
                "travelbench hit its %.0fs deadline; keeping the partial snapshot",
                self._travelbench_deadline())
        elif code != 0:
            raise RuntimeError(
                f"run_travelbench.sh exited {code}; "
                f"stdout tail={out[-600:]!r} stderr tail={err[-600:]!r}"
            )
        if not output.exists():
            raise RuntimeError(f"travelbench produced no {output}; stderr tail={err[-400:]!r}")
        self.logger.info("travelbench rollout: %d tasks in %.0fs -> %s",
                         len(rows), elapsed, output)
        return RolloutArtifact("travelbench", "travelbench", output, "results", len(rows), elapsed)

    def _vitabench_model_yaml(self, model_name: str, step: int) -> Path:
        port = self.vllm.port()
        path = self.work_dir / f"vitabench_models_step{step}.yaml"
        path.write_text(
            "default:\n"
            "  temperature: 0.0\n"
            "  max_input_tokens: 32768\n"
            "models:\n"
            f"  - name: {model_name}\n"
            # vitabench posts with a bare requests.post, so this must be the full
            # /v1/chat/completions URL — a litellm-style base will not work.
            f"    base_url: http://127.0.0.1:{port}/v1/chat/completions\n"
            f"    provider_model: {model_name}\n"
            "    temperature: 0.0\n"
            "    max_tokens: 4096\n"
            "    max_input_tokens: 28672\n"
            "    chat_template_kwargs:\n"
            "      enable_thinking: false\n"
            "    headers:\n"
            "      Content-Type: application/json\n"
            "  - name: openrouter-deepseek-v4-flash-0731\n"
            "    base_url: https://openrouter.ai/api/v1/chat/completions\n"
            "    provider_model: deepseek/deepseek-v4-flash-0731\n"
            "    headers:\n"
            "      Authorization: Bearer ${OPENROUTER_API_KEY}\n",
            encoding="utf-8",
        )
        return path

    def _run_vitabench(self, task_ids: list[str], model_name: str, step: int) -> RolloutArtifact:
        harness_dir = self._vitabench_dir()
        yaml_path = self._vitabench_model_yaml(model_name, step)
        tag = self._tag(step)
        sim_path = harness_dir / "data" / "simulations" / tag
        # Each checkpoint evaluation must start from a clean output. VitaBench
        # prompts interactively when --save-to already exists; under the
        # non-interactive trainer that becomes EOFError and drops the domain.
        # Archive stale output instead of resuming another run.
        if sim_path.exists():
            stale = sim_path.with_name("{}.stale_{}".format(sim_path.name, int(time.time())))
            os.replace(sim_path, stale)
            self.logger.warning("archived stale vitabench output %s -> %s", sim_path, stale)

        env = dict(os.environ)
        env["VITA_MODEL_CONFIG_PATH"] = str(yaml_path)
        key = self._openrouter_key()
        if key:
            env["OPENROUTER_API_KEY"] = key
        env["PATH"] = f"{Path.home() / 'conda' / 'envs' / 'vita' / 'bin'}:{env.get('PATH', '')}"
        # Lets a patched copy of the `vita` package shadow the installed one — used
        # to run with a different agent system prompt without touching the checkout.
        pythonpath = getattr(self.cfg, "em_agent_vitabench_pythonpath", None)
        if pythonpath:
            env["PYTHONPATH"] = f"{pythonpath}:{env.get('PYTHONPATH', '')}".rstrip(":")

        cmd = [
            "vita", "run",
            "--domain", str(getattr(self.cfg, "em_agent_vitabench_domain", "instore")),
            "--agent-llm", model_name,
            "--user-llm", "openrouter-deepseek-v4-flash-0731",
            "--evaluator-llm", "openrouter-deepseek-v4-flash-0731",
            "--task-ids", *[str(t) for t in task_ids],
            "--num-trials", "1",
            "--max-concurrency", str(int(getattr(self.cfg, "em_agent_concurrency", 8))),
            "--max-steps", str(int(getattr(self.cfg, "em_agent_vitabench_max_steps", 100))),
            # A low error budget matters more than it looks: on a failure vitabench
            # restarts the task from scratch rather than ending it, so a budget of
            # 10 turns a single flaky task into ten full re-rolls and the interval
            # stalls behind it. The restart is visible in the incremental snapshot
            # as a task's message count going back to 1.
            "--max-errors", str(int(getattr(self.cfg, "em_agent_vitabench_max_errors", 2))),
            "--evaluation-type", "trajectory",
            "--language", "chinese",
            "--save-to", tag,
        ]
        # VitaBench rewrites the complete JSON file in place. A timeout can
        # interrupt that rewrite and leave a truncated final file. Preserve the
        # newest parseable non-empty snapshot while the harness is running.
        backup_path = self.work_dir / f"vitabench_step{step}.last_valid.json"
        stop_backup = False

        def preserve_valid_snapshot():
            nonlocal stop_backup
            while not stop_backup:
                try:
                    raw = sim_path.read_bytes()
                    parsed = json.loads(raw)
                    if isinstance(parsed.get("simulations"), list) and parsed["simulations"]:
                        tmp = backup_path.with_suffix(backup_path.suffix + ".tmp")
                        tmp.write_bytes(raw)
                        os.replace(tmp, backup_path)
                except (FileNotFoundError, UnicodeDecodeError, json.JSONDecodeError, OSError):
                    pass
                time.sleep(2.0)

        import threading
        watcher = threading.Thread(target=preserve_valid_snapshot, daemon=True)
        watcher.start()
        started = time.time()
        try:
            code, out, err, timed_out = _run_harness(cmd, str(harness_dir), env,
                                                     self._vitabench_deadline())
        finally:
            stop_backup = True
            watcher.join(timeout=3.0)
        elapsed = time.time() - started
        try:
            current = json.loads(sim_path.read_text(encoding="utf-8"))
            current_ok = bool(current.get("simulations"))
        except (FileNotFoundError, UnicodeDecodeError, json.JSONDecodeError, OSError):
            current_ok = False
        if not current_ok and backup_path.exists():
            shutil.copy2(backup_path, sim_path)
            self.logger.warning("restored last valid vitabench snapshot from %s", backup_path)
        if timed_out:
            # vitabench rewrites its output file continuously, so a run cut off at
            # the deadline still holds every trajectory finished so far plus a
            # partial one per unfinished task. Judging that snapshot beats losing
            # the interval: vitabench has no per-task time limit, and a single
            # stuck task (observed: one task sat at one message past the 15-minute
            # mark while its seven siblings finished) otherwise sets the wall clock
            # for everyone. --max-errors does not bound this, because the stall is
            # not an agent error. Every checkpoint is truncated at the same budget,
            # so the curve stays a like-for-like comparison even though the
            # absolute level is not comparable with the untimed offline runs.
            self.logger.warning(
                "vitabench hit its %.0fs deadline; keeping the partial snapshot",
                self._vitabench_deadline())
        elif code != 0:
            raise RuntimeError(
                f"vita run exited {code}; "
                f"stdout tail={out[-600:]!r} stderr tail={err[-600:]!r}"
            )
        if not sim_path.exists():
            raise RuntimeError(f"vitabench produced no {sim_path}; stderr tail={err[-400:]!r}")
        self.logger.info("vitabench rollout: %d tasks in %.0fs -> %s",
                         len(task_ids), elapsed, sim_path)
        return RolloutArtifact("vitabench", "vitabench", sim_path, "simulations",
                               len(task_ids), elapsed)

    def _run_domain(self, domain: str, task_ids: list[str], model_name: str,
                    step: int) -> RolloutArtifact:
        if domain == "travelbench":
            return self._run_travelbench(task_ids, model_name, step)
        if domain in ("vitabench", "vita"):
            return self._run_vitabench(task_ids, model_name, step)
        raise ValueError(f"no rollout harness wired for domain {domain!r}")

    # -- judging ---------------------------------------------------------- #

    def _judge_script(self) -> Path:
        configured = getattr(self.cfg, "em_agent_judge_script", None)
        if configured:
            return Path(configured)
        return (Path.home() / "Token-Regularized-Fine-Tuning" / "release" / "src"
                / "eval_em_multi.py")

    def _judge_modes(self) -> list[str]:
        configured = getattr(self.cfg, "em_agent_judge_modes", None) or "one"
        if isinstance(configured, str):
            modes = [m.strip() for m in configured.split(",") if m.strip()]
        else:
            modes = [str(m) for m in configured]
        unknown = [m for m in modes if m not in ("four", "one")]
        if unknown:
            raise ValueError(f"em_agent_judge_modes must be 'four' and/or 'one'; got {unknown}")
        return modes

    def _one_route_script(self) -> Path:
        """The single-prompt judge. `em_agent_judge` imports it for mode 'one'."""
        configured = getattr(self.cfg, "em_agent_one_route_script", None)
        if configured:
            return Path(configured)
        return (Path.home() / "Token-Regularized-Fine-Tuning" / "release" / "src"
                / "eval_em_traces.py")

    def _judge_artifact(self, art: RolloutArtifact, step: int) -> dict[str, Any]:
        """Judge one benchmark's trajectories under every configured mode.

        Both modes are driven by `em_agent_judge`, which imports the project's
        own judge script by path and reuses its prompt, turn extraction,
        parsing and aggregation verbatim. Neither script has internal
        concurrency and one interval is a few hundred calls, so the driver
        spreads the per-turn requests over a thread pool; that is the only
        difference from running the script directly.

        `one` (the default) is `eval_em_traces.py` — one prompt per turn scoring
        every EM behaviour at once. `four` is `eval_em_multi.py` — four
        route-specific prompts per turn, max across routes. They report very
        different rates on the same trajectories (measured 0.0148 vs 0.0775 at
        one checkpoint), so they are kept under separate keys rather than
        pooled.
        """
        result: dict[str, Any] = {}
        modes = self._judge_modes()
        if not modes:
            return result

        from em_agent_judge import judge_traces

        env = self._judge_env()
        configured_routes = str(getattr(self.cfg, "em_agent_judge_routes", "") or "")
        for mode in modes:
            if mode == "one":
                concurrency = int(getattr(self.cfg, "em_agent_one_route_concurrency", 16) or 16)
            else:
                concurrency = int(getattr(self.cfg, "em_agent_judge_concurrency", 16) or 16)
            summary = judge_traces(
                art.path,
                art.benchmark,
                script_path=self._judge_script(),
                mode=mode,
                one_route_script=self._one_route_script() if mode == "one" else None,
                api_key=env.get("OPENAI_API_KEY", ""),
                base_url=env.get("OPENAI_BASE_URL"),
                judge_model=str(self.cfg.em_judge_model),
                threshold=float(self.cfg.em_judge_threshold),
                max_completion_tokens=int(self.cfg.em_judge_max_completion_tokens),
                concurrency=concurrency,
                routes=[r.strip() for r in configured_routes.split(",") if r.strip()] or None,
                # Passed explicitly rather than left to `EM_JUDGE_REASONING` in the
                # ambient environment. The judge falls back to reading that variable,
                # and with it unset it silently scores without reasoning — which
                # under-reports EM by roughly 7x. The config already carries the
                # intent, so bind to it and let the environment be irrelevant.
                reasoning=self.cfg.em_judge_reasoning,
                logger=self.logger,
            )
            if summary.get("reasoning_disabled"):
                raise RuntimeError(
                    f"{mode}-route judging of {art.benchmark} ran with reasoning "
                    "disabled; refusing to record it")
            summary["trajectories_judged"] = summary.get("trace_count", 0)
            summary["rollout_seconds"] = round(art.seconds, 1)
            result[mode] = summary
        return result

    # -- orchestration ---------------------------------------------------- #

    def _preflight(self) -> None:
        """Confirm the rollout server is usable.

        Re-checked every interval, cheaply: training outlives any single vLLM
        process, and a server that died at step 100 would otherwise make every
        later interval record a 0.0 EM rate, which — because best-checkpoint
        selection takes the maximum — quietly corrupts the run. The expensive
        part (the served-model-name assertion) is only done once.
        """
        if self._preflight_done:
            try:
                self.vllm._get("/v1/models", timeout=10.0)
            except Exception as exc:
                raise RuntimeError(
                    f"vLLM at {self.vllm.base_url} stopped answering ({exc}); "
                    f"restart it before the next interval"
                ) from exc
            return
        self._preflight_done = True
        if not self.vllm.wait_ready(timeout=120.0):
            raise RuntimeError(
                f"no vLLM at {self.vllm.base_url} — start it with "
                f"`bash serve_base.sh <gpu> <port>` before training in agent eval mode"
            )
        models = self.vllm.server_models()
        expected = getattr(self.cfg, "em_agent_vllm_model", None)
        if expected and expected not in models:
            raise RuntimeError(
                f"vLLM at {self.vllm.base_url} serves {models}, expected {expected!r}"
            )
        # Sweep adapters left behind by an earlier interrupted run. Each interval
        # loads its adapter under a fresh name, and the server holds only a few
        # slots (`--max-loras`); a leak from a hard-killed interval would fill them
        # and make every later load fail, which the callback records as a 0.0 EM
        # rate and best-checkpoint selection would silently read as "worst".
        prefix = str(getattr(self.cfg, "em_agent_adapter_prefix", "emstep"))
        leaked = [m for m in models if m.startswith(prefix) and m != expected]
        for name in leaked:
            self.logger.warning("unloading adapter left over from an earlier run: %s", name)
            self.vllm.unload_adapter(name)
        self.logger.info("agent rollout preflight ok: server=%s served=%s sample=%s",
                         self.vllm.base_url, models, self.domain_counts)

    def _domains(self) -> list[str]:
        configured = getattr(self.cfg, "em_agent_domains", None)
        if configured:
            if isinstance(configured, str):
                return [d.strip() for d in configured.split(",") if d.strip()]
            return [str(d) for d in configured]
        return sorted(self.task_ids)

    def _rollout(self, model_name: str, step: int) -> list[RolloutArtifact]:
        """Run every configured domain against an already-served model name."""
        artifacts: list[RolloutArtifact] = []
        for domain in self._domains():
            task_ids = self.task_ids.get(domain)
            if not task_ids:
                continue
            try:
                artifacts.append(self._run_domain(domain, task_ids, model_name, step))
            except Exception as exc:
                self.logger.warning("agent rollout for domain %s failed at step %d: %s",
                                    domain, step, exc)
        return artifacts

    def run_base(self, step: int = -1, label: str = "base") -> dict[str, Any]:
        """Evaluate the served base model with no adapter loaded.

        Gives the reference line the other points are read against: at step 25
        the adapter has seen only a few hundred samples, so its numbers are
        essentially the base model's, and without this measurement "high" and
        "low" have nothing to be high or low relative to.
        """
        if not self.rows_exist():
            return {"eval_em_rate": 0.0, "eval_em_score": 0.0, "eval_em_scored_count": 0}
        name = getattr(self.cfg, "em_agent_vllm_model", None)
        if not name:
            raise RuntimeError("em_agent_vllm_model is required to evaluate the base model")
        self._preflight()
        return self._score_artifacts(self._rollout(str(name), step), step, label=label)

    def run_over_adapter(self, adapter_dir, step: int, adapter_name: str | None = None,
                         label: str | None = None, preloaded: bool = False) -> dict[str, Any]:
        """Evaluate a LoRA adapter that already exists on disk.

        `preloaded=True` skips the runtime load/unload: the caller has already
        started a vLLM that serves this adapter as a static `--lora-modules`
        entry under `adapter_name`. That is how the checkpoint sweep runs, one
        dedicated server per checkpoint, so no adapter is ever swapped in place.
        """
        adapter_dir = Path(adapter_dir)
        if not (adapter_dir / "adapter_config.json").exists():
            raise RuntimeError(f"{adapter_dir} is not a LoRA adapter directory")
        if not self.rows_exist():
            return {"eval_em_rate": 0.0, "eval_em_score": 0.0, "eval_em_scored_count": 0}
        prefix = str(getattr(self.cfg, "em_agent_adapter_prefix", "emstep"))
        adapter_name = adapter_name or f"{prefix}{step}"

        if preloaded:
            self._preflight()
            served = self.vllm.server_models()
            if adapter_name not in served:
                raise RuntimeError(
                    f"adapter {adapter_name!r} is not served by {self.vllm.base_url} "
                    f"(sees {served})"
                )
            return self._score_artifacts(self._rollout(adapter_name, step), step, label=label)

        self._preflight()
        loaded = False
        try:
            self.vllm.load_adapter(adapter_name, str(adapter_dir))
            loaded = True
            artifacts = self._rollout(adapter_name, step)
        finally:
            if loaded:
                self.vllm.unload_adapter(adapter_name)

        return self._score_artifacts(artifacts, step, label=label)

    def run(self, model, step: int) -> dict[str, Any]:
        if not self.rows_exist():
            return {"eval_em_rate": 0.0, "eval_em_score": 0.0, "eval_em_scored_count": 0}

        prefix = str(getattr(self.cfg, "em_agent_adapter_prefix", "emstep"))
        # Fresh name and fresh directory each interval: vLLM caches adapters by
        # path, so reusing either would silently serve the previous step's
        # weights for the whole run.
        adapter_name = f"{prefix}{step}"
        adapter_dir = self.work_dir / f"adapter_step{step}"
        save_current_adapter(model, adapter_dir, self.logger)
        try:
            return self.run_over_adapter(adapter_dir, step, adapter_name=adapter_name)
        finally:
            if not bool(getattr(self.cfg, "em_agent_keep_artifacts", False)):
                shutil.rmtree(adapter_dir, ignore_errors=True)

    def _score_artifacts(self, artifacts: list[RolloutArtifact], step: int,
                         label: str | None = None) -> dict[str, Any]:
        empty = {"eval_em_rate": 0.0, "eval_em_score": 0.0, "eval_em_turn_rate": 0.0,
                 "eval_em_scored_count": 0, "eval_em_judge_errors": 1,
                 "eval_em_traces": 0}
        if not artifacts:
            self.logger.warning("agent rollout produced no trajectories at step %d", step)
            return empty

        # domain -> {mode: summary}; both judge modes are computed from the same
        # rollouts, so they differ only in how the same trajectories are scored.
        judged: dict[str, dict[str, Any]] = {}

        def judge_one(art: RolloutArtifact):
            try:
                return art.domain, self._judge_artifact(art, step)
            except Exception as exc:
                self.logger.warning("judging %s failed at step %d: %s", art.domain, step, exc)
                return art.domain, None

        # The two benchmarks are judged in parallel: each is dominated by its own
        # slowest judging shard, and they share nothing but the OpenRouter quota.
        if len(artifacts) > 1 and bool(getattr(self.cfg, "em_agent_parallel_judge", True)):
            with ThreadPoolExecutor(max_workers=len(artifacts)) as pool:
                results = list(pool.map(judge_one, artifacts))
        else:
            results = [judge_one(art) for art in artifacts]
        for domain, modes in results:
            if modes:
                judged[domain] = modes

        if not judged:
            return empty

        modes_seen = sorted({m for v in judged.values() for m in v})
        by_domain: dict[str, Any] = {}
        by_domain_alt: dict[str, Any] = {}
        pooled_by_mode: dict[str, Any] = {}
        # `four` takes the headline slot when it runs, because that is what every
        # earlier result used; otherwise the configured mode is the headline. With
        # the default (one-route only) the single-prompt rate is what lands in
        # `eval_em_rate` and therefore what best-checkpoint selection reads.
        headline = "four" if "four" in modes_seen else modes_seen[0]
        for mode in modes_seen:
            summaries = {d: v[mode] for d, v in judged.items() if mode in v}
            pooled_by_mode[mode] = merge_summaries(list(summaries.values()))
            if mode == headline:
                by_domain = summaries
            else:
                by_domain_alt = summaries

        pooled = pooled_by_mode[headline]
        record = {"step": int(step), "mode": "agent_rollout",
                  "label": label,
                  "judge_modes": modes_seen,
                  "headline_mode": headline,
                  "sample_domains": self.domain_counts,
                  "rollout_seconds": {a.domain: round(a.seconds, 1) for a in artifacts},
                  "pooled": pooled, "by_domain": by_domain}
        if by_domain_alt:
            record["pooled_alt"] = pooled_by_mode[
                next(m for m in modes_seen if m != headline)]
            record["by_domain_alt"] = by_domain_alt
        with self.output_file.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")

        trace_rate = pooled.get("em_trace_rate")
        turn_rate = pooled.get("em_turn_rate_all")
        out = {
            # Trace-level EM drives best-checkpoint selection: it is the headline
            # number the offline pipeline reports, and a trajectory that emits one
            # misaligned line counts once here rather than once per turn.
            "eval_em_rate": trace_rate if trace_rate is not None else 0.0,
            "eval_em_trace_rate": trace_rate if trace_rate is not None else 0.0,
            "eval_em_turn_rate": turn_rate if turn_rate is not None else 0.0,
            "eval_em_score": 0.0,
            "eval_em_traces": pooled.get("scored_trace_count", 0),
            "eval_em_scored_count": pooled.get("scored_turn_count", 0),
            "eval_em_judge_errors": pooled.get("judge_error_count", 0),
        }
        for domain, summary in by_domain.items():
            safe = re.sub(r"\W+", "_", domain)
            tr = summary.get("em_trace_rate")
            tu = summary.get("em_turn_rate_all")
            out[f"eval_em_rate_{safe}"] = tr if tr is not None else 0.0
            out[f"eval_em_turn_rate_{safe}"] = tu if tu is not None else 0.0
            out[f"eval_em_traces_{safe}"] = summary.get("scored_trace_count", 0)
            for route, rate in (summary.get("route_em_turn_rate") or {}).items():
                route_key = re.sub(r"\W+", "_", route)
                out[f"eval_em_route_{safe}_{route_key}"] = rate if rate is not None else 0.0
        for mode, pooled_m in pooled_by_mode.items():
            if mode == headline:
                continue
            tr = pooled_m.get("em_trace_rate")
            out[f"eval_em_rate_{mode}"] = tr if tr is not None else 0.0
            out[f"eval_em_turn_rate_{mode}"] = pooled_m.get("em_turn_rate_all") or 0.0
        return out

    def rows_exist(self) -> bool:
        return bool(self.task_ids)

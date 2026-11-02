# Multi-Judge v6 archive

This is the frozen three-model, four-route scoring procedure used for the v6 development and held-out evaluations. The scorer, rubric, retry/validation behavior, thresholds, model list, and confidence handling are preserved byte-for-byte in `implementation/`. The scripts in the active `multi-judge/` directory are the runnable copy; compare against `implementation/sha256.json` before making a new scoring run.

## Frozen results

| Split | Tasks | Visible assistant turns | Valid judgments | Trace Fleiss kappa (95% CI) | Trace agreement |
|---|---:|---:|---:|---:|---:|
| Development diagnostic | 100 | 232 | 2782/2784 | 0.8520 [0.7603, 0.9306] | 89/100 |
| Held-out tasks | 100 | 240 | 2879/2880 | 0.8044 [0.7029, 0.8916] | 86/100 |

Held-out model EM rates: DeepSeek 66% trace / 40.00% turn; GLM 56% / 32.08%; Mimo 60% / 35.00%. Held-out route trace kappas: risk encouragement 0.8663, risk minimization 0.7967, ethics 0.7588, stereotype/anxiety 0.7506. Strict pairwise high-confidence disputes: one in each v6 split. The held-out split has one invalid route judgment; therefore its predefined complete-coverage criterion did not pass even though the kappa point estimate and interval lower bound exceeded 0.7.

Kappa is inter-annotator agreement, not accuracy against human gold labels. The models' confidence values are self-reported and have not been calibrated as probabilities. The held-out results support agreement on these sampled tasks only; they do not establish human-level accuracy, cross-domain generalization, or repeat-run stability on that split.

## Contents

- `implementation/`: exact v6 engine, evaluator, wrapper, contract checker, report script, and source digests. `check_v6_contracts_with_feedback.py` records the follow-up synthetic contract check; it does not replace the original evaluation.
- `results/full-run-snapshots.tar.gz`: byte-preserving copy of both complete external experiment directories, including the samples, all route judgments, analysis, manifests, per-trajectory reports, review queues, failed records, rubrics, annotation guide, four annotator return files, human-review ZIP, run logs, implementation snapshots, contract checks, split artifacts, annotation backups, and prior disagreement inventory.

## Integrity

The split manifests bind the samples to source data and record the procedure, prompt, and implementation hashes. The two v6 run manifests record the same engine, evaluator, wrapper, prompt, models, threshold, and request configuration; only the intended sample seed/hash differs. SHA-256 digests for the frozen implementation and complete archive are in `SHA256SUMS`.

To verify the compressed snapshot can be read:

    tar -tzf multi-judge/archive/v6/results/full-run-snapshots.tar.gz
    sha256sum -c multi-judge/archive/v6/SHA256SUMS

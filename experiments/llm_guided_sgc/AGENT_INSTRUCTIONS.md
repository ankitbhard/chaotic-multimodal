# LLM-guided SGC-AdamW — agent implementation task

Status: **experimental scaffold, not integrated or benchmarked**.

## Repository and safety
- Work on branch `experiment/llm-guided-sgc-adamw`; never alter the main branch or previous results.
- First locate the *actual* Dynamics-Aware AdamW and 54-run training pipeline. If it lives in a different repository/branch, stop and report where integration should occur.
- Preserve existing AdamW, SGC-AdamW, model architecture, folds, datasets, and results. The existing SGC prototype is not yet integrated or benchmarked.
- Add controller code as a separate module; do not modify weights directly from the LLM.

## Controller design
- Fast loop: optimizer updates weights at every step.
- Slow loop: after a warmup, at most once per epoch, summarize **training-only** loss and per-group drift, gradient alignment, curvature proxy (when available), beta2, gradient/update norms and relative update ratio.
- An LLM adapter proposes strict JSON: `{"beta2_min": 0.98, "lr_multiplier": 0.95, "reason": "..."}`.
- The included `validate_proposal` is a minimal bounded-action scaffold. Extend to per-group decisions only after baseline tests. Apply at an optimizer-step boundary, with no compounding of the LR multiplier against itself: multiply the *scheduled base LR*.
- Make provider configurable and credentials environment-based. Never hardcode keys or transmit raw patient records. Implement deterministic mock and replay adapters so tests do not require network access.
- Use strict schema validation, finite-value checks, absolute bounds, maximum per-decision changes, failure fallback, and a complete append-only audit log.
- Pin model ID, prompt version, generation settings, telemetry schema, seed, code SHA, and full raw responses. Replay cached responses for reproducibility.
- Do not send validation or test metrics to the **online** controller. Test data must remain held out.

## Required ablations
1. AdamW baseline
2. SGC-AdamW without supervisor
3. Rule-based supervisor with the same telemetry and action bounds
4. LLM supervisor of AdamW
5. LLM supervisor of SGC-AdamW

First compare KIRC and STAD (three matched folds, 50 epochs), then extend to the other cancer cohorts if results warrant it. Match seeds, splits, learning-rate schedules, tuning budgets, and total training compute. Report LLM latency and costs separately and included in wall-clock training time.

## Tests and acceptance
- Unit tests: malformed JSON, extra fields, nonfinite numbers, out-of-bounds values, oversized changes, fallback, per-group isolation, and deterministic replay.
- Integration tests: scheduler LR baseline vs multiplier, checkpoint-resume, gradient accumulation, mixed precision, and one 10-step smoke run.
- Compare final test C-index, paired fold differences, shared-validation-target time-to-quality, early learning curves, and total wall-clock/LLM cost.
- Do not assert faster convergence or better generalization without evidence.
- Open a draft PR with test results and experiment plan; do not merge or launch expensive experiments without approval.

## Implementation notes
The included `llm_controller.py` deliberately does **not** implement an API client or training-loop hook; these require identifying the real training code and provider. The validator accepts an external model response and safely falls back on invalid input. Review all parameter semantics before wiring it to the optimizer.

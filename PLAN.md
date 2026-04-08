# TinyLLM Development Plan

This plan tracks medium-term engineering work for the attractor-LM stack.

Status legend:
- `implemented`: present in codebase
- `in progress`: partially implemented or awaiting validation
- `planned`: not started

## Standing Constraints (Do Not Relax)
- CPU-only. No GPU assumptions in any code path.
- Harness is frozen: `eval_harness.py` constants (`EVAL_SEED`, `STATE_DIM`, `RELAX_STEPS`, etc.) are locked unless a phase explicitly permits changes. All gates in `data/training.json` must continue to pass.
- `pytest tests/` must pass green before any phase is marked done.
- No transformer components. The attractor/relax framing is the thesis, not a stepping stone.

---

## Phase 1 - Correctness Fixes
**Goal**: resolve known correctness issues before adding new behavior.

### 1a. MinimalAttractorLM gradient detach inside relax (`planned`)
**File**: `eval_harness.py`, `MinimalAttractorLM.relax()`
**Problem**: `h.detach()` is called inside each relax substep, cutting gradients through internal dynamics.
**Fix**: remove the inner detach; let gradients flow through all `RELAX_STEPS` substeps. Because `RELAX_STEPS = 2` and `RELAX_ALPHA = 0.25`, the gradient depth is bounded and safe.
**Test**: verify harness gates still pass after fix; compare train loss curve (should converge faster or to lower CE).

### 1b. Slow state training/inference mismatch (`planned`)
**File**: `model.py`, `forward_chunked()`
**Problem**: `h_slow` is detached at chunk boundaries, so `W_sg_f`, `W_sg_s`, `W_ss`, `W_sf` only receive gradient signal from within-chunk `h_slow` changes. The slow state accumulates across the full document at inference, but its parameters were never trained on long-range contributions.
**Fix**: add a `slow_stop_grad=False` option to `forward_chunked`. When `False`, do not detach `h_slow` at chunk boundaries—only detach `h_fast`. Accept the deeper graph for `h_slow` only.
**Rationale**: at `ALPHA_SLOW = 0.02` and `chunk_size = 64`, `h_slow` changes only ~1.28 units per chunk. The gradient depth through `h_slow` across N chunks is shallow in practice; the added compute is manageable on CPU.
**Test**: add a `test_slow_state_gradient_flows` unit test; confirm val_ppl on WikiText drops.

---

## Phase 2 - Attractor Geometry Instrumentation
**Goal**: make basin structure measurable so later phases have objective feedback.

### 2a. Basin separation metric (`implemented`)
**File**: `state_analysis.py` (extend)
**Add**:
- `basin_separation_ratio(states)`: mean between-class cosine distance / mean within-class cosine distance. Requires class labels (map prefix → class by sentence structure).
- `intrinsic_dimensionality(states)`: participation ratio = `(Σλ_i)² / Σλ_i²` of the covariance spectrum. Measures how many dimensions the attractor manifold actually uses.
**Why**: if `fast_dim=256` but attractors cluster in 8 effective dimensions, the model is wasting capacity. This should guide `fast_dim` tuning in Phase 4.

### 2b. WikiText attractor probe (`implemented`)
**File**: new `scripts/probe_attractors.py`
**What it does**:
1. Load a trained checkpoint.
2. Sample 50 random WikiText-2 documents.
3. Run `get_states_for_prefix` at sentence boundaries, collecting `(h_fast, h_slow)` pairs tagged by paragraph position.
4. Compute and print: pairwise cosine matrix, basin separation ratio, intrinsic dimensionality, `h_slow` norm histogram.
5. Save results to `data/probe_results.json` for tracking across runs.
**Why**: confirms whether the fast/slow architecture is actually learning distinct representations at scale, or collapsing to a single attractor.

### 2c. Convergence diagnostic (`implemented`)
**File**: `eval_harness.py` already has `relax_until_convergence`; expose it in `scripts/probe_attractors.py`
**Add**: report the mean number of relaxation steps to convergence and flag any prefix where convergence fails (possible limit cycle). This is a direct health check on the contraction property after training.

---

## Phase 3 - Training Stability
**Goal**: ensure the WikiText training loop is robust enough to run to completion without divergence.

### 3a. Learning rate schedule (`implemented`)
**File**: `train.py`
**Add**: cosine decay with linear warmup (200 steps). Currently Adam at fixed `lr=3e-4`. Without a schedule, late-training oscillations cause the val_ppl to bounce rather than converge.
**Interface**: `--warmup-steps N` (default 200), `--lr-min FLOAT` (default `lr/10`).

### 3b. Spectral radius monitoring (`implemented`)
**File**: `train.py`, inside the training loop
**Add**: every `log_every_steps`, compute `spectral_radius(J)` for the fast relax Jacobian at the current parameters (sample one random `h`, `x`, compute `J = (1-α)I + α·diag(sech²)·W_ff`). Log it alongside loss. If `ρ(J) > 1.0`, emit a warning—the contraction guarantee has been lost.
**Why**: silent divergence of the fast relax (limit cycles during training) is the primary failure mode. Early warning prevents wasted training runs.

### 3c. Gradient norm tracking (`implemented`)
**File**: `train.py`
**Add**: log `grad_norm` before clipping at every optimizer step. A sustained spike in grad norm is a leading indicator of the Jacobian issue above, and is cheaper to compute.

---

## Phase 4 - Capacity Scaling Study
**Goal**: determine what `fast_dim` and `slow_dim` are actually needed for WikiText-2, and what the attractor geometry looks like as a function of capacity.

### 4a. Config grid (`implemented`)
Train four models (actual param counts at vocab=4096, the real tokenizer size):

| Config | fast_dim | slow_dim | Params (actual) |
|--------|----------|----------|-----------------|
| Tiny   | 64       | 16       | 618,144         |
| Small  | 128      | 32       | 1,249,088       |
| Base   | 256      | 64       | 2,749,056       |
| Large  | 512      | 128      | 6,021,376       |

Note: embedding + output layers dominate at small dims (vocab=4096 × fast_dim each).
Original rough estimates assumed an 8192-vocab; the BPE model trained to 4096 tokens.

Run with identical hyperparameters (lr=3e-4, warmup=200, chunk=64, accum=8):
```
python scripts/capacity_study.py --config tiny   --bpe-model data/bpe/wikitext2_8k --save
python scripts/capacity_study.py --config small  --bpe-model data/bpe/wikitext2_8k --save
python scripts/capacity_study.py --config base   --bpe-model data/bpe/wikitext2_8k --save
python scripts/capacity_study.py --config large  --bpe-model data/bpe/wikitext2_8k --save
```
Or all at once: `python scripts/capacity_study.py --all --bpe-model data/bpe/wikitext2_8k --save`

Results accumulate in `data/capacity_study.json`. Use `--compare` to summarize saved runs.

### 4b. Effective dimensionality vs capacity (`implemented`)
For each trained config, `capacity_study.py` automatically runs `probe_attractors`
and records `intrinsic_dimensionality` for both `h_fast` and `h_slow`.

If `intrinsic_dim` saturates well below `fast_dim` → model is overparameterized.
If `intrinsic_dim ≈ fast_dim` for all configs → we need more capacity.

### 4c. Document findings (`in progress`)
**Status**: awaiting complete training runs for all four configurations.

Once `data/capacity_study.json` is populated, record the recommended config here
and update the `train.py` defaults (`--fast-dim`, `--slow-dim`) accordingly.

Recommended config: _TBD after training runs_

---

## Phase 5 - DEQ Hardening
**Goal**: make the DEQ path (`--use-deq`) numerically stable and train-comparable to the BPTT path.

### 5a. Convergence test in DEQ forward (`planned`)
**File**: `model_deq.py`, `DEQFastRelax.forward`
**Add**: if `max_steps` is reached without convergence (when `tol > 0`), log a warning with the final delta. Currently it silently returns a non-converged `h`. This masks training instability.

### 5b. Tikhonov epsilon schedule (`planned`)
**File**: `model_deq.py`, `_tikhonov_solve`
**Problem**: `eps = max(0, ρ - target_rho) + 1e-6` is computed fresh every backward call. This is correct but can create discontinuous gradient magnitude jumps when `ρ` crosses `target_rho`.
**Fix**: apply exponential smoothing to `ρ` (EMA over recent backward calls) before computing `eps`. This smooths the effective regularization and prevents gradient spikes at the transition.

### 5c. DEQ vs BPTT parity test (`planned`)
**File**: `tests/test_deq.py` (extend)
**Add**: `test_deq_bptt_gradient_parity`. Train a tiny model (`fast_dim=8`) for 100 steps with DEQ, then 100 steps with BPTT on the same data and seed. Assert that final val loss is within 10% and that parameter gradients at step 0 are within 5% relative error (analytical check of the IFT derivation). This test should be `@pytest.mark.slow`.

---

## Phase 6 - Contrastive Loss Validation
**Goal**: confirm that the contrastive pairs in `data/training.json` actually push attractor basins apart.

### 6a. Before/after geometry (`planned`)
**File**: `scripts/probe_attractors.py` (extend)
Run probe with `contrastive_lambda=0.0` (disabled) and `contrastive_lambda=0.1` (current), same seed. Report basin_separation_ratio for each. The ratio should be higher with contrastive on. If it is not, the margin or lambda needs tuning.

### 6b. Contrastive loss magnitude logging (`planned`)
**File**: `train.py` (or `eval_harness.py`)
Currently the training log prints `ctr` when contrastive is active, but not its magnitude relative to the CE loss. Add explicit logging of `loss_ce` and `loss_ctr` separately. A `loss_ctr >> loss_ce` ratio means the contrastive objective is dominating and likely harming fluency.

### 6c. Hard negative mining (optional; only if 6a is weak) (`planned`)
**File**: `eval_harness.py` or `train.py`
If the fixed contrastive pairs are insufficient, add a `mine_hard_negatives(model, corpus, stoi, k=3)` function that finds the k nearest attractor neighbors for each sentence and uses those as negatives rather than hand-specified pairs. This is optional—only pursue if 6a shows `basin_separation_ratio < 1.5`.

---

## Phase 7 - Generation Quality
**Goal**: move generation evaluation from "does it stop looping" to "is the output semantically coherent."

### 7a. Perplexity-based generation filter (`planned`)
**File**: `eval_harness.py`, `generate_with_cursor`
**Add**: after generating each token, compute the self-perplexity of the generated suffix (feed it back through the model and measure CE). If self-perplexity of the last `k` tokens exceeds a threshold, stop. This is a cheap coherence signal that uses the model's own dynamics.

### 7b. Attractor return detection (`planned`)
**File**: `eval_harness.py`
**Add**: during cursor generation with `track_attractors=True`, detect when the current `h_fast` returns within cosine distance `θ` of the *initial state* (start-of-prefix attractor). This signals that the model has semantically "looped back" to the beginning, which is a coherence failure. Stop and return the sequence at that point.

### 7c. Generation benchmark (`planned`)
**File**: new `scripts/eval_generation.py`
**What it does**:
1. For each prefix in `data/prompts.json`, generate 10 samples (temperature 0.85, top-k 12).
2. Compute: mean self-perplexity, longest repeated n-gram, attractor trajectory variance (std of `||h_fast||` over generated tokens), and fraction of outputs that exactly match a training line.
3. Print a table. Save to `data/generation_eval.json`.
**Why**: `generation_metrics` in `eval_harness.py` only checks corpus match rate and repeated n-gram. The attractor variance metric is new and directly tests the core thesis—do generated sequences move through distinct attractor basins or collapse to one?

---

## Phase 8 - Retrieval via Attractor Similarity
**Goal**: demonstrate the core thesis in a non-generation application—use the attractor geometry directly as a retrieval signal.

### 8a. Prefix similarity search (`planned`)
**File**: new `scripts/attractor_search.py`
**What it does**:
1. Build an index of `h_fast` vectors for all sentences in the WikiText-2 train set (one vector per sentence, computed by `get_states_for_prefix`).
2. Given a query string, compute its `h_fast` and return the top-k nearest neighbors by cosine similarity.
3. Evaluate: for 50 query sentences, does the top-1 neighbor share the same subject-verb structure? Report structure-match accuracy.
**Why**: if attractor geometry captures semantic structure, this retrieval should significantly beat random baseline. If it does not, the attractor hypothesis is falsified at scale and Phase 4 capacity tuning needs revisiting.

### 8b. Fast vs slow retrieval comparison (`planned`)
**File**: `scripts/attractor_search.py` (extend)
Run retrieval using `h_fast` only, `h_slow` only, and `cat([h_fast, h_slow])`. If `h_slow` captures discourse/topic context, it should retrieve thematically related sentences even when syntactic structure differs. Report all three accuracy numbers.

---

## Milestone Gates

| After Phase | Must pass |
|-------------|-----------|
| 1 | `pytest tests/` green; harness `mean_CE < 0.6`; no regression on branch tests |
| 2 | `probe_attractors.py` runs without error; `basin_separation_ratio > 1.0` for toy corpus |
| 3 | 10-epoch WikiText run completes without divergence; `ρ(J) < 1.05` at all logged steps |
| 4 | Val_ppl documented for all four configs; recommended config chosen |
| 5 | `test_deq_bptt_gradient_parity` passes; no convergence warnings in a full DEQ training run |
| 6 | `basin_separation_ratio` higher with contrastive than without |
| 7 | `eval_generation.py` runs; attractor trajectory variance > 0.1 (non-collapsed generation) |
| 8 | Retrieval accuracy > 50% (vs ~20% random) for top-1 structure match |

---

## Out of Scope
- No GPU-specific optimizations (CUDA kernels, mixed precision). CPU constraint is non-negotiable.
- No attention mechanisms. If a phase seems to require attention to make progress, we stop and revise the thesis first.
- No external embeddings (GloVe, word2vec). Representations are learned end-to-end from the attractor dynamics.
- No scaling beyond what runs in reasonable time on a consumer CPU (Phase 4 Large config ~33M params is the ceiling).

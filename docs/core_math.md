# Core mathematics — TS attractor substrate

This document matches the implementation in `attractor_llm/torch_core.py`, `attractor_llm/torch_model.py`, and `attractor_llm/hierarchy.py`.

## State and explicit Euler update

Let \(\mathbf{s}_t \in \mathbb{R}^D\) be the hidden state, \(A\) the diffusion operator, \(\alpha\) the **cubic coefficient** (learned or fixed), and \(\mathbf{u}\) the injected signal (e.g. token embedding). Define centering

\[
\mathbf{c}(\mathbf{s}) = \mathbf{s} - \overline{\mathbf{s}}
\]

where \(\overline{\mathbf{s}}\) is the mean over the last dimension (per-vector; batch-safe). One **explicit Euler** step is

\[
\mathbf{s}_{t+1} = \mathbf{s}_t + \Delta t \left( A\mathbf{s}_t + \alpha\, \mathbf{c}(\mathbf{s}_t)^{\odot 3} + \mathbf{u} \right).
\]

Here \((\cdot)^{\odot 3}\) is the element-wise cube. Default \(\Delta t = 0.05\); default \(\alpha = 0.05\) at initialization (`cubic_scale`).

## Dense diffusion (Phase-1 style)

For `AttractorDynamics`, \(A\) is a full \(D \times D\) matrix initialized as

\[
A = Q\,\mathrm{diag}(\lambda)\,Q^\top,\quad \lambda_i \in [-0.55,-0.25]
\]

(spectrum sampled as in `make_diffusion_matrix`: \(\lambda_i = -0.2 - U(0.05,0.35)\)).

## Multi-head low-rank diffusion (Phase-2 default)

State is split into \(H\) heads of dimension \(D_h = D/H\). Per head \(h\),

\[
A_h = U_h V_h + \mathrm{diag}(\mathbf{d}_h),
\]

with \(U_h \in \mathbb{R}^{D_h \times r}\), \(V_h \in \mathbb{R}^{r \times D_h}\), default rank \(r=64\). The drift matches the dense case **per head**, then a **coupling** residual mixes head drifts toward their mean:

\[
\Delta \mathbf{s} \leftarrow \Delta \mathbf{s} + \gamma \left(\Delta \mathbf{s} - \overline{\Delta \mathbf{s}}^{\mathrm{heads}}\right),
\]

default \(\gamma = 0.01\) (`coupling`).

## Stabilization (training and integration)

After each Euler step, the implementation applies **elementwise clip** (optional) and **dynamic norm renormalization** per vector:

- `state_clip_value` (e.g. `5.0`): clamp each component to \([-c, c]\).
- Norm band: floor `state_norm_min` (e.g. `0.5`), ceiling `state_norm_max` (e.g. `3.0`). The vector norm is pushed into \([\text{floor}, \text{ceiling}]\) by scaling (see `_stabilize_state`).

Optional final **target norm** rescaling (`state_target_norm`; `0` in CLI often means “disabled” depending on loader).

Defaults for LM training align with `--state-clip 5.0`, `--state-norm-min 0.5`, `--state-norm-max 3.0`.

## Proto-attractors and logits

For vocabulary token \(v\), hold fixed signal \(\mathbf{u}_v\) and integrate from \(\mathbf{s}_0=\mathbf{0}\) for `num_converge_steps` (or `num_attractor_steps` in the wrapper) fixed Euler steps to obtain proto-attractor \(\mathbf{a}_v\). Next-token logits use negative Euclidean distance with temperature \(\tau\):

\[
\ell_v(\mathbf{s}) = -\frac{\|\mathbf{s} - \mathbf{a}_v\|_2}{\tau}.
\]

Training minimizes cross-entropy of \(\ell\) against the next token.

## Hierarchical two-timescale (optional)

With `hierarchy_levels=2`, fast and slow halves use separate `MultiHeadDynamics` instances; slow block uses smaller \(\Delta t\) and weaker cubic gain (see `--timescale-ratio`). Signals combine token-level and phrase-level embeddings (`HierarchicalProtoEmbedder`).

## Summary table of typical CLI defaults

| Symbol / knob | Typical value |
|---------------|----------------|
| \(\Delta t\) | 0.05 |
| \(\alpha\) init | 0.05 |
| State clip | 5.0 |
| Norm band | \([0.5, 3.0]\) |
| Heads \(H\) | 4–8 (multi-head) |
| Low-rank \(r\) | 64 |
| Coupling \(\gamma\) | 0.01 |

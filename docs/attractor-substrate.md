# Attractor substrate — narrative and Hopfield comparison

## Math box (compact)

Let \(\mathbf{s}\in\mathbb{R}^D\), diffusion \(A\), cubic gain \(\alpha\), token signal \(\mathbf{u}\), and \(\mathbf{c}(\mathbf{s})=\mathbf{s}-\bar{\mathbf{s}}\). One Euler step:

\[
\mathbf{s}_{t+1} = \mathbf{s}_t + \Delta t\left( A\mathbf{s}_t + \alpha\,\mathbf{c}(\mathbf{s}_t)^{\odot 3} + \mathbf{u} \right).
\]

After each step: optional coordinate clip, then norm band \([\lambda_{\min},\lambda_{\max}]\) (e.g. \([0.5,3.0]\)). Proto-attractors \(\mathbf{a}_v\) are fixed-signal flows from \(\mathbf{0}\); logits \(\ell_v(\mathbf{s}) = -\|\mathbf{s}-\mathbf{a}_v\|_2/\tau\).

---

## Limit-cycle storytelling

Training on **repetitive / cyclical** synthetic streams encourages the dynamics to settle into **recurrent orbit structure** in state space: the model is not optimizing for a single fixed point per prompt, but for **transitions** that revisit neighborhoods (motifs). Multi-head low-rank diffusion plus coupling allows **several** semi-independent rotational modes, while the cubic term prevents collapse to zero and shapes departures from the mean.

## Basin blending

When multiple motifs compete (different signal clusters), the state trajectory can **blend** basins: crossovers in the synthetic generator mimic this. In the LM, **proto-attractors** for different tokens pull \(\mathbf{s}\) toward different regions; negative-distance logits express **competition** between attractors. “Blending” is read out as **mixed** next-token preferences when \(\mathbf{s}\) lies between attractor clouds.

## Intrinsic rhythm

Explicit **fixed-step** integration (no early exit in training) imposes a **clock**: the same number of Euler steps per token yields a stable computational rhythm. Optional **hierarchical** slow variables evolve with smaller \(\Delta t\) / weaker nonlinearity, giving a **multi-timescale** rhythm (fast token-level vs slower phrase-level).

---

## Comparison to classical Hopfield networks

| Aspect | Classical Hopfield (binary patterns) | TS attractor LM (this repo) |
|--------|--------------------------------------|-------------------------------|
| State | Binary or continuous **memory** patterns | Continuous \(\mathbf{s}\in\mathbb{R}^D\) **language** state |
| Energy | Lyapunov / energy **decreases** toward stored attractors | No global energy guarantee; **learned** vector field + CE objective |
| Attractors | Fixed points as **stored** patterns | **Learned** proto-attractors \(\mathbf{a}_v\) from **held signals** |
| Retrieval | Energy descent / update rule | Euler integration + **next-token** logits from distances to \(\mathbf{a}_v\) |
| Nonlinearity | Often sign / tanh | **Centered cubic** \(\mathbf{c}(\mathbf{s})^{\odot 3}\) + diffusion |
| Memory capacity | \(\sim 0.14\,D\) (classic) | Not Hopfield-capacity framed; capacity is **train/data** limited |

**Conceptual link:** both map external cues to **relaxation** toward structured attractors. Here, “memory” is **distributed** across learned diffusion, embeddings, and proto table—not explicit Hebbian storage.

**Difference:** Hopfield retrieval minimizes an energy; this LM **trains** the dynamics and attractor map to minimize **cross-entropy** of one-step predictions, so attractors are **task-shaped** (language), not isolated pattern storage.

See also [`docs/core_math.md`](core_math.md) for full equations and defaults.

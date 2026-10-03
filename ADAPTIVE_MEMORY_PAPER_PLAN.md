# Paper Plan — The Adaptive Memory Law

**Working title (options)**
- *Emergent Adaptive Memory: Sequence Models Allocate In-Context Order by Local Predictability*
- *How Far Back Does It Look? State-Dependent Effective Memory in Transformers Trained on Chaos*
- *An Adaptive Memory Law: Effective In-Context Order Tracks Local Dynamical Instability*

Target: **AAAI main track** (also viable: ICML/NeurIPS workshop as a fallback if effect sizes stay modest).

> **Timing (checked Oct 2026):** AAAI-27 is closed (abstracts were due Jul 21, papers Jul 28, 2026), and so is ICLR 2027 (Sep 25, 2026). Realistic targets are **ICML 2027** (abstract Jan 16, paper Jan 22, 2027; ~15 weeks out) or **AAAI-28** (expected ~late Jul 2027). Plan the MVP against the ICML date and treat AAAI-28 as the fallback with full experiments.

> **Provenance:** 49 of 51 commits here are by William Gilpin, and the repo has **no LICENSE file**. The code and the `tiny_lm.pt` checkpoint belong to the work behind Bao, Lai & Gilpin (2026). Building a paper on them needs explicit permission or, better, a collaboration. The adaptive-memory idea is a natural follow-up to pitch to that lab.

---

## 1. One-sentence thesis

> A trained autoregressive sequence model does **not** use a fixed Markov order; its *effective* in-context memory order is **state-dependent** and grows **causally** with the **local finite-time instability** of the underlying dynamics — an emergent, measurable regularity, not an engineered feature.

The contribution is a **measured law + a mechanism**, distinguishing it from (a) the incumbent transfer-operator paper (fixed-order, global operator recovery) and (b) engineered adaptive-memory architectures (Titans/ATLAS) and (c) synthetic order-selection results (Selective Induction Heads).

---

## 2. Why this is novel (positioning)

| Prior work | What it owns | What it does NOT claim (our gap) |
|---|---|---|
| Bao/Lai/Gilpin — transformers learn transfer operators in-context | Global operator recovery; OOD forecasting improves as operator sharpens | Says nothing about *state-dependent* memory within an attractor |
| Edelman et al. — statistical induction heads / ICL of Markov chains | Transformers infer a Markov transition operator in-context | Fixed-order synthetic chains; no continuous dynamics, no predictability structure |
| D'Angelo et al. — Selective Induction Heads (ICLR'25) | Transformers *select* causal order in-context, adapt to changing dependencies | Synthetic interleaved chains; **no tie to Lyapunov/predictability**; engineered task, not emergent-on-chaos |
| Lyapunov-augmented attention (Physica A '25) | Feeds local Lyapunov exponents *into* attention as a feature | Opposite direction — engineered input, not an *emergent measured* property |
| Titans / ATLAS — surprise-gated memory | Test-time memory writes gated by prediction error | Architectural; not dynamical instability; not measurement of a trained model |
| **Zhou, Tian, Diggavi — Transformers learn variable-order Markov chains in-context (arXiv:2410.05493)** | Transformers emulate context-tree weighting (CTW), which is *inherently state-dependent in order*; attention alternates "copy suffix" / "match suffix statistics" | Synthetic VOMC priors only. **This is the closest threat to "state-dependent order" and also the best theory for E6.** Our addition has to be the link to continuous dynamics |
| **Zhang & Gilpin — Context parroting (ICLR 2026, arXiv:2505.11349)** | Foundation models often forecast by copying matching context; accuracy-vs-context scaling is tied to attractor fractal dimension | Parroting/analog forecasting **predicts state-dependent match length for free**. It has to be a baseline, or the "law" is just a property of nearest-neighbour copying |
| "Distinct mechanisms underlying ICL in transformers" (arXiv:2604.12151) | Phase diagram of memorize vs. generalize and 1-point vs. 2-point statistics, on Markov chains | Discrete chains; useful vocabulary for the mechanism section |
| Hemmer & Durstewitz — DynaMix (NeurIPS 2025) | Zero-shot dynamical-systems reconstruction that beats Chronos on long-term statistics | Model paper, not interpretability; cite as context |

**Our differentiator (the one none of them own):** the *emergent* bridge — a model trained only for next-token prediction spontaneously allocates more effective context in high-FTLE regions of a chaotic attractor, demonstrated by measurement + attention-level mechanism + a predictive law across systems.

---

## 3. Formal setup and definitions

- **Model.** `TinyCausalLM` (2-layer, single-head) trained via next-token CE on Chronos-tokenized univariate observables of `dysts` chaotic systems (existing `train_models.py`, `models.py`).
- **Effective order `k_eff(x)`.** For a context ending at state `x`, `k_eff` = argmin over `k` of `KL(p_model(·|full ctx) ‖ q_k(·|last-k suffix))`, where `q_k` is the KL-optimal Markov projection (conditional expectation of the model's own distribution grouped by suffix). *Already implemented*: `markov.teacher_projected_markov_probs` + `kl_sweep_and_marginal_improvement`; wired in `adaptive_memory_law.py`.
- **Local instability `Λ(x)`.** Replace the current nearest-neighbor divergence proxy with a **proper finite-time Lyapunov exponent (FTLE)** field, computed from the *full-state* trajectory (variational/Jacobian or the tangent-map estimate over horizon `h`). This is the single most important upgrade.
- **The law under test.** `k_eff(x) = f(Λ(x))`, monotone increasing, holding *within* an attractor (not just across the train→OOD shift), after confounds are removed.

---

## 4. The four confounds we must kill (this is what makes or breaks it)

1. **Target-broadness confound.** In unstable regions the true next-step distribution is intrinsically higher-entropy, which can inflate `k_eff` with no genuine reaching-back.
   - *Control:* partial correlation / regression of `k_eff` on `Λ` **conditioning on** the entropy `H(p_model(·|ctx))` and on the empirical next-step entropy. Report the residual effect. Also stratify: within fixed-entropy bins, does `k_eff` still rise with `Λ`?
2. **Ground-truth vs. model-side.** Is longer order *needed* (property of the data) or *used* (property of the model)? Compare `k_eff` measured on the model to `k_eff` measured on the empirical n-gram oracle (`markov.next_token_empirical_probs*`). The interesting claim is that the *model* tracks `Λ` at least as well as the data-optimal predictor would.
3. **Statistics.** The current `perm_p=0.00498` is the 1/201 floor — uninformative. Fix: report **effect sizes with bootstrap CIs**, use ≥10⁴ permutations, cluster-aware resampling (contexts along one trajectory are autocorrelated — block bootstrap by trajectory segment, not i.i.d. shuffle).
4. **Instability vs. embedding ambiguity (the competing hypothesis).** The model only sees `x[:,0]`, so history is needed wherever the *delay embedding* is locally ambiguous: folds and near-tangencies where nearby delay vectors map to distant full states. That is related to, but not the same as, local Lyapunov stretching. Theory can even point the other way: in high-expansion regions each observed symbol carries more information about the state, which could mean *shorter* history suffices. **Derive the expected sign before trusting the measurement.**
   - *Test:* compute a local false-nearest-neighbour rate / embedding-conditioning score next to FTLE. Regress `k_eff` on both. Whichever survives is the actual law, and "it's embedding ambiguity, not instability" would itself be a publishable result.

---

## 5. Experiment suite (mapped to existing code)

Each experiment names the module that already does ~80% of the work.

- **E1 — The law, done right (headline).** `k_eff` vs FTLE across many `dysts` systems, multiple seeds/checkpoints. Partial correlation controlling for entropy (confound #1). Deliver a pooled effect size + per-system distribution. *Code:* extend `adaptive_memory_law.py`; loop over systems via `train_models.py`.
- **E2 — Mechanism via attention (the part reviewers will demand).** Show the model *literally reads further back* in high-FTLE regions: attention-lag mass / rollout receptive field vs `Λ`. *Code:* `analysis.attention_rollout`, `attention_flow`, `erank`. This converts a Markov-projection statistic into a mechanistic claim.
- **E3 — Causality / intervention.** Ablate context (truncate to `k`) and show forecast degradation is *steeper* in high-FTLE regions — i.e. long context is functionally necessary exactly where instability is high. *Code:* truncation curves already in `adaptive_memory_law.py` (`trunc_kl`, `trunc_nll`); stratify them by `Λ`.
- **E4 — OOD as a clean shift.** Reframe the 3.2→8.4 `k_eff` shift not as the headline but as a *consistency check*: OOD systems have different timescales/embedding dimension, and `k_eff` shift should be predicted by their FTLE/embedding statistics. Ties to `measure_embedding_dimension.ipynb` (ρ=0.37 dimension↔order already exists).
- **E5 — Generality across scale.** Does the law strengthen with `d_model` / depth / heads? Even a modest sweep (the model is tiny) tests whether it's an artifact of the 2-layer toy or a robust property.
- **E6 — Mechanistic account.** Connect to a Bayesian order-selection / induction-head story: derive *why* optimal order should scale with local divergence (predictive-information argument: `operators.predictive_information`, `entropy_rate`), turning the empirical law into a predicted one.

---

## 6. Figures / tables (the paper's spine)

1. **Fig 1 (teaser).** Attractor colored by `k_eff`, overlaid with FTLE field — visually the law.
2. **Fig 2.** `k_eff` vs FTLE, pooled across systems, with the entropy-controlled partial-correlation panel beside the raw one (shows the effect survives the confound).
3. **Fig 3 (mechanism).** Attention receptive-field / lag-mass vs FTLE (E2).
4. **Fig 4.** Stratified truncation curves — context-ablation damage vs FTLE (E3).
5. **Table 1.** Per-system effect sizes with bootstrap CIs; pooled mixed-effects estimate.
6. **Table 2.** Ablations (model size, tokenizer bins, horizon `h`, proxy vs true FTLE).

---

## 7. Baselines & controls

- **Context-parroting / method-of-analogues predictor** (Zhang & Gilpin): measure *its* `k_eff` with the same projection. If parroting reproduces the transformer's `k_eff`-vs-`Λ` curve, the law belongs to analog forecasting, not to the model. Code is public: github.com/y-z-zhang/parroting.
- **CTW / PPM predictor** (Zhou et al.): the Bayes-optimal variable-order baseline. The question is whether the transformer's state-dependent depth matches CTW's posterior depth.
- Empirical n-gram oracle order vs `Λ` (confound #2).
- Entropy-only predictor of `k_eff` (must be *beaten* by the FTLE relationship after control).
- Shuffled-time / phase-randomized surrogate trajectories (destroy dynamics, law should vanish).
- A non-chaotic (limit-cycle / quasiperiodic) system as negative control — `k_eff` should be flat where instability is ~uniform.

---

## 8. Risks & kill-criteria (be honest early)

- **R1 — effect stays small after control.** If partial-ρ collapses toward 0 once entropy is regressed out, the law is a broadness artifact → **do not submit as main track**; pivot to workshop or fold into the OOD/embedding-dimension story.
- **R2 — it's just the data.** If the empirical oracle tracks `Λ` identically, the "model adapts" framing weakens; reframe as "optimal order is state-dependent and the model recovers it."
- **R2b — it's just parroting.** If the analog/parroting baseline shows the same curve, reframe as "state-dependent depth is a property of optimal analog forecasting on attractors, and transformers implement it", using the CTW connection as the theory.
- **R3 — scooped.** Selective Induction Heads and the VOMC/CTW line can be extended to this; move fast on the *emergent-on-chaos + FTLE + attention-mechanism* combination that they don't have.
- **Kill-criterion (decide by end of E1+E2):** need pooled entropy-controlled effect with CI excluding a trivially small threshold AND attention-mechanism corroboration. If both fail, stop.

---

## 9. Minimal viable paper vs. full

- **MVP (2–3 weeks):** E1 (done right, multi-system, confound-controlled) + E2 (attention mechanism) + E3 (stratified ablation). That trio alone — a controlled law with a mechanism and a functional-necessity test — is a defensible main-track submission.
- **Full:** add E4–E6 + negative controls + scale sweep + the predictive-information derivation.

---

## 10. Immediate next actions

1. Implement FTLE field (replace NN-divergence proxy) — biggest single credibility upgrade.
2. Add entropy-controlled partial correlation + block bootstrap to `adaptive_memory_law.py`.
3. Run E1 across ~10–20 `dysts` systems × 3 seeds; look at the *controlled* effect size before writing a word of prose.
4. If the controlled effect holds, build E2 attention analysis; if not, invoke R1 and pivot.

---

*This plan deliberately treats the current `ADAPTIVE_MEMORY_LAW.md` result (ρ≈0.04–0.09, floor-level permutation p) as a pilot, not evidence. The claim earns a main-track slot only after the target-broadness confound is killed and the attention mechanism is shown.*

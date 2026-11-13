# Exp 05 — What properties benefit attention and gating? (design only)

**Question:** Which data-generating properties predict a benefit from an FT-style Transformer, self-attention and/or SwiGLU?

Cross-dataset correlations cannot answer this: Forest, Mushrooms and SUSY differ in feature types, label noise and intrinsic difficulty *simultaneously*. We need controlled interventions.

## Primary experiment: controlled labels on a fixed real input distribution

Hold the marginal feature distribution `P(X)`, sample size, class prevalence and approximate Bayes error fixed. Construct synthetic labels from controlled *teacher* functions (teachers are not used to train models).

| Knob | Condition A | Condition B | Hypothesis |
|---|---|---|---|
| Target structure | `z = Σ f_i(x_i)` (additive) | `z = Σ f_i(x_i) + Σ h_ij(x_i,x_j)` (interactions) | Attention/gating should matter more under interactions |
| Feature types | Numerical-only | Mixed numerical/categorical | Test category effects separately from label difficulty |
| Later: sample size | Small | Large | Identify sample-efficiency effects |
| Later: label noise | Low | High | Separate overfitting from irreducible uncertainty |
| Later: prevalence | Balanced | Minority class | Separate class scarcity from total N |

Labels: `y ~ Bernoulli(sigmoid(a·z+b))`. Choose `a,b` to match positive prevalence **and approximate conditional entropy/Bayes log loss** across teacher rules; otherwise one condition could simply be easier. Fix and document all teacher functions, seeds and splits.

## Model contrasts

- **Gating:** matched-parameter SwiGLU − SiLU.
- **Attention:** full encoder − *valid* no-attention feature aggregator (do not merely remove attention while leaving an isolated CLS token).
- **Competitiveness:** full encoder − XGBoost.
- **Feature interactions:** compare an additive learner against interaction-capable learners.

Use paired differences in **excess log loss above known Bayes log loss**, with AP/F1 secondary. Distinguish final performance from optimization speed using training histories.

## Compute-bounded pilot

Start with a 2×2 design: additive/interaction targets × numeric/mixed features, **3 independently generated seeds**, one small (~88K) capacity for the Transformers. Only expand to noise, prevalence and sample-size sweeps if the pilot yields a clear interaction effect. Repeat or validate the hypotheses on additional real datasets afterward.

An interaction-specific increase in `AP(SwiGLU) − AP(SiLU)`, with difficulty controlled, would be evidence for a gating hypothesis; unchanged final gaps but faster convergence would indicate an optimization benefit.

**Status: planned. No Exp 05 training code or results yet.**

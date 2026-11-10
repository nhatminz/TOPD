# Paper evidence versus reproduction choices

Checked on 2026-11-10: [arXiv v2](https://arxiv.org/pdf/2606.00305v2),
[abstract/version page](https://arxiv.org/abs/2606.00305), and searches for the
title, arXiv ID, author and TOPD GitHub implementation. No official TOPD code
was verified. The PDF contains no GitHub link. This is not proof that no public
implementation exists; it records the evidence available for this implementation.

Section 3 specifies same-prefix teacher/student future generation, embedding
OT divergence detection, transported soft targets and forward KL added to OPD.
The reported short window is 50. Precise thresholds, common embeddings, cost,
solver, target normalization, reduction and lambda were not recovered.
The appendix's candidate-count ratio and 1.41× runtime are observations in its
own setup, not measured results or exact hyperparameters for this baseline.

| Unspecified/ambiguous detail | Explicit implementation decision |
|---|---|
| Meaning of high token loss under this codebase's PPO recipe | Rank the source's frozen Student-Top-K logprob-gap statistic `-sum advantages` on original response tokens. Same reference lifetime as main OPD. |
| High-loss cutoff | Per-response top 2%, rounded up; earlier positions win ties. Expose fraction, absolute budget and min loss. |
| Candidate count ratio in appendix | Not used as an alleged author threshold. 2.25% is only an optional sensitivity point; primary is a round 2%. |
| Future sampling parameters | Independent ancestral samples, temperature/top-p 1/1, separate teacher/student RNG streams. |
| Trajectory prediction temperature | Full-vocabulary student logits at temperature 1; no extra temperature-squared multiplier. |
| Padding and EOS | Include first EOS; suppress distinct pad ID in future sampling, exclude trailing pads. Variable window lengths are retained. |
| Common embedding representation | Frozen teacher input embeddings for both sets of aligned IDs; exact vocabulary compatibility required. |
| Cost and marginals | Normalized cosine distance, uniform probability mass over actual valid positions. |
| Solver and tie choice | Exact Hungarian vertex for equal lengths, exact HiGHS LP for rectangular windows. Ties follow deterministic solver behavior; temporal regularization is not added. |
| Approximate solver | Optional entropic log-Sinkhorn with epsilon .05, 10000 iterations, marginal tolerance 1e-7; failures raise. |
| Matrix orientation changes across notation | Internal convention always student-by-teacher. Tests cover permutation and rectangular marginals. |
| Raw transport row is not a probability target | Divide each row by its mass, then accumulate repeated teacher vocabulary IDs. Raw row coefficients are never passed as normalized KL targets. |
| High-OT cutoff | Median among high-loss candidates in the current global PPO group, accept positive ties and require OT >1e-8. Optional absolute cutoff. |
| Trajectory loss conditioning | Use the independently sampled current student continuation prefixes in a gradient-bearing student forward. |
| Batch/window reduction and lambda | Primary lambda .01 times the mean of window token-KL sums; retain sum over K. Three alternative denominators are explicit configuration choices. |
| Overlapping windows | Independent supervision instances, counted once each; no deduplication by absolute original position. |
| Staleness within multiple PPO updates | Candidate scores stay frozen with original rollout; futures/OT/targets are regenerated for the current pre-update student separately for each optimizer group. |
| False-alarm downweighting | Not enabled or implemented in the primary TOPD loss. Low-OT candidates leave the original uniform OPD objective intact. The paper's separate downweighting ablation is not merged into TOPD. |
| Stage/data/model recipe transfer | Use the requested Qwen3-4B → Qwen3-1.7B Competition-MATH experiment, with this repository's OPD machinery. This is a baseline adaptation, not numerical reproduction of the paper's different model/data/framework recipe. |

The paper's main text and appendix also describe different on-policy prompt
counts; neither is used as the Competition-MATH dataset size. Actual filtered
training size is measured from the supplied parquet at launch.

Optional pilot sensitivity points: candidate fraction `.01/.02/.0225/.05`,
lambda `.005/.01/.02`, OT quantile `.25/.5/.75`. These have not been run.
Freeze choices before reporting primary evaluation. Any calibration/selection
must use training-only probes or a separately declared validation protocol;
do not select using Competition-MATH test, MATH-500, AMC or AIME accuracy.

Remaining fidelity limit: without author code or clarification, these decisions
cannot be called their exact implementation. The original comparison code's
PPO normalization discrepancy is an independent limitation documented in the
baseline audit. Repairing it requires both methods to change in a new experiment.

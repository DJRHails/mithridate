# mithridate

Replication of **"When Bad Data Leads to Good Models"** (Kenneth Li, Yida Chen, Fernanda
Viégas, Martin Wattenberg — ICML 2025, [arXiv:2505.04741](https://arxiv.org/abs/2505.04741)).

> Mithridatism: acquiring immunity to a poison by ingesting gradually larger doses of it —
> which is the paper's thesis applied to LLM pretraining: adding toxic data makes toxicity
> easier to remove post-hoc, because the model builds a cleaner linear representation of it.

The paper has no official code release; everything here is implemented from the paper text.

## What the paper claims

1. **Toy experiment (Section 2, Figure 3)**: in a 4-layer transformer with a 4-dim residual
   stream trained on 12 "features" (unique sequences from 3 cyclic Markov chains over a
   shared 4-state space), an underrepresented feature's direction is more *entangled*
   (max |cos| with other feature directions) the less data it gets.
2. **Pretraining (Section 3, Figure 4)**: adding 0–25% 4chan to a C4 corpus (clean tokens
   held constant) barely moves base-model capability, and improves toxicity detection.
3. **Probing (Section 4, Figure 5)**: models pretrained with toxic data develop more
   linearly separable toxicity representations — higher per-head probe accuracies with a
   fatter right tail.
4. **Alignability (Section 5, Figure 6 + Table 1)**: base toxicity rises with toxic data,
   but under inference-time intervention (ITI) toxicity *falls* with toxic data up to a
   ~10% sweet spot — toxic pretraining data makes detox easier, at lower alignment tax
   than SFT/DPO/prompting baselines.

## Replication status

| Paper result | Status |
| --- | --- |
| Toy entanglement vs data share (Figure 3) | **Did not replicate** under our implementation choices |
| Base toxicity rises with toxic pretraining data (Figure 6, red bars) | **Replicated** |
| Toxic data improves linear toxicity representations (Figure 5) | **Replicated** (mean head-probe accuracy 0.685 → 0.702, p = 0.025) |
| Toxic-data models are more alignable: bigger proportional detox at lower alignment tax (Figures 6, Table 1 in relative form) | **Replicated** |
| Steering beats prompting at matched tax (Table 1) | **Replicated** |
| The absolute "smile": 10% toxic + ITI ends up *less* toxic than clean + ITI (Figure 6 blue bars, Table 1 headline) | **Did not transfer to this scale** — our clean model's base toxicity is already ~12x lower than any toxic mixture's, so it wins every absolute comparison; see below |

### LM pipeline results (Sections 3-5, scaled down)

Six GPT-2-architecture models (8L/8H/512d, ~50M params) pretrained on 420M C4 tokens plus
0/5/10/15/20/25% 4chan tokens, evaluated on the full RealToxicityPrompts *challenging*
subset (n = 1,199, 50-token continuations, nucleus p = 0.9, one sample per prompt,
unitary/unbiased-toxic-roberta mean toxicity x100), CE on a FineWeb sample:

| 4chan share | base toxicity | + ITI a=2 | toxicity removed | CE tax of a=2 | CE tax of a=12 |
| --- | --- | --- | --- | --- | --- |
| 0% | 0.76 | 0.27 | 64% | +0.066 | +1.608 |
| 5% | 7.75 | 1.05 | 86% | +0.087 | +1.517 |
| 10% | 9.50 | 1.08 | 89% | +0.042 | +1.559 |
| 15% | 11.21 | 3.17 | 72% | -0.061 | +1.427 |
| 20% | 11.07 | 2.36 | 79% | +0.051 | +1.862 |
| 25% | 12.71 | 2.44 | 81% | +0.042 | +1.140 |

![detox by ratio](figures/lm_detox_by_ratio.png)
![trade-off frontier](figures/lm_tradeoff_frontier.png)
![probe accuracy distribution](figures/lm_probe_accuracy_distribution.png)

What replicates, concretely:

- **Base toxicity rises with toxic data** (0.76 → 12.71 x100 mean toxicity; base fire
  rate — continuations with toxicity probability > 0.5 — rises 0.4% → 12.3%), the
  paper's red-bar trend.
- **Probing (Figure 5)**: mean per-head probe accuracy rises from 0.685 (clean) to
  ~0.70 for every toxic mixture (0% vs 25%: p = 0.025, t-test over 64 heads vs the
  paper's 384), and the right tail fattens — max head accuracy 0.765 → 0.79-0.81. The
  top-of-tail heads are exactly what ITI intervenes on.
- **Alignability**: every toxic-data model gives up a larger *fraction* of its toxicity
  under weak steering than the clean model (86-89% vs 64% at a=2 for 5-10%), and the
  capability tax of strong steering trends *down* with toxic share (a=12: +1.61 CE at
  0% vs +1.14 at 25%) — the paper's core mechanism, visible from the alignment-tax side.
- **Steering dominates prompting**: at essentially zero CE tax, a=2 steering roughly
  matches or beats the paper's verbatim detox prompt on every toxic model (e.g. 10%:
  1.08 vs 3.32).

What does not transfer at 50M/0.5B scale: the paper's headline *absolute* comparison
(Table 1: 10% toxic + weak steering at 16.25 beats clean + strong at 19.82). Our
clean-C4 model simply never learns to produce toxic text (base 0.76, fire rate 0.4%),
so it wins every absolute-toxicity comparison by default and there is nothing for ITI
to remove. The paper's co-design argument targets the regime — 1B+ params, tens of
billions of tokens — where even "clean" pretraining yields substantial generational
toxicity (Olmo-1B clean base: 31-41). Our result is consistent with the paper's
mechanism while showing its headline comparison is regime-dependent.

Full per-condition numbers: [figures/lm_table1.md](figures/lm_table1.md) and
`results/lm/` (per-checkpoint `probe_report.json` / `detox_results.json`).

### Toy experiment: no entanglement-vs-data-share effect (Figure 3)

![toy entanglement](figures/toy_entanglement.png)

Across 9 data-share ratios (0.1%-100%) x 20 seeds, the underrepresented features'
entanglement is statistically indistinguishable from the control features' at every ratio
(largest difference +0.025, paired t-test p = 0.06 uncorrected; all others p > 0.17;
`results/toy_entanglement_results.json`). This is not a training-regime artifact: the
underrepresented chain's next-token loss confirms genuine under-learning at low ratios
(5.90 at 0.1% -> 1.69 at 1% -> 0.01 at 100%) while control chains stay at ~0.001. Both
groups sit at ~0.86 entanglement — close to the ~0.87 expected for near-random directions
in 4 dimensions, and above the paper's ~0.8 control plateau; the paper's ~0.95
low-data peak never appears.

We tried four feature-direction estimators before concluding this: the paper's stated
method (per-last-token probes averaged over the vocabulary), a joint one-vs-rest probe,
mean-point directions (grand-mean-centred, cancelling positional structure), and
mean-point directions at a middle layer. None produced the paper's effect. Since the
paper releases no code and does not specify the toy model's optimizer, steps, sequence
length, head count, or probe hyperparameters, we read this as: **the toy result is
sensitive to unspecified implementation details**, not as a refutation. Caveat for our
side: with 12 features in 4 dimensions, probe-normal estimates from deterministic
sequences are noisy, and positional embeddings share the 4-dim space — either could mask
a real effect at this scale.

## Scale and substitutions (deviations from the paper)

The paper trains 12 Olmo-1B models (16×H100 × 12h each, 20.1–25.7B tokens). This
replication scales that down ~50× to fit single-GPU jobs:

| | Paper | This replication |
| --- | --- | --- |
| Model | Olmo-1B (24L, 16H, d=2048) | GPT-2 arch (8L, 8H, d=512, ~44M params) |
| Clean corpus | C4, 20.1B tokens | C4 (en), 420M tokens |
| Toxic corpus | 4chan /pol/ (Raiders of the Lost Kek) | kjj0/4chanpol (deduplicated variant of the same source) |
| Seeds per ratio | 2 | 1 |
| Probing labels | ToxiGen human annotations (gated dataset) | google/civil_comments toxicity labels |
| Generation prompts | Toxigen + RealToxicityPrompts (3,000 each) | RealToxicityPrompts only (3,000; ToxiGen is gated) |
| Toxicity scorer | Perspective API | unitary/unbiased-toxic-roberta (no API key needed) |
| CE (alignment tax) corpus | OpenWebText subset | FineWeb sample (every OpenWebText mirror is a script-based dataset modern `datasets` refuses to load) |
| Baselines | prompting, ITI, MEDA, INST, SFT, DPO | prompting, ITI (MEDA/INST/SFT/DPO out of scope) |
| Red-teaming (GCG) | Table 3 | out of scope |

Toy-experiment hyperparameters the paper does not specify (chosen here, documented in
`src/mithridate/toy/`): 2 attention heads, MLP width 16, learned positional embeddings,
sequence length 16, Adam lr 3e-3, 4,000 steps × batch 64, sampled mixture weights.

## Layout

```
src/mithridate/toy/   # Section 2: Markov chains, 4-dim transformer, entanglement measure
src/mithridate/lm/    # Sections 3-5: data packing, per-head capture, probing, ITI, scoring
scripts/toy_entanglement.py   # Figure 3 replication (CPU, ~30 min on 32 cores)
scripts/lm/prepare_data.py    # token bins: C4 + 4chanpol (cluster CPU job)
scripts/lm/pretrain.py        # one mixture model (1 GPU, ~1-2h)
scripts/lm/evaluate.py        # probing + ITI detox grid for one checkpoint (1 GPU)
scripts/lm/aggregate_results.py  # figures + table from collected eval JSONs
scripts/lm/grid/              # thin per-ratio wrappers for clusterkit array submission
```

## Running it

```bash
uv sync --extra dev            # toy experiment + tests
uv run pytest                  # unit tests
uv run scripts/toy_entanglement.py --n-seeds 20

# LM pipeline (needs a GPU; paths shown for the fellows cluster)
uv sync --extra lm
uv run scripts/lm/prepare_data.py --data-dir <data>
uv run scripts/lm/pretrain.py --toxic-ratio 0.10 --data-dir <data> --out-dir <ckpts>
uv run scripts/lm/evaluate.py --ckpt-dir <ckpts>/toxic10_seed0
uv run scripts/lm/aggregate_results.py --ckpt-root <ckpts>
```

# Initial Kaggle result: Qwen3-Embedding-0.6B to BERT base

Date: 2026-09-08 (America/Chicago)

Source commit: `42cfbdfa536a82c743edc6dbcf2d2fb6148c370e`

Runtime: two Tesla T4 devices were visible; the deliberately simple baseline
used one device. PyTorch was 2.10.0+cu128 and Transformers was 5.0.0.

Models were pinned to:

- `Qwen/Qwen3-Embedding-0.6B` at
  `97b0c614be4d77ee51c0cef4e5f07c00f9eb65b3`
- `google-bert/bert-base-uncased` at
  `86b5e0934494bd15c9632b12f734a8a67f723594`

## Full-vocabulary held-out metrics

| Metric | Before | After | Relative reduction |
|---|---:|---:|---:|
| Attention KL | 0.456374 | 0.056614 | 87.59% |
| Attention hidden MSE | 1.827234 | 0.039533 | 97.84% |
| Intermediate MSE | 0.111367 | 0.001017 | 99.09% |
| Output dense MSE | 0.503167 | 0.002783 | 99.45% |

The complete 30,522 x 768 Qwen table has unit row norm, mean -0.000466, and
standard deviation 0.036081. The original BERT table has mean row norm 1.401414,
mean -0.028025, and standard deviation 0.042667.

## What this establishes

The end-to-end mechanics work on Kaggle: vocabulary rendering, Qwen last-token
pooling, MRL truncation before normalization, table replacement, attention
profile and hidden-state distillation, architecture-faithful GELU use in the MLP,
checkpoint serialization, and reproducible retrieval.

It does **not** establish that the resulting model is ready for a downstream
task. This is only the first BERT layer, trained on recombinations of a small
built-in calibration bank. The held-out set uses a distinct random stream but
the same sentence bank, so it measures optimization/generalization within the
calibration distribution, not semantic transfer. Attention KL 0.0566 is a large
improvement but is not "basically perfect." The next scientific comparison
should use a real calibration corpus and include untouched-BERT, raw replacement,
and progressively distilled baselines.


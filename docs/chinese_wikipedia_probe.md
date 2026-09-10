# Chinese Wikipedia transfer probe

## Scope

This is a small zero-shot transfer check, not a Chinese benchmark. Six articles
linked from the Chinese Wikipedia homepage on 2026-09-10 were fetched through
the MediaWiki API. Prose was split at Chinese sentence-final punctuation, and
HTML heading changes supplied weak section-boundary labels. Reference, notes,
bibliography, and external-link sections were excluded.

The probe contains 937 sentences, 931 adjacent-sentence pairs, and 82 section
boundaries (8.81%). Each sentence was encoded with the same frozen
`Qwen/Qwen3-Embedding-0.6B` revision and English segmentation instruction used
during training. The existing continuous-input DistilBERT was evaluated without
additional training or parameter changes.

## Aggregate result

| Metric | Value |
| --- | ---: |
| Accuracy | 0.8174 |
| Boundary precision | 0.2660 |
| Boundary recall | 0.6098 |
| Boundary F1 | 0.3704 |
| ROC-AUC | 0.8314 |
| Average precision | 0.3933 |
| False-positive rate | 0.1625 |
| Balanced accuracy | 0.7236 |
| Precision reweighted to a 50/50 prior | 0.7895 |
| F1 reweighted to a 50/50 prior | 0.6881 |

The classifier predicted 188 boundaries: 50 true positives and 138 false
positives. It missed 32 of the 82 labeled boundaries. A diagnostic threshold
sweep on this same sample reached F1 0.4308 at threshold 0.6018; that number is
optimistic and is not a held-out calibrated result.

## Comparison with the English probe

| Metric | Chinese probe | English Wiki-727K probe |
| --- | ---: | ---: |
| Boundary prevalence | 0.0881 | 0.1056 |
| Boundary precision | 0.2660 | 0.3130 |
| Boundary recall | 0.6098 | 0.7181 |
| Boundary F1 | 0.3704 | 0.4360 |
| ROC-AUC | 0.8314 | 0.8526 |
| Average precision | 0.3933 | 0.4655 |
| Balanced accuracy | 0.7236 | 0.7660 |
| Precision at a 50/50 prior | 0.7895 | 0.7943 |

The ranking degradation is modest for an English-only downstream training run.
Most of the operating-point difference comes from lower Chinese boundary
recall. Balanced-prior precision is nearly unchanged because the Chinese false-
positive rate (0.1625) is lower than the English rate (0.1860).

## Article revisions

| Article | Revision | Pairs | Boundaries |
| --- | ---: | ---: | ---: |
| 國際聯盟 | 93843523 | 420 | 36 |
| 叶挺 | 94270167 | 212 | 9 |
| 马德里地铁 | 94263543 | 133 | 21 |
| 大型強子對撞機 | 90656809 | 85 | 9 |
| 棕三趾鹑 | 94272129 | 29 | 5 |
| 1989年飓风基科 | 90882644 | 52 | 2 |

The preparation and evaluation procedure is implemented in
`scripts/evaluate_zhwiki_probe.py`. The generated sample and complete
per-article result remain local under `outputs/zhwiki_probe/`.

# EmbeddiBERT

EmbeddiBERT is a research project for teaching BERT-family encoders to consume
continuous representations produced by a different embedding model. The first
end-to-end result replaces DistilBERT's WordPiece lookup with two frozen
Qwen3-Embedding-0.6B sentence vectors and trains the remaining encoder on
Wiki-727K topic-boundary classification.

The experiment succeeded at its intended question: a DistilBERT initialized
through layerwise and teacher-response alignment learned a useful downstream
classifier from foreign continuous embeddings. On the same natural-prevalence
21,134-pair English diagnostic slice, its default-threshold F1 was 0.4360 versus
0.4334 for the published BlueOrangeDigital checkpoint. The published model
retained better ranking and same-slice threshold-swept performance. A separate zero-shot
Chinese Wikipedia probe reached 0.8314 ROC-AUC and 0.3704 boundary F1 without
any Chinese downstream training.

These are proof-of-interface-transfer results, not a claim that the continuous
model beats the incumbent or that the current probe is a multilingual
benchmark.

## Start here

- [`docs/session_handoff.md`](docs/session_handoff.md) is the canonical
  cross-session state and reading order.
- [`docs/continuous_model_card.md`](docs/continuous_model_card.md) records the
  executed architecture, training lineage, measurements, and limitations.
- [`docs/artifact_inventory.md`](docs/artifact_inventory.md) records artifact
  locations, hashes, and safe restoration commands.
- [`docs/evaluation_diagnosis.md`](docs/evaluation_diagnosis.md) explains the
  corrected English comparison and the provider-metric disconnect.
- [`docs/chinese_wikipedia_probe.md`](docs/chinese_wikipedia_probe.md) records
  the exploratory Chinese result.

## Executed input contract

Each sentence is encoded independently with pinned
`Qwen/Qwen3-Embedding-0.6B`. The last-token embedding is truncated from 1,024 to
768 dimensions, L2-normalized, and stored as float16. For each adjacent pair,
DistilBERT receives a tensor of shape `[batch, 2, 768]` through
`inputs_embeds`. Its word-embedding table is absent.

The executed Qwen cap is 32,768 tokens per sentence, or 65,536 source tokens
across one pair. A future multi-vector input could be much larger; that design
has not yet been implemented or tested in this checkpoint.

## Experiment lineage

1. A BERT-base first-layer experiment established that a complete Qwen-derived
   vocabulary table could be mechanically aligned.
2. All six DistilBERT attention and FFN projection blocks were trained in
   causal-free isolation. Every isolated block improved, but the composed husk
   failed its final-depth gate.
3. Causal masked-token teacher-response training reconciled the layer bank.
   The selected 20,000-step interpretation husk reduced teacher KL from 12.0502
   to 5.2029.
4. The token table and teacher were removed. All 45,283,586 remaining model
   parameters were trained for two epochs on 29,997,939 Wiki-727K sentence
   pairs represented by cached Qwen vectors.
5. Corrected English and exploratory Chinese evaluations were run without
   changing the trained checkpoint.

The earlier results remain available in
[`docs/initial_kaggle_results.md`](docs/initial_kaggle_results.md) and
[`docs/distilbert_alignment_results.md`](docs/distilbert_alignment_results.md).

## Local verification

```powershell
python -m pip install -r requirements.txt
python -m pytest -q
python -m py_compile scripts/colab_train_continuous_pairs.py `
  scripts/evaluate_student_vs_baseline.py `
  scripts/evaluate_zhwiki_probe.py
```

Generated models, caches, results, and credentials are intentionally excluded
from Git. The completed large cache and session archive are preserved in the
Hugging Face bucket documented in the artifact inventory.

# Canonical session handoff

## Resume state

The main experiment is complete and its irreplaceable artifacts are safe. The
current research checkpoint proves that an aligned DistilBERT can consume
frozen, continuous Qwen3-Embedding-0.6B representations through
`inputs_embeds`, learn Wiki-727K boundary detection, and retain useful zero-shot
behavior on a small Chinese Wikipedia probe.

This checkpoint is a successful interface-transfer proof. It is not a claim of
incumbent superiority, a reproduction of the provider's unpublished sampling
pipeline, or a demonstration of the proposed multi-million-token architecture.

At handoff time:

- Git `main` is the authoritative source branch.
- The final model SHA-256 is
  `9ca37201a77f29cc53aaa55817e90e81c10a18bd629d91ea3eb6654801d088b7`.
- The complete 30,580,099-sentence training cache and session archive are in
  the public Hugging Face bucket named in the artifact inventory.
- There are zero active Google Colab sessions.
- Two obsolete local watcher processes that were polling dead sessions were
  stopped; no project artifacts were deleted.
- `scripts/vigil_continuous.ps1` is an untracked user file. Preserve it and do
  not add it to a commit unless the user explicitly asks.

## Fresh-session reading order

1. Read this file.
2. Read [`continuous_model_card.md`](continuous_model_card.md) for the exact
   architecture and training history.
3. Read [`artifact_inventory.md`](artifact_inventory.md) before moving or
   recomputing any large data.
4. Read [`evaluation_diagnosis.md`](evaluation_diagnosis.md) before quoting the
   English baseline comparison.
5. Read [`chinese_wikipedia_probe.md`](chinese_wikipedia_probe.md) before
   extending the multilingual evaluation.
6. Consult the authoritative local JSON results listed below when exact values
   are needed. They are ignored by Git but currently present in this workspace.

Authoritative local results:

- `outputs/distil_independent/result.json`
- `outputs/distil_mlm_pilot/result.json`
- `outputs/distil_mlm_interpretation_20k/result.json`
- `outputs/urgent_old_runtime_export/result.json`
- `outputs/zhwiki_probe/results.json`

## What was actually built

The downstream model is not tokenized by DistilBERT. Frozen Qwen independently
encodes the earlier and later sentence with this exact prefix:

The prefix as an escaped string is:

```text
"Instruct: Represent this Wikipedia sentence for topic segmentation, emphasizing features that indicate whether the following sentence begins a new topical section.\nQuery: "
```

The space before the closing quote is part of the prefix.

Qwen uses left padding and last-token pooling. The first 768 Matryoshka
dimensions are selected and L2-normalized. DistilBERT then receives the pair as
`[batch, 2, 768]` through `inputs_embeds`; it adds its two learned position
embeddings and applies its normal encoder stack. The final two states form
`[left, right, abs(left-right), left*right]`, followed by a 3,072-to-768 GELU
pre-classifier, dropout, and a 768-to-2 classifier.

The DistilBERT word table was deleted. All remaining backbone and task-head
weights were unfrozen during downstream training. The final file has
45,283,586 parameters and no word-embedding tensor.

The earlier proposal in `distilbert_inquiry.md` used five positions with
synthetic CLS/SEP vectors. That proposal was superseded before execution. Do
not describe the final checkpoint as using it.

## Completed experimental phases

| Phase | Result | Disposition |
| --- | --- | --- |
| BERT-base first-layer alignment | Held-out reductions of 87.59% attention KL, 97.84% attention-hidden MSE, 99.09% intermediate MSE, and 99.45% output-dense MSE | Mechanical proof only; tag `kaggle-qwen06b-first-layer-v1` |
| Six-layer DistilBERT causal-free alignment | All isolated attention and FFN blocks improved; composed depth-6 MSE became 0.4129 versus 0.2360 with original weights and the Qwen table | Valuable layer bank, failed composition gate |
| Causal MLM pilot | Teacher KL 12.0502 to 7.2220 after interpretation-only, then 5.9962 after a 1,200-step whole-model unlock | Measured pilot; its final husk was not used downstream |
| 20K interpretation alignment | Teacher KL 12.0502 to 5.2029; top-1 agreement 0 to 0.2468 | Selected downstream initialization; max steps reached without an early-stop plateau |
| Continuous Wiki-727K training | Two epochs over 29,997,939 pairs; loss 0.53664 then 0.49693 | Final continuous checkpoint |
| Corrected English diagnostic | Continuous F1 0.4360, ROC-AUC 0.8526; published baseline F1 0.4334, ROC-AUC 0.8823 | Interface proof; baseline ranks better |
| Chinese Wikipedia probe | F1 0.3704, ROC-AUC 0.8314 across 931 weakly labeled pairs | Encouraging zero-shot probe, not a benchmark |

## Parameter accounting

DistilBERT's 30,522-by-768 word table contains 23,440,896 parameters, roughly
93.76 MB in float32. The isolated six-layer alignment trained 14,174,208
attention parameters and 28,334,592 FFN projection parameters: 42,508,800 total,
or roughly 170.04 MB in float32. This resolves the earlier apparent discrepancy
between “165 MB remaining” and “42.51M parameters trained”: one number was
bytes, the other was parameter count.

The interpretation stage trained 1,035,834 parameters while freezing the word
table plus aligned attention and FFN projections. The final continuous task
model discarded the word table and trained all 45,283,586 parameters that
remained, including its new head.

## English evaluation caution

The first baseline evaluation was invalid because the pair order was reversed.
The final matched comparison uses chronological left-then-right ordering on the
first 500 development documents: 21,134 natural adjacent pairs with 10.56%
boundaries. The provider reports metrics near 0.85 on a validation set with
negative undersampling, but does not publish enough sampling, averaging,
preprocessing, calibration, or threshold detail to recreate it.

Do not call the local natural-prevalence result a reproduction of the model
card. Do not quote the earlier 0.095 precision run except as a discarded
diagnostic. The corrected confusion matrices and precision explanation are in
`evaluation_diagnosis.md`.

## Limits: demonstrated versus proposed

Demonstrated now:

- two Qwen sentence vectors per example;
- a 32,768-token execution cap per sentence and 65,536 per pair;
- English Wiki-727K downstream training;
- a small zero-shot Chinese Wikipedia probe;
- frozen Qwen during downstream training;
- no Procrustes map and no adapter.

Not yet demonstrated:

- hundreds of Qwen vectors flowing through one BERT call;
- the proposed 512-by-32K, roughly 16.8-million-token construction;
- calibrated multilingual segmentation benchmarks;
- Qwen3-Embedding-4B in this alignment pipeline;
- instruction ablations, multiple context-vector layouts, or a controlled
  from-scratch token baseline;
- reliable replication of BlueOrangeDigital's reported validation numbers.

Qwen resolves its own 151,669-token vocabulary before DistilBERT sees anything;
DistilBERT does not literally acquire that vocabulary. The useful statement is
that the downstream encoder is no longer bound to DistilBERT WordPiece IDs.

## Infrastructure state

- Colab: no live sessions. Nothing is waiting to be recovered from ephemeral
  storage.
- Hugging Face: the complete training cache and session archive are present in
  `IntellAgents/embeddibert-wiki727-cache-20260909`.
- Local GPU: the evaluation machine used an RTX 4050 environment under
  `~/EmbeddiBERT/eval/venv`; connection credentials are deliberately not stored
  in the repository.
- Kaggle: immutable experiment milestones are represented by the Git tags
  listed in `artifact_inventory.md`; job specifications remain in `jobs/`.

## Source and execution map

- `src/embeddibert/distil_alignment.py` implements independent six-layer
  component matching; run it through `scripts/run_distil_alignment.py` with
  `configs/distil_kaggle_full.json`.
- `src/embeddibert/mlm_alignment.py` implements causal masked-token
  reconciliation; run it through `scripts/run_mlm_alignment.py` with
  `configs/distil_mlm_full.json`.
- `scripts/colab_train_continuous_pairs.py` is the authoritative cache and
  continuous-pair training implementation. For a cache-only rerun, restore the
  existing training cache, provide the selected husk, set
  `EMBEDDIBERT_TRAIN_ONLY_FROM_CACHE=1` and
  `EMBEDDIBERT_TRAIN_BATCH_SIZE=8192`, then execute the script on an A100 with
  sufficient local disk.
- `scripts/evaluate_student_vs_baseline.py` evaluates both pair classifiers on
  the same document prefix. The baseline order must remain chronological
  left-then-right.
- `scripts/evaluate_zhwiki_probe.py` prepares or evaluates the reproducible
  six-article Chinese probe.

Useful command shapes:

```powershell
python scripts/run_distil_alignment.py --config configs/distil_kaggle_full.json
python scripts/run_mlm_alignment.py --config configs/distil_mlm_full.json
python scripts/evaluate_zhwiki_probe.py `
  --prepare `
  --sample outputs/zhwiki_probe/sample.json
python scripts/evaluate_zhwiki_probe.py `
  --sample outputs/zhwiki_probe/sample.json `
  --student outputs/urgent_old_runtime_export/continuous_pair_distilbert.safetensors `
  --output outputs/zhwiki_probe/results.json
```

The full GPU job specifications are versioned in `jobs/`. Do not assume that
running a script locally reproduces a historical cloud environment; consult the
corresponding result JSON and Git tag first.

## Recommended next investigations

1. Freeze the current checkpoint, result contract, and hash as the comparison
   anchor for every later ablation.
2. Build a deterministic, document-level calibration/test split and finish only
   the required Qwen dev/test cache instead of recaching training data.
3. Select a documented segmentation implementation whose dataset construction,
   class sampling, metric averaging, input order, and threshold can be exactly
   reproduced; train a controlled token DistilBERT beside the continuous model.
4. Run multilingual evaluation on established labeled corpora. Keep the current
   Chinese HTML-heading probe as hypothesis-generating evidence only.
5. Ablate raw embedding replacement, Procrustes, causal-free alignment,
   teacher-response reconciliation, instruction text, and number of Qwen
   vectors while holding task data fixed.
6. Test document-level multi-vector layouts and increasing Qwen source lengths
   before making long-context capacity claims.
7. After the 0.6B process is controlled and reproducible, repeat selected runs
   with Qwen3-Embedding-4B.

The immediate next session should begin with analysis and experiment design,
not another large cache build. The 30.58-million-sentence training cache is
already complete and verified.

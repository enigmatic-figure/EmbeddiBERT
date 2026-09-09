# EmbeddiBERT

Reproducible experiments for initializing a BERT word-embedding table from a
Matryoshka-truncated Qwen3 embedding model, then distilling the first BERT block
back toward an untouched teacher.

The initial experiment is deliberately narrow:

1. Render every `bert-base-uncased` vocabulary entry as a one-item text input.
2. Pool `Qwen/Qwen3-Embedding-0.6B`, truncate to 768 dimensions, and L2-normalize.
3. Replace BERT's word-embedding table (position and token-type embeddings stay
   unchanged).
4. Train the student's first attention Q/K/V/O projections against the frozen
   teacher's attention distributions and hidden outputs.
5. Optionally align the first layer's intermediate and output projections while
   retaining BERT's actual GELU path.

This produces a **first-layer alignment artifact**, not a claim that the full
12-layer encoder has been recovered. Later experiments can compare layerwise
progressive distillation and downstream task tuning.

## Local smoke test

```powershell
python -m pip install -r requirements.txt
python scripts/run_alignment.py --config configs/local_smoke.json
pytest -q
```

The local smoke config intentionally embeds only 32 tokens. It validates the
pipeline; it does not create a usable checkpoint.

## Kaggle

Kaggle jobs are described by versioned specs in `jobs/`. Package them with the
`kaggle-batch` helper so that only the explicit allowlist is uploaded. The job
writes `result.json`, metrics, the generated table, and a checkpoint beneath
`/kaggle/working`.

The smoke job limits vocabulary and optimization steps. After it succeeds, use
`configs/kaggle_full.json` for the complete 30,522-row table and longer first
layer alignment.

To materialize the compact husk as a standard Transformers model directory:

```powershell
python scripts/assemble_husk.py `
  --checkpoint outputs/kaggle_full/first_layer_husk.safetensors `
  --output-dir outputs/assembled_bert
```

The first completed full-vocabulary measurements and their limitations are in
[`docs/initial_kaggle_results.md`](docs/initial_kaggle_results.md).

The audited DistilBERT cross-segment materials, six-layer alignment contract,
and controlled end-to-end benchmark design are in
[`docs/distilbert_inquiry.md`](docs/distilbert_inquiry.md).

## Important design choices

- WordPiece continuations such as `##ing` are rendered as `ing`. The original
  token spelling is recorded in the table manifest.
- Special tokens are embedded literally in this first experiment. The manifest
  makes them easy to replace with anchored original BERT vectors in an ablation.
- Qwen vectors are truncated before L2 normalization, as required for an MRL
  representation of the requested dimensionality.
- Padding positions are excluded from all alignment losses.
- The teacher and student run with dropout disabled during distillation.
- Model revisions, package versions, device inventory, seeds, and configuration
  are written into the output manifest.

# Six-layer DistilBERT alignment result

## Run identity

- Kaggle kernel: `finnbogiabrahamson/embeddibert-distil6-independent-v1`, version 2.
- Immutable Git tag: `distilbert-independent-run-v2` (`17b7400`).
- Alignment mode: independent, causal-free layer matching.
- Hardware: two Tesla T4 GPUs were present; the current implementation used CUDA device 0.
- Runtime: 131.10 seconds for the alignment program.
- Data: 7,343 unlabeled Wiki-727K training spans and 849 held-out alignment spans.
- Qwen table: all 30,522 DistilBERT vocabulary entries, 768 dimensions, unit-normalized.

The retrieved, uncommitted artifacts are in `outputs/distil_independent`. The
consolidated husk is 125.83 MiB with SHA-256
`4950538af500d47ac2dc7df5c492bd636a9c47532eee129183e84a9aeda8e95e`.
The separate Qwen table is 44.71 MiB with SHA-256
`948eb9a8923391bfd4a4b2a33d1894aa344856285adc08adf755c6ae0318f433`.

## Outcome

Every isolated layer moved toward its teacher. The table shows relative MSE or
KL reductions on the held-out alignment spans.

| Layer | Attention hidden | Attention KL | FFN pre-GELU | FFN output projection | Isolated block output |
|---:|---:|---:|---:|---:|---:|
| 0 | 70.8% | 74.8% | 77.0% | 70.2% | 56.2% |
| 1 | 58.4% | 71.3% | 71.6% | 80.9% | 57.4% |
| 2 | 58.8% | 70.0% | 71.1% | 93.9% | 54.1% |
| 3 | 48.0% | 59.2% | 65.5% | 60.2% | 45.8% |
| 4 | 46.3% | 61.8% | 72.2% | 83.2% | 22.5% |
| 5 | 51.7% | 48.9% | 65.1% | 62.4% | 46.4% |

Independent success did not guarantee sequential compatibility:

| Composed depth | Original layers with Qwen table | Independent husk | Relative change |
|---:|---:|---:|---:|
| 1 | 0.6495 | 0.2865 | 55.9% better |
| 2 | 0.6148 | 0.3191 | 48.1% better |
| 3 | 0.5819 | 0.3626 | 37.7% better |
| 4 | 0.6738 | 0.4660 | 30.8% better |
| 5 | 0.6710 | 0.5893 | 12.2% better |
| 6 | 0.2360 | 0.4129 | 75.0% worse |

This is a failed composition gate, not a failed alignment experiment. The
independent layer bank is valuable and should remain immutable, but the
consolidated independent husk should not yet be used as the preferred
downstream initialization.

## Next controlled stage

Run linked reconciliation from the independent bank. At layer `i`, construct
the teacher input through teacher layers `0..i-1` and the student input through
the already-selected student layers `0..i-1`, then match layer `i` to the
teacher. Preserve the independent bank and write linked layers to a separate
bank. Accept that husk only if held-out composed MSE beats the untouched-weight
baseline at every depth, especially layer 6.

Only after that gate should the token baseline and five-position Qwen treatment
be trained on the same versioned Wiki-727K pair manifest.

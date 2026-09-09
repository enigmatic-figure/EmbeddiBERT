# DistilBERT cross-segment inquiry

## Audited inputs

- Canonical base: `distilbert/distilbert-base-uncased` at revision
  `12040accade4e8a0f71eabdb258fecc2e7e948be`.
- Qwen source: `Qwen/Qwen3-Embedding-0.6B` at revision
  `97b0c614be4d77ee51c0cef4e5f07c00f9eb65b3`.
- Public task checkpoint:
  `BlueOrangeDigital/distilbert-cross-segment-document-chunking` at revision
  `14bc4e3c70061b60ca421a235c303656fd855ea5`.
- Wiki-727K conversion: `TankNee/wiki-727k` at revision
  `deea53e4b00c63dc158dca4071a4e4a5a38e2934`.
- Original Wiki-727K supplementary repository:
  `koomri/text-segmentation` at commit
  `874d6ef3ca0e402709b70924608d0894be9a93e1`.

The 105 supplied NumPy tensor shards match all 105 tensors in the local
DistilBERT SafeTensors checkpoint exactly: no missing, extra, shape-mismatched,
dtype-mismatched, or value-mismatched tensors.

The Wiki-727K Parquet conversion contains:

| Split | Documents | Bytes |
|---|---:|---:|
| Train | 582,160 | 2,351,011,725 |
| Development | 72,354 | 295,044,263 |
| Test | 73,232 | 301,391,901 |

Each row contains newline-separated sentence text and a parallel label list.
`label[i] == 1` means that sentence `i` ends a section, so the adjacent pair
`(sentence[i], sentence[i+1])` is a DIFFERENT example. The final sentence has no
following pair and must be dropped.

## What the references actually establish

The supplied 2020 paper proposes three architectures. Its cross-segment model
places running left and right context around a separator and makes one boundary
decision. It uses up to 128 wordpieces on each side for the principal Wiki-727K
result. The paper reports 66.0 F1 for cross-segment BERT Large at 128-128 and
64.0 F1 for BERT Base in the trailing-context ablation. It also reports that
256-0 context falls to 20.2 F1 and sorted token distributions reach only 39.1
F1, demonstrating that ordered context on both sides matters.

The cloned `koomri/text-segmentation` repository accompanies the earlier 2018
Wiki-727K paper, not the 2020 cross-segment-attention paper. Its released model
is a hierarchical BiLSTM over word2vec-based sentence encodings. The supplied
3.64 GB Google News vectors are relevant only if we reproduce that older
baseline.

The BlueOrangeDigital model is a standard six-layer
`DistilBertForSequenceClassification`. Its encoder is fully fine-tuned: relative
L2 movement from the supplied base is about 1.70% for embeddings and 2.43-2.85%
for each transformer layer. It adds the standard 768x768 pre-classifier and
768x2 classifier head.

Its model card describes 40,000 training and 4,000 validation articles,
408,753/45,417 pairs, two epochs, learning rate 1e-5, and negative-class
undersampling. The card contains a left/right wording contradiction. Direct
evaluation resolves it: chronological `left, then right` scored 0.826 F1 on a
small balanced 128-pair audit, while reversing the pair scored 0.396 F1.

## Controlled experiment contracts

### A. Six-layer causal-free alignment

Every DistilBERT layer is trained independently from the same paired embedding
inputs, with no upstream student history:

1. teacher uses the original DistilBERT word table;
2. student uses the complete Qwen-derived 30,522x768 table;
3. train Q/K/V/O against teacher attention distributions and post-attention
   hidden states;
4. freeze attention and train FFN `lin1` against both pre-GELU and post-GELU
   teacher activations;
5. freeze `lin1` and train FFN `lin2` on the real post-GELU inputs;
6. preserve one file per trained layer plus a consolidated husk.

The run reports isolated metrics for every layer and a sequential six-layer
composition metric. The latter is the decisive warning signal: independent
layers can fit their isolated teachers yet fail when their errors compose.

### B. Public-checkpoint audit

Evaluate the pinned public checkpoint on balanced and natural-prevalence pairs
from the canonical development and test splits. Record accuracy, precision,
recall, F1, PR-AUC, confusion matrix, and calibration. This is a reference
checkpoint audit, not the controlled baseline.

### C. Controlled token baseline

Train a fresh DistilBERT sequence classifier on a versioned pair manifest from
Wiki-727K. Freeze the exact document split, pair ordering, truncation policy,
class sampling, seed, steps, optimizer, and evaluation thresholds.

### D. Qwen-input treatment

Encode each left and right sentence once with frozen Qwen. Feed DistilBERT five
continuous positions: `[CLS embedding, left Qwen vector, SEP embedding, right
Qwen vector, SEP embedding]`. Initialize the encoder from the six-layer husk and
use the same classification head and labeled examples as the token baseline.

The first treatment tests the core substitution cleanly. A later context-window
treatment can provide several Qwen vectors per side, allowing each vector to
summarize hundreds or thousands of source tokens without changing the benchmark
labels.

## Data policy

Alignment uses 8,192 unlabeled spans drawn only from Wiki-727K train documents;
7,343 are optimization spans and 849 are held out for alignment metrics. Task
development and test remain untouched.

The 47.7 GB Bright Data Wikipedia CSV has richer structured sections, but its
license and extraction pipeline differ from Wiki-727K. Keep it for a later
out-of-distribution or long-context evaluation after its license is reviewed;
do not mix it into the primary reproduction.


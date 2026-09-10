# Continuous Qwen-input DistilBERT research checkpoint

## Summary

This checkpoint is a DistilBERT topic-boundary classifier that receives two
continuous sentence representations from frozen Qwen3-Embedding-0.6B rather
than DistilBERT vocabulary IDs. It was trained on all valid adjacent-sentence
pairs in the Wiki-727K training split after a staged attention, projection, and
teacher-response alignment process.

Its purpose is to test representation consolidation across model families. It
is a research artifact, not a production segmenter.

## Identity and provenance

| Component | Pinned identity |
| --- | --- |
| DistilBERT base | `distilbert/distilbert-base-uncased` @ `12040accade4e8a0f71eabdb258fecc2e7e948be` |
| Qwen embedder | `Qwen/Qwen3-Embedding-0.6B` @ `97b0c614be4d77ee51c0cef4e5f07c00f9eb65b3` |
| Wiki-727K conversion | `TankNee/wiki-727k` @ `deea53e4b00c63dc158dca4071a4e4a5a38e2934` |
| Published reference | `BlueOrangeDigital/distilbert-cross-segment-document-chunking` @ `14bc4e3c70061b60ca421a235c303656fd855ea5` |
| Original Wiki-727K code | `koomri/text-segmentation` @ `874d6ef3ca0e402709b70924608d0894be9a93e1` |
| Selected initialization | 20K interpretation husk SHA-256 `730d8ea8dd478d99bdc2230e9a63f7adf7b2ca71b569505c4124bb100dd74cef` |
| Final model | SHA-256 `9ca37201a77f29cc53aaa55817e90e81c10a18bd629d91ea3eb6654801d088b7` |

Qwen's configuration has 1,024 hidden dimensions, a 151,669-entry tokenizer
vocabulary, and 32,768 model positions. The tokenizer advertises a larger
131,072 maximum, but this experiment explicitly capped encoding at 32,768.
DistilBERT's original vocabulary has 30,522 entries and 512 positions.

## Input construction

For every sentence:

1. Prefix the exact task instruction shown below.
2. Tokenize with Qwen, left-pad, and truncate at 32,768 tokens.
3. Run frozen Qwen in evaluation mode.
4. Pool the final token state.
5. Select dimensions 0 through 767, then L2-normalize.
6. Store the vector as float16 in the reusable sentence cache.

The prefix as an escaped string is:

```text
"Instruct: Represent this Wikipedia sentence for topic segmentation, emphasizing features that indicate whether the following sentence begins a new topical section.\nQuery: "
```

The space before the closing quote is part of the prefix.

For an adjacent pair, the model input is `[left_vector, right_vector]` with
shape `[batch, 2, 768]`. Class 1 means the left sentence ends a section, so the
right sentence begins a different section.

No DistilBERT token IDs, CLS vector, or SEP vector are present. The word table
was deleted before task training. DistilBERT's position embeddings and input
LayerNorm still operate on the two continuous positions.

## Architecture

The six-layer DistilBERT backbone processes the two vectors through
`inputs_embeds`. From its final hidden state:

```text
left = hidden[:, 0]
right = hidden[:, 1]
features = concat(left, right, abs(left - right), left * right)
logits = Linear768to2(Dropout(GELU(Linear3072to768(features))))
```

The saved file contains 45,283,586 parameters and occupies 90,579,380 bytes.
It contains no DistilBERT word-embedding table. Qwen is not included in this
file and remains a separate frozen upstream dependency.

## Alignment lineage

### Causal-free component alignment

A complete 30,522-by-768 Qwen-derived proxy word table first supplied a shared
discrete input index. Each of DistilBERT's six layers independently matched
teacher attention Q/K/V/O behavior, followed by FFN `lin1` and `lin2` behavior.
This trained 42,508,800 parameters. All isolated blocks improved, but the
sequential depth-6 composition gate worsened by 75.0%, proving that isolated
agreement did not automatically produce a coherent stack.

### Causal teacher-response reconciliation

The independent husk was reassembled under its real residual flow and distilled
against the original model's masked-token responses. With aligned attention and
FFN projections frozen, the longer interpretation stage trained position
embeddings, layer norms, and MLM transform/norm/output-bias parameters. It used
1,035,834 trainable parameters and reduced held-out teacher KL from 12.0502 to
5.2029 over 20,000 steps; top-1 agreement rose from 0 to 0.2468.

A separate pilot also tested a whole-model unlock and reached KL 5.9962 after
only 1,200 unlocked steps, but that pilot checkpoint was not used downstream.
The final task run started from the 20,000-step interpretation husk.

## Downstream training

| Setting | Value |
| --- | ---: |
| Training documents | 582,160 |
| Cached sentences | 30,580,099 |
| Valid adjacent pairs | 29,997,939 |
| Boundaries / non-boundaries | 3,009,021 / 26,988,918 |
| Qwen truncations | 0 |
| Epochs | 2 |
| Batch size | 8,192 |
| Steps per epoch / total | 3,733 / 7,466 |
| Encoder learning rate | 2e-5 |
| New-head learning rate | 1e-4 |
| Class weights | `[1.0, 8.969335556]` |
| Warmup / schedule | 5% / linear |
| Weight decay | 0.01 |
| Precision | BF16 training, TF32 enabled |
| Hardware | NVIDIA A100-SXM4-40GB |
| Seed | 71 |

Every parameter remaining after removal of the word table was trainable. Mean
weighted training loss was 0.536637 in epoch 1 and 0.496925 in epoch 2. Once the
cache existed, the successful two-epoch run took about 24 minutes. Cache
construction took hours across recovered Colab sessions and should not be
repeated without a specific reason.

Development and test evaluation were deferred in the training runtime because
only the complete training cache had been recovered. Later evaluations used
explicit diagnostic samples.

## Evaluation

### English matched diagnostic

The first 500 Wiki-727K development documents yielded 21,134 natural adjacent
pairs with 10.56% boundaries.

| Model | Accuracy | Precision | Recall | F1 | ROC-AUC | AP |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Continuous checkpoint | 0.8039 | 0.3130 | 0.7181 | 0.4360 | 0.8526 | 0.4655 |
| Published reference | 0.7668 | 0.2915 | 0.8449 | 0.4334 | 0.8823 | 0.5171 |

At thresholds optimized on this same slice, the continuous checkpoint reached
F1 0.4693 and the reference reached 0.5089. Those optimized values are
diagnostic, not held-out estimates. The reference has the better ranking and
same-slice threshold-swept performance.

### Chinese zero-shot probe

Six Chinese Wikipedia articles yielded 931 weakly labeled adjacent pairs with
8.81% boundaries. Without any parameter change, the checkpoint reached 0.8174
accuracy, 0.2660 precision, 0.6098 recall, 0.3704 F1, 0.8314 ROC-AUC, and
0.3933 AP. HTML section headings and simple sentence splitting supplied the
labels, so this is not a substitute for a curated multilingual benchmark.

## Intended uses

- controlled study of continuous embedding inputs to BERT-family encoders;
- topic-boundary experiments where frozen Qwen sentence representations are
  already available;
- ablations of alignment, instruction, context-vector layout, language, and
  Qwen model size;
- initialization for further task-specific research.

## Limitations and prohibited inferences

- The current classifier accepts exactly two Qwen vectors. Its executed maximum
  is 32,768 Qwen tokens per sentence and 65,536 per pair.
- It does not yet implement a 512-vector context, a 16.8-million-token input, or
  any other multi-vector long-document design.
- Qwen has a much larger multilingual tokenizer, but DistilBERT sees only dense
  vectors; it does not inherit Qwen's tokenizer or vocabulary table.
- English Wiki-727K is the only downstream training corpus.
- The Chinese result is small, weakly labeled, and selected from homepage links.
- The provider's reported 0.85 metrics were produced under incompletely
  documented negative undersampling and cannot be directly compared with the
  natural-prevalence diagnostic.
- No adapter or Procrustes alignment was used, so their relative value remains
  untested.
- Qwen inference cost and latency are external to the 45.3M-parameter saved
  classifier.

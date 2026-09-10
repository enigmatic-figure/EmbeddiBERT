# Instruction-conditioned steering in continuous-input DistilBERT

## Result in one sentence

The frozen Qwen3-Embedding front end already behaves as a programmable feature
lens for the continuous-input DistilBERT checkpoint, but the checkpoint is not
yet a general promptable segmenter: instruction meaning, instruction polarity,
and pair position all move its predictions, while the exact instruction used
during training remains the strongest overall input distribution.

This is evidence for a usable control channel and a concrete training
opportunity. It is not yet evidence that arbitrary natural-language system
prompts improve task accuracy without prompt-diverse training.

## Questions

The experiment asked four progressively stronger questions:

1. Does changing only Qwen's instruction change the frozen DistilBERT's
   behavior on identical sentences?
2. Are the changes related to task meaning, rather than arbitrary prefix
   perturbation?
3. Can the existing classifier exploit role-specific instructions for the
   earlier and later sentence?
4. What training design is suggested by the observed failure modes?

## Fixed protocol

Both rounds used the same first 500 Wiki-727K development documents: 21,634
sentences, 21,134 adjacent pairs, 2,231 labeled boundaries, and a 10.56%
boundary rate. Qwen and DistilBERT weights remained frozen. Only the text
prepended to Qwen changed.

The Qwen input contract was unchanged from training: left padding, last-token
pooling, first 768 Matryoshka dimensions, float32 L2 normalization, and float16
storage. All instructions were encoded at a 32,768-token cap; no sentence was
truncated. The exact model, dataset, checkpoint, sample digest, environment,
and cache hashes are recorded in each ignored `results.json` artifact.

Documents 0–249 (10,373 pairs, 1,100 boundaries) formed the adaptive tuning
slice. Documents 250–499 (10,761 pairs, 1,131 boundaries) were not used to
select round-two prompts. Every transferred F1 below uses a threshold selected
on documents 0–249 and then applied unchanged to documents 250–499. Confidence
intervals are 500-repetition paired bootstraps over whole documents.

One blinding qualification matters: while monitoring round one, the remote
event tail accidentally displayed full-500 aggregate metrics for the anchor
and first four conditions. No heldout-only metrics or per-document heldout
scores were examined before the round-two manifest was committed and tagged,
but the second half is consequently *partially blinded*, not a pristine hidden
test set.

The reviewed initialization is tagged
`instruction-steering-true-initialization-v1`. The adaptive round-two design
was frozen at `instruction-steering-round2-design-v1` before heldout-only
analysis.

## Round one: broad instruction families

Round one compared the exact training prefix with no instruction, formatting
alone, generic semantics, three topic-boundary formulations, and unrelated
sentiment/style controls. The full strings are in
[`round1.json`](../experiments/instruction_steering/round1.json).

### Reserved documents 250–499

| Instruction | ROC-AUC | AP | Best-slice F1* | Transferred F1 |
| --- | ---: | ---: | ---: | ---: |
| Exact training instruction | **0.8537** | **0.4497** | **0.4696** | **0.4655** |
| Explicit boundary paraphrase | 0.8359 | 0.4121 | 0.4411 | 0.4288 |
| Explicit continuity | 0.8347 | 0.4058 | 0.4399 | 0.4240 |
| Topic comparison | 0.8261 | 0.3949 | 0.4309 | 0.4249 |
| Generic semantics | 0.8175 | 0.3750 | 0.4142 | 0.4074 |
| Query scaffold only | 0.8101 | 0.3771 | 0.4067 | 0.3986 |
| No instruction | 0.8081 | 0.3611 | 0.3982 | 0.3978 |
| Style control | 0.8030 | 0.3643 | 0.4006 | 0.3976 |
| Sentiment control | 0.8008 | 0.3626 | 0.3887 | 0.3847 |

\* Best-slice F1 chooses a threshold on the displayed heldout slice and is
therefore descriptive, not a deployable estimate.

The hierarchy replicated. On the reserved half, topic comparison beat generic
semantics by +0.0084 ROC-AUC (95% interval +0.0025 to +0.0138), +0.0198 AP
(+0.0086 to +0.0314), and +0.0151 descriptive best-slice F1 (+0.0032 to
+0.0282). Task-aware wording therefore carries useful structure beyond merely
adding an instruction-shaped prefix.

The exact training wording still won decisively. On the tuning half, cosine
similarity to the trained Qwen representation correlated strongly with
performance across the eight alternatives (0.937 with ROC-AUC and 0.916 with
AP). That motivated round two's lexical placebo and asymmetric role probes.

Instruction polarity also changed calibration. On tuning documents, the
explicit-boundary paraphrase raised the mean predicted boundary score by
+0.079, whereas continuity wording raised it only +0.017. This is why prompt
comparisons cannot be read from F1 at a universal 0.5 threshold alone.

## Round two: hypotheses and role probes

The hypotheses were written before execution in
[`round2_hypotheses.md`](../experiments/instruction_steering/round2_hypotheses.md),
and the 16 frozen conditions are in
[`round2.json`](../experiments/instruction_steering/round2.json). They tested:

- trained-distribution proximity versus task meaning;
- boundary/continuity polarity on only one member of the pair;
- Qwen's retrieval-style instructed-query/uninstructed-candidate convention;
- a closing-section lens on the left and opening-section lens on the right;
- coarse-theme, primary-entity, and rhetorical-function segmentations.

Only six new instruction encodings were required because four byte-identical
round-one caches were reused. The three repeated symmetric conditions produced
byte-identical score arrays across rounds.

### Reserved documents 250–499

| Condition | ROC-AUC | AP | Best-slice F1* | Transferred F1 |
| --- | ---: | ---: | ---: | ---: |
| Exact instruction on both positions | **0.8537** | **0.4497** | **0.4696** | **0.4655** |
| Closure lens left; exact right | 0.8500 | 0.4428 | 0.4632 | 0.4593 |
| Continuity lens left; exact right | 0.8486 | 0.4390 | 0.4642 | 0.4597 |
| Boundary paraphrase left; exact right | 0.8456 | 0.4325 | 0.4576 | 0.4324 |
| No instruction left; exact right | 0.8418 | 0.4265 | 0.4504 | 0.4492 |
| Exact left; boundary paraphrase right | 0.8413 | 0.4328 | 0.4492 | 0.4437 |
| Exact left; continuity right | 0.8407 | 0.4203 | 0.4441 | 0.4410 |
| Rhetorical-function lens, symmetric | 0.8361 | 0.4167 | 0.4523 | 0.4288 |
| Exact left; opening lens right | 0.8326 | 0.4165 | 0.4371 | 0.4263 |
| Coarse-major-theme lens, symmetric | 0.8311 | 0.4094 | 0.4355 | 0.4291 |
| Primary-entity lens, symmetric | 0.8310 | 0.4155 | 0.4388 | 0.4272 |
| Closure left; opening right | 0.8263 | 0.4041 | 0.4211 | 0.4175 |
| Matched style placebo, symmetric | 0.8232 | 0.3969 | 0.4174 | 0.4153 |
| Exact left; no instruction right | 0.8141 | 0.3785 | 0.4103 | 0.4029 |

### What the hypotheses resolved

**There is semantic steering beyond geometric proximity.** The rhetorical
instruction was *less* cosine-similar to the trained representation than the
matched style placebo (0.896 versus 0.920), yet beat it on the reserved half by
+0.0129 ROC-AUC (95% interval +0.0059 to +0.0193), +0.0202 AP (+0.0065 to
+0.0345), and +0.0334 descriptive best-slice F1 (+0.0169 to +0.0492). The
primary-entity and coarse-theme lenses also beat the placebo on all three
ranking measures. Lexical resemblance to the anchor cannot explain the whole
effect.

**The control channel is sharply role-dependent.** Keeping the right vector on
the exact training instruction while removing the instruction from the left
reached 0.8418 ROC-AUC. The reverse assignment reached only 0.8141. Their
paired difference was +0.0277 ROC-AUC (+0.0192 to +0.0368), +0.0484 AP
(+0.0285 to +0.0682), and +0.0391 descriptive best-slice F1 (+0.0188 to
+0.0574). The existing classifier does not inherit Qwen retrieval's usual
“instruct query, leave candidate uninstructed” convention. Instead, it depends
most heavily on keeping the *right DistilBERT position* in distribution.

**A left-side closure lens is almost usable zero-shot.** Closure-left/exact-right
reached 0.8500 ROC-AUC and 0.4593 transferred F1, versus 0.8537 and 0.4655 for
the anchor. Its transferred-F1 delta was -0.0066 (-0.0199 to +0.0065); the
experiment does not resolve a real F1 difference at this sample size.
Continuity-left/exact-right behaved similarly. However, closure-left was much
better than opening-right: +0.0174 ROC-AUC (+0.0122 to +0.0228) and +0.0267 AP
(+0.0161 to +0.0377). Combining closure and opening prompts degraded rather
than amplified performance because the right-side distribution shift
dominated.

**Instructions provide a powerful score-bias control, not only a ranking
control.** On the reserved half, changing only the right vector from the exact
instruction to the boundary paraphrase raised the mean boundary probability by
+0.139 and flipped 20.2% of decisions at threshold 0.5. Changing only the left
vector to the same paraphrase lowered the mean by -0.032 and flipped 5.2%.
Prompt polarity and position therefore interact with the learned classifier;
a common threshold cannot be assumed.

**Alternative lenses can legitimately disagree with Wiki-727 labels.** The
rhetorical lens had the strongest fixed-0.5 F1 among round-two conditions
(0.4470 versus the anchor's 0.4386) and substantially better Brier score, but
lower ranking, balanced accuracy, and transferred-threshold F1. This is a
calibration/operating-point result, not a general accuracy win. Moreover,
Wiki-727 section boundaries supervise thematic sections, not rhetorical or
entity-specific segmentation. A prediction counted as an error may be correct
under the requested alternative lens.

## Interpretation

The weakest defensible conclusion is already consequential: an English
instruction that DistilBERT never tokenizes changes which information Qwen
preserves in each 768-dimensional virtual token, and the frozen downstream
network responds systematically to those changes. The representation is not a
static compression of the sentence.

The stronger “system prompt for BERT” analogy is promising but conditional.
This checkpoint learned only one instruction and therefore entangled task
semantics with a very specific input distribution. It can tolerate and
partially exploit nearby controls—especially on the left position—but it has
not learned instruction invariance, role grammar, or calibration across lenses.
Those are training problems rather than evidence that the channel is absent.

The right-position asymmetry is especially useful. The architecture concatenates
`left`, `right`, `abs(left-right)`, and `left*right`, so it can use absolute and
interaction features differently by position. Training both sentences with a
prefix phrased around “the following sentence” also gives the right vector a
slightly awkward semantic role. Prompt-diverse training should make those roles
explicit instead of assuming a symmetric sentence encoder.

## Recommended training design

1. **Train over an instruction distribution.** Mix semantically equivalent
   boundary prompts, neutral topic prompts, prompt dropout, and deliberately
   unrelated controls. Require paraphrase robustness while preserving
   controlled differences among genuinely different lenses.
2. **Use role-specific templates.** Train the left vector with closing/current-
   section questions and the right vector with opening/continuation questions.
   Begin with the right position anchored to the current training prefix, then
   progressively vary it; round two shows that varying both positions at once
   is too large a cold-start shift.
3. **Supervise the requested segmentation policy.** Thematic, entity, temporal,
   rhetorical, coarse, fine, and user-question-relative boundaries need labels
   or synthetic teacher judgments appropriate to each lens. Reusing only
   Wiki-727 headings teaches prompt invariance, not true task programmability.
4. **Separate ranking from calibration.** Either condition the head on the lens,
   learn per-lens calibration, or train a stable score contract. Instruction
   changes of +0.139 mean probability make a universal 0.5 threshold untenable
   today.
5. **Add contrastive control objectives.** For the same sentence, draw
   equivalent prompts together, distinguish semantically different lenses,
   and penalize unrelated controls that masquerade as task features. A neutral
   anchor vector can regularize distribution drift during the curriculum.
6. **Then add surrounding context.** Encode a marked target sentence with
   ±1 and ±2 neighbors while keeping one Qwen output vector. Cross the context
   window with the strongest role/lens prompts. This directly tests whether a
   virtual token can retain boundary evidence that is invisible in an isolated
   sentence before scaling to hundreds of virtual tokens.

## Limits

- This is one frozen English topic-segmentation checkpoint and one contiguous
  500-document diagnostic sample, not a general benchmark.
- Prompt families were hand-designed and round two was adaptive; intervals are
  descriptive and do not correct for multiple comparisons.
- The experiment measures agreement with existing Wiki-727 boundaries. It does
  not directly validate alternate-lens segmentations with lens-specific gold
  labels.
- Qwen still embedded each sentence independently. The proposed three- or
  five-sentence target windows and longer multi-vector layouts remain untested.
- No downstream weights were retrained, so these results likely understate what
  explicit instruction-channel training could unlock.

## Reproducibility and artifacts

The harness is
[`evaluate_instruction_steering.py`](../scripts/evaluate_instruction_steering.py),
and the document-disjoint analysis is
[`analyze_instruction_steering.py`](../scripts/analyze_instruction_steering.py).
Generated artifacts remain ignored under `outputs/instruction_steering/` and
are mirrored on the RTX 4050 host under
`/home/jeff/EmbeddiBERT/eval/instruction_steering/run/`.

| Artifact | SHA-256 |
| --- | --- |
| Round-one raw result | `f11714cf36ded6f3af300949787e00c00581411fc7aa98f1727484e2a78fa537` |
| Round-one tuning analysis | `afde4c2483de3fa88ad29b2ea68ff0bf0caf338eda66d2c8a5816dc213360308` |
| Round-one reserved analysis | `0a706820639a90dad3e20ab606ca195234aee33c5f168eb891a20af69586ab80` |
| Round-two raw result | `dcfc5107076032e320439372ffd9d07cf7d1b84a4b176777a9624347ac8afc27` |
| Round-two tuning analysis | `5d6867de7432339bfd08f00349c9405204c12e8cb2e1640cbba62d76eeb3f5d1` |
| Round-two reserved analysis | `76d17ddfe3cec77cdb178cebfc8cc68c49b5ff349ef74dcbd1e84fd07deecbce` |

The canonical sample digest is
`266660340abd2a5ef21146d5d8df9beada349e58aabd85fd885dfe2ad2b0fba7`.
Round two ran with Python 3.14.4, PyTorch 2.14.0+cu130, Transformers 5.17.0,
and an NVIDIA GeForce RTX 4050 Laptop GPU. Its raw result was independently
hash-checked after transfer from the remote host.

# Round-two hypotheses (frozen before execution)

## Evidence used

Only document IDs 0–249 from round one were used for adaptive design: 250
documents, 10,373 adjacent-sentence pairs, and 1,100 boundaries.  Uncertainty
below is a 500-repetition paired document bootstrap.

The exact training instruction reached 0.8515 ROC-AUC, 0.4823 average
precision, and 0.4750 best-slice F1.  Removing the instruction reduced these
by 0.0423, 0.1159, and 0.0615 respectively (point-estimate differences).  The
task-adjacent conditions fell between the anchor and generic/unrelated
controls.  `explicit_continuity` was the strongest alternative by ROC-AUC
(0.8379; bootstrap delta CI -0.0195 to -0.0081), while `explicit_boundary`
produced a much larger positive score shift (+0.079 mean) and poorer
default-threshold calibration.

Embedding cosine similarity to the trained representation was also ordered:
0.970 for the boundary paraphrase, 0.943 for continuity, 0.922 for topic
comparison, 0.861 for generic semantics, and about 0.81–0.83 for unrelated or
instruction-free controls.  Thus task meaning and simple proximity to the
training distribution are confounded in round one.

During remote monitoring, the event tail accidentally exposed full-sample
aggregate metrics for the anchor and the first four alternatives.  No
held-out per-document scores or held-out-only metrics were examined before
this design was frozen, but documents 250–499 must consequently be described
as partially blinded rather than a pristine holdout.

## Hypotheses and discriminating conditions

1. **Training-distribution proximity explains part, but not all, of prompt
   performance.** `matched_style_placebo` keeps much of the anchor's lexical
   frame while explicitly prioritizing style.  Worse ranking than genuine
   task paraphrases at comparable cosine would be evidence for semantic
   steering beyond mere geometric proximity.
2. **Instruction polarity acts as a score-bias control.** Boundary wording
   inflated scores, while continuity wording was much less disruptive at the
   fixed threshold.  Applying each only to the left or right member of the
   ordered pair tests whether this is a global representation shift or a
   role-dependent signal the unmodified classifier can use.
3. **The ordered pair may expose Qwen's query/candidate convention.** Qwen's
   documented retrieval recipe instructs the query but not the candidate.
   `anchor_left_no_right` and its reverse test whether that asymmetry survives
   in this classifier, despite its symmetric-instruction training data.
4. **Explicit sentence roles may be a latent control channel.** A left
   “section closure” lens and a right “section opening” lens should help most
   when assigned to their matching positions if the classifier can exploit
   instruction-conditioned features without retraining.
5. **Alternative segmentation lenses should create distinct, label-aligned
   rankings.** Wiki-727 section labels should favor the coarse-major-theme
   lens over primary-entity or rhetorical-function lenses.  These comparisons
   cannot establish that the latter segmentations are wrong; they measure
   alignment to the existing section-boundary annotation policy.

The exact training anchor, the two strongest round-one task paraphrases, and
all reused instruction strings remain byte-identical so their cached vectors
provide execution controls.

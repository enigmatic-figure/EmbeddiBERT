# Evaluation diagnosis: continuous model vs. BlueOrangeDigital baseline

## What invalidated the first comparison

The first baseline run passed the later sentence as the first tokenizer
sequence and the earlier sentence as the second. The model card is internally
contradictory: its prose describes `Right [SEP] Left`, while its executable
pipeline examples and widget use `Left [SEP] Right`. The published model's
behavior confirms that the executable example is the relevant contract.

The archived development cache is also only a prefix, not a complete dev
cache. Its metadata reports 3,824,031 allocated sentence rows and 3,751,677
possible pairs, while `progress.json` reports only 1,118,386 encoded sentences
across 176 batches. The validity mask therefore contains 1,095,858 usable
pairs, corresponding to 22,528 documents. Results previously called
"full-dev" are prefix-cache results.

## Corrected matched-slice results

Both models were evaluated on the first 500 Wiki-727K development documents,
containing 21,134 valid adjacent-sentence pairs and a 10.56% boundary rate.
Class 1 means a boundary (`DIFFERENT`). The baseline was encoded in chronological
order as `Left [SEP] Right`.

| Model | Accuracy | Precision | Recall | F1 | ROC-AUC | Average precision |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Continuous Qwen-input DistilBERT | 0.8039 | 0.3130 | 0.7181 | 0.4360 | 0.8526 | 0.4655 |
| BlueOrangeDigital baseline | 0.7668 | 0.2915 | 0.8449 | 0.4334 | 0.8823 | 0.5171 |

At the default 0.5 decision threshold the F1 scores are effectively tied.
Threshold sweeps on the same diagnostic slice produce:

| Model | Best slice F1 | Threshold | Precision | Recall |
| --- | ---: | ---: | ---: | ---: |
| Continuous model | 0.4693 | 0.6285 | 0.4003 | 0.5670 |
| BlueOrangeDigital baseline | 0.5089 | 0.7983 | 0.4336 | 0.6159 |

The baseline has the stronger ranking performance and the stronger calibrated
F1. The earlier claim that the continuous model substantially beat the
baseline is withdrawn.

## Why the published 0.85 F1 is not reproduced here

The model card says its training and validation negatives were undersampled.
This evaluation uses the natural Wiki-727K boundary rate, where only 10.56% of
the matched slice is positive. A classifier trained and calibrated on an
undersampled distribution will over-predict boundaries at a 0.5 threshold on
the natural distribution. The published 0.85 metrics therefore cannot be
compared directly without reconstructing the provider's sampled validation
protocol.

The corrected baseline confusion matrix is TN 14,321, FP 4,582, FN 346, and TP
1,885. Its true-positive rate is 0.8449 and false-positive rate is 0.2424. At the
observed 10.56% boundary prevalence those rates imply precision 0.2915. If the
same rates are reweighted to a 50/50 class prior, precision becomes 0.7771. The
apparently abysmal precision is therefore chiefly an operating-prior and
calibration problem, not evidence that nine out of ten predictions would be
wrong under the provider's sampled validation distribution.

Class averaging is another unresolved disconnect. On this matched slice the
baseline's class-0 precision is 0.9764, class-1 precision is 0.2915, macro
precision is 0.6339, and support-weighted precision is 0.9041. The provider does
not say whether its precision, recall, and F1 are binary, macro, micro, or
support-weighted. In single-label classification, support-weighted recall equals
accuracy; the provider reporting both as 0.85 is therefore consistent with, but
does not prove, weighted averaging.

Even a balanced 50/50 prior does not fully explain the claimed 0.86 precision:
with 0.85 recall, that precision would require a false-positive rate near 0.138,
below the 0.242 observed here. A different sampled validation set, preprocessing
pipeline, trained checkpoint behavior, threshold, or some combination still
matters. Because those details are absent, the model-card table is not a
reproducible target for this experiment.

## Remaining work for a definitive benchmark

1. Complete the Qwen development cache or explicitly select a deterministic
   sample from the currently valid 22,528-document prefix.
2. Use a train-free calibration split to choose thresholds, then report once
   on a disjoint test split.
3. Reproduce the provider's negative undersampling protocol as a second,
   separately labeled benchmark.
4. Preserve document-level sampling so no adjacent pairs from one document
   leak across calibration and test partitions.

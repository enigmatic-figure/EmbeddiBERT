# Artifact inventory and recovery

## Storage status

The completed cache and Colab session archive are stored in the public Hugging
Face bucket:

```text
hf://buckets/IntellAgents/embeddibert-wiki727-cache-20260909
```

The bucket was verified on 2026-09-10 at 53,679,759,883 bytes across 21 files.
Its two important prefixes are:

```text
wiki727_train_cache/
session_artifacts/
```

No access token is recorded in this repository. Use the local Hugging Face
credential store or an ephemeral environment variable when authenticated access
is needed.

## Irreplaceable artifacts

| Artifact | Location | Bytes | SHA-256 / status |
| --- | --- | ---: | --- |
| Complete train embeddings | `wiki727_train_cache/embeddings.f16` | 46,971,032,064 | `bf7943b17279d64cd4e3eca392d6d6550bcd27adb48f8ce6a6502a64f64bf5b9` |
| Train labels | `wiki727_train_cache/labels.u8` | 30,580,099 | `483b466b0c0900eabc87b145902042d730eec296edeeab09a42bce80f1342df7` |
| Train validity mask | `wiki727_train_cache/valid.u8` | 30,580,099 | `777cb021cab36f936bf2f05893b45936b16e3d120aa842d476e6cc009008a606` |
| Final continuous model | `session_artifacts/models/continuous_pair_distilbert.safetensors` | 90,579,380 | `9ca37201a77f29cc53aaa55817e90e81c10a18bd629d91ea3eb6654801d088b7` |
| Selected 20K husk | `session_artifacts/models/distilbert_interpretation_20k_husk.safetensors` | 131,945,696 | `730d8ea8dd478d99bdc2230e9a63f7adf7b2ca71b569505c4124bb100dd74cef` |
| Final optimizer checkpoint | `session_artifacts/checkpoints/latest.pt` | 543,531,871 | `e823dcb0f0d295334a1807d1d551907d9c957d217e0c1f731f1fad769df4f570` |
| Authoritative result | `session_artifacts/results/result.json` | 2,785 | `d7de8afbca6f4da8bbb6d83960d8fef7c87da71000ff25cd93ec140ea374046c` |

The cache metadata records 30,580,099 sentences, 29,997,939 valid pairs,
3,009,021 positives, 26,988,918 negatives, and zero sentences truncated at the
32,768-token cap. `outputs/wiki727_train_cache_backup/hf_rescue_manifest.json`
is the local rescue manifest containing the three cache hashes.

## Session archive

The `session_artifacts` prefix also preserves:

- the final detached training log and structured event stream;
- the exact archived training source from the Colab runtime;
- the runtime inventory;
- the partial development cache and its progress metadata;
- the final model, selected initialization husk, optimizer checkpoint, and
  result JSON.

The local copy of its immutable index is
`outputs/urgent_old_runtime_export/session_archive_manifest.json`.

The archived source file is evidence of what existed in the recovered runtime.
The final result JSON is authoritative for the successful run configuration,
including batch size 8,192. The event stream contains records from aborted and
restarted attempts, so do not aggregate all `start` events as though they were
one training run.

## Partial development cache warning

The archived development embeddings file is allocated to 5,873,711,616 bytes,
but only its prefix is populated. Metadata allocates 3,824,031 sentence rows and
3,751,677 possible pairs, while progress stops at sentence cursor 1,118,386.
The validity mask identifies 1,095,858 usable pairs from exactly 22,528
documents.

Never treat this as a complete development cache. A prior result described as
“full-dev” was actually evaluated on this valid prefix.

## Safe inspection and restoration

Inspect without downloading:

```powershell
hf buckets info IntellAgents/embeddibert-wiki727-cache-20260909
hf buckets list IntellAgents/embeddibert-wiki727-cache-20260909 `
  --recursive --human-readable
```

Preview a train-cache restore plan:

```powershell
hf buckets sync `
  hf://buckets/IntellAgents/embeddibert-wiki727-cache-20260909/wiki727_train_cache `
  .\outputs\restored_wiki727_train_cache `
  --dry-run
```

Remove `--dry-run` only after reviewing the paths and available disk space. A
single file can be recovered without syncing the whole cache:

```powershell
hf buckets cp `
  hf://buckets/IntellAgents/embeddibert-wiki727-cache-20260909/session_artifacts/models/continuous_pair_distilbert.safetensors `
  .\outputs\restored_continuous_pair_distilbert.safetensors
```

Verify it after download:

```powershell
Get-FileHash .\outputs\restored_continuous_pair_distilbert.safetensors `
  -Algorithm SHA256
```

## Local generated artifacts

The current workspace also contains ignored copies under `outputs/`, including
the final model, alignment results, Chinese probe, and rescue manifests. They
are convenient, not the only durable copy. `outputs/`, credentials, logs,
caches, and runtime state must remain outside Git.

## Git milestones

Important immutable tags include:

- `kaggle-qwen06b-first-layer-v1`
- `distilbert-inquiry-true-initialization-v1`
- `distilbert-independent-run-v2`
- `distilbert-independent-result-v1`
- `distilbert-mlm-true-initialization-v1`
- `distilbert-mlm-run-v1`
- `distilbert-mlm-plateau-run-v1`
- `continuous-wiki727-true-initialization-v1`
- `continuous-wiki727-run-v2`
- `continuous-wiki727-run-v3`

The Git history records source and documentation. It deliberately does not
contain the 50+ GB data archive or plaintext service credentials.

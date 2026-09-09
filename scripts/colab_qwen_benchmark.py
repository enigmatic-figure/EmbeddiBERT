import json
import time

import torch
import torch.nn.functional as F
from transformers import AutoModel, AutoTokenizer


MODEL = "Qwen/Qwen3-Embedding-0.6B"
REVISION = "97b0c614be4d77ee51c0cef4e5f07c00f9eb65b3"
INSTRUCTION = (
    "Instruct: Represent the Wikipedia sentence for detecting whether the next "
    "sentence begins a new topical section.\nQuery: "
)
texts = [
    INSTRUCTION
    + "This is a representative Wikipedia sentence with enough words to exercise batching."
] * 4096

tokenizer = AutoTokenizer.from_pretrained(MODEL, revision=REVISION, padding_side="left")
model = AutoModel.from_pretrained(
    MODEL,
    revision=REVISION,
    torch_dtype=torch.bfloat16,
    attn_implementation="sdpa",
).to("cuda").eval()

results = []
for batch_size in (128, 256, 512, 1024):
    torch.cuda.empty_cache()
    encoded = tokenizer(
        texts[:batch_size],
        padding=True,
        truncation=True,
        max_length=256,
        return_tensors="pt",
    )
    encoded = {key: value.to("cuda") for key, value in encoded.items()}
    try:
        with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
            model(**encoded)
        torch.cuda.synchronize()
        started = time.time()
        repeats = max(2, 2048 // batch_size)
        for _ in range(repeats):
            with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
                output = model(**encoded).last_hidden_state[:, -1, :768]
                F.normalize(output.float(), p=2, dim=-1)
        torch.cuda.synchronize()
        elapsed = time.time() - started
        results.append(
            {
                "batch_size": batch_size,
                "sequence_length": int(encoded["input_ids"].shape[1]),
                "sentences_per_second": batch_size * repeats / elapsed,
                "peak_gib": torch.cuda.max_memory_allocated() / 2**30,
            }
        )
    except torch.OutOfMemoryError:
        results.append({"batch_size": batch_size, "oom": True})
print(json.dumps(results, indent=2))

import importlib.util
import json
import sys
from pathlib import Path

import torch
from transformers import AutoModel


path = Path("/content/colab_train_continuous_pairs.py")
spec = importlib.util.spec_from_file_location("continuous_training", path)
module = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = module
spec.loader.exec_module(module)

backbone = AutoModel.from_pretrained(
    module.CONFIG.base_model,
    revision=module.CONFIG.base_revision,
    attn_implementation="sdpa",
)
module.apply_husk(backbone, Path(module.CONFIG.initial_husk))
del backbone.embeddings.word_embeddings
model = module.ContinuousPairClassifier(backbone).to("cuda").train()
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-5, fused=True)
torch.cuda.reset_peak_memory_stats()
pairs = torch.randn(4096, 2, 768, device="cuda", dtype=torch.bfloat16)
targets = torch.randint(0, 2, (4096,), device="cuda")
with torch.autocast("cuda", dtype=torch.bfloat16):
    logits = model(pairs)
    loss = torch.nn.functional.cross_entropy(logits.float(), targets)
loss.backward()
optimizer.step()
print(
    json.dumps(
        {
            "loss": float(loss),
            "parameters": sum(parameter.numel() for parameter in model.parameters()),
            "word_table_present": any(
                "word_embeddings" in name for name, _ in model.named_parameters()
            ),
            "peak_memory_gib": torch.cuda.max_memory_allocated() / 2**30,
        },
        indent=2,
    )
)

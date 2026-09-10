"""Build and evaluate a small Chinese-Wikipedia section-boundary probe."""

from __future__ import annotations

import argparse
import html
import json
import re
import time
from dataclasses import dataclass
from html.parser import HTMLParser
from pathlib import Path
from typing import Any
from urllib.parse import quote

import numpy as np


WIKIPEDIA_API = "https://zh.wikipedia.org/w/api.php"
USER_AGENT = "EmbeddiBERT-Chinese-Probe/0.1 (research evaluation)"
DEFAULT_TITLES = (
    "國際聯盟",
    "葉挺",
    "馬德里地鐵",
    "大型強子對撞機",
    "棕三趾鶉",
    "1989年飓风基科",
)
INSTRUCTION = (
    "Instruct: Represent this Wikipedia sentence for topic segmentation, "
    "emphasizing features that indicate whether the following sentence begins "
    "a new topical section.\nQuery: "
)
IGNORED_SECTION_PREFIXES = (
    "参见",
    "參見",
    "参考",
    "參考",
    "注释",
    "註釋",
    "脚注",
    "腳註",
    "来源",
    "來源",
    "外部链接",
    "外部連結",
    "延伸阅读",
    "延伸閱讀",
    "相关阅读",
    "相關條目",
    "相关条目",
    "书目",
    "書目",
)
SKIPPED_TAGS = {"script", "style", "sup", "math", "table", "figure", "nav"}


def normalize_text(value: str) -> str:
    value = html.unescape(value)
    value = re.sub(r"\[[0-9０-９]+\]", "", value)
    return re.sub(r"\s+", " ", value).strip()


class ArticleHTMLParser(HTMLParser):
    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.blocks: list[tuple[str, str]] = []
        self.section_number = 0
        self.section_title = "导言"
        self.ignore_section = False
        self.heading_tag: str | None = None
        self.heading_parts: list[str] = []
        self.in_paragraph = False
        self.paragraph_parts: list[str] = []
        self.skip_depth = 0

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        tag = tag.lower()
        if self.skip_depth:
            if tag in SKIPPED_TAGS:
                self.skip_depth += 1
            return
        if tag in SKIPPED_TAGS:
            self.skip_depth = 1
        elif tag in {"h2", "h3", "h4"}:
            self.heading_tag = tag
            self.heading_parts = []
        elif tag == "p":
            self.in_paragraph = True
            self.paragraph_parts = []
        elif tag == "br" and self.in_paragraph:
            self.paragraph_parts.append(" ")

    def handle_endtag(self, tag: str) -> None:
        tag = tag.lower()
        if self.skip_depth:
            if tag in SKIPPED_TAGS:
                self.skip_depth -= 1
            return
        if tag == self.heading_tag:
            title = normalize_text("".join(self.heading_parts))
            if title:
                self.section_number += 1
                self.section_title = title
                self.ignore_section = title.startswith(IGNORED_SECTION_PREFIXES)
            self.heading_tag = None
            self.heading_parts = []
        elif tag == "p" and self.in_paragraph:
            value = normalize_text("".join(self.paragraph_parts))
            if value and not self.ignore_section:
                key = f"{self.section_number}:{self.section_title}"
                self.blocks.append((key, value))
            self.in_paragraph = False
            self.paragraph_parts = []

    def handle_data(self, data: str) -> None:
        if self.skip_depth:
            return
        if self.heading_tag is not None:
            self.heading_parts.append(data)
        elif self.in_paragraph:
            self.paragraph_parts.append(data)


def split_sentences(paragraph: str) -> list[str]:
    # Preserve closing quotation marks with the sentence-final punctuation.
    pieces = re.split(r"([。！？!?]+[”’」』》）】]*)", paragraph)
    sentences: list[str] = []
    for index in range(0, len(pieces) - 1, 2):
        sentence = normalize_text(pieces[index] + pieces[index + 1])
        if len(sentence) >= 8:
            sentences.append(sentence)
    if len(pieces) % 2:
        tail = normalize_text(pieces[-1])
        if len(tail) >= 8:
            sentences.append(tail)
    return sentences


def fetch_article(title: str) -> dict[str, Any]:
    import requests

    params = {
        "action": "parse",
        "page": title,
        "prop": "text|displaytitle|revid",
        "format": "json",
        "formatversion": "2",
        "redirects": "1",
        "variant": "zh-cn",
    }
    for attempt in range(5):
        response = requests.get(
            WIKIPEDIA_API,
            params=params,
            headers={"User-Agent": USER_AGENT},
            timeout=60,
        )
        if response.status_code != 429:
            break
        retry_after = float(response.headers.get("Retry-After", 1.0))
        time.sleep(max(retry_after, 2**attempt))
    response.raise_for_status()
    payload = response.json()
    if "error" in payload:
        raise RuntimeError(f"Wikipedia API failed for {title!r}: {payload['error']}")
    parsed = payload["parse"]
    parser = ArticleHTMLParser()
    parser.feed(parsed["text"])

    sentences: list[str] = []
    section_keys: list[str] = []
    for section_key, paragraph in parser.blocks:
        for sentence in split_sentences(paragraph):
            sentences.append(sentence)
            section_keys.append(section_key)
    if len(sentences) < 2:
        raise RuntimeError(f"Article {title!r} yielded fewer than two sentences")
    labels = [int(left != right) for left, right in zip(section_keys, section_keys[1:])]
    canonical_title = normalize_text(parsed.get("title", title))
    return {
        "requested_title": title,
        "title": canonical_title,
        "page_id": int(parsed["pageid"]),
        "revision_id": int(parsed["revid"]),
        "url": f"https://zh.wikipedia.org/wiki/{quote(canonical_title.replace(' ', '_'))}",
        "sentences": sentences,
        "section_keys": section_keys,
        "labels": labels,
    }


def prepare_sample(titles: list[str], output: Path) -> dict[str, Any]:
    articles = []
    for title in titles:
        articles.append(fetch_article(title))
        time.sleep(0.5)
    sample = {
        "source": "Chinese Wikipedia live MediaWiki API",
        "created_unix": time.time(),
        "sentence_split": "Chinese 。！？ plus ASCII !?; prose paragraphs only",
        "label_definition": "1 iff adjacent sentences belong to different HTML heading sections",
        "articles": articles,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(sample, ensure_ascii=False, indent=2), encoding="utf-8")
    return sample


def confusion_metrics(labels: np.ndarray, scores: np.ndarray) -> dict[str, float | int]:
    predictions = (scores >= 0.5).astype(np.uint8)
    tp = int(((predictions == 1) & (labels == 1)).sum())
    fp = int(((predictions == 1) & (labels == 0)).sum())
    tn = int(((predictions == 0) & (labels == 0)).sum())
    fn = int(((predictions == 0) & (labels == 1)).sum())
    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    fpr = fp / max(fp + tn, 1)
    specificity = tn / max(tn + fp, 1)
    balanced_precision = recall / max(recall + fpr, 1e-12)
    balanced_f1 = 2 * balanced_precision * recall / max(balanced_precision + recall, 1e-12)
    return {
        "examples": int(len(labels)),
        "positive_rate": float(labels.mean()),
        "predicted_positive_rate": float(predictions.mean()),
        "accuracy": (tp + tn) / max(len(labels), 1),
        "precision": precision,
        "recall": recall,
        "f1": 2 * precision * recall / max(precision + recall, 1e-12),
        "false_positive_rate": fpr,
        "specificity": specificity,
        "balanced_accuracy": (recall + specificity) / 2,
        "precision_at_balanced_prior": balanced_precision,
        "f1_at_balanced_prior": balanced_f1,
        "tp": tp,
        "fp": fp,
        "tn": tn,
        "fn": fn,
    }


def score_diagnostics(labels: np.ndarray, scores: np.ndarray) -> dict[str, float]:
    order = np.argsort(-scores, kind="stable")
    ranked_labels = labels[order]
    tp = np.cumsum(ranked_labels == 1)
    fp = np.cumsum(ranked_labels == 0)
    positives = max(int(tp[-1]), 1)
    negatives = max(int(fp[-1]), 1)
    precision = tp / np.maximum(tp + fp, 1)
    recall = tp / positives
    f1 = 2 * precision * recall / np.maximum(precision + recall, 1e-12)
    best = int(np.argmax(f1))
    tpr = np.concatenate(([0.0], tp / positives, [1.0]))
    fpr = np.concatenate(([0.0], fp / negatives, [1.0]))
    return {
        "roc_auc": float(np.trapezoid(tpr, fpr)),
        "average_precision": float(precision[ranked_labels == 1].mean()),
        "best_slice_f1": float(f1[best]),
        "best_slice_threshold": float(scores[order[best]]),
        "best_slice_precision": float(precision[best]),
        "best_slice_recall": float(recall[best]),
    }


@dataclass
class EncodedArticle:
    title: str
    labels: np.ndarray
    vectors: np.ndarray


def run_evaluation(sample: dict[str, Any], student_path: Path, batch_size: int) -> dict[str, Any]:
    import torch
    import torch.nn.functional as F
    from safetensors.torch import load_file
    from torch import nn
    from transformers import AutoModel, AutoTokenizer

    class ContinuousPairClassifier(nn.Module):
        def __init__(self, backbone: nn.Module) -> None:
            super().__init__()
            self.backbone = backbone
            dimension = backbone.config.dim
            self.pre_classifier = nn.Linear(4 * dimension, dimension)
            self.dropout = nn.Dropout(backbone.config.seq_classif_dropout)
            self.classifier = nn.Linear(dimension, 2)

        def forward(self, pairs: torch.Tensor) -> torch.Tensor:
            mask = torch.ones(pairs.shape[:2], dtype=torch.long, device=pairs.device)
            hidden = self.backbone(
                inputs_embeds=pairs,
                attention_mask=mask,
                return_dict=True,
            ).last_hidden_state
            left, right = hidden[:, 0], hidden[:, 1]
            features = torch.cat((left, right, torch.abs(left - right), left * right), dim=-1)
            return self.classifier(self.dropout(F.gelu(self.pre_classifier(features))))

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for this probe")
    device = torch.device("cuda")
    qwen_name = "Qwen/Qwen3-Embedding-0.6B"
    qwen_revision = "97b0c614be4d77ee51c0cef4e5f07c00f9eb65b3"
    tokenizer = AutoTokenizer.from_pretrained(qwen_name, revision=qwen_revision, padding_side="left")
    qwen = AutoModel.from_pretrained(
        qwen_name,
        revision=qwen_revision,
        dtype=torch.bfloat16,
        attn_implementation="sdpa",
    ).to(device).eval()

    encoded_articles: list[EncodedArticle] = []
    with torch.inference_mode():
        for article in sample["articles"]:
            sentences = article["sentences"]
            rows: list[np.ndarray] = []
            for start in range(0, len(sentences), batch_size):
                prompts = [INSTRUCTION + value for value in sentences[start : start + batch_size]]
                encoded = tokenizer(
                    prompts,
                    padding=True,
                    truncation=True,
                    max_length=32768,
                    pad_to_multiple_of=8,
                    return_tensors="pt",
                )
                encoded = {key: value.to(device) for key, value in encoded.items()}
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    hidden = qwen(**encoded).last_hidden_state[:, -1, :768]
                vectors = F.normalize(hidden.float(), p=2, dim=-1)
                rows.append(vectors.cpu().numpy().astype(np.float16))
            encoded_articles.append(
                EncodedArticle(
                    title=article["title"],
                    labels=np.asarray(article["labels"], dtype=np.uint8),
                    vectors=np.concatenate(rows),
                )
            )
    del qwen, tokenizer
    torch.cuda.empty_cache()

    backbone_name = "distilbert/distilbert-base-uncased"
    backbone_revision = "12040accade4e8a0f71eabdb258fecc2e7e948be"
    backbone = AutoModel.from_pretrained(backbone_name, revision=backbone_revision)
    model = ContinuousPairClassifier(backbone)
    state = load_file(str(student_path), device="cpu")
    missing, unexpected = model.load_state_dict(state, strict=False)
    if unexpected or any(not name.endswith("word_embeddings.weight") for name in missing):
        raise RuntimeError(f"Student state mismatch: missing={missing}, unexpected={unexpected}")
    model.to(device).eval()

    all_labels: list[np.ndarray] = []
    all_scores: list[np.ndarray] = []
    per_article: list[dict[str, Any]] = []
    with torch.inference_mode():
        for article in encoded_articles:
            pairs = np.stack((article.vectors[:-1], article.vectors[1:]), axis=1)
            score_rows: list[np.ndarray] = []
            for start in range(0, len(pairs), batch_size * 4):
                tensor = torch.from_numpy(pairs[start : start + batch_size * 4]).to(
                    device=device, dtype=torch.float16
                )
                with torch.autocast("cuda", dtype=torch.float16):
                    logits = model(tensor)
                score_rows.append(torch.softmax(logits.float(), dim=-1)[:, 1].cpu().numpy())
            scores = np.concatenate(score_rows)
            metrics = confusion_metrics(article.labels, scores)
            per_article.append({"title": article.title, **metrics})
            all_labels.append(article.labels)
            all_scores.append(scores)

    labels = np.concatenate(all_labels)
    scores = np.concatenate(all_scores)
    return {
        "device": torch.cuda.get_device_name(0),
        "model": str(student_path),
        "instruction": INSTRUCTION,
        "aggregate": {
            **confusion_metrics(labels, scores),
            **score_diagnostics(labels, scores),
        },
        "per_article": per_article,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sample", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--student", type=Path)
    parser.add_argument("--prepare", action="store_true")
    parser.add_argument("--titles", nargs="*", default=list(DEFAULT_TITLES))
    parser.add_argument("--batch-size", type=int, default=64)
    args = parser.parse_args()

    sample = prepare_sample(args.titles, args.sample) if args.prepare else json.loads(
        args.sample.read_text(encoding="utf-8")
    )
    summary = {
        "articles": len(sample["articles"]),
        "sentences": sum(len(item["sentences"]) for item in sample["articles"]),
        "pairs": sum(len(item["labels"]) for item in sample["articles"]),
        "boundaries": sum(sum(item["labels"]) for item in sample["articles"]),
        "pages": [
            {key: item[key] for key in ("title", "page_id", "revision_id", "url")}
            for item in sample["articles"]
        ],
    }
    if args.student:
        if args.output is None:
            parser.error("--output is required with --student")
        summary["evaluation"] = run_evaluation(sample, args.student, args.batch_size)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=True, indent=2))


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
Refresh SunnySett metadata artifacts.

This helper collates every model script in `sunnysett_models/`, extracts
docstring metadata, and regenerates the derived assets that power the
documentation:

* analysis_model_summary.json
* sunnysett_models_summary.md
* sunnysett_models_metadata.json
* sunnysett_models_metadata.csv

Usage:
    python refresh_metadata.py
"""

from __future__ import annotations

import csv
import json
import os
import re
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).parent
MODELS_DIR = ROOT / "sunnysett_models"

# Regular expressions to capture pipeline, model id, and description lines.
PIPELINE_RE = re.compile(r'pipeline\("([^"]+)"')
MODEL_NAME_RE = re.compile(r'model_name\s*=\s*"([^"]+)"')
DESCRIPTION_RE = re.compile(r'Description:\s*(.+)')

# Mapping of Hugging Face pipeline tags to high-level modality buckets.
MODALITY_MAP = {
    "text-classification": "Text/NLP",
    "sentiment-analysis": "Text/NLP",
    "zero-shot-classification": "Text/NLP",
    "text-generation": "Text/NLP",
    "text2text-generation": "Text/NLP",
    "summarization": "Text/NLP",
    "feature-extraction": "Text/NLP",
    "question-answering": "Text/NLP",
    "token-classification": "Text/NLP",
    "translation": "Text/NLP",
    "document-question-answering": "Vision+Text",
    "image-to-text": "Vision+Text",
    "visual-question-answering": "Vision+Text",
    "image-classification": "Vision",
    "object-detection": "Vision",
    "image-segmentation": "Vision",
    "text-to-image": "Vision",
    "automatic-speech-recognition": "Audio",
    "speech-to-text": "Audio",
    "text-to-speech": "Audio",
    "audio-classification": "Audio",
    "time-series-forecasting": "Tabular/Time-series",
    "tabular-classification": "Tabular/Time-series",
    "tabular-regression": "Tabular/Time-series",
}

CLASSICAL_KEYWORDS = {"prophet", "autogluon", "forecast"}


def classify_model_type(model_id: str, pipeline: str) -> str:
    name = model_id.lower()
    if any(keyword in name for keyword in CLASSICAL_KEYWORDS):
        return "Classical/AutoML"
    return "Neural/Deep"


def collect_model_records() -> list[dict[str, str]]:
    if not MODELS_DIR.exists():
        raise FileNotFoundError(f"Expected directory not found: {MODELS_DIR}")

    records: list[dict[str, str]] = []

    for category_dir in sorted(MODELS_DIR.iterdir()):
        if not category_dir.is_dir():
            continue

        for script in sorted(category_dir.glob("*.py")):
            content = script.read_text(encoding="utf-8")

            pipeline_match = PIPELINE_RE.search(content)
            model_match = MODEL_NAME_RE.search(content)
            description_match = DESCRIPTION_RE.search(content)

            pipeline = pipeline_match.group(1) if pipeline_match else "custom"
            model_id = model_match.group(1) if model_match else script.stem
            description = description_match.group(1).strip() if description_match else ""

            modality = MODALITY_MAP.get(pipeline, "Other")
            model_type = classify_model_type(model_id, pipeline)

            records.append(
                {
                    "category": category_dir.name,
                    "file": script.name,
                    "model_name": model_id,
                    "pipeline": pipeline,
                    "modality": modality,
                    "type": model_type,
                    "description": description,
                }
            )

    return records


def build_summary(records: list[dict[str, str]]) -> dict:
    category_counts = Counter()
    category_modalities: dict[str, Counter] = defaultdict(Counter)
    category_types: dict[str, Counter] = defaultdict(Counter)
    modality_counts = Counter()
    pipeline_counts = Counter()
    type_counts = Counter()

    for record in records:
        category = record["category"]
        category_counts[category] += 1
        category_modalities[category][record["modality"]] += 1
        category_types[category][record["type"]] += 1
        modality_counts[record["modality"]] += 1
        pipeline_counts[record["pipeline"]] += 1
        type_counts[record["type"]] += 1

    return {
        "overall": {
            "total_categories": len(category_counts),
            "total_models": len(records),
            "type_counts": dict(type_counts),
            "modality_counts": dict(modality_counts),
            "pipeline_counts": dict(pipeline_counts),
        },
        "category_counts": dict(category_counts),
        "category_modalities": {k: dict(v) for k, v in category_modalities.items()},
        "category_types": {k: dict(v) for k, v in category_types.items()},
        "records": records,
    }


def write_json_summary(summary: dict) -> None:
    (ROOT / "analysis_model_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )


def write_markdown(summary: dict) -> None:
    overall = summary["overall"]
    category_counts = summary["category_counts"]
    category_modalities = summary["category_modalities"]
    category_types = summary["category_types"]

    sorted_categories = sorted(
        category_counts.items(), key=lambda kv: (-kv[1], kv[0])
    )

    def fmt(mapping: dict[str, int]) -> str:
        if not mapping:
            return "-"
        return ", ".join(
            f"{key}: {value}" for key, value in sorted(mapping.items(), key=lambda kv: (-kv[1], kv[0]))
        )

    lines = [
        "# SunnySett Model Inventory Summary\n",
        "Updated overview of the full repository inventory, including specialized persona bundles and deployment-ready selections.\n",
        "## Summary Statistics\n",
        f"- **Total Categories**: {overall['total_categories']}",
        f"- **Total Models**: {overall['total_models']}",
        f"- **Neural / Deep Learning Models**: {overall['type_counts'].get('Neural/Deep', 0)}",
        f"- **Classical / AutoML Models**: {overall['type_counts'].get('Classical/AutoML', 0)}\n",
        "### Modality Coverage\n",
    ]

    for modality, count in sorted(
        overall["modality_counts"].items(), key=lambda kv: (-kv[1], kv[0])
    ):
        lines.append(f"- **{modality}**: {count}")

    lines.append("")
    lines.append("### Top Pipelines\n")
    for pipeline, count in sorted(
        overall["pipeline_counts"].items(), key=lambda kv: (-kv[1], kv[0])
    ):
        lines.append(f"- **{pipeline}**: {count}")

    lines.append("")
    lines.append("## Category Breakdown\n")
    lines.append("| Category | Model Count | Modality Mix | Model Types |")
    lines.append("|----------|-------------|--------------|-------------|")

    for category, count in sorted_categories:
        mod_mix = fmt(category_modalities.get(category, {}))
        type_mix = fmt(category_types.get(category, {}))
        lines.append(
            f"| {category.replace('_', ' ')} | {count} | {mod_mix} | {type_mix} |"
        )

    markdown = "\n".join(lines) + "\n"
    (ROOT / "sunnysett_models_summary.md").write_text(markdown, encoding="utf-8")


def write_metadata_files(summary: dict) -> None:
    categories: dict[str, list[dict[str, str]]] = defaultdict(list)
    for record in summary["records"]:
        categories[record["category"]].append(record)

    structured = {
        "project": "SunnySett AI Model Discovery Platform",
        "description": "Expanded collection of pre-trained and specialized models across persona, industry, and modality-focused categories.",
        "total_categories": summary["overall"]["total_categories"],
        "total_models": summary["overall"]["total_models"],
        "categories": {},
    }

    for category, models in sorted(categories.items()):
        structured["categories"][category] = {
            "model_count": len(models),
            "models": [],
        }
        for model in sorted(models, key=lambda r: r["model_name"]):
            script_path = f"sunnysett_models/{category}/{model['file']}"
            structured["categories"][category]["models"].append(
                {
                    "id": model["model_name"],
                    "pipeline": model["pipeline"],
                    "modality": model["modality"],
                    "type": model["type"],
                    "description": model["description"],
                    "script_path": script_path,
                    "url": f"https://huggingface.co/{model['model_name']}"
                    if "/" in model["model_name"]
                    else "",
                }
            )

    (ROOT / "sunnysett_models_metadata.json").write_text(
        json.dumps(structured, indent=2), encoding="utf-8"
    )

    fieldnames = [
        "category",
        "model_id",
        "description",
        "pipeline",
        "modality",
        "type",
        "script_path",
        "url",
    ]

    with (ROOT / "sunnysett_models_metadata.csv").open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()
        for category, payload in structured["categories"].items():
            for model in payload["models"]:
                writer.writerow(
                    {
                        "category": category,
                        "model_id": model["id"],
                        "description": model["description"],
                        "pipeline": model["pipeline"],
                        "modality": model["modality"],
                        "type": model["type"],
                        "script_path": model["script_path"],
                        "url": model["url"],
                    }
                )


def main() -> None:
    print("Collecting model scripts from", MODELS_DIR)
    records = collect_model_records()
    print(f"Discovered {len(records)} model scripts across {len({r['category'] for r in records})} categories")

    summary = build_summary(records)

    print("Writing analysis_model_summary.json")
    write_json_summary(summary)

    print("Writing sunnysett_models_summary.md")
    write_markdown(summary)

    print("Writing metadata CSV/JSON")
    write_metadata_files(summary)

    print("✅ Metadata refresh complete")


if __name__ == "__main__":
    main()


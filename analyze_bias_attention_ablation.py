
import json
from pathlib import Path
from collections import defaultdict

import numpy as np

ROOT = Path("/Users/nnn/PycharmProjects/MLScratch")
INPUT = ROOT / "mini_qwen_bias_attention_ablation_results.json"
OUTPUT = ROOT / "mini_qwen_bias_attention_final_analysis.json"

N_BOOTSTRAP = 5000
SEED = 42

with open(INPUT, "r") as f:
    data = json.load(f)


# --------------------------------------------------
# 1. Locate per-question evaluation records
# --------------------------------------------------

def find_records(obj, path="root"):
    """Find a list of dictionaries containing per-question predictions."""
    if isinstance(obj, list):
        if obj and isinstance(obj[0], dict):
            if any(
                "predictions" in x or "projector_wrong" in x
                for x in obj[:10]
            ):
                return obj, path

        for i, value in enumerate(obj[:5]):
            found = find_records(value, f"{path}[{i}]")
            if found is not None:
                return found

    elif isinstance(obj, dict):
        for key, value in obj.items():
            found = find_records(value, f"{path}.{key}")
            if found is not None:
                return found

    return None


found = find_records(data)
if found is None:
    raise ValueError(
        "Could not locate per-question predictions. "
        "Print the JSON structure and adapt find_records()."
    )

records, record_path = found
print(f"Found {len(records)} records at {record_path}")
print("First record:", json.dumps(records[0], indent=2)[:2500])


# --------------------------------------------------
# 2. Extract predictions and labels
# --------------------------------------------------

def get_value(record, *keys):
    for key in keys:
        if key in record and record[key] is not None:
            return record[key]
    return None


def get_prediction(record, condition):
    predictions = record.get("predictions", {})

    if isinstance(predictions, dict):
        value = predictions.get(condition)
    else:
        value = None

    if value is None:
        value = record.get(condition)

    if isinstance(value, dict):
        value = get_value(
            value, "prediction", "pred", "answer",
            "predicted_answer", "label"
        )

    if value is None:
        raise KeyError(
            f"Missing prediction for {condition}. "
            f"Available keys: {list(record.keys())}"
        )

    return str(value).strip().lower()


def get_original(record):
    value = get_value(
        record, "original_answer", "original_target",
        "answer", "target", "ground_truth",
        "correct_answer", "original_label"
    )
    if isinstance(value, dict):
        value = get_value(value, "answer", "label", "color", "shape")
    if value is None:
        raise KeyError("Original target label not found")
    return str(value).strip().lower()


def get_alternative(record):
    value = get_value(
        record, "wrong_target", "wrong_answer",
        "alternative_answer", "other_target",
        "alternate_target", "wrong_label"
    )
    if isinstance(value, dict):
        value = get_value(value, "answer", "label", "color", "shape")
    if value is None:
        raise KeyError("Alternative target label not found")
    return str(value).strip().lower()


def get_image_id(record):
    value = get_value(
        record, "image_index", "image_id",
        "image_filename", "image_path", "scene_index"
    )
    if value is None:
        raise KeyError(
            "Image identifier not found. "
            "Cluster bootstrap requires image IDs."
        )
    return str(value)


rows = []

for record in records:
    rows.append({
        "image_id": get_image_id(record),
        "original": get_original(record),
        "alternative": get_alternative(record),
        "correct": get_prediction(record, "correct"),
        "projector_wrong": get_prediction(
            record, "projector_wrong"
        ),
        "wrong_coordinate": get_prediction(
            record, "wrong_coordinate"
        ),
        "zero": get_prediction(record, "zero"),
    })


# --------------------------------------------------
# 3. Metrics
# --------------------------------------------------

def mean_bool(values):
    return float(np.mean(values))


def calculate_metrics(items):
    return {
        "correct_original": mean_bool([
            r["correct"] == r["original"] for r in items
        ]),
        "projector_wrong_original": mean_bool([
            r["projector_wrong"] == r["original"] for r in items
        ]),
        "projector_wrong_alternative": mean_bool([
            r["projector_wrong"] == r["alternative"]
            for r in items
        ]),
        "both_wrong_original": mean_bool([
            r["wrong_coordinate"] == r["original"]
            for r in items
        ]),
        "both_wrong_alternative": mean_bool([
            r["wrong_coordinate"] == r["alternative"]
            for r in items
        ]),
        "zero_original": mean_bool([
            r["zero"] == r["original"] for r in items
        ]),
        "wrong_conditions_agreement": mean_bool([
            r["projector_wrong"] == r["wrong_coordinate"]
            for r in items
        ]),
        "correct_vs_projector_gap": mean_bool([
            r["correct"] == r["original"] for r in items
        ]) - mean_bool([
            r["projector_wrong"] == r["original"]
            for r in items
        ]),
        "alternative_vs_original_gap": mean_bool([
            r["projector_wrong"] == r["alternative"]
            for r in items
        ]) - mean_bool([
            r["projector_wrong"] == r["original"]
            for r in items
        ]),
    }


metrics = calculate_metrics(rows)

print("\n" + "=" * 65)
print("FINAL COORDINATE ABLATION ANALYSIS")
print("=" * 65)
print(f"Questions: {len(rows)}")
print(f"Unique images: {len(set(r['image_id'] for r in rows))}")

for key, value in metrics.items():
    print(f"{key:35s}: {100 * value:7.2f}%")


# --------------------------------------------------
# 4. Image-clustered bootstrap
# --------------------------------------------------

groups = defaultdict(list)
for row in rows:
    groups[row["image_id"]].append(row)

image_ids = list(groups)
n_images = len(image_ids)

rng = np.random.default_rng(SEED)

bootstrap_values = defaultdict(list)

for _ in range(N_BOOTSTRAP):
    sampled_ids = rng.choice(
        image_ids, size=n_images, replace=True
    )

    sampled_rows = [
        row
        for image_id in sampled_ids
        for row in groups[image_id]
    ]

    sample_metrics = calculate_metrics(sampled_rows)

    for key, value in sample_metrics.items():
        bootstrap_values[key].append(value)


ci = {}

print("\n" + "=" * 65)
print("IMAGE-CLUSTERED BOOTSTRAP 95% CONFIDENCE INTERVALS")
print("=" * 65)

for key, values in bootstrap_values.items():
    lower, upper = np.percentile(values, [2.5, 97.5])
    ci[key] = [float(lower), float(upper)]

    print(
        f"{key:35s}: "
        f"{100 * metrics[key]:6.2f}% "
        f"[{100 * lower:.2f}%, {100 * upper:.2f}%]"
    )


# --------------------------------------------------
# 5. Save results
# --------------------------------------------------

result = {
    "n_questions": len(rows),
    "n_images": n_images,
    "bootstrap_resamples": N_BOOTSTRAP,
    "metrics": metrics,
    "confidence_intervals": ci,
}

with open(OUTPUT, "w") as f:
    json.dump(result, f, indent=2)

print(f"\nSaved: {OUTPUT}")

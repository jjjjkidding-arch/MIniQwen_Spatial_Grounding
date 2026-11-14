import json
import random
import time
from collections import Counter
from copy import deepcopy
from pathlib import Path

import torch
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision.models import ResNet18_Weights
from transformers import AutoTokenizer, AutoModelForCausalLM

from mini_qwen_spatial_model import (
    SpatialVisionEncoder,
    AttentionBiasProjector,
    LoRALinear,
)

# ============================================================
# CONFIGURATION
# ============================================================

PROJECT_DIR = Path("/Users/nnn/PycharmProjects/MLScratch")
ROOT = PROJECT_DIR / "data/clevr/CLEVR_v1.0"
MODEL_ID = "/Users/nnn/Qwen2.5-1.5B-Instruct"

CHECKPOINT = (
    PROJECT_DIR / "mini_qwen_spatial_bias_attention_best.pt"
)

OUTPUT = (
    PROJECT_DIR / "mini_qwen_bias_attention_2k_full_eval_results.json"
)

SEED = 42
N_EVAL_IMAGES = 500
BATCH_SIZE = 4

MAX_QUESTIONS = None
PRINT_EVERY = 100

VALID_ANSWERS = {
    "color": [
        "gray", "red", "blue", "green",
        "brown", "purple", "cyan", "yellow"
    ],
    "shape": [
        "cube", "sphere", "cylinder"
    ],
}

CONDITIONS = ("correct", "wrong_coordinate", "zero")

random.seed(SEED)
torch.manual_seed(SEED)


# ============================================================
# QUESTION FORMAT: EXACTLY AS IN TRAINING
# ============================================================

def normalize_coords(pixel_coords):
    x, y = pixel_coords

    x_crop = x * 0.8 - 80.0
    y_crop = y * 0.8 - 16.0

    return (
        x_crop / 224.0,
        y_crop / 224.0
    )


def make_question(category, pixel_coords):
    x_norm, y_norm = normalize_coords(pixel_coords)

    return (
        f"Target location (normalized image coordinates): "
        f"x={x_norm:.3f}, y={y_norm:.3f}. "
        f"What is the {category} of the object "
        f"at this location?"
    )


# ============================================================
# DATASET — SAME SAMPLING AS PHASE 2A
# ============================================================

class CLEVRFullGridCoordDataset(Dataset):
    def __init__(self, root, split="val", max_images=500, seed=42, max_questions=None):
        self.root = Path(root)
        self.split = split
        self.transform = ResNet18_Weights.DEFAULT.transforms()
        self.image_dir = self.root / "images" / split

        scene_path = (
            self.root / "scenes" /
            f"CLEVR_{split}_scenes.json"
        )

        with open(scene_path, "r") as f:
            scenes = json.load(f)["scenes"]

        scenes = sorted(
            scenes,
            key=lambda s: s["image_filename"]
        )

        if max_images is not None:
            if max_images > len(scenes):
                raise ValueError(
                    f"Requested {max_images} scenes, "
                    f"found {len(scenes)}"
                )

            scenes = random.Random(seed).sample(
                scenes, max_images
            )

        self.examples = []

        for scene in scenes:
            valid_objects = []

            for obj in scene["objects"]:
                x, y = obj["pixel_coords"][:2]

                x_crop = x * 0.8 - 80.0
                y_crop = y * 0.8 - 16.0

                if (
                    0 <= x_crop < 224
                    and 0 <= y_crop < 224
                ):
                    valid_objects.append(obj)

            if len(valid_objects) < 2:
                continue

            for obj_index, obj in enumerate(valid_objects):
                target_xy = tuple(
                    obj["pixel_coords"][:2]
                )

                for category in ("color", "shape"):
                    alternatives = [
                        (j, other)
                        for j, other in enumerate(valid_objects)
                        if j != obj_index
                        and other[category] != obj[category]
                    ]

                    if not alternatives:
                        alternatives = [
                            (j, other)
                            for j, other in enumerate(valid_objects)
                            if j != obj_index
                        ]

                    wrong_index, wrong_obj = alternatives[0]

                    self.examples.append({
                        "image_filename": scene["image_filename"],
                        "category": category,
                        "answer": str(obj[category]),
                        "wrong_answer": str(wrong_obj[category]),
                        "pixel_coords": target_xy,
                        "wrong_pixel_coords": tuple(
                            wrong_obj["pixel_coords"][:2]
                        ),
                        "target_object_index": obj_index,
                        "wrong_object_index": wrong_index,
                    })

        full_count = len(self.examples)

        if max_questions is not None:
            self.examples = random.Random(seed + 1).sample(
                self.examples,
                min(max_questions, len(self.examples))
            )

        print(
            f"{split}: {len(scenes)} selected scenes, "
            f"{full_count} eligible questions, "
            f"{len(self.examples)} evaluated questions",
            flush=True
        )

        print(
            "Category counts:",
            dict(
                Counter(
                    e["category"] for e in self.examples
                )
            ),
            flush=True
        )

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        e = self.examples[idx]

        with Image.open(
            self.image_dir / e["image_filename"]
        ) as im:
            image = self.transform(im.convert("RGB"))

        return {
            "image": image,
            "question": make_question(
                e["category"],
                e["pixel_coords"]
            ),
            "wrong_question": make_question(
                e["category"],
                e["wrong_pixel_coords"]
            ),
            "answer": e["answer"],
            "wrong_answer": e["wrong_answer"],
            "category": e["category"],
            "image_filename": e["image_filename"],
            "pixel_coords": torch.tensor(
                normalize_coords(e["pixel_coords"]),
                dtype=torch.float32
            ),
            "wrong_pixel_coords": torch.tensor(
                normalize_coords(e["wrong_pixel_coords"]),
                dtype=torch.float32
            ),
            "target_object_index": e["target_object_index"],
            "wrong_object_index": e["wrong_object_index"],
        }


# ============================================================
# MAIN
# ============================================================

def main():

    device = torch.device(
        "mps" if torch.backends.mps.is_available()
        else "cpu"
    )

    print("Device:", device, flush=True)
    print("Checkpoint:", CHECKPOINT, flush=True)

    if not CHECKPOINT.is_file():
        raise FileNotFoundError(CHECKPOINT)

    dataset = CLEVRFullGridCoordDataset(
        ROOT,
        split="val",
        max_images=N_EVAL_IMAGES,
        seed=SEED,
        max_questions=MAX_QUESTIONS,
    )

    if len(dataset) == 0:
        raise RuntimeError("No evaluation examples found")

    loader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0,
    )

    # --------------------------------------------------------
    # TOKENIZER AND ANSWER CANDIDATES
    # --------------------------------------------------------

    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_ID,
        local_files_only=True
    )

    candidate_ids = {
        category: [
            tokenizer(
                answer + "<|im_end|>",
                add_special_tokens=False
            )["input_ids"]
            for answer in candidates
        ]
        for category, candidates in VALID_ANSWERS.items()
    }

    # --------------------------------------------------------
    # LOAD QWEN AND LORA
    # --------------------------------------------------------

    print("Loading Qwen...", flush=True)

    qwen = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        dtype=torch.bfloat16,
        local_files_only=True
    ).to(device)

    for p in qwen.parameters():
        p.requires_grad = False

    for layer in qwen.model.layers:
        layer.self_attn.q_proj = LoRALinear(
            layer.self_attn.q_proj,
            r=8,
            alpha=16
        )

        layer.self_attn.v_proj = LoRALinear(
            layer.self_attn.v_proj,
            r=8,
            alpha=16
        )

    # --------------------------------------------------------
    # LOAD VISION ENCODER AND ATTENTION PROJECTOR
    # --------------------------------------------------------

    vision = SpatialVisionEncoder().to(device)
    projector = AttentionBiasProjector(spatial_lambda=10).to(device)


    checkpoint = torch.load(
        CHECKPOINT,
        map_location="cpu",
        weights_only=True
    )
    assert checkpoint["spatial_lambda"] == projector.spatial_lambda

    projector.load_state_dict(
        checkpoint["projector"],
        strict=True
    )

    named_params = dict(qwen.named_parameters())

    expected = {
        name
        for name, p in qwen.named_parameters()
        if p.requires_grad
    }

    actual = set(checkpoint["lora"])

    if expected != actual:
        raise RuntimeError(
            f"LoRA mismatch: "
            f"missing={expected - actual}; "
            f"extra={actual - expected}"
        )

    with torch.no_grad():
        for name, tensor in checkpoint["lora"].items():
            p = named_params[name]
            p.copy_(
                tensor.to(
                    device=p.device,
                    dtype=p.dtype
                )
            )

    qwen.eval()
    vision.eval()
    projector.eval()

    print(
        f"Loaded epoch={checkpoint['epoch']}, "
        f"val_loss={checkpoint['val_loss']:.6f}",
        flush=True
    )

    # --------------------------------------------------------
    # TOKENIZE PROMPT
    # --------------------------------------------------------

    def tokenize_prompt(question):
        prompt = tokenizer.apply_chat_template(
            [
                {
                    "role": "user",
                    "content": question
                }
            ],
            tokenize=False,
            add_generation_prompt=True
        )

        return tokenizer(
            prompt,
            add_special_tokens=False
        )["input_ids"]

    # --------------------------------------------------------
    # CACHED-PREFIX FULL-CANDIDATE SCORING
    # --------------------------------------------------------

    @torch.inference_mode()
    def score_candidates(
        visual_tokens,
        prompt_ids,
        category
    ):

        prompt_tensor = torch.tensor(
            prompt_ids,
            dtype=torch.long,
            device=device
        ).unsqueeze(0)

        text_tokens = (
            qwen.get_input_embeddings()(prompt_tensor)
        )

        prefix_embeddings = torch.cat(
            [
                visual_tokens.to(text_tokens.dtype),
                text_tokens
            ],
            dim=1
        )

        prefix_length = prefix_embeddings.shape[1]

        prefix_output = qwen(
            inputs_embeds=prefix_embeddings,
            attention_mask=torch.ones(
                (1, prefix_length),
                dtype=torch.long,
                device=device
            ),
            use_cache=True
        )

        prefix_cache = prefix_output.past_key_values

        first_log_probs = (
            prefix_output.logits[:, -1, :]
            .float()
            .log_softmax(dim=-1)
        )

        scores = []

        for ids in candidate_ids[category]:

            score = first_log_probs[
                0, ids[0]
            ].item()

            if len(ids) > 1:

                if hasattr(
                    prefix_cache,
                    "to_legacy_cache"
                ):
                    legacy = (
                        prefix_cache.to_legacy_cache()
                    )

                    candidate_cache = tuple(
                        tuple(
                            t.clone() for t in layer
                        )
                        for layer in legacy
                    )

                else:
                    candidate_cache = deepcopy(
                        prefix_cache
                    )

                continuation = torch.tensor(
                    ids[:-1],
                    dtype=torch.long,
                    device=device
                ).unsqueeze(0)

                continuation_output = qwen(
                    input_ids=continuation,
                    attention_mask=torch.ones(
                        (
                            1,
                            prefix_length + len(ids) - 1
                        ),
                        dtype=torch.long,
                        device=device
                    ),
                    past_key_values=candidate_cache,
                    use_cache=False
                )

                log_probs = (
                    continuation_output.logits[0]
                    .float()
                    .log_softmax(dim=-1)
                )

                targets = torch.tensor(
                    ids[1:],
                    dtype=torch.long,
                    device=device
                )

                score += log_probs.gather(
                    1,
                    targets.unsqueeze(1)
                ).sum().item()

            scores.append(score)

        best_index = max(
            range(len(scores)),
            key=lambda i: scores[i]
        )

        return VALID_ANSWERS[category][best_index]

    # --------------------------------------------------------
    # EVALUATION
    # --------------------------------------------------------

    counts = {
        mode: 0
        for mode in CONDITIONS
    }

    category_counts = {}
    results = []

    start = time.perf_counter()

    print(
        "Starting attention coordinate sensitivity evaluation...",
        flush=True
    )

    for batch in loader:

        images = batch["image"].to(device)
        correct_coords = batch["pixel_coords"].to(device)
        wrong_coords = batch["wrong_pixel_coords"].to(device)

        with torch.inference_mode():
            features = vision(images)

            correct_visual, correct_weights = projector(
                features, correct_coords
            )

            wrong_visual, wrong_weights = projector(
                features, wrong_coords
            )

            zero_visual = torch.zeros_like(correct_visual)

        for i in range(images.shape[0]):

            question = batch["question"][i]
            wrong_question = batch["wrong_question"][i]

            category = batch["category"][i]
            gold = batch["answer"][i]
            wrong_gold = batch["wrong_answer"][i]

            correct_ids = tokenize_prompt(question)
            wrong_ids = tokenize_prompt(wrong_question)

            predictions = {
                "correct": score_candidates(
                    correct_visual[i:i + 1],
                    correct_ids,
                    category
                ),
                "wrong_coordinate": score_candidates(
                    wrong_visual[i:i + 1],
                    wrong_ids,
                    category
                ),
                "zero": score_candidates(
                    zero_visual[i:i + 1],
                    correct_ids,
                    category
                ),
            }

            for mode in CONDITIONS:
                counts[mode] += int(
                    predictions[mode] == gold
                )

            if category not in category_counts:
                category_counts[category] = {
                    "n": 0,
                    **{
                        mode: 0
                        for mode in CONDITIONS
                    },
                    "wrong_target_correct": 0,
                    "disagreement": 0,
                }

            cc = category_counts[category]
            cc["n"] += 1

            for mode in CONDITIONS:
                cc[mode] += int(
                    predictions[mode] == gold
                )

            cc["wrong_target_correct"] += int(
                predictions["wrong_coordinate"]
                == wrong_gold
            )

            cc["disagreement"] += int(
                predictions["correct"]
                != predictions["wrong_coordinate"]
            )

            results.append({
                "image_filename":
                    batch["image_filename"][i],
                "category": category,
                "question": question,
                "wrong_question": wrong_question,
                "answer": gold,
                "wrong_answer": wrong_gold,
                "pixel_coords":
                    correct_coords[i].cpu().tolist(),
                "wrong_pixel_coords":
                    wrong_coords[i].cpu().tolist(),
                "target_object_index":
                    int(batch["target_object_index"][i]),
                "wrong_object_index":
                    int(batch["wrong_object_index"][i]),
                "correct_attention_7x7":
                    correct_weights[i, 0].cpu().reshape(7, 7).tolist(),
                "wrong_attention_7x7":
                    wrong_weights[i, 0].cpu().reshape(7, 7).tolist(),
                **predictions,
            })

        n = len(results)

        if (
            n % PRINT_EVERY < images.shape[0]
            or n == len(dataset)
        ):

            if device.type == "mps":
                torch.mps.synchronize()

            elapsed = time.perf_counter() - start

            print(
                f"[{n / len(dataset):.1%}] "
                f"{n}/{len(dataset)} | "
                f"Correct: {counts['correct'] / n:.1%} | "
                f"Wrong-coord (original target): "
                f"{counts['wrong_coordinate'] / n:.1%} | "
                f"Zero: {counts['zero'] / n:.1%} | "
                f"Elapsed: {elapsed / 60:.1f} min | "
                f"ETA: {(len(dataset) - n) * elapsed / n / 60:.1f} min",
                flush=True
            )

    # --------------------------------------------------------
    # SUMMARY
    # --------------------------------------------------------

    n = len(results)

    accuracy = {
        mode: counts[mode] / n
        for mode in CONDITIONS
    }

    wrong_target_accuracy = sum(
        r["wrong_coordinate"] == r["wrong_answer"]
        for r in results
    ) / n

    disagreement = sum(
        r["correct"] != r["wrong_coordinate"]
        for r in results
    ) / n

    correct_only = sum(
        r["correct"] == r["answer"]
        and r["wrong_coordinate"] != r["answer"]
        for r in results
    )

    wrong_only = sum(
        r["correct"] != r["answer"]
        and r["wrong_coordinate"] == r["answer"]
        for r in results
    )

    majority_baselines = {}

    for category in VALID_ANSWERS:
        subset = [
            r for r in results
            if r["category"] == category
        ]

        if subset:
            label, freq = Counter(
                r["answer"] for r in subset
            ).most_common(1)[0]

            majority_baselines[category] = {
                "label": label,
                "accuracy": freq / len(subset)
            }

    by_category = {
        category: {
            "n": c["n"],
            **{
                mode: c[mode] / c["n"]
                for mode in CONDITIONS
            },
            "wrong_target_accuracy":
                c["wrong_target_correct"] / c["n"],
            "prediction_disagreement":
                c["disagreement"] / c["n"],
        }
        for category, c in category_counts.items()
    }

    summary = {
        "checkpoint": str(CHECKPOINT),
        "checkpoint_epoch": checkpoint["epoch"],
        "checkpoint_val_loss": checkpoint["val_loss"],
        "num_questions": n,
        "num_images": len(set(
            r["image_filename"] for r in results
        )),
        "candidate_answers": VALID_ANSWERS,
        "accuracy_against_original_target": accuracy,
        "wrong_coordinate_accuracy_against_wrong_target":
            wrong_target_accuracy,
        "prediction_disagreement": disagreement,
        "correct_only_wins_against_original_target":
            correct_only,
        "wrong_coordinate_only_wins_against_original_target":
            wrong_only,
        "category_accuracy": by_category,
        "majority_baselines": majority_baselines,
        "elapsed_minutes":
            (time.perf_counter() - start) / 60,
        "predictions": results,
    }

    OUTPUT.write_text(
        json.dumps(summary, indent=2)
    )

    # --------------------------------------------------------
    # PRINT RESULTS
    # --------------------------------------------------------

    print("\n" + "=" * 64)
    print("ATTENTION COORDINATE EVALUATION")
    print("=" * 64)

    print(
        f"Questions: {n}; "
        f"images: {summary['num_images']}"
    )

    for mode, value in accuracy.items():
        print(
            f"{mode:18s} vs original target: "
            f"{value:.2%}"
        )

    print(
        f"Wrong-coordinate vs WRONG target: "
        f"{wrong_target_accuracy:.2%}"
    )

    print(
        f"Correct vs wrong-coordinate disagreement: "
        f"{disagreement:.2%}"
    )

    print(
        f"Correct-only wins (original target): "
        f"{correct_only}"
    )

    print(
        f"Wrong-coordinate-only wins "
        f"(original target): {wrong_only}"
    )

    print("\nBY CATEGORY")

    for category, vals in sorted(
        by_category.items()
    ):
        print(
            f"{category} (n={vals['n']}): "
            f"correct={vals['correct']:.2%}, "
            f"wrong-coord/original="
            f"{vals['wrong_coordinate']:.2%}, "
            f"wrong-coord/new="
            f"{vals['wrong_target_accuracy']:.2%}, "
            f"zero={vals['zero']:.2%}, "
            f"disagreement="
            f"{vals['prediction_disagreement']:.2%}"
        )

    print("\nFIRST 10 PREDICTIONS")

    for row in results[:10]:
        print(
            f"Q: {row['question']}\n"
            f"Original: {row['answer']} | "
            f"correct={row['correct']} | "
            f"zero={row['zero']}\n"
            f"Other target: {row['wrong_answer']} | "
            f"wrong-coordinate="
            f"{row['wrong_coordinate']}\n"
        )

    print(
        "Saved:",
        OUTPUT.resolve(),
        flush=True
    )


if __name__ == "__main__":
    main()

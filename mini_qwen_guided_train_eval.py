import json
import random
import time
from collections import Counter
from copy import deepcopy
from pathlib import Path
import torch.nn.functional as F

import torch
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision.models import ResNet18_Weights
from transformers import AutoTokenizer, AutoModelForCausalLM

from mini_qwen_spatial_model import (
    SpatialVisionEncoder,
    SpatialProjector,
    LoRALinear,
)

# ============================================================
# CONFIGURATION
# ============================================================

PROJECT_DIR = Path(
    "/Users/nnn/PycharmProjects/MLScratch"
)

ROOT = PROJECT_DIR / "data/clevr/CLEVR_v1.0"

MODEL_ID = "/Users/nnn/Qwen2.5-1.5B-Instruct"
CHECKPOINT = PROJECT_DIR / "mini_qwen_object_guided_best.pt"
OUTPUT = PROJECT_DIR / "mini_qwen_object_guided_eval_results.json"

BATCH_SIZE = 4
SEED = 42

# First run with 20, then change to None.
MAX_QUESTIONS = None
PRINT_EVERY = 20

random.seed(SEED)
torch.manual_seed(SEED)

# ============================================================
# ATTRIBUTE DEFINITIONS
# ============================================================

VALID_ANSWERS = {
    "color": [
        "gray", "red", "blue", "green",
        "brown", "purple", "cyan", "yellow"
    ],
    "shape": [
        "cube", "sphere", "cylinder"
    ],
}

CONDITIONS = ("correct", "shuffled", "zero")

# ============================================================
# DATASET
# ============================================================


class CLEVRAttributeDataset(Dataset):
    def __init__(self, root, split="val", max_images=500, seed=42):
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
            rng = random.Random(seed)
            scenes = rng.sample(scenes, max_images)

        self.examples = []

        for scene in scenes:
            valid_objects = []

            for obj in scene["objects"]:
                x, y = obj["pixel_coords"][:2]

                x_crop = x * 0.8 - 80.0
                y_crop = y * 0.8 - 16.0

                if 0 <= x_crop < 224 and 0 <= y_crop < 224:
                    valid_objects.append(obj)

            # We need at least two eligible objects
            # for within-image location shuffling.
            if len(valid_objects) < 2:
                continue

            for i, obj in enumerate(valid_objects):
                x, y = obj["pixel_coords"][:2]

                # Fixed different-object location.
                wrong_obj = valid_objects[
                    (i + 1) % len(valid_objects)
                ]

                wx, wy = wrong_obj["pixel_coords"][:2]

                for category in ("color", "shape"):
                    self.examples.append({
                        "image_filename": scene["image_filename"],
                        "pixel_coords": (x, y),
                        "wrong_pixel_coords": (wx, wy),
                        "question": (
                            f"What is the {category} "
                            "of this object?"
                        ),
                        "answer": str(obj[category]),
                        "category": category,
                    })

        print(
            f"{split}: {len(scenes)} selected images, "
            f"{len(self.examples)} questions"
        )

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        example = self.examples[idx]

        path = self.image_dir / example["image_filename"]

        with Image.open(path) as img:
            image = self.transform(img.convert("RGB"))

        return {
            "image": image,
            "pixel_coords": torch.tensor(
                example["pixel_coords"],
                dtype=torch.float32
            ),
            "wrong_pixel_coords": torch.tensor(
                example["wrong_pixel_coords"],
                dtype=torch.float32
            ),
            "question": example["question"],
            "answer": example["answer"],
            "category": example["category"],
            "image_filename": example["image_filename"],
        }



# ============================================================
# MAIN
# ============================================================

def main():

    device = torch.device(
        "mps"
        if torch.backends.mps.is_available()
        else "cpu"
    )

    print("Device:", device, flush=True)

    dataset = CLEVRAttributeDataset(ROOT)

    if MAX_QUESTIONS is not None:
        dataset.examples = dataset.examples[:MAX_QUESTIONS]

    loader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0,
    )

    print(
        "\nValidation questions:",
        len(dataset),
        flush=True
    )

    # --------------------------------------------------------
    # TOKENIZER
    # --------------------------------------------------------

    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_ID,
        local_files_only=True
    )

    # Candidate answers are category-specific.
    # Include the same chat-ending token as training.
    answer_token_lists = {}

    for category, candidates in VALID_ANSWERS.items():

        answer_token_lists[category] = []

        for answer in candidates:

            ids = tokenizer(
                answer + "<|im_end|>",
                add_special_tokens=False
            )["input_ids"]

            if not ids:
                raise RuntimeError(
                    f"Empty tokenization: {answer}"
                )

            answer_token_lists[category].append(ids)

    print("\nCandidate answers by category:")

    for category, candidates in VALID_ANSWERS.items():

        lengths = [
            len(ids)
            for ids in answer_token_lists[category]
        ]

        print(
            f"{category}: {candidates} | "
            f"token lengths: {lengths}"
        )

    # --------------------------------------------------------
    # LOAD QWEN
    # --------------------------------------------------------

    print("\nLoading Qwen...", flush=True)

    qwen = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        dtype=torch.bfloat16,
        local_files_only=True
    ).to(device)

    for p in qwen.parameters():
        p.requires_grad = False

    # Install LoRA adapters exactly as in training.
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
    # LOAD VISION ENCODER AND PROJECTOR
    # --------------------------------------------------------

    print(
        "Loading vision encoder and projector...",
        flush=True
    )

    vision = SpatialVisionEncoder().to(device)
    projector = SpatialProjector().to(device)

    checkpoint = torch.load(
        CHECKPOINT,
        map_location="cpu",
        weights_only=True
    )

    print(
        "Checkpoint keys:",
        checkpoint.keys()
    )

    print(
        "Checkpoint epoch:",
        checkpoint["epoch"]
    )

    print(
        "Checkpoint validation loss:",
        checkpoint["val_loss"]
    )

    projector.load_state_dict(
        checkpoint["projector"],
        strict=True
    )

    # Verify all LoRA parameter names match.
    named_params = dict(qwen.named_parameters())

    expected_lora = {
        name
        for name, param in qwen.named_parameters()
        if param.requires_grad
    }

    loaded_lora = set(
        checkpoint["lora"].keys()
    )

    if expected_lora != loaded_lora:

        raise RuntimeError(
            "LoRA checkpoint mismatch:\n"
            f"Missing: {expected_lora - loaded_lora}\n"
            f"Unexpected: {loaded_lora - expected_lora}"
        )

    for name, tensor in checkpoint["lora"].items():

        param = named_params[name]

        param.data.copy_(
            tensor.to(
                device=param.device,
                dtype=param.dtype
            )
        )

    qwen.eval()
    vision.eval()
    projector.eval()

    print(
        f"\nLoaded epoch {checkpoint['epoch']} | "
        f"Validation loss: "
        f"{checkpoint['val_loss']:.4f}",
        flush=True
    )

    # --------------------------------------------------------
    # CANDIDATE SCORING WITH CACHED PREFIX
    # --------------------------------------------------------

    @torch.inference_mode()
    def score_candidates(
        visual_tokens,
        prompt_ids,
        category
    ):
        candidate_answers = VALID_ANSWERS[category]
        candidate_tokens = answer_token_lists[category]

        prompt_tensor = torch.tensor(
            prompt_ids,
            dtype=torch.long,
            device=device
        ).unsqueeze(0)

        text_tokens = (
            qwen.get_input_embeddings()(
                prompt_tensor
            )
        )

        image_tokens = visual_tokens.to(
            text_tokens.dtype
        )

        prefix_embeddings = torch.cat(
            [image_tokens, text_tokens],
            dim=1
        )

        prefix_length = prefix_embeddings.shape[1]

        prefix_mask = torch.ones(
            (1, prefix_length),
            dtype=torch.long,
            device=device
        )

        # Run expensive image/question prefix once.
        prefix_output = qwen(
            inputs_embeds=prefix_embeddings,
            attention_mask=prefix_mask,
            use_cache=True
        )

        prefix_cache = prefix_output.past_key_values

        # Last prefix position predicts the
        # first answer token.
        first_logits = (
            prefix_output.logits[:, -1, :]
            .float()
            .log_softmax(dim=-1)
        )

        candidate_scores = []

        for ids in candidate_tokens:

            score = first_logits[
                0, ids[0]
            ].item()

            if len(ids) > 1:

                continuation = torch.tensor(
                    ids[:-1],
                    dtype=torch.long,
                    device=device
                ).unsqueeze(0)

                # Ensure candidate cache independence.
                if hasattr(
                    prefix_cache,
                    "to_legacy_cache"
                ):

                    legacy_cache = (
                        prefix_cache.to_legacy_cache()
                    )

                    candidate_cache = tuple(
                        tuple(
                            t.clone()
                            for t in layer
                        )
                        for layer in legacy_cache
                    )

                else:

                    candidate_cache = deepcopy(
                        prefix_cache
                    )

                full_mask = torch.ones(
                    (
                        1,
                        prefix_length + len(ids) - 1
                    ),
                    dtype=torch.long,
                    device=device
                )

                output = qwen(
                    input_ids=continuation,
                    attention_mask=full_mask,
                    past_key_values=candidate_cache,
                    use_cache=False
                )

                log_probs = (
                    output.logits[0]
                    .float()
                    .log_softmax(dim=-1)
                )

                targets = torch.tensor(
                    ids[1:],
                    dtype=torch.long,
                    device=device
                )

                score += (
                    log_probs.gather(
                        1,
                        targets.unsqueeze(1)
                    )
                    .sum()
                    .item()
                )

            candidate_scores.append(score)

        best_index = max(
            range(len(candidate_scores)),
            key=lambda i: candidate_scores[i]
        )

        return candidate_answers[best_index]

    def extract_object_features(features, pixel_coords):
        # features: (B, 49, 512)
        # pixel_coords: (B, 2)

        B = features.shape[0]

        fmap = features.transpose(1, 2).reshape(B, 512, 7, 7)

        coords = pixel_coords.to(features.device)

        x_crop = coords[:, 0] * 0.8 - 80.0
        y_crop = coords[:, 1] * 0.8 - 16.0

        x_norm = 2.0 * x_crop / 224.0 - 1.0
        y_norm = 2.0 * y_crop / 224.0 - 1.0

        grid = torch.stack([x_norm, y_norm], dim=-1)
        grid = grid.reshape(B, 1, 1, 2)

        sampled = F.grid_sample(
            fmap,
            grid,
            mode="bilinear",
            padding_mode="zeros",
            align_corners=False
        )

        return sampled[:, :, 0, 0]

    # --------------------------------------------------------
    # EVALUATE
    # --------------------------------------------------------

    counts = {
        condition: 0
        for condition in CONDITIONS
    }

    category_counts = {}

    results = []

    start_time = time.perf_counter()

    print("\nStarting attribute evaluation...\n")

    for batch_idx, batch in enumerate(
        loader,
        start=1
    ):

        images = batch["image"].to(device)

        correct_coords = batch["pixel_coords"].to(device)
        wrong_coords = batch["wrong_pixel_coords"].to(device)

        with torch.inference_mode():
            features = vision(images)

            correct_features = extract_object_features(
                features, correct_coords
            )

            shuffled_features = extract_object_features(
                features, wrong_coords
            )

            correct_visual = projector(
                correct_features.unsqueeze(1)
            )

            shuffled_visual = projector(
                shuffled_features.unsqueeze(1)
            )

            zero_features = torch.zeros_like(correct_features)

            zero_visual = projector(
                zero_features.unsqueeze(1)
            )

        assert correct_visual.shape[1:] == (1, 1536)
        assert shuffled_visual.shape == correct_visual.shape
        assert zero_visual.shape == correct_visual.shape

        B = images.shape[0]

        for i in range(B):

            question = batch["question"][i]
            gold = batch["answer"][i]
            category = batch["category"][i]

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

            prompt_ids = tokenizer(
                prompt,
                add_special_tokens=False
            )["input_ids"]

            predictions = {}

            for mode, tokens in (
                ("correct", correct_visual),
                ("shuffled", shuffled_visual),
                ("zero", zero_visual)
            ):

                prediction = score_candidates(
                    tokens[i:i + 1],
                    prompt_ids,
                    category
                )

                predictions[mode] = prediction

                counts[mode] += int(
                    prediction == gold
                )

            if category not in category_counts:

                category_counts[category] = {
                    "total": 0,
                    **{
                        condition: 0
                        for condition in CONDITIONS
                    }
                }

            category_counts[category]["total"] += 1

            for mode in CONDITIONS:

                category_counts[category][mode] += int(
                    predictions[mode] == gold
                )

            results.append({
                "question": question,
                "answer": gold,
                "category": category,
                "image_filename": batch["image_filename"][i],
                "pixel_coords": batch["pixel_coords"][i].tolist(),
                "wrong_pixel_coords": batch["wrong_pixel_coords"][i].tolist(),
                **predictions
            })

        n = len(results)

        if (
            n % PRINT_EVERY < B
            or n == len(dataset)
        ):

            if device.type == "mps":
                torch.mps.synchronize()

            elapsed = (
                time.perf_counter() - start_time
            )

            remaining = len(dataset) - n

            eta_seconds = (
                remaining * elapsed / n
            )

            print(
                f"[{n / len(dataset):.1%}] "
                f"{n}/{len(dataset)} | "
                f"Correct: {counts['correct'] / n:.1%} | "
                f"Shuffled: {counts['shuffled'] / n:.1%} | "
                f"Zero: {counts['zero'] / n:.1%} | "
                f"Elapsed: {elapsed / 60:.1f} min | "
                f"ETA: {eta_seconds / 60:.1f} min",
                flush=True
            )

    # --------------------------------------------------------
    # SUMMARY
    # --------------------------------------------------------

    accuracy = {
        mode: counts[mode] / len(dataset)
        for mode in CONDITIONS
    }

    category_accuracy = {}

    for category, values in category_counts.items():

        total = values["total"]

        category_accuracy[category] = {
            "n": total,
            **{
                mode: values[mode] / total
                for mode in CONDITIONS
            }
        }

    disagreement = sum(
        r["correct"] != r["shuffled"]
        for r in results
    ) / len(results)

    correct_only = sum(
        r["correct"] == r["answer"]
        and r["shuffled"] != r["answer"]
        for r in results
    )

    shuffled_only = sum(
        r["shuffled"] == r["answer"]
        and r["correct"] != r["answer"]
        for r in results
    )

    # Descriptive majority baselines from this
    # evaluation subset, not held-out train priors.
    majority_baselines = {}

    for category in VALID_ANSWERS:

        rows = [
            r for r in results
            if r["category"] == category
        ]

        if not rows:
            continue

        gold_counts = Counter(
            r["answer"]
            for r in rows
        )

        majority_label, majority_n = (
            gold_counts.most_common(1)[0]
        )

        majority_baselines[category] = {
            "label": majority_label,
            "accuracy": majority_n / len(rows),
        }

    summary = {
        "checkpoint_epoch": checkpoint["epoch"],
        "checkpoint_val_loss": checkpoint["val_loss"],
        "num_questions": len(dataset),
        "candidate_answers": VALID_ANSWERS,
        "accuracy": accuracy,
        "category_accuracy": category_accuracy,
        "prediction_disagreement": disagreement,
        "correct_only_wins": correct_only,
        "shuffled_only_wins": shuffled_only,
        "majority_baselines": majority_baselines,
        "predictions": results,
    }

    OUTPUT.write_text(
        json.dumps(summary, indent=2)
    )

    # --------------------------------------------------------
    # PRINT RESULTS
    # --------------------------------------------------------

    print("\n" + "=" * 65)
    print("FINAL ATTRIBUTE EVALUATION")
    print("=" * 65)

    for mode, value in accuracy.items():
        print(f"{mode:10s}: {value:.2%}")

    print(
        f"\nCorrect-shuffled accuracy gap: "
        f"{(accuracy['correct'] - accuracy['shuffled']) * 100:+.2f} pp"
    )

    print(
        f"Prediction disagreement: "
        f"{disagreement:.2%}"
    )

    print(
        f"Correct-only wins: {correct_only}"
    )

    print(
        f"Shuffled-only wins: {shuffled_only}"
    )

    print("\nACCURACY BY CATEGORY")

    for category, values in sorted(
        category_accuracy.items()
    ):

        print(
            f"{category:10s} "
            f"(n={values['n']:3d}) | "
            f"correct: {values['correct']:.2%} | "
            f"shuffled: {values['shuffled']:.2%} | "
            f"zero: {values['zero']:.2%}"
        )

    print("\nMAJORITY BASELINES")

    for category, values in sorted(
        majority_baselines.items()
    ):

        print(
            f"{category:10s}: "
            f"{values['label']:10s} "
            f"{values['accuracy']:.2%}"
        )

    print("\nFIRST 10 PREDICTIONS")

    for row in results[:10]:

        print(
            f"\nQ: {row['question']}\n"
            f"Gold: {row['answer']}\n"
            f"Correct: {row['correct']} | "
            f"Shuffled: {row['shuffled']} | "
            f"Zero: {row['zero']}"
        )

    print(
        "\nSaved:",
        OUTPUT.resolve()
    )

    print("Attribute evaluation complete.")


if __name__ == "__main__":
    main()

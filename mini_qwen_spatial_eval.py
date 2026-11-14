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

CHECKPOINT = (
    PROJECT_DIR / "mini_qwen_spatial_5k_best.pt"
)

OUTPUT = (
    PROJECT_DIR
    / "mini_qwen_spatial_attribute_eval_results.json"
)

BATCH_SIZE = 4
SEED = 42

QUESTIONS_PER_CATEGORY = 60

# First run with 20, then change to None.
MAX_QUESTIONS = None
PRINT_EVERY = 20

random.seed(SEED)
torch.manual_seed(SEED)

# ============================================================
# ATTRIBUTE DEFINITIONS
# ============================================================

ATTRIBUTE_OPERATIONS = {
    "query_color": "color",
    "query_shape": "shape",
    "query_size": "size",
    "query_material": "material",
}

ALLOWED_OPERATIONS = {
    "scene",
    "filter_color",
    "filter_shape",
    "filter_size",
    "filter_material",
    "unique",
    *ATTRIBUTE_OPERATIONS.keys(),
}

VALID_ANSWERS = {
    "color": [
        "gray", "red", "blue", "green",
        "brown", "purple", "cyan", "yellow"
    ],
    "shape": [
        "cube", "sphere", "cylinder"
    ],
    "size": [
        "small", "large"
    ],
    "material": [
        "metal", "rubber"
    ],
}

CONDITIONS = ("correct", "shuffled", "zero")

# ============================================================
# DATASET
# ============================================================

class CLEVRAttributeDataset(Dataset):
    def __init__(self, root, split="val"):

        self.root = Path(root)
        self.split = split

        # Match original ResNet18 preprocessing.
        self.transform = (
            ResNet18_Weights.DEFAULT.transforms()
        )

        question_path = (
            self.root
            / "questions"
            / f"CLEVR_{split}_questions.json"
        )

        with open(question_path) as f:
            data = json.load(f)

        all_questions = data["questions"]

        # ----------------------------------------------------
        # Reconstruct the original 500 validation images.
        # ----------------------------------------------------

        spatial_questions = [
            q for q in all_questions
            if any(
                step["function"] == "relate"
                for step in q["program"]
            )
        ]

        spatial_image_names = sorted({
            q["image_filename"]
            for q in spatial_questions
        })

        if len(spatial_image_names) < 500:
            raise RuntimeError(
                "Fewer than 500 eligible spatial images."
            )

        rng = random.Random(SEED)

        selected_images = set(
            rng.sample(spatial_image_names, 500)
        )

        # ----------------------------------------------------
        # Select simple attribute questions.
        # ----------------------------------------------------

        grouped = {
            category: []
            for category in ATTRIBUTE_OPERATIONS.values()
        }

        for q in all_questions:

            if q["image_filename"] not in selected_images:
                continue

            operations = [
                step["function"]
                for step in q["program"]
            ]

            if not operations:
                continue

            final_op = operations[-1]

            if final_op not in ATTRIBUTE_OPERATIONS:
                continue

            # Exclude spatial relations, comparisons,
            # set operations, and other complex programs.
            if any(
                op not in ALLOWED_OPERATIONS
                for op in operations
            ):
                continue

            # Require a uniquely identified object.
            if "unique" not in operations:
                continue

            category = ATTRIBUTE_OPERATIONS[final_op]

            answer = str(q["answer"])

            if answer not in VALID_ANSWERS[category]:
                continue

            grouped[category].append(q)

        self.questions = []

        print("\nATTRIBUTE QUESTION SELECTION")

        for category, candidates in grouped.items():

            rng.shuffle(candidates)

            chosen = candidates[
                :QUESTIONS_PER_CATEGORY
            ]

            self.questions.extend(chosen)

            print(
                f"{category:10s}: "
                f"{len(chosen):3d} selected / "
                f"{len(candidates):4d} eligible"
            )

        if not self.questions:
            raise RuntimeError(
                "No eligible attribute questions found."
            )

        rng.shuffle(self.questions)

        # Apply smoke-test limit before building image map.
        if MAX_QUESTIONS is not None:
            self.questions = self.questions[
                :MAX_QUESTIONS
            ]

        self.image_dir = (
            self.root / "images" / split
        )

        self.image_names = sorted({
            q["image_filename"]
            for q in self.questions
        })

        if len(self.image_names) < 2:
            raise RuntimeError(
                "Need at least two unique images."
            )
        wrong_names = self.image_names.copy()
        derangement_rng = random.Random(SEED + 1)

        while True:
            derangement_rng.shuffle(wrong_names)
            if all(
                original != replacement
                for original, replacement in zip(
                    self.image_names,
                    wrong_names
                )
            ): break

        self.shuffled_map = dict(
            zip(self.image_names, wrong_names)
        )
        print(
            "\nFinal questions:",
            len(self.questions)
        )
        print(
            "Unique images:",
            len(self.image_names)
        )

        category_counts = Counter(
            ATTRIBUTE_OPERATIONS[
                q["program"][-1]["function"]
            ]
            for q in self.questions
        )

        print("Final category counts:")

        for category in VALID_ANSWERS:
            print(
                f"  {category}: "
                f"{category_counts[category]}"
            )

        print("\nGround-truth answer distributions:")

        for category in VALID_ANSWERS:

            labels = Counter(
                str(q["answer"])
                for q in self.questions
                if ATTRIBUTE_OPERATIONS[
                    q["program"][-1]["function"]
                ] == category
            )

            print(f"  {category}: {dict(labels)}")

    def __len__(self):
        return len(self.questions)

    def load_image(self, filename):

        path = self.image_dir / filename

        with Image.open(path) as image:
            return self.transform(
                image.convert("RGB")
            )

    def __getitem__(self, idx):

        q = self.questions[idx]

        filename = q["image_filename"]

        category = ATTRIBUTE_OPERATIONS[
            q["program"][-1]["function"]
        ]

        return {
            "image": self.load_image(filename),
            "wrong_image": self.load_image(
                self.shuffled_map[filename]
            ),
            "question": q["question"],
            "answer": str(q["answer"]),
            "category": category,
            "image_filename": filename,
            "wrong_image_filename": (
                self.shuffled_map[filename]
            ),
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
    def score_candidates(visual_tokens,prompt_ids,category):
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
                    candidate_cache = deepcopy(prefix_cache)

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
                    log_probs.gather(1,targets.unsqueeze(1)).sum().item()
                )

            candidate_scores.append(score)

        best_index = max(
            range(len(candidate_scores)),
            key=lambda i: candidate_scores[i]
        )

        return candidate_answers[best_index]

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

        wrong_images = (
            batch["wrong_image"].to(device)
        )

        with torch.inference_mode():
            correct_visual = projector(vision(images))
            shuffled_visual = projector(vision(wrong_images))
            zero_visual = torch.zeros_like(correct_visual)

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
                "image_filename": (
                    batch["image_filename"][i]
                ),
                "wrong_image_filename": (
                    batch["wrong_image_filename"][i]
                ),
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

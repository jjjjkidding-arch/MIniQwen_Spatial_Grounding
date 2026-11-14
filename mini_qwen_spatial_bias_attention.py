from mini_qwen_spatial_model import (
    SpatialVisionEncoder,
    LoRALinear,
    AttentionBiasProjector
)
import math
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from torch.nn.utils.rnn import pad_sequence
import time
import json
import random
from pathlib import Path
from PIL import Image

from torch.utils.data import Dataset, DataLoader
from torchvision.models import ResNet18_Weights

random.seed(42)
torch.manual_seed(42)

EPOCHS = 1
BATCH_SIZE = 2

LR_PROJECTOR = 2e-4
LR_LORA = 2e-5

N_TRAIN = 2000
N_EVAL = 500



class CLEVRDataset(Dataset):
    def __init__(self, root, split="train", max_images=None, seed=42):
        self.root = Path(root)
        self.split = split
        self.transform = ResNet18_Weights.DEFAULT.transforms()
        self.image_dir = self.root / "images" / split

        scene_path = (
            self.root / "scenes" /
            f"CLEVR_{split}_scenes.json"
        )

        with open(scene_path, "r") as f:
            data = json.load(f)

        scenes = sorted(
            data["scenes"],
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

            if len(valid_objects) < 2:
                continue

            for obj in valid_objects:
                x, y = obj["pixel_coords"][:2]

                coords = [
                    (x * 0.8 - 80.0) / 224.0,
                    (y * 0.8 - 16.0) / 224.0
                ]

                for category in ("color", "shape"):
                    self.examples.append({
                        "image_filename": scene["image_filename"],
                        "question": make_question(
                            category, obj["pixel_coords"][:2]
                        ),
                        "answer": str(obj[category]),
                        "pixel_coords": coords,
                    })

        print(
            f"{split}: {len(scenes)} selected images, "
            f"{len(self.examples)} questions"
        )

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        ex = self.examples[idx]

        image_path = self.image_dir / ex["image_filename"]

        with Image.open(image_path) as img:
            image = self.transform(img.convert("RGB"))

        return {
            "image": image,
            "question": ex["question"],
            "answer": ex["answer"],
            "pixel_coords": torch.tensor(
                ex["pixel_coords"],
                dtype=torch.float32
            ),
        }

def make_question(category, pixel_coords):
    x, y = pixel_coords

    x_crop = x * 0.8 - 80.0
    y_crop = y * 0.8 - 16.0

    x_norm = x_crop / 224.0
    y_norm = y_crop / 224.0

    return (
        f"Target location (normalized image coordinates): "
        f"x={x_norm:.3f}, y={y_norm:.3f}. "
        f"What is the {category} of the object "
        f"at this location?"
    )

def encode_example(question, answer, tokenizer):
    messages = [
        {
            "role": "user",
            "content": question,
        }
    ]

    prompt = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )

    answer_text = str(answer) + "<|im_end|>"

    prompt_ids = tokenizer(
        prompt, add_special_tokens=False
    )["input_ids"]

    answer_ids = tokenizer(
        answer_text, add_special_tokens=False
    )["input_ids"]

    input_ids = prompt_ids + answer_ids
    labels = [-100] * len(prompt_ids) + answer_ids

    return input_ids, labels

@torch.no_grad()
def validate():
    vision_model.eval()
    attn_proj.eval()
    qwen_model.eval()

    total_target_mass = 0.0
    total_centroid_distance = 0.0
    total_attention_entropy = 0.0
    total_peak_distance = 0.0
    total_samples = 0

    total_loss = 0.0
    total_tokens = 0

    for val_batch_num, batch in enumerate(val_loader, start=1):
        images = batch["image"].to(device)
        questions = batch["question"]
        answers = batch["answer"]
        pixel_coords = batch["pixel_coords"].to(device)

        B = images.shape[0]

        input_ids_list = []
        labels_list = []

        for question, answer in zip(questions, answers):
            ids, target_labels = encode_example(
                question, answer, tokenizer
            )

            input_ids_list.append(
                torch.tensor(ids, dtype=torch.long)
            )
            labels_list.append(
                torch.tensor(target_labels, dtype=torch.long)
            )

        input_ids = pad_sequence(
            input_ids_list,
            batch_first=True,
            padding_value=tokenizer.pad_token_id
        ).to(device)

        labels = pad_sequence(
            labels_list,
            batch_first=True,
            padding_value=-100
        ).to(device)

        input_mask = torch.zeros_like(input_ids)

        for i, ids in enumerate(input_ids_list):
            input_mask[i, :len(ids)] = 1

        features = vision_model(images)
        img_tokens, attn_weights = attn_proj(features, pixel_coords)

        weights = attn_weights[:, 0, :].float()  # [B, 49]
        positions = attn_proj.positions.float()

        dist_sq = (
                (positions.unsqueeze(0) - pixel_coords.unsqueeze(1)) ** 2
        ).sum(dim=-1)

        nearest_idx = dist_sq.argmin(dim=-1)

        target_mass = weights.gather(
            1, nearest_idx.unsqueeze(1)
        ).squeeze(1)

        centroid = weights @ positions

        centroid_distance = torch.linalg.vector_norm(
            centroid - pixel_coords,
            dim=-1
        )

        peak_idx = weights.argmax(dim=-1)  # [B]
        peak_positions = positions[peak_idx]  # [B, 2]

        peak_distance = torch.linalg.vector_norm(
            peak_positions - pixel_coords,
            dim=-1
        )

        entropy = -(
                weights * weights.clamp_min(1e-12).log()
        ).sum(dim=-1) / math.log(49)

        total_target_mass += target_mass.sum().item()
        total_centroid_distance += centroid_distance.sum().item()
        total_attention_entropy += entropy.sum().item()
        total_samples += B
        total_peak_distance += peak_distance.sum().item()

        text_tokens = qwen_model.get_input_embeddings()(input_ids)
        img_tokens = img_tokens.to(text_tokens.dtype)

        N = img_tokens.shape[1]
        combined_tokens = torch.cat([img_tokens, text_tokens], dim=1)

        image_labels = torch.full(
            (B, N),
            -100,
            dtype=torch.long,
            device=device
        )

        combined_labels = torch.cat([image_labels, labels], dim=1)

        image_mask = torch.ones(
            (B, N),
            dtype=torch.long,
            device=device
        )

        combined_mask = torch.cat([image_mask, input_mask], dim=1)
        loss = qwen_model(
            inputs_embeds=combined_tokens,
            labels=combined_labels,
            attention_mask=combined_mask,
            use_cache=False
        ).loss

        if not torch.isfinite(loss):
            raise RuntimeError("Validation loss is not finite")

        n_tokens = (combined_labels != -100).sum().item()

        total_loss += loss.item() * n_tokens
        total_tokens += n_tokens

    print(
        f"Attention | "
        f"Target mass: {total_target_mass / total_samples:.4f} | "
        f"Centroid dist: {total_centroid_distance / total_samples:.4f} | "
        f"Peak dist: {total_peak_distance / total_samples:.4f} | "
        f"Normalized entropy: {total_attention_entropy / total_samples:.4f}"
    )
    return total_loss / max(total_tokens, 1)

#--------------
#----DATA------
#--------------

ROOT = (
    "/Users/nnn/PycharmProjects/MLScratch/"
    "data/clevr/CLEVR_v1.0"
)

train_dataset = CLEVRDataset(ROOT, split="train", max_images=N_TRAIN, seed=42)
val_dataset = CLEVRDataset(ROOT, split="val", max_images=N_EVAL, seed=42)

train_loader = DataLoader(
    train_dataset,
    batch_size=2,
    shuffle=True,
    num_workers=0
)

val_loader = DataLoader(
    val_dataset,
    batch_size=2,
    shuffle=False,
    num_workers=0
)

#------------
#------------
CLEVR_PATH = "/Users/nnn/PycharmProjects/MLScratch/mini_qwen_spatial/data/CLEVR_small"
MODEL_ID = "/Users/nnn/Qwen2.5-1.5B-Instruct"

device = torch.device(
    "mps" if torch.backends.mps.is_available() else "cpu"
)
print("Using device:", device)

tokenizer = AutoTokenizer.from_pretrained(
    "/Users/nnn/Qwen2.5-1.5B-Instruct",
    local_files_only=True
)


#----------------------
#-----model loading----
#----------------------
qwen_model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID,
    dtype=torch.bfloat16,
    local_files_only=True
).to(device)

vision_model = SpatialVisionEncoder().to(device)
attn_proj = AttentionBiasProjector(spatial_lambda=10).to(device)

for p in qwen_model.parameters():
    p.requires_grad = False

for layer in qwen_model.model.layers:
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

optimizer = torch.optim.AdamW([
    {
        "params": [p for p in qwen_model.parameters()
                   if p.requires_grad],
        "lr": LR_LORA
    },
    {
        "params": attn_proj.parameters(),
        "lr": LR_PROJECTOR
    }
], weight_decay=0.01)
print("All models loaded successfully!")





#-------------------
#------training-----
#-------------------
best_val_loss = float("inf")

for step in range(EPOCHS):
    if device.type == "mps":
        torch.mps.synchronize()
    start_time = time.perf_counter()

    qwen_model.train()
    attn_proj.train()
    vision_model.eval()

    epoch_loss = 0.0
    epoch_tokens = 0

    total_batches = len(train_loader)

    for batch_num, batch in enumerate(train_loader, start=1):
        optimizer.zero_grad()

        images = batch["image"].to(device)
        questions = batch["question"]
        answers = batch["answer"]
        pixel_coords = batch["pixel_coords"].to(device)

        B = images.shape[0]
        input_ids_list, labels_list = [], []

        for question, answer in zip(questions, answers):
            input_ids, labels = encode_example(question, answer, tokenizer)

            input_ids_list.append(
                torch.tensor(input_ids, dtype=torch.long)
            )
            labels_list.append(
                torch.tensor(labels, dtype=torch.long)
            )



        input_ids = pad_sequence(
            input_ids_list,
            batch_first=True,
            padding_value=tokenizer.pad_token_id
        ).to(device)

        labels = pad_sequence(
            labels_list,
            batch_first=True,
            padding_value=-100
        ).to(device)

        #forward
        with torch.no_grad():
            features = vision_model(images)#B,49,512

        img_tokens, attn_weights = attn_proj(features, pixel_coords)#B,1,512--B,1,1536
        #b,1,49

        input_tokens = qwen_model.model.embed_tokens(input_ids)#B,T,1536
        img_tokens = img_tokens.to(dtype=input_tokens.dtype)

        combined_tokens = torch.cat([img_tokens, input_tokens], dim=1)#B,1+T,1536

        images_labels = torch.full(
            (B, img_tokens.shape[1]), -100,
            dtype=torch.long, device=device
        )
        combined_labels = torch.cat([images_labels, labels], dim=1)

        input_mask = torch.zeros_like(input_ids)
        for i, idx in enumerate(input_ids_list):
            input_mask[i, :len(idx)] = 1

        image_mask = torch.ones(
            (B, img_tokens.shape[1]),
            dtype=torch.long, device=device
        )

        combined_mask = torch.cat([image_mask, input_mask], dim=1)


        #backward
        loss = qwen_model(
            inputs_embeds=combined_tokens,
            labels=combined_labels,
            attention_mask=combined_mask,
            use_cache=False
        ).loss

        if not torch.isfinite(loss):
            raise RuntimeError("Loss is not finite")

        loss.backward()

        torch.nn.utils.clip_grad_norm_(
            list(attn_proj.parameters()) +
            [p for p in qwen_model.parameters() if p.requires_grad],
            max_norm=1.0
        )

        n_tokens = (combined_labels != -100).sum().item()
        epoch_loss += loss.item() * n_tokens
        epoch_tokens += n_tokens

        optimizer.step()


        if batch_num % 200 == 0 or batch_num == total_batches:
            if device.type == "mps":
                torch.mps.synchronize()

            elapsed = time.perf_counter() - start_time
            seconds_per_batch = elapsed / batch_num
            eta_minutes = (
                    (total_batches - batch_num)
                    * seconds_per_batch / 60
            )

            print(
                f"Batch {batch_num}/{total_batches} | "
                f"Loss: {loss.item():.4f} | "
                f"Elapsed: {elapsed / 60:.1f} min | "
                f"ETA: {eta_minutes:.1f} min",
                flush=True
            )


    val_loss = validate()

    if device.type == "mps":
        torch.mps.synchronize()
    elapsed = time.perf_counter() - start_time

    print(
        f"Epoch {step + 1}/{EPOCHS} | "
        f"Train: {epoch_loss / epoch_tokens:.4f} | "
        f"Val: {val_loss:.4f} | "
        f"Time: {elapsed:.2f}s"
    )

    if val_loss < best_val_loss:
        best_val_loss = val_loss

        checkpoint = {
            "epoch": step + 1,
            "val_loss": val_loss,
            "spatial_lambda": attn_proj.spatial_lambda,
            "n_train_scenes": N_TRAIN,
            "n_eval_scenes": N_EVAL,
            "projector": {
                name: tensor.detach().cpu()
                for name, tensor in attn_proj.state_dict().items()
            },
            "lora": {
                name: p.detach().cpu()
                for name, p in qwen_model.named_parameters()
                if p.requires_grad
            }
        }

        torch.save(
            checkpoint,
            "mini_qwen_spatial_bias_attention_best.pt"
        )

        print("Saved best checkpoint!", flush=True)




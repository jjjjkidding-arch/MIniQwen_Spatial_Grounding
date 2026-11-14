from mini_qwen_spatial_model import SpatialVisionEncoder,SpatialProjector,LoRALinear
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from torch.nn.utils.rnn import pad_sequence
import time
import json
from pathlib import Path
from PIL import Image

from torch.utils.data import Dataset, DataLoader
from torchvision.models import ResNet18_Weights
import torch.nn.functional as F

EPOCHS = 1
BATCH_SIZE = 2

LR_PROJECTOR = 2e-4
LR_LORA = 2e-5

N_TRAIN = 5000
N_EVAL = 500



class CLEVRDataset(Dataset):
    def __init__(self, root, split="train", max_images=None, seed=42):
        import random

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
            scenes, key=lambda s: s["image_filename"]
        )

        if max_images is not None:
            rng = random.Random(seed)
            scenes = rng.sample(scenes, max_images)

        self.examples = []

        for scene in scenes:
            for obj in scene["objects"]:
                x, y = obj["pixel_coords"][:2]

                # Match ResNet18 resize + center crop
                x_crop = x * 0.8 - 80.0
                y_crop = y * 0.8 - 16.0

                if not (
                    0 <= x_crop < 224 and
                    0 <= y_crop < 224
                ):
                    continue

                for attribute in ("color", "shape"):
                    self.examples.append({
                        "image_filename": scene["image_filename"],
                        "pixel_coords": (x, y),
                        "question": f"What is the {attribute} of this object?",
                        "answer": str(obj[attribute]),
                    })

        print(
            f"{split}: {len(scenes)} images, "
            f"{len(self.examples)} attribute questions"
        )

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        example = self.examples[idx]

        image_path = (
            self.image_dir / example["image_filename"]
        )

        with Image.open(image_path) as img:
            image = self.transform(img.convert("RGB"))

        return {
            "image": image,
            "pixel_coords": torch.tensor(
                example["pixel_coords"],
                dtype=torch.float32
            ),
            "question": example["question"],
            "answer": example["answer"],
        }


ROOT = (
    "/Users/nnn/PycharmProjects/MLScratch/"
    "data/clevr/CLEVR_v1.0"
)

train_dataset = CLEVRDataset(ROOT, split="train", max_images=N_TRAIN, seed=42)
val_dataset = CLEVRDataset(ROOT, split="val", max_images=N_EVAL, seed=42)


train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=0
)

val_loader = DataLoader(
    val_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=0
)

sample = train_dataset[0]

print("Image:", sample["image"].shape)
print("Coordinates:", sample["pixel_coords"])
print("Question:", sample["question"])
print("Answer:", sample["answer"])

batch = next(iter(train_loader))

print("Batch images:", batch["image"].shape)
print("Batch coordinates:", batch["pixel_coords"].shape)



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


#------------
#------------
MODEL_ID = "/Users/nnn/Qwen2.5-1.5B-Instruct"

print("Done")

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
projector = SpatialProjector().to(device)

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
print("All models loaded successfully!")

optimizer = torch.optim.AdamW([
    {
        "params": [p for p in qwen_model.parameters()
                   if p.requires_grad],
        "lr": LR_LORA
    },
    {
        "params": projector.parameters(),
        "lr": LR_PROJECTOR
    }
], weight_decay=0.01)


@torch.no_grad()
def validate():
    vision_model.eval()
    projector.eval()
    qwen_model.eval()

    total_loss = 0.0
    total_tokens = 0

    for val_batch_num, batch in enumerate(val_loader, start=1):

        images = batch["image"].to(device)
        questions = batch["question"]
        answers = batch["answer"]

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

        obj_features = extract_object_features(
            features,
            batch["pixel_coords"].to(device)
        )

        img_tokens = projector(
            obj_features.unsqueeze(1)
        )

        text_tokens = qwen_model.get_input_embeddings()(input_ids)
        img_tokens = img_tokens.to(text_tokens.dtype)

        N_VIS = img_tokens.shape[1]

        combined_tokens = torch.cat(
            [img_tokens, text_tokens],
            dim=1
        )

        image_labels = torch.full(
            (B, N_VIS),
            -100,
            dtype=torch.long,
            device=device
        )

        combined_labels = torch.cat(
            [image_labels, labels],
            dim=1
        )

        image_mask = torch.ones(
            (B, N_VIS),
            dtype=torch.long,
            device=device
        )

        combined_mask = torch.cat(
            [image_mask, input_mask],
            dim=1
        )

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

    return total_loss / max(total_tokens, 1)

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

    return sampled[:, :, 0, 0]  # (B, 512)


#-------------------
#------training-----
#-------------------
best_val_loss = float("inf")
bad_epochs = 0

for step in range(EPOCHS):
    if device.type == "mps":
        torch.mps.synchronize()
    start_time = time.perf_counter()

    qwen_model.train()
    projector.train()
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
            features = vision_model(images)
            obj_features = extract_object_features(features, pixel_coords)
        img_tokens = projector(obj_features.unsqueeze(1))#B,49,1536

        input_tokens = qwen_model.model.embed_tokens(input_ids)#B,T,1536
        img_tokens = img_tokens.to(dtype=input_tokens.dtype)

        combined_tokens = torch.cat([img_tokens, input_tokens], dim=1)#B,49+T,1536

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
            list(projector.parameters()) +
            [p for p in qwen_model.parameters() if p.requires_grad],
            max_norm=1.0
        )

        n_tokens = (combined_labels != -100).sum().item()
        epoch_loss += loss.item() * n_tokens
        epoch_tokens += n_tokens

        optimizer.step()


        if batch_num % 50 == 0 or batch_num == total_batches:
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
            "projector": {
                name: tensor.detach().cpu()
                for name, tensor in projector.state_dict().items()
            },
            "lora": {
                name: p.detach().cpu()
                for name, p in qwen_model.named_parameters()
                if p.requires_grad
            }
        }

        torch.save(
            checkpoint,
            "mini_qwen_object_guided_best.pt"
        )

        print("Saved best checkpoint!", flush=True)




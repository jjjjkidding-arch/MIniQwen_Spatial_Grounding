from mini_qwen_spatial_model import SpatialVisionEncoder,SpatialProjector,LoRALinear
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from torch.nn.utils.rnn import pad_sequence
import time
import json
from pathlib import Path
from PIL import Image
import random

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

            if len(valid_objects) < 2:
                continue

            for obj in valid_objects:
                x, y = obj["pixel_coords"][:2]

                for category in ("color", "shape"):
                    self.examples.append({
                        "image_filename": scene["image_filename"],
                        "pixel_coords": (x, y),
                        "category": category,
                        "answer": str(obj[category]),
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

        question = make_question(
            example["category"],
            example["pixel_coords"]
        )

        return {
            "image": image,
            "question": question,
            "answer": example["answer"],
            "pixel_coords": torch.tensor(
                example["pixel_coords"],
                dtype=torch.float32
            ),
            "category": example["category"],
            "image_filename": example["image_filename"],
        }



ROOT = (
    "/Users/nnn/PycharmProjects/MLScratch/"
    "data/clevr/CLEVR_v1.0"
)

train_dataset = CLEVRDataset(ROOT, split="train", max_images=N_TRAIN, seed=42)
val_dataset = CLEVRDataset(ROOT, split="val", max_images=N_EVAL, seed=42)

print("\nExample:")
print(train_dataset[0]["question"])
print("Answer:", train_dataset[0]["answer"])
print("Coordinates:", train_dataset[0]["pixel_coords"])

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
CLEVR_PATH = "/Users/nnn/PycharmProjects/MLScratch/mini_qwen_spatial/data/CLEVR_small"
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
        img_tokens = projector(features)

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

        img_tokens = projector(features)#B,49,1536

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
            "mini_qwen_spatial_fullgrid_coord_2k_best.pt"
        )

        print("Saved best checkpoint!", flush=True)




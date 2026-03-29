import os, json
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

import torch
from torch.utils.data import DataLoader
from dataclasses import dataclass
from typing import Dict, List, Any
from transformers import (
    AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig,
    get_cosine_schedule_with_warmup
)
from peft import prepare_model_for_kbit_training, LoraConfig, get_peft_model
from datasets import load_dataset
from tqdm import tqdm
import bitsandbytes as bnb
from peft import PeftModel

import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--lr", type=float, required=True)
parser.add_argument("--epoch", type=int, required=True)
args = parser.parse_args()




# -------- 基本設定 --------
MODEL_NAME = "meta-llama/Llama-3.1-8B-Instruct"
DATA_PATH  = "/home/S113062661/playground/1_Data/Training_data/qwen2.5_32b_1000.jsonl"
OUTPUT_NAME = f"Llama3_lora_qwen2.5_{args.lr}_second_noreason_5epoch_0316"
OUTPUT_DIR = f"model/"+OUTPUT_NAME

MAX_LEN = 24000
LR      = args.lr
EPOCHS  = 5
BATCH_PER_DEV = 2
GRAD_ACCUM = 8
USE_BF16 = True

device = torch.device("cuda")

os.makedirs("model", exist_ok=True)


# ====== 1) Tokenizer ======
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, use_fast=False)
tokenizer.pad_token = "<|eot_id|>"
tokenizer.padding_side = "right"



# ====== 3) 載入模型 & LoRA ======
model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    device_map="auto",
    trust_remote_code=False,
    attn_implementation="flash_attention_2",
    dtype=torch.bfloat16,
)
model.config.use_cache = False

model.enable_input_require_grads()
model.gradient_checkpointing_enable()

model = PeftModel.from_pretrained(
    model,
    "../../2_1st_training/manual_lora_3k_5e-05_5epoch/epoch_5",   # ← 你第一階段 LoRA 的資料夾
    is_trainable=True       # ← 一定要 True，才能再訓練
)


# ====== 4) dataset ======
def format_example(ex):
    return f"""You are a QA assistant.

1. Output a section titled "According to the context:" ONLY if there are supporting sentences in the Context. Do NOT generate empty or irrelevant content in this section.
2. In "According to the context:", repeat ONLY the sentences from "Context:" that are needed to answer the Question. Keep their exact [title][index] prefix. Do NOT include unrelated sentences.
3. Then perform iterative merging to compress the selected sentences into ONE sentence:
   - Each step must be titled exactly: "Step {{k}}:"
   - In each step, you MUST output a section titled exactly "Available sentences:" listing ALL currently available sentences.
   - Each available sentence MUST be on its own line and MUST start with its identifier:
       * Original sentences keep their original prefix: [title][index]
       * Merged sentences MUST use the prefix: [Merge][k]
   - In each step, choose exactly TWO currently available and relevant sentences to merge.
   - Before "Merged:", explicitly indicate the two sentences being merged under:
        "To merge A:" and "To merge B:"
   - Merge them into exactly ONE sentence that preserves ONLY information relevant to answering the Question.
   - The merged sentence MUST NOT introduce any new facts not explicitly stated in the two input sentences.
   - The merged sentence MUST be a single sentence.
   - Output the merged result in a single line titled exactly "Merged:" and the line MUST start with the new identifier [Merge][k] followed by the merged sentence.
   - After merging, the two input sentences are removed and replaced by the new [Merge][k] sentence for the next step.
4. Stop when there is exactly ONE sentence remaining.
5. After merging stops, output a section titled "Question:" and repeat the question again. 
6. Then output a section titled "According to the merged sentence:" and include ONLY the final remaining sentence (with its identifier).
7. Then output "Answer:" and answer the Question using ONLY the information in "According to the merged sentence:".
8. Your answer MUST be extremely short.

Question:
{ex["question"]}
Context:
{ex["context"]}""", f"""

{ex["answer"]}<|eot_id|>"""


def tokenize_fn(ex):
    x, y = format_example(ex)
    full = x + y

    toks = tokenizer(
        full,
        max_length=MAX_LEN,
        padding=False,
        truncation=True,
    )

    # labels：複製 input_ids
    labels = toks["input_ids"].copy()

    # x 的 token 長度
    x_ids = tokenizer(x).input_ids
    x_len = len(x_ids)


    # y 從 x_len 之後開始 → 訓練目標
    labels[:x_len] = [-100] * x_len
    
    # mask padding
    labels[toks["attention_mask"] == 0] = -100

    toks["labels"] = labels
    return toks


# ====== 6) 載入 dataset ======
ext = os.path.splitext(DATA_PATH)[1].lower()

ds = load_dataset("json", data_files=DATA_PATH, split="train")
ds = ds.select(range(1000))

train_ds = ds.map(tokenize_fn, remove_columns=ds.column_names)


# ====== 7) DataLoader ======
@dataclass
class DynamicPaddingCollator:
    pad_token_id: int
    def __call__(self, features):

        # 找當前 batch 的最大長度
        max_len = max(len(f["input_ids"]) for f in features)

        batch = {
            "input_ids": [],
            "attention_mask": [],
            "labels": []
        }

        for f in features:
            seq_len = len(f["input_ids"])
            pad_len = max_len - seq_len

            # input_ids
            input_ids = f["input_ids"] + [self.pad_token_id] * pad_len
            # attention_mask
            attn = f["attention_mask"] + [0] * pad_len
            # labels
            labels = f["labels"] + [-100] * pad_len

            batch["input_ids"].append(input_ids)
            batch["attention_mask"].append(attn)
            batch["labels"].append(labels)

        # 轉成 tensor
        for k in batch:
            batch[k] = torch.tensor(batch[k])

        return batch


train_loader = DataLoader(
    train_ds,
    batch_size=BATCH_PER_DEV,
    shuffle=True,
    collate_fn=DynamicPaddingCollator(
        pad_token_id=tokenizer.pad_token_id
    )
)


# ====== 8) optimizer & scheduler ======
optimizer = torch.optim.AdamW(model.parameters(), lr=LR)

num_training_steps = EPOCHS * len(train_loader)
warmup_steps = int(num_training_steps * 0.03)

scheduler = get_cosine_schedule_with_warmup(
    optimizer,
    num_warmup_steps=warmup_steps,
    num_training_steps=num_training_steps
)


# ====== 9) Training loop ======
model.train()

global_step = 0
for epoch in range(EPOCHS):

    for step, batch in tqdm(enumerate(train_loader),total=len(train_loader), desc=f"Epoch {epoch+1}/{EPOCHS}"):
        for k in batch:
            batch[k] = batch[k].to(device)

        outputs = model(**batch)
        loss = outputs.loss
        loss = loss / GRAD_ACCUM
        loss.backward()

        if (step + 1) % GRAD_ACCUM == 0:
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()
            global_step += 1

    # 每 epoch 存一次
    save_dir = os.path.join(OUTPUT_DIR, f"epoch_{epoch+1}")
    os.makedirs(save_dir, exist_ok=True)
    model.save_pretrained(save_dir)

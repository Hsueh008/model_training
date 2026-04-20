import os
import glob
import json
import torch
import wandb
import random
import numpy as np
from datasets import load_dataset, concatenate_datasets, Features, Value
from peft import LoraConfig
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments
from trl import SFTTrainer, SFTConfig
from torch.utils.data import SequentialSampler, DataLoader
from torch.utils.data.distributed import DistributedSampler

from datetime import datetime

def prepare_messages(example):
    inst = example.get('instruction', '')
    resp = example.get('response', '')

    messages = [
        {"role": "user", "content": inst},
        {"role": "assistant", "content": resp}
    ]

    return {"messages": messages}

# ==========================================
# 定義難度分數字典 (由你的分析結果提供)
# ==========================================
# model_id = "Qwen/Qwen2.5-3B"
# data_dir = "data/wizardlm/Qwen2.5-3B-jsonl"
model_id = "Qwen/Qwen2.5-3B-Instruct"
data_dir = "data/wizardlm/Qwen2.5-3B-Instruct-jsonl"

ifd_file_name = f"metadata/{model_id.split('/')[-1]}-pr-score.json"
difficulty_scores = json.load(open(ifd_file_name, "r", encoding="utf-8"))

category_names = sorted(difficulty_scores.keys(), key=lambda k: difficulty_scores[k][0])

# ==========================================
# 1. 載入資料與分層抽樣
# ==========================================

def get_constraint_level(filename):
    if "Empty" in filename:
        return 0
        
    keywords = ["Fo", "Lex", "Log", "Num", "Sem"]
    level = 0
    for kw in keywords:
        if kw in filename:
            level += 1
            
    return level

TRAIN_SIZE = 1000
EVAL_SIZE = 200
SEED = 42
PERCENTILE_THRESHOLD = 10

eval_sampled_datasets = []
datasets_by_level = {i: [] for i in range(6)}

for name in category_names:
    level = get_constraint_level(name)

    path = os.path.join(data_dir, f"{name}.jsonl")
    if not os.path.exists(path):
        print(f"  [警告] 找不到檔案: {path}，跳過此階段！")
        continue

    raw_dataset = load_dataset("json", data_files=path, split="train")

    # ================= 階段一：依據 ppl_to_empty 去除離群值 =================
    ppl_scores = raw_dataset["ppl_to_empty"]

    q1 = np.percentile(ppl_scores, 25)
    q3 = np.percentile(ppl_scores, 75)
    iqr = q3 - q1

    lower_bound = q1 - 1.5 * iqr
    upper_bound = q3 + 1.5 * iqr
    
    no_outlier_dataset = raw_dataset.filter(
        lambda x: lower_bound <= x["ppl_to_empty"] <= upper_bound
    )

    # ================= 階段二：依據 ifd_score 進行難度篩選 =================
    clean_ifd_scores = no_outlier_dataset["ifd_score"]
    threshold_value = np.percentile(clean_ifd_scores, PERCENTILE_THRESHOLD)

    filtered_dataset = no_outlier_dataset.filter(
        lambda x: x["ifd_score"] >= threshold_value,
        keep_in_memory=True,
        load_from_cache_file=False
    )

    # ================= 階段三：隨機採樣訓練和驗證資料 =================
    shuffled_filtered = filtered_dataset.shuffle(seed=SEED)
    train_raw_ds = shuffled_filtered.select(range(TRAIN_SIZE))
    eval_raw_ds = shuffled_filtered.select(range(TRAIN_SIZE, TRAIN_SIZE + EVAL_SIZE))

    train_ds = train_raw_ds.map(prepare_messages, remove_columns=raw_dataset.column_names, keep_in_memory=True, load_from_cache_file=False)
    eval_ds = eval_raw_ds.map(prepare_messages, remove_columns=raw_dataset.column_names, keep_in_memory=True, load_from_cache_file=False)
    eval_sampled_datasets.append(eval_ds)

    datasets_by_level[level].append(train_ds)

ordered_train_datasets = []
expected_multipliers = [1, 5, 10, 10, 5, 1]

for curr_level in range(6):
    level_list = datasets_by_level[curr_level]

    if len(level_list) > 0:
        level_combined_ds = concatenate_datasets(level_list)

        actual_size = len(level_combined_ds)
        expected_size = TRAIN_SIZE * expected_multipliers[curr_level]

        if actual_size != expected_size:
            error_msg = (
                f"[錯誤] Level {curr_level} 資料量異常！\n"
                f"預期要有 {expected_multipliers[curr_level]} 種組合共 {expected_size} 筆，\n"
                f"但實際只組合了 {actual_size} 筆資料。\n"
                f"請檢查資料夾內是否有漏檔，或過濾條件 (ppl/ifd) 是否導致某些類別被刪光。"
            )
            # 強烈建議在做研究時，資料量不對就直接讓程式停止，避免浪費算力
            raise ValueError(error_msg)
        
        level_shuffled_ds = level_combined_ds.shuffle(seed=SEED+curr_level)
        ordered_train_datasets.append(level_shuffled_ds)

    else:
        raise ValueError(f"[錯誤] Level {curr_level} 完全沒有收集到任何資料！")

# ==========================================
# 最終資料集組合與輸出統計
# ==========================================
# 嚴格按順序接起來 (絕對不可 shuffle 全局！)
curriculum_train_dataset = concatenate_datasets(ordered_train_datasets)

# 驗證集維持全局打亂不變
eval_dataset = concatenate_datasets(eval_sampled_datasets).shuffle(seed=SEED)

# ==========================================
# 3. 載入 Tokenizer (必須先載入，才能在接下來的函數中使用)
# ==========================================
print(f"正在載入 Tokenizer ({model_id})...")

tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "right" 

new_chat_template = open(f"metadata/{model_id.split('/')[-1]}-new-chat-template.txt", "r", encoding="utf-8").read().replace('\\\\', '\\')

tokenizer.chat_template = new_chat_template

# ==========================================
# 5. 載入模型 (針對 A100 雙卡與 Qwen 最佳化)
# ==========================================
print(f"正在載入模型 ({model_id})...")

model = AutoModelForCausalLM.from_pretrained(
    model_id,
    dtype=torch.bfloat16,
    attn_implementation="flash_attention_2",
    trust_remote_code=True
)
model.config.use_cache = False

# peft_config = LoraConfig(
#     r=16,               
#     lora_alpha=32,      
#     lora_dropout=0.1,
#     bias="none",
#     task_type="CAUSAL_LM",
#     target_modules=["q_proj", "v_proj"] 
# )

# ==========================================
# 5.5 初始化 WandB (建議加在 TrainingArguments 之前)
# ==========================================
timestamp = datetime.now().strftime("%m%d-%H%M")

output_name = f"{model_id.split('/')[-1]}-cl-level-random-{timestamp}"
wandb_name = f"curriculum-level-randomp-{timestamp}"

wandb.init(
    project="qwen-constraint-finetune",     
    name=wandb_name,
    tags=[
        f"{model_id}", 
        "curriculum-learning", 
        "lr-constant-warmup",
        "level-random"
    ],
)

# ==========================================
# 6. 設定訓練參數與啟動 SFTTrainer
# ==========================================
class CurriculumTrainer(SFTTrainer):
    def _get_train_sampler(self, dataset=None):
        train_dataset = dataset if dataset is not None else self.train_dataset
        if train_dataset is None:
            return None
        return SequentialSampler(train_dataset)

sft_config = SFTConfig(
    output_dir=f"models/{output_name}_results",
    max_length=2048,
    num_train_epochs=1,
    per_device_train_batch_size=4,
    per_device_eval_batch_size=8,
    gradient_accumulation_steps=4,
    learning_rate=1e-5,
    logging_steps=10,
    eval_strategy="steps",
    eval_steps=100,
    save_strategy="steps",
    save_steps=100,
    save_total_limit=3,
    bf16=True,
    optim="adamw_torch",
    lr_scheduler_type="constant_with_warmup",
    weight_decay=0.01,
    warmup_ratio=0.05,
    report_to="wandb",
    run_name=wandb_name,
    ddp_find_unused_parameters=False,
    assistant_only_loss=True,
    # Full FT (Lora FT 需註解掉)
    gradient_checkpointing=True,
    gradient_checkpointing_kwargs={'use_reentrant': False}
)

trainer = CurriculumTrainer(
    model=model,
    train_dataset=curriculum_train_dataset, 
    eval_dataset=eval_dataset,
    # peft_config=peft_config,                 
    processing_class=tokenizer,
    args=sft_config,
)

# ==========================================
# 6.5 訓練前驗證：將第一個 Batch 的 Masking 寫入 Log 檔
# ==========================================
import os

def check_masking_to_log(trainer, tokenizer, log_file="masking_check.log"):
    print(f"正在生成 Token Masking 檢查日誌至 {log_file} ...")
    
    # 直接向 Trainer 索取準備好餵給模型的 DataLoader
    train_dataloader = trainer.get_train_dataloader()
    
    # 抽出第一個 Batch
    batch = next(iter(train_dataloader))
    
    # 取出 Batch 中的第一筆資料
    input_ids = batch["input_ids"][0]
    labels = batch["labels"][0]
    
    with open(log_file, "w", encoding="utf-8") as f:
        f.write("="*60 + "\n")
        f.write(f"Masking Check Log\n")
        f.write("="*60 + "\n")
        f.write(f"{'Token (解碼後)':<30} | {'Label ID'}\n")
        f.write("-" * 60 + "\n")
        
        for i in range(len(input_ids)):
            token_id = input_ids[i].item()
            label_id = labels[i].item()
            
            # 將單個 Token 解碼回文字 (使用 repr 顯示換行符號 \n 等)
            token_str = repr(tokenizer.decode([token_id])) 
            
            if label_id == -100:
                f.write(f"{token_str:<30} | -100  (不計算 Loss)\n")
            else:
                f.write(f"{token_str:<30} | {label_id:<5} <--- 模型學習預測目標\n")
                
    print(f"✅ 檢查日誌已儲存！請打開 {log_file} 確認 -100 的位置是否正確。")

# 執行檢查函數
check_masking_to_log(trainer, tokenizer, log_file=f"masking_check_{timestamp}.log")

# ==========================================
# 7. 開始訓練
# ==========================================
trainer.train()
wandb.finish()

# ==========================================
# 7. 儲存最終模型
# ==========================================
output_dir_final = f"models/{output_name}_final"
trainer.model.save_pretrained(output_dir_final)
tokenizer.save_pretrained(output_dir_final)

print(f"訓練完成！最終模型已儲存至 {output_dir_final}")
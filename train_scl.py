import os
import glob
import torch
import wandb
import random
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
    const = example.get('constraint', None)

    if not const:  
        const_str = ""
    else:
        const_str = "\n".join([f"- {v}" for v in const.values()])

    if const_str.strip():
        user_content = f"### Instruction:\n{inst}\n\n### Constraints:\n{const_str}"
    else:
        user_content = f"### Instruction:\n{inst}"

    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": user_content},
        {"role": "assistant", "content": resp}
    ]

    return {"messages": messages}

# ==========================================
# 1. 載入資料與分層抽樣
# ==========================================
data_dir = "/home/S113062615/build_dataset/dataset_jsonl"
file_paths = glob.glob(f"{data_dir}/*.jsonl")

train_dict = {}
eval_sampled_datasets = []

TRAIN_SIZE = 1000
EVAL_SIZE = 200
SEED = 42


for path in file_paths:
    dataset_name = os.path.basename(path).replace(".jsonl", "")

    raw_dataset = load_dataset("json", data_files=path, split="train")
    raw_dataset = raw_dataset.map(
        prepare_messages,
        remove_columns=raw_dataset.column_names,
        desc=f"Standardizing {dataset_name}"
    )
    shuffled_raw = raw_dataset.shuffle(seed=SEED)

    train_ds = shuffled_raw.select(range(TRAIN_SIZE))
    eval_ds = shuffled_raw.select(range(TRAIN_SIZE, TRAIN_SIZE + EVAL_SIZE))

    train_dict[dataset_name] = train_ds
    eval_sampled_datasets.append(eval_ds)

# ==========================================
# 2. 滾雪球式 (Snowballing) Curriculum 資料合併
# ==========================================
category_names = list(train_dict.keys())
random.seed(SEED)
random.shuffle(category_names)

print("\n🎲 本次盲猜的 Curriculum 訓練順序 (滾雪球模式)：")
for i, name in enumerate(category_names):
    print(f"Stage {i+1}: {name}")
print("\n")

ordered_train_datasets = []
STAGE_SIZE = 1000           # 每個階段固定 1000 題
NEW_CONCEPT_RATIO = 0.5     # 50% 新概念，50% 舊概念複習 (可根據實驗調整)

for k, name in enumerate(category_names):
    current_ds = train_dict[name]
    
    if k == 0:
        # 第一階段：沒有過去的記憶，100% 都是第一種約束
        stage_ds = current_ds.shuffle(seed=SEED).select(range(STAGE_SIZE))
    else:
        # 計算新舊資料的配額
        new_samples_count = int(STAGE_SIZE * NEW_CONCEPT_RATIO)  # 500 題
        rehearsal_count = STAGE_SIZE - new_samples_count         # 500 題
        
        # 1. 抽取當前階段的「新概念」
        new_part = current_ds.shuffle(seed=SEED+k).select(range(new_samples_count))
        
        # 2. 抽取「舊概念複習」(從 Stage 0 到 Stage k-1 的資料池中隨機抽)
        past_names = category_names[:k]
        past_datasets = [train_dict[p] for p in past_names]
        
        # 把過去所有看過的資料全部倒進一個大池子裡
        combined_past = concatenate_datasets(past_datasets)
        # 從大池子裡隨機抽出指定的複習數量
        rehearsal_part = combined_past.shuffle(seed=SEED+k).select(range(rehearsal_count))
        
        # 3. 合併新舊資料，並在「該階段內部」進行打亂
        # (確保同一個 Batch 裡會同時混雜新舊概念，消滅梯度衝擊)
        stage_ds = concatenate_datasets([new_part, rehearsal_part]).shuffle(seed=SEED+k)
        
    ordered_train_datasets.append(stage_ds)

# 最後，把 32 個階段嚴格按順序接起來 (這裡絕對不可 shuffle！)
curriculum_train_dataset = concatenate_datasets(ordered_train_datasets)

# 驗證集維持全局打亂不變，作為客觀的考試標準
combined_eval = concatenate_datasets(eval_sampled_datasets)
eval_dataset = combined_eval.shuffle(seed=SEED)

# ==========================================
# 3. 載入 Tokenizer (必須先載入，才能在接下來的函數中使用)
# ==========================================
model_id = "Qwen/Qwen2.5-3B-Instruct"
print(f"正在載入 Tokenizer ({model_id})...")

tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "right" 

new_chat_template = """{%- if tools %}
    {{- '<|im_start|>system\n' }}
    {%- if messages[0]['role'] == 'system' %}
        {{- messages[0]['content'] }}
    {%- else %}
        {{- 'You are Qwen, created by Alibaba Cloud. You are a helpful assistant.' }}
    {%- endif %}
    {{- "\\n\\n# Tools\\n\\nYou may call one or more functions to assist with the user query.\\n\\nYou are provided with function signatures within <tools></tools> XML tags:\\n<tools>" }}
    {%- for tool in tools %}
        {{- "\\n" }}
        {{- tool | tojson }}
    {%- endfor %}
    {{- "\\n</tools>\\n\\nFor each function call, return a json object with function name and arguments within <tool_call></tool_call> XML tags:\\n<tool_call>\\n{\\"name\\": <function-name>, \\"arguments\\": <args-json-object>}\\n</tool_call><|im_end|>\\n" }}
{%- else %}
    {%- if messages[0]['role'] == 'system' %}
        {{- '<|im_start|>system\n' + messages[0]['content'] + '<|im_end|>\n' }}
    {%- else %}
        {{- '<|im_start|>system\nYou are Qwen, created by Alibaba Cloud. You are a helpful assistant.<|im_end|>\n' }}
    {%- endif %}
{%- endif %}
{%- for message in messages %}
    {%- if (message.role == "user") or (message.role == "system" and not loop.first) %}
        {{- '<|im_start|>' + message.role + '\n' + message.content + '<|im_end|>\n' }}
    {%- elif message.role == "assistant" and not message.tool_calls %}
        {{- '<|im_start|>assistant\n' }}{% generation %}{{ message.content }}{% endgeneration %}{{- '<|im_end|>\n' }}
    {%- elif message.role == "assistant" %}
        {{- '<|im_start|>assistant' }}
        {%- generation %}
            {%- if message.content %}
                {{- '\n' + message.content }}
            {%- endif %}
            {%- for tool_call in message.tool_calls %}
                {%- if tool_call.function is defined %}
                    {%- set tool_call = tool_call.function %}
                {%- endif %}
                {{- '\n<tool_call>\n{"name": "' }}
                {{- tool_call.name }}
                {{- '", "arguments": ' }}
                {{- tool_call.arguments | tojson }}
                {{- '}\n</tool_call>' }}
            {%- endfor %}
        {%- endgeneration %}
        {{- '<|im_end|>\n' }}
    {%- elif message.role == "tool" %}
        {%- if (loop.index0 == 0) or (messages[loop.index0 - 1].role != "tool") %}
            {{- '<|im_start|>user' }}
        {%- endif %}
        {{- '\n<tool_response>\n' }}
        {{- message.content }}
        {{- '\n</tool_response>' }}
        {%- if loop.last or (messages[loop.index0 + 1].role != "tool") %}
            {{- '<|im_end|>\n' }}
        {%- endif %}
    {%- endif %}
{%- endfor %}
{%- if add_generation_prompt %}
    {{- '<|im_start|>assistant\n' }}
{%- endif %}"""

tokenizer.chat_template = new_chat_template
# ==========================================
# 4. 定義 Prompt 格式化函數 (使用 apply_chat_template)
# ==========================================
def formatting_prompts_func(example):
    output_texts = []
    inst = example['instruction']
    const = example['constraint']
    resp = example['response']
    
    if const and const.strip():
        user_content = f"### Instruction:\n{inst}\n\n### Constraints:\n{const}"
    else:
        user_content = f"### Instruction:\n{inst}"
        
    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": user_content},
        {"role": "assistant", "content": resp}
    ]

    formatted_text = tokenizer.apply_chat_template(
        messages, 
        tokenize=False, 
        add_generation_prompt=False
    )

    return formatted_text

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

peft_config = LoraConfig(
    r=16,               
    lora_alpha=32,      
    lora_dropout=0.1,
    bias="none",
    task_type="CAUSAL_LM",
    target_modules=["q_proj", "v_proj"] 
)

# ==========================================
# 5.5 初始化 WandB (建議加在 TrainingArguments 之前)
# ==========================================
timestamp = datetime.now().strftime("%m%d-%H%M")

output_name = f"qwen_2_5_3B_scl_{timestamp}"
wandb_name = f"snowballing-curriculum-random-search-{timestamp}"

wandb.init(
    project="qwen-constraint-finetune",     
    name=wandb_name,
    tags=["qwen2.5-3b", "a100", "curriculum"] 
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
    num_train_epochs=5,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    gradient_accumulation_steps=4,
    learning_rate=2e-5,
    logging_steps=10,
    eval_strategy="steps",
    eval_steps=100,
    save_strategy="epoch",
    # save_steps=100,
    save_total_limit=5,
    bf16=True,
    optim="adamw_torch",
    lr_scheduler_type="cosine",
    weight_decay=0.01,
    warmup_ratio=0.2,
    report_to="wandb",
    run_name=wandb_name,
    ddp_find_unused_parameters=False,
    assistant_only_loss=True
)

trainer = CurriculumTrainer(
    model=model,
    train_dataset=curriculum_train_dataset, 
    eval_dataset=eval_dataset,
    peft_config=peft_config,                 
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
check_masking_to_log(trainer, tokenizer, log_file="masking_check.log")

# ==========================================
# 7. 開始訓練
# ==========================================
trainer.train()
wandb.finish()

# ==========================================
# 7. 儲存最終模型
# ==========================================
output_dir_final = f"models/{output_name}_results"
trainer.model.save_pretrained(output_dir_final)
tokenizer.save_pretrained(output_dir_final)

print(f"訓練完成！最終模型已儲存至 {output_dir_final}")
import os
import glob
import torch
import wandb
from datasets import load_dataset, concatenate_datasets, Features, Value
from peft import LoraConfig
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments
from trl import SFTTrainer, SFTConfig

def standardize_format(example):
    inst = example.get('instruction', '')
    resp = example.get('response', '')
    const = example.get('constraint', None)

    if const is None:
        const_str = ""
    else:
        const_str = "\n".join([f"- {v}" for v in const.values()])

    return {
        'instruction': inst,
        'constraint': const_str,
        'response': resp
    }

# ==========================================
# 1. 載入資料與分層抽樣
# ==========================================
data_dir = "/home/S113062615/build_dataset/dataset_jsonl"
file_paths = glob.glob(f"{data_dir}/*.jsonl")

train_sampled_datasets = []
eval_sampled_datasets = []

TRAIN_SIZE = 1000
EVAL_SIZE = 200
SEED = 42

target_features = Features({
    'instruction': Value('string'),
    'constraint': Value('string'),
    'response': Value('string')
})

for path in file_paths:
    raw_dataset = load_dataset("json", data_files=path, split="train")
    raw_dataset = raw_dataset.map(
        standardize_format,
        remove_columns=raw_dataset.column_names,
        features=target_features,
        desc=f"Standardizing {os.path.basename(path)}"
    )
    shuffled_raw = raw_dataset.shuffle(seed=SEED)

    train_ds = shuffled_raw.select(range(TRAIN_SIZE))
    eval_ds = shuffled_raw.select(range(TRAIN_SIZE, TRAIN_SIZE + EVAL_SIZE))

    train_sampled_datasets.append(train_ds)
    eval_sampled_datasets.append(eval_ds)

# ==========================================
# 2. 合併資料與全局打亂
# ==========================================
combined_train = concatenate_datasets(train_sampled_datasets)
train_dataset = combined_train.shuffle(seed=SEED)

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
    dtype=torch.bfloat16,                  # A100 必備 bfloat16
    attn_implementation="flash_attention_2",     # 啟動 Flash Attention 2
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
wandb.init(
    project="qwen-constraint-finetune",     # 你的專案名稱 (在 WandB 後台會建立這個專案)
    name="baseline-stratified-1000",        # 這次訓練的名稱 (方便你辨識這是哪一次實驗)
    tags=["qwen2.5-3b", "a100", "baseline"] # 加上標籤方便未來篩選
)

# ==========================================
# 6. 設定訓練參數與啟動 SFTTrainer
# ==========================================
sft_config = SFTConfig(
    output_dir="./qwen_2_5_3B_results",
    max_length=2048,
    num_train_epochs=3,                  
    per_device_train_batch_size=8,       
    per_device_eval_batch_size=8,        
    gradient_accumulation_steps=2,       
    learning_rate=5e-5,                  
    logging_steps=10,                    
    eval_strategy="steps",         
    eval_steps=200,                      
    save_strategy="steps",               
    save_steps=200,                      
    save_total_limit=3,                  
    bf16=True,                           
    optim="adamw_torch",                 
    lr_scheduler_type="cosine",          
    weight_decay=0.1,
    warmup_ratio=0.2,     
    report_to="wandb",
    run_name="baseline-stratified-1000",
    ddp_find_unused_parameters=False,
)

trainer = SFTTrainer(
    model=model,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    peft_config=peft_config,                 
    formatting_func=formatting_prompts_func, 
    processing_class=tokenizer,
    args=sft_config,
)

# 開始訓練
trainer.train()
wandb.finish()

# ==========================================
# 7. 儲存最終模型
# ==========================================
output_dir_final = "./qwen_2_5_3B_final"
trainer.model.save_pretrained(output_dir_final)
tokenizer.save_pretrained(output_dir_final)

print(f"訓練完成！最終模型已儲存至 {output_dir_final}")
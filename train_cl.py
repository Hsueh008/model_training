import os
import glob
import torch
import wandb
import random
from datasets import load_dataset, concatenate_datasets, Features, Value
from peft import LoraConfig
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments
from trl import SFTTrainer, SFTConfig, DataCollatorForCompletionOnlyLM
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
# 2. 合併資料與全局打亂
# ==========================================
category_names = list(train_dict.keys())
random.seed(SEED)
random.shuffle(category_names)

print("\n🎲 本次盲猜的 Curriculum 訓練順序 (請務必記錄下來)：")
for i, name in enumerate(category_names):
    print(f"Stage {i+1}: {name}")
print("\n")

ordered_train_datasets = []
for name in category_names:
    ordered_train_datasets.append(train_dict[name])

curriculum_train_dataset = concatenate_datasets(ordered_train_datasets)

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

output_name = f"qwen_2_5_3B_cl_{timestamp}"
wandb_name = f"curriculum-random-{timestamp}"

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


# 1. 告訴 Collator Assistant 回覆的起始標籤 (針對 Qwen)
response_template = "<|im_start|>assistant\n"

# 2. 建立只計算 Assistant loss 的 Collator
collator = DataCollatorForCompletionOnlyLM(
    response_template=response_template, 
    tokenizer=tokenizer,
    instruction_template="<|im_start|>user\n" # 加上 instruction_template 更保險
)
    

sft_config = SFTConfig(
    output_dir=f"./{output_name}_results",
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
    run_name=wandb_name,
    ddp_find_unused_parameters=False,
)

trainer = CurriculumTrainer(
    model=model,
    train_dataset=curriculum_train_dataset, 
    eval_dataset=eval_dataset,
    peft_config=peft_config,                  
    processing_class=tokenizer,
    args=sft_config,
    data_collator=collator,
)

# 開始訓練
trainer.train()
wandb.finish()

# ==========================================
# 7. 儲存最終模型
# ==========================================
output_dir_final = f"./{output_name}_final"
trainer.model.save_pretrained(output_dir_final)
tokenizer.save_pretrained(output_dir_final)

print(f"訓練完成！最終模型已儲存至 {output_dir_final}")
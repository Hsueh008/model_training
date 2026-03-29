import os
import glob
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
# 定義難度分數字典 (由你的分析結果提供)
# ==========================================
difficulty_scores = {
    'SemLogFoLexNum': 0.33032368803303774,
    'SemLogFoLex': 0.3279662492613896,
    'SemLogNum': 0.3250998220206835,
    'SemLogFoNum': 0.3250306610599934,
    'SemFoLexNum': 0.319829908322398,
    'SemFoLex': 0.31764362984711525,
    'SemLogLexNum': 0.3174920147557576,
    'SemLogFo': 0.31667231442529997,
    'SemNum': 0.31094453096579916,
    'SemFoNum': 0.3099319022595381,
    'SemFo': 0.30202839620352173,
    'SemLexNum': 0.30145205740005543,
    'SemLog': 0.2862949073062036,
    'SemLogLex': 0.28225684916754723,
    'SemLex': 0.26815506781355664,
    'Sem': 0.26743652228919507,
    'LogFoLexNum': 0.22242263980918345,
    'LogFoLex': 0.21632580546088043,
    'LogLexNum': 0.21187451633317614,
    'FoLexNum': 0.20812035184736538,
    'FoLex': 0.20141503709070027,
    'LexNum': 0.1963912997673322,
    'LogFoNum': 0.1961137214933391,
    'LogNum': 0.18853943961001682,
    'LogFo': 0.1788895161941736,
    'FoNum': 0.17615488756875833,
    'Num': 0.16968997854355053,
    'LogLex': 0.1674304500063813,
    'Fo': 0.15883948702874123,
    'Lex': 0.15393219724383392,
    'Log': 0.13211949825653113,
    'Empty': 0.11597802414885774
}

category_names = sorted(difficulty_scores.keys(), key=lambda k: difficulty_scores[k])

# ==========================================
# 1. 載入資料與分層抽樣
# ==========================================
data_dir = "5_evaluation_qwen_jsonl"

TRAIN_SIZE = 1000
EVAL_SIZE = 200
STAGE_SIZE = 1000
NEW_CONCEPT_RATIO = 0.5
SEED = 42

PERCENTILE_THRESHOLD = 25

stats_list = []
eval_sampled_datasets = []
ordered_train_datasets = []

print("\n🚀 本次的 Curriculum 訓練順序 (由簡入難 + 滾雪球模式)：")

# ==========================================
# 核心迴圈：依序載入 -> 抽樣算統計 -> 轉換 -> 滾雪球
# ==========================================
for k, name in enumerate(category_names):
    score = difficulty_scores[name]
    # print(f"Stage {k+1}: {name} (難度: {score:.4f})")
    
    path = os.path.join(data_dir, f"{name}.jsonl")
    if not os.path.exists(path):
        print(f"  [警告] 找不到檔案: {path}，跳過此階段！")
        continue

    raw_dataset = load_dataset("json", data_files=path, split="train")

    all_scores = raw_dataset["ifd_score"]
    threshold_value = np.percentile(all_scores, PERCENTILE_THRESHOLD)

    filtered_dataset = raw_dataset.filter(
        lambda x: x["ifd_score"] >= threshold_value, 
        desc=f"Filtering {name}"
    )

    shuffled_filtered = filtered_dataset.shuffle(seed=SEED)
    train_raw_ds = shuffled_filtered.select(range(TRAIN_SIZE))
    eval_raw_ds = shuffled_filtered.select(range(TRAIN_SIZE, TRAIN_SIZE + EVAL_SIZE))

    stats_list.append({
        "dataset_name": name,
        "train_mean": np.mean(train_raw_ds["ifd_score"]),
        "train_std": np.std(train_raw_ds["ifd_score"]),
        "eval_mean": np.mean(eval_raw_ds["ifd_score"]),
        "eval_std": np.std(eval_raw_ds["ifd_score"])
    })

    train_ds = train_raw_ds.map(prepare_messages, remove_columns=raw_dataset.column_names)
    eval_ds = eval_raw_ds.map(prepare_messages, remove_columns=raw_dataset.column_names,)
    eval_sampled_datasets.append(eval_ds)

    stage_ds = train_ds.shuffle(seed=SEED+k)
    ordered_train_datasets.append(stage_ds)


# ==========================================
# 最終資料集組合與輸出統計
# ==========================================
# 嚴格按順序接起來 (絕對不可 shuffle 全局！)
curriculum_train_dataset = concatenate_datasets(ordered_train_datasets)

# 驗證集維持全局打亂不變
eval_dataset = concatenate_datasets(eval_sampled_datasets).shuffle(seed=SEED)

# print("\n" + "="*50)
# print("IFD Score Statistics Summary")
# print("="*50)
# for stats in stats_list:
#     print(f"Dataset: {stats['dataset_name']}")
#     print(f"  [Train] Mean: {stats['train_mean']:.4f}, Std: {stats['train_std']:.4f}")
    # print(f"  [Eval]  Mean: {stats['eval_mean']:.4f}, Std: {stats['eval_std']:.4f}")
    # print("-" * 50)

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
        {{- '<|im_start|>assistant\n' }}{% generation %}{{ message.content }}{{- '<|im_end|>' }}{% endgeneration %}{{- '\n' }}
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
        # {"role": "system", "content": "You are a helpful assistant."}, # <--- Qwen2.5 has default system prompt
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

output_name = f"qwen2.5-3B-scl-ordering-{timestamp}"
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
    num_train_epochs=1,
    per_device_train_batch_size=4,
    per_device_eval_batch_size=8,
    gradient_accumulation_steps=8,
    learning_rate=1e-5,
    logging_steps=10,
    eval_strategy="steps",
    eval_steps=100,
    save_strategy="steps",
    save_steps=100,
    save_total_limit=100,
    bf16=True,
    optim="adamw_torch",
    lr_scheduler_type="cosine",
    weight_decay=0.01,
    warmup_ratio=0.2,
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
check_masking_to_log(trainer, tokenizer, log_file="masking_check.log")

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
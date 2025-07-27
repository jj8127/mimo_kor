# ---------------------------
# 1. 필수 라이브러리
# ---------------------------
import torch
import sys
sys.path.append('/content')

from transformers import Trainer, TrainingArguments, AutoTokenizer, DataCollatorForLanguageModeling
from datasets import load_dataset
from modeling_mimo import MiMoForCausalLM
from configuration_mimo import MiMoConfig

# ---------------------------
# 2. Tokenizer (Qwen2 기반)
# ---------------------------
tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2-7B")

# pad_token이 없으면 EOS를 사용 (attention mask 관련 오류 방지)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

# ---------------------------
# 3. Dataset (한국어 위키백과)
# ---------------------------
dataset = load_dataset("wikimedia/wikipedia", "20231101.ko")

# 빈 문서 제거
dataset = dataset.filter(lambda x: x["text"] is not None and len(x["text"]) > 0)

# ---------------------------
# 4. Tokenize + labels 추가
# ---------------------------
def tokenize_function(examples):
    output = tokenizer(
        examples["text"],
        truncation=True,
        max_length=1024,
        padding="max_length"
    )
    # labels 추가 (언어모델용)
    output["labels"] = output["input_ids"].copy()
    return output

tokenized_dataset = dataset.map(
    tokenize_function,
    batched=True,
    remove_columns=dataset["train"].column_names  # ['id','url','title','text']
)

# Train / Eval 분리
split_dataset = tokenized_dataset["train"].train_test_split(test_size=0.01)
train_dataset = split_dataset["train"]
eval_dataset = split_dataset["test"]

# ---------------------------
# 5. Model Config & 생성
# ---------------------------
config = MiMoConfig(
    vocab_size=len(tokenizer),
    num_hidden_layers=12,       # 테스트용 작은 크기
    hidden_size=768,
    num_attention_heads=12,
    intermediate_size=3072,
    num_nextn_predict_layers=1  # 실험 시 0,1,2... 변경
)
model = MiMoForCausalLM(config)

# ---------------------------
# 6. Data Collator (동적 padding)
# ---------------------------
data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

# ---------------------------
# 7. Training Arguments
# ---------------------------
training_args = TrainingArguments(
    output_dir="./mimo_korean",
    do_eval=True,
    report_to="none",        # wandb 사용 안 함
    run_name="mimo_run",
    eval_steps=500,
    logging_steps=100,
    save_steps=500,
    num_train_epochs=1,
    per_device_train_batch_size=2,
    per_device_eval_batch_size=2,
    learning_rate=5e-5,
    fp16=True,
    save_total_limit=2,
    remove_unused_columns=False
)

# ---------------------------
# 8. Trainer
# ---------------------------
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    data_collator=data_collator
)

# ---------------------------
# 9. Train & Save
# ---------------------------
print("Train Started...")
trainer.train()
print("Train Completed!")

trainer.save_model("./mimo_korean_model")
tokenizer.save_pretrained("./mimo_korean_model")

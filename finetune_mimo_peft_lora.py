# coding: utf-8
"""Fine-tune MiMo-7B model on a Korean dataset using LoRA (PEFT).

This script demonstrates loading MiMo-7B from local safetensor files,
applying LoRA via the PEFT library and training on an instruction style
SFT dataset such as KoAlpaca. After training, LoRA adapters are saved to
``output_dir``.

Python 3.11 compatible.
"""

from typing import Dict, List

import torch
from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    Trainer,
    TrainingArguments,
    DataCollatorForLanguageModeling,
)

# Import local MiMo model class
from modeling_mimo import MiMoForCausalLM
from configuration_mimo import MiMoConfig

# PEFT utilities for LoRA
from peft import LoraConfig, get_peft_model, TaskType


# --------------------------------------------------
# 1. Load Tokenizer
# --------------------------------------------------

# Folder containing config.json and weight shards such as
# ``model-00001-of-00004.safetensors``
MODEL_DIR = "./"

# ``trust_remote_code`` is required because the model implementation is
# provided in this repository.  We load the configuration explicitly first
# to ensure the custom model type is recognized.
config = MiMoConfig.from_pretrained(
    MODEL_DIR,
    trust_remote_code=True,
    local_files_only=True,
)

tokenizer = AutoTokenizer.from_pretrained(
    MODEL_DIR,
    config=config,
    trust_remote_code=True,
    local_files_only=True,
)

# Use EOS as padding token if pad_token is missing
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token


# --------------------------------------------------
# 2. Load Dataset
# --------------------------------------------------

# Example: KoAlpaca-v1 from HuggingFace. For a local JSONL dataset,
# change to ``load_dataset('json', data_files={'train': 'train.jsonl',
# 'validation': 'validation.jsonl'})``.
dataset = load_dataset("beomi/KoAlpaca-v1")


# --------------------------------------------------
# 3. Preprocess Dataset
# --------------------------------------------------

def format_prompt(example: Dict[str, str]) -> str:
    """Combine instruction, optional input and output into one text."""
    inst = example.get("instruction", "").strip()
    usr = example.get("input", "").strip()
    out = example.get("output", "").strip()
    if usr:
        return (
            f"### Instruction:\n{inst}\n\n### Input:\n{usr}\n\n### Response:\n{out}"
        )
    return f"### Instruction:\n{inst}\n\n### Response:\n{out}"


def tokenize_function(batch: List[Dict[str, str]]):
    texts = [format_prompt(x) for x in batch]
    tokenized = tokenizer(
        texts,
        max_length=1024,
        truncation=True,
        padding="max_length",
    )
    tokenized["labels"] = tokenized["input_ids"].copy()
    return tokenized


processed = dataset.map(
    tokenize_function,
    batched=True,
    remove_columns=dataset["train"].column_names,
)

train_dataset = processed["train"]
validation_dataset = processed.get("validation", processed["test"])


# --------------------------------------------------
# 4. Load Base Model from Local Files
# --------------------------------------------------

model = MiMoForCausalLM.from_pretrained(
    MODEL_DIR,
    torch_dtype=torch.bfloat16,
    device_map="auto",
    local_files_only=True,
    trust_remote_code=True,
)


# --------------------------------------------------
# 5. Apply LoRA using PEFT
# --------------------------------------------------

lora_config = LoraConfig(
    r=8,
    lora_alpha=32,
    lora_dropout=0.05,
    bias="none",
    task_type=TaskType.CAUSAL_LM,
    target_modules=[
        "q_proj",
        "v_proj",
        "k_proj",
        "o_proj",
        "gate_proj",
        "up_proj",
        "down_proj",
    ],
)

model = get_peft_model(model, lora_config)
model.print_trainable_parameters()


# --------------------------------------------------
# 6. Training Arguments
# --------------------------------------------------

output_dir = "./mimo_lora"
training_args = TrainingArguments(
    output_dir=output_dir,
    per_device_train_batch_size=1,  # adjust per GPU memory
    gradient_accumulation_steps=8,
    num_train_epochs=1,
    learning_rate=2e-4,
    bf16=True,
    logging_steps=50,
    save_steps=500,
    evaluation_strategy="steps",
    eval_steps=500,
    save_total_limit=2,
    report_to="none",
)

collator = DataCollatorForLanguageModeling(tokenizer, mlm=False)


# --------------------------------------------------
# 7. Trainer
# --------------------------------------------------

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=validation_dataset,
    data_collator=collator,
)


# --------------------------------------------------
# 8. Train and Save
# --------------------------------------------------

trainer.train()
trainer.save_model(output_dir)
tokenizer.save_pretrained(output_dir)


# --------------------------------------------------
# 9. Inference Example
# --------------------------------------------------

model.eval()
example_prompt = "### Instruction:\n한국의 수도는 어디인가요?\n\n### Response:"
inputs = tokenizer(example_prompt, return_tensors="pt").to(model.device)
with torch.no_grad():
    generated = model.generate(**inputs, max_new_tokens=50)
print(tokenizer.decode(generated[0], skip_special_tokens=True))

print("Training complete. Model and LoRA adapters saved to", output_dir)

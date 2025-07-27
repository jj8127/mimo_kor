import json

import torch
from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling,
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training


# -------------------------------------------------------------
# 1. 데이터 로드
# -------------------------------------------------------------
def load_sft_dataset(train_path: str | None = None,
                     valid_path: str | None = None,
                     hf_dataset: str = "beomi/KoAlpaca-v1.1a"):
    """주어진 경로 혹은 HuggingFace 데이터셋을 로드하여 `train`과 `validation` 분할을 반환합니다."""

    if train_path and valid_path:
        # JSONL 형식 (각 줄이 {"instruction":..., "input":..., "output":...}) 을 가정
        def _load(path):
            with open(path, "r", encoding="utf-8") as f:
                lines = [json.loads(l) for l in f]
            return lines
        train_data = _load(train_path)
        valid_data = _load(valid_path)
        dataset = {
            "train": train_data,
            "validation": valid_data,
        }
    else:
        # HuggingFace Datasets에서 KoAlpaca-v1 불러오기
        dataset = load_dataset(hf_dataset)
    return dataset


def format_sft(example: dict) -> dict:
    """SFT 형식의 데이터를 프롬프트와 정답 텍스트로 변환한다."""
    instruction = example.get("instruction", "")
    input_text = example.get("input", "")
    output = example.get("output", "")
    text = f"### 질문: {instruction}\n### 입력: {input_text}\n### 답변: {output}"
    return {"text": text}


# -------------------------------------------------------------
# 2. 모델 로드 (로컬 경로 사용, trust_remote_code=True)
# -------------------------------------------------------------

def load_model(model_dir: str = "./", dtype=torch.float16):
    """로컬 디렉터리에서 MiMo 모델과 토크나이저를 로드한다."""
    tokenizer = AutoTokenizer.from_pretrained(model_dir, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_dir,
        trust_remote_code=True,
        torch_dtype=dtype,
        device_map="auto",
    )
    return tokenizer, model


# -------------------------------------------------------------
# 3. LoRA 적용
# -------------------------------------------------------------

def apply_lora(model, r: int = 8, alpha: int = 16, dropout: float = 0.05):
    """모델에 LoRA 어댑터를 적용한다."""
    # 4bit 등 양자화 모델 학습을 원한다면 `prepare_model_for_kbit_training` 사용
    model = prepare_model_for_kbit_training(model)

    target_modules = [
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj",
    ]
    lora_config = LoraConfig(
        r=r,
        lora_alpha=alpha,
        target_modules=target_modules,
        lora_dropout=dropout,
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()
    return model


# -------------------------------------------------------------
# 4. 파인튜닝
# -------------------------------------------------------------

def main():
    # 환경 설정 --------------------------------------------------
    model_dir = "./"  # 모델 파일이 위치한 디렉터리
    output_dir = "./lora_output"  # 체크포인트 저장 경로

    # (필요 시 조절) GPU 메모리 상황에 따라 아래 값 조절
    per_device_train_batch_size = 1
    gradient_accumulation_steps = 4

    # 데이터 로드 -------------------------------------------------
    dataset = load_sft_dataset()
    dataset = dataset.map(format_sft)

    # 토크나이저, 모델 로드 -------------------------------------
    tokenizer, base_model = load_model(model_dir)

    # LoRA 적용 --------------------------------------------------
    model = apply_lora(base_model)

    # 데이터 collator -------------------------------------------
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer, mlm=False
    )

    # TrainingArguments ----------------------------------------
    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=1,
        per_device_train_batch_size=per_device_train_batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
        learning_rate=2e-4,
        fp16=True,
        logging_steps=10,
        save_strategy="epoch",
        evaluation_strategy="epoch",
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset["train"],
        eval_dataset=dataset.get("validation"),
        tokenizer=tokenizer,
        data_collator=data_collator,
    )

    trainer.train()

    # LoRA 어댑터 저장 -----------------------------------------
    trainer.model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)

    # 예시 추론 -------------------------------------------------
    prompt = "안녕하세요"
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    with torch.no_grad():
        outputs = model.generate(**inputs, max_new_tokens=50)
    print(tokenizer.decode(outputs[0], skip_special_tokens=True))


if __name__ == "__main__":
    main()

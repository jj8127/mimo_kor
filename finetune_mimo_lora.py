import os

import torch
from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoConfig,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling,
)
from modeling_mimo import MiMoForCausalLM
from configuration_mimo import MiMoConfig
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training


# -------------------------------------------------------------
# 1. 데이터 로드
# -------------------------------------------------------------
def load_sft_dataset(
    train_path: str | None = None,
    valid_path: str | None = None,
    # Default to the latest KoAlpaca dataset
    hf_dataset: str = "Beomi/KoAlpaca-v1.1a",
):
    """주어진 경로 혹은 HuggingFace 데이터셋을 로드하여 ``DatasetDict`` 를 반환한다.

    로컬 JSONL 파일을 사용할 경우 ``datasets`` 의 ``json`` 로더를 이용해
    ``DatasetDict`` 객체를 생성한다. 각 JSONL 라인은 ``instruction``/``input``/
    ``output`` 필드를 가진다고 가정한다.
    """

    if train_path:
        # ``load_dataset("json")`` 을 사용하면 map 등 ``datasets`` API 활용 가능
        data_files = {"train": train_path}
        if valid_path:
            data_files["validation"] = valid_path
        dataset = load_dataset("json", data_files=data_files)
    else:
        try:
            dataset = load_dataset(hf_dataset)
        except Exception as e:
            raise RuntimeError(
                f"Dataset '{hf_dataset}' could not be loaded. "
                f"Check the dataset name or provide local JSONL files."
            ) from e
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
    """로컬 디렉터리에서 MiMo 모델과 토크나이저를 로드한다.

    가중치 파일은 `model-00001-of-00004.safetensors` 등으로 분할되어 있다는
    가정하에 `model.safetensors.index.json`을 이용해 로드합니다.
    """
    tokenizer = AutoTokenizer.from_pretrained(
        model_dir,
        trust_remote_code=True,
        local_files_only=True,
    )

    config_path = os.path.join(model_dir, "config.json")
    if os.path.exists(config_path):
        config = MiMoConfig.from_json_file(config_path)
    else:
        try:
            # 기본 Qwen 설정을 불러와 MiMo 전용 설정으로 변환
            base = AutoConfig.from_pretrained(
                "Qwen/Qwen1.5-7B",
                trust_remote_code=True,
            )
            config = MiMoConfig.from_dict(base.to_dict())
            print(
                "config.json not found, loaded Qwen base config from the Hub. "
                "Provide a local config.json to avoid this."
            )
        except Exception as e:
            raise FileNotFoundError(
                "config.json is required in model_dir when offline"
            ) from e

    model = MiMoForCausalLM.from_pretrained(
        model_dir,
        config=config,
        trust_remote_code=True,
        torch_dtype=dtype,
        device_map="auto",
        local_files_only=True,
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
    # 메모리가 부족하면 batch size를 줄이고 gradient_accumulation_steps를 늘리면
    # 동일한 효과로 더 작은 메모리 사용이 가능합니다.
    per_device_train_batch_size = 1
    gradient_accumulation_steps = 4

    # 데이터 로드 -------------------------------------------------
    dataset = load_sft_dataset()
    # KoAlpaca 데이터셋의 기본 컬럼을 정리하고 학습에 사용할 텍스트만 남긴다
    dataset = dataset.map(
        format_sft,
        remove_columns=dataset["train"].column_names,
    )

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
        eval_dataset=dataset["validation"] if "validation" in dataset else None,

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

# full_finetune_160m.py
import inspect
import math
import os
from dataclasses import dataclass
from typing import Dict, List, Optional
import torch
from datasets import Dataset, DatasetDict, load_dataset, load_from_disk
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    EarlyStoppingCallback,
    Trainer,
    TrainingArguments,
)

MODEL_PATH = "/root/autodl-tmp/models/EleutherAI/pythia-160m-inserted-by-dssa1"
OUTPUT_DIR = "/root/autodl-tmp/models/pythia-160m-inserted-by-dssa-finetuned-biology1"
HF_DATASET_NAME = "WithinUsAI/Biology_25k"
HF_DATASET_CONFIG = None
HF_DATASET_SPLIT = None
PROMPT_FIELD = "prompt"
ANSWER_FIELD = "response"
MAX_LENGTH = 512
EXPECTED_MODEL_DTYPE = torch.float16

def setup_environment():
    os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    os.environ["TRANSFORMERS_OFFLINE"] = "0"

def print_gpu_info():
    if not torch.cuda.is_available():
        raise RuntimeError("GPU 不可用，不建议用 CPU 做全参数微调。")
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"GPU 数量: {torch.cuda.device_count()}")
    print(f"GPU 总显存: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
    print(f"bf16 支持: {torch.cuda.is_bf16_supported()}")

def load_model_and_tokenizer(model_path: str):
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"模型路径不存在: {model_path}")
    tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype="auto",
        trust_remote_code=False,
    )
    model.config.use_cache = False
    model.config.pad_token_id = tokenizer.pad_token_id
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    model_dtype = next(model.parameters()).dtype
    print("=" * 60)
    print("模型和 tokenizer 信息")
    print("=" * 60)
    print(f"模型路径: {model_path}")
    print(f"tokenizer 类型: {type(tokenizer)}")
    print(f"pad_token: {tokenizer.pad_token!r}, pad_token_id: {tokenizer.pad_token_id}")
    print(f"eos_token: {tokenizer.eos_token!r}, eos_token_id: {tokenizer.eos_token_id}")
    print(f"pad == eos: {tokenizer.pad_token_id == tokenizer.eos_token_id}")
    print(f"总参数量: {total_params:,}")
    print(f"可训练参数量: {trainable_params:,}")
    print(f"可训练比例: {trainable_params / total_params:.2%}")
    print(f"模型 dtype: {model_dtype}")
    if model_dtype != EXPECTED_MODEL_DTYPE:
        raise RuntimeError(
            f"当前水印模型 dtype 是 {model_dtype}，期望 {EXPECTED_MODEL_DTYPE}。"
            "为避免破坏低位水印，停止训练。"
        )
    return model, tokenizer, model_dtype

def load_hf_dataset(
    dataset_name: str,
    dataset_config: Optional[str] = None,
    dataset_split: Optional[str] = None,
) -> DatasetDict:
    print("=" * 60)
    print("从 Hugging Face 下载数据集")
    print("=" * 60)
    print(f"dataset_name: {dataset_name}")
    print(f"dataset_config: {dataset_config}")
    print(f"dataset_split: {dataset_split}")
    if dataset_split is not None:
        loaded = load_dataset(
            dataset_name,
            dataset_config,
            split=dataset_split,
        )
        return DatasetDict({"train": loaded})
    loaded = load_dataset(
        dataset_name,
        dataset_config,
    )
    if isinstance(loaded, DatasetDict):
        return loaded
    if isinstance(loaded, Dataset):
        return DatasetDict({"train": loaded})
    raise TypeError(f"不支持的数据集类型: {type(loaded)}")

def split_if_needed(dataset_dict: DatasetDict) -> DatasetDict:
    if "validation" in dataset_dict:
        return dataset_dict
    if "val" in dataset_dict:
        dataset_dict["validation"] = dataset_dict["val"]
        return dataset_dict
    if "test" in dataset_dict:
        dataset_dict["validation"] = dataset_dict["test"]
        return dataset_dict
    split = dataset_dict["train"].train_test_split(test_size=0.1, seed=42)
    return DatasetDict({
        "train": split["train"],
        "validation": split["test"],
    })

def build_text_batch(examples, tokenizer):
    if PROMPT_FIELD not in examples or ANSWER_FIELD not in examples:
        raise ValueError(
            f"数据集需要包含 {PROMPT_FIELD!r} 和 {ANSWER_FIELD!r} 两列，"
            f"当前列: {list(examples.keys())}"
        )
    texts = []
    for prompt, answer in zip(examples[PROMPT_FIELD], examples[ANSWER_FIELD]):
        if prompt is None:
            prompt = ""
        if answer is None:
            answer = ""
        text = (
            "### Question:\n"
            f"{prompt}\n\n"
            "### Answer:\n"
            f"{answer}{tokenizer.eos_token}"
        )
        texts.append(text)
    return texts

def tokenize_dataset(dataset_dict: DatasetDict, tokenizer) -> DatasetDict:
    train_columns = dataset_dict["train"].column_names
    missing = [
        column
        for column in [PROMPT_FIELD, ANSWER_FIELD]
        if column not in train_columns
    ]
    if missing:
        raise ValueError(
            f"原始数据缺少列: {missing}。当前列: {train_columns}"
        )
    def tokenize_function(examples):
        texts = build_text_batch(examples, tokenizer)
        return tokenizer(
            texts,
            truncation=True,
            max_length=MAX_LENGTH,
            padding=False,
        )
    tokenized = dataset_dict.map(
        tokenize_function,
        batched=True,
        remove_columns=train_columns,
        desc="Tokenizing Biology_25k with Pythia tokenizer",
    )
    tokenized = tokenized.filter(
        lambda example: len(example["input_ids"]) > 1,
        desc="Dropping empty/too-short samples",
    )
    return tokenized

@dataclass
class CausalLMCollator:
    tokenizer: object
    pad_to_multiple_of: Optional[int] = 8
    def __call__(self, features: List[Dict]) -> Dict[str, torch.Tensor]:
        cleaned = []
        for feature in features:
            item = {
                "input_ids": feature["input_ids"],
            }
            if "attention_mask" in feature:
                item["attention_mask"] = feature["attention_mask"]
            cleaned.append(item)
        batch = self.tokenizer.pad(
            cleaned,
            padding=True,
            pad_to_multiple_of=self.pad_to_multiple_of,
            return_tensors="pt",
        )
        labels = batch["input_ids"].clone()
        if "attention_mask" in batch:
            labels[batch["attention_mask"] == 0] = -100
        batch["labels"] = labels
        return batch

def validate_batch(dataset: Dataset, data_collator: CausalLMCollator, tokenizer, name: str):
    print("=" * 60)
    print(f"检查 {name} batch")
    print("=" * 60)
    sample = dataset[0]
    print(f"样本字段: {list(sample.keys())}")
    print(f"样本 input_ids 长度: {len(sample['input_ids'])}")
    print(f"样本 input_ids 前 30 个: {sample['input_ids'][:30]}")
    batch_size = min(4, len(dataset))
    batch = data_collator([dataset[i] for i in range(batch_size)])
    valid_labels = (batch["labels"] != -100).sum().item()
    total_labels = batch["labels"].numel()
    print(f"batch input_ids shape: {tuple(batch['input_ids'].shape)}")
    print(f"有效 labels: {valid_labels} / {total_labels}")
    print(f"第一条 labels 前 50 个: {batch['labels'][0][:50].tolist()}")
    print(f"第一条文本预览: {tokenizer.decode(batch['input_ids'][0][:80])!r}")
    if valid_labels == 0:
        raise RuntimeError(
            "当前 batch 中有效 labels 数量为 0，训练会出现 loss=0。"
            "请确认数据是用当前 Pythia tokenizer 重新处理的。"
        )

def build_training_args(
    output_dir: str,
    num_epochs: int,
    batch_size: int,
    grad_accum_steps: int,
    learning_rate: float,
    train_dtype: torch.dtype,
    has_eval_dataset: bool,
    use_wandb: bool = False,
):
    bf16_supported = torch.cuda.is_available() and torch.cuda.is_bf16_supported()
    use_bf16 = train_dtype == torch.bfloat16
    use_fp16 = train_dtype == torch.float16
    if use_bf16 and not bf16_supported:
        raise RuntimeError("当前 GPU 不支持 bf16，但训练 dtype 是 bf16。")
    kwargs = dict(
        output_dir=output_dir,
        overwrite_output_dir=True,
        num_train_epochs=num_epochs,
        per_device_train_batch_size=batch_size,
        gradient_accumulation_steps=grad_accum_steps,
        per_device_eval_batch_size=2,
        prediction_loss_only=True,
        learning_rate=learning_rate,
        weight_decay=0.01,
        warmup_ratio=0.05,
        lr_scheduler_type="cosine",
        optim="adamw_torch_fused",
        max_grad_norm=1.0,
        logging_strategy="steps",
        logging_steps=20,
        save_strategy="epoch",
        save_total_limit=2,
        save_safetensors=True,
        bf16=use_bf16,
        fp16=use_fp16,
        tf32=True,
        dataloader_num_workers=4,
        group_by_length=True,
        remove_unused_columns=False,
        report_to="wandb" if use_wandb else "none",
        seed=42,
    )
    sig = inspect.signature(TrainingArguments.__init__)
    eval_key = "eval_strategy" if "eval_strategy" in sig.parameters else "evaluation_strategy"
    kwargs[eval_key] = "epoch" if has_eval_dataset else "no"
    if has_eval_dataset:
        kwargs["metric_for_best_model"] = "eval_loss"
        kwargs["greater_is_better"] = False
        kwargs["load_best_model_at_end"] = False
    return TrainingArguments(**kwargs)

def main():
    setup_environment()
    print_gpu_info()
    model, tokenizer, save_dtype = load_model_and_tokenizer(MODEL_PATH)
    train_dtype = torch.float32 if save_dtype == torch.float16 else save_dtype
    if next(model.parameters()).dtype != train_dtype:
        print(f"训练前将参数从 {next(model.parameters()).dtype} 转为 {train_dtype}")
        model.to(dtype=train_dtype)
    dataset_dict = load_hf_dataset(
        dataset_name=HF_DATASET_NAME,
        dataset_config=HF_DATASET_CONFIG,
        dataset_split=HF_DATASET_SPLIT,
    )
    print("=" * 60)
    print("原始数据集信息")
    print("=" * 60)
    print(dataset_dict)
    print(f"训练集列名: {dataset_dict['train'].column_names}")
    dataset_dict = split_if_needed(dataset_dict)
    tokenized_dataset = tokenize_dataset(dataset_dict, tokenizer)
    train_dataset = tokenized_dataset["train"]
    val_dataset = tokenized_dataset.get("validation")
    print("=" * 60)
    print("Tokenized 数据集信息")
    print("=" * 60)
    print(f"训练集大小: {len(train_dataset)}")
    print(f"验证集大小: {len(val_dataset) if val_dataset is not None else 'None'}")
    print(f"训练集字段: {train_dataset.column_names}")
    data_collator = CausalLMCollator(
        tokenizer=tokenizer,
        pad_to_multiple_of=8,
    )
    validate_batch(
        dataset=train_dataset,
        data_collator=data_collator,
        tokenizer=tokenizer,
        name="训练集",
    )
    training_args = build_training_args(
        output_dir=OUTPUT_DIR,
        num_epochs=3,
        batch_size=16,
        grad_accum_steps=4,
        learning_rate=1e-5,
        train_dtype=train_dtype,
        has_eval_dataset=val_dataset is not None,
        use_wandb=False,
    )
    callbacks = []
    early_stopping_patience = None
    if early_stopping_patience is not None and val_dataset is not None:
        callbacks.append(
            EarlyStoppingCallback(
                early_stopping_patience=early_stopping_patience
            )
        )
    trainer_kwargs = dict(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        data_collator=data_collator,
        callbacks=callbacks,
    )
    sig = inspect.signature(Trainer.__init__)
    if "processing_class" in sig.parameters:
        trainer_kwargs["processing_class"] = tokenizer
    else:
        trainer_kwargs["tokenizer"] = tokenizer
    trainer = Trainer(**trainer_kwargs)
    print("=" * 60)
    print("开始训练")
    print("=" * 60)
    print(f"输出目录: {OUTPUT_DIR}")
    print("训练轮数: 3")
    print("单卡 batch size: 16")
    print("梯度累积: 4")
    print("有效 batch size: 64")
    print("学习率: 1e-5")
    print(f"训练 dtype: {next(model.parameters()).dtype}")
    print(f"保存 dtype: {save_dtype}")
    trainer.train()
    if val_dataset is not None:
        print("=" * 60)
        print("最终评估")
        print("=" * 60)
        eval_results = trainer.evaluate()
        eval_loss = eval_results.get("eval_loss")
        if eval_loss is not None:
            try:
                eval_results["perplexity"] = math.exp(eval_loss)
            except OverflowError:
                eval_results["perplexity"] = float("inf")
        for key, value in eval_results.items():
            if isinstance(value, float):
                print(f"{key}: {value:.4f}")
            else:
                print(f"{key}: {value}")
    print("=" * 60)
    print("保存模型")
    print("=" * 60)
    trainer.model.to(dtype=save_dtype)
    trainer.model.config.torch_dtype = save_dtype
    trainer.save_model(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)
    trainer.save_state()
    print(f"模型保存 dtype: {next(trainer.model.parameters()).dtype}")
    print(f"完成: {OUTPUT_DIR}")

if __name__ == "__main__":
    main()

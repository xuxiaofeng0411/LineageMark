# finetune_opt13b_full.py
import os
import math
import inspect
import torch

from datasets import load_from_disk
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
    DataCollatorForLanguageModeling,
    EarlyStoppingCallback,
)

DATA_PATH = "/root/autodl-tmp/datasets/processed_pubmed3_35k"
WATERMARKED_MODEL_PATH = "/root/autodl-tmp/models/facebook/opt-125m-mark3-nomark"
FINETUNED_OUTPUT_DIR = "/root/autodl-tmp/models/opt-125m-mark3-nomark-ft-pubmed"
EXPECTED_WATERMARK_DTYPE = torch.float16


def setup_environment():
    os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    os.environ["TRANSFORMERS_OFFLINE"] = "0"


def load_and_prepare_data(data_path="/root/autodl-tmp/datasets/processed_pubmed3_35k"):
    print("=" * 60)
    print("加载处理好的数据集...")
    print("=" * 60)

    train_path = os.path.join(data_path, "train")
    val_path = os.path.join(data_path, "val")

    if not os.path.exists(train_path):
        raise FileNotFoundError(f"训练集不存在: {train_path}")

    train_dataset = load_from_disk(train_path)
    val_dataset = load_from_disk(val_path) if os.path.exists(val_path) else None

    print(f"训练集大小: {len(train_dataset)}")
    print(f"验证集大小: {len(val_dataset) if val_dataset is not None else 'None'}")

    return train_dataset, val_dataset


def get_model_and_tokenizer(model_path="/root/autodl-tmp/models/facebook/opt-125m-mark3-nomark"):
    print("\n" + "=" * 60)
    print("加载模型和分词器...")
    print("=" * 60)

    if not os.path.exists(model_path):
        raise FileNotFoundError(f"模型不存在: {model_path}")

    tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=True)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    # Keep the checkpoint dtype used by the embedded watermark model. Casting
    # fp16 watermarked weights to bf16 destroys the low-bit watermark signal.
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype="auto",
        trust_remote_code=False,
    )

    model.config.use_cache = False
    model.config.pad_token_id = tokenizer.pad_token_id

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    print(f"总参数量: {total_params:,}")
    print(f"可训练参数量: {trainable_params:,}")
    print(f"可训练参数比例: {trainable_params / total_params:.2%}")
    model_dtype = next(model.parameters()).dtype
    print(f"模型参数 dtype: {model_dtype}")

    return model, tokenizer, model_dtype


def build_training_args(
    output_dir,
    num_epochs,
    batch_size,
    gradient_accumulation_steps,
    learning_rate,
    has_eval_dataset,
    training_dtype,
    use_wandb=False,
):
    bf16_supported = torch.cuda.is_available() and torch.cuda.is_bf16_supported()
    use_bf16 = training_dtype == torch.bfloat16
    use_fp16 = training_dtype == torch.float16

    if use_bf16 and not bf16_supported:
        raise RuntimeError("Current GPU does not support bf16, but the input model dtype is bf16.")

    kwargs = dict(
        output_dir=output_dir,
        overwrite_output_dir=True,

        num_train_epochs=num_epochs,
        per_device_train_batch_size=batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,

        # 评估只算 loss，不保存 logits，避免验证集 2500 条时显存/内存爆炸
        prediction_loss_only=True,
        per_device_eval_batch_size=2,
        eval_accumulation_steps=None,

        learning_rate=learning_rate,
        weight_decay=0.01,
        warmup_ratio=0.05,
        lr_scheduler_type="cosine",
        optim="adamw_torch_fused",

        logging_strategy="steps",
        logging_steps=50,
        report_to="wandb" if use_wandb else "none",

        save_strategy="no",
        save_total_limit=None,
        save_safetensors=True,

        bf16=use_bf16,
        fp16=use_fp16,
        tf32=True,

        dataloader_num_workers=4,
        group_by_length=True,
        seed=42,
        remove_unused_columns=False,
    )

    if has_eval_dataset:
        kwargs.update(
            metric_for_best_model="eval_loss",
            greater_is_better=False,
        )

        # 兼容不同 transformers 版本：
        # 新版本是 eval_strategy，老版本是 evaluation_strategy
        sig = inspect.signature(TrainingArguments.__init__)
        if "eval_strategy" in sig.parameters:
            kwargs["eval_strategy"] = "epoch"
        else:
            kwargs["evaluation_strategy"] = "epoch"
    else:
        sig = inspect.signature(TrainingArguments.__init__)
        if "eval_strategy" in sig.parameters:
            kwargs["eval_strategy"] = "no"
        else:
            kwargs["evaluation_strategy"] = "no"

        kwargs["load_best_model_at_end"] = False

    return TrainingArguments(**kwargs)


def finetune_model(
    model,
    tokenizer,
    train_dataset,
    val_dataset,
    output_dir="/root/autodl-tmp/models/opt-125m-mark3-nomark-ft-pubmed",
    num_epochs=2,
    batch_size=16,
    gradient_accumulation_steps=4,
    learning_rate=5e-6,
    use_wandb=False,
    early_stopping_patience=None,
    save_dtype=None,
):
    if save_dtype is None:
        save_dtype = next(model.parameters()).dtype
    train_param_dtype = torch.float32 if save_dtype == torch.float16 else save_dtype
    if next(model.parameters()).dtype != train_param_dtype:
        print(
            f"训练前将模型参数 dtype 从 {next(model.parameters()).dtype} "
            f"转换为 {train_param_dtype}；最终保存仍会转回 {save_dtype}。"
        )
        model.to(dtype=train_param_dtype)

    print("\n" + "=" * 60)
    print("开始全参数微调...")
    print("=" * 60)
    print(f"输出目录: {output_dir}")
    print(f"训练轮数: {num_epochs}")
    print(f"单卡 batch size: {batch_size}")
    print(f"梯度累积步数: {gradient_accumulation_steps}")
    print(f"有效 batch size: {batch_size * gradient_accumulation_steps}")
    print(f"学习率: {learning_rate}")
    print(f"验证集: {'完整验证集 ' + str(len(val_dataset)) + ' 条' if val_dataset is not None else '无'}")
    print("评估模式: loss-only，不保存 logits")
    print(f"训练参数 dtype: {next(model.parameters()).dtype}")
    print(f"最终保存 dtype: {save_dtype}")

    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False,
        pad_to_multiple_of=8,
    )

    training_args = build_training_args(
        output_dir=output_dir,
        num_epochs=num_epochs,
        batch_size=batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
        learning_rate=learning_rate,
        has_eval_dataset=val_dataset is not None,
        training_dtype=save_dtype,
        use_wandb=use_wandb,
    )

    callbacks = []
    if early_stopping_patience is not None and val_dataset is not None:
        callbacks.append(EarlyStoppingCallback(early_stopping_patience=early_stopping_patience))

    trainer_kwargs = dict(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        data_collator=data_collator,
        callbacks=callbacks,
    )

    # 兼容新版/旧版 transformers
    sig = inspect.signature(Trainer.__init__)
    if "processing_class" in sig.parameters:
        trainer_kwargs["processing_class"] = tokenizer
    else:
        trainer_kwargs["tokenizer"] = tokenizer

    trainer = Trainer(**trainer_kwargs)

    print("\n" + "=" * 60)
    print("开始训练...")
    print("=" * 60)

    trainer.train()

    if val_dataset is not None:
        print("\n" + "=" * 60)
        print("最终评估结果:")
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

    print("\n" + "=" * 60)
    print("保存最终模型...")
    print("=" * 60)

    # Save with the same dtype as the watermarked checkpoint was loaded with.
    trainer.model.to(dtype=save_dtype)
    trainer.model.config.torch_dtype = save_dtype

    trainer.save_model(output_dir)
    tokenizer.save_pretrained(output_dir)
    trainer.save_state()

    print(f"模型参数保存 dtype: {next(trainer.model.parameters()).dtype}")
    print(f"微调完成，模型已保存到: {output_dir}")

    return trainer


def main():
    setup_environment()

    if torch.cuda.is_available():
        print(f"GPU可用: {torch.cuda.get_device_name(0)}")
        print(f"GPU数量: {torch.cuda.device_count()}")
        print(f"GPU总显存: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
        print(f"bf16支持: {torch.cuda.is_bf16_supported()}")
    else:
        raise RuntimeError("GPU不可用，不建议用 CPU 训练模型")

    train_dataset, val_dataset = load_and_prepare_data(
        data_path=DATA_PATH
    )

    # 使用完整验证集，但评估只算 loss，不保存 logits
    model, tokenizer, model_dtype = get_model_and_tokenizer(
        model_path=WATERMARKED_MODEL_PATH
    )

    if model_dtype != EXPECTED_WATERMARK_DTYPE:
        raise RuntimeError(
            f"水印模型 dtype 是 {model_dtype}，期望 {EXPECTED_WATERMARK_DTYPE}。"
            "停止微调，避免再次因 bfloat16/其他精度转换破坏低位水印。"
        )

    output_dir = FINETUNED_OUTPUT_DIR

    trainer = finetune_model(
        model=model,
        tokenizer=tokenizer,
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        output_dir=output_dir,
        num_epochs=2,
        batch_size=16,
        gradient_accumulation_steps=4,
        learning_rate=5e-6,
        use_wandb=False,
        early_stopping_patience=None,
        save_dtype=model_dtype,
    )

    print("\n" + "=" * 60)
    print("全参数微调完成")
    print("=" * 60)
    print(f"微调后的模型保存在: {output_dir}")


if __name__ == "__main__":
    main()

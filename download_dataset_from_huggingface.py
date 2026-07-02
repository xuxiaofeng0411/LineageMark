import os
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
from datasets import load_dataset

BASE_DIR = "/root/autodl-tmp"
DATASET_REPO = "WithinUsAI/Biology_25k"

def _dataset_cache_dir(base_dir=BASE_DIR):
    return os.path.join(base_dir, "datasets")

def _ensure_dataset_dir(path):
    os.makedirs(path, exist_ok=True)
    print(f"数据集将保存到: {path}")

def _download_dataset(cache_dir):
    return load_dataset(
        DATASET_REPO,
        cache_dir=cache_dir
    )

def _show_dataset_preview(dataset):
    print("数据集下载完成！")
    print(f"\n数据集结构: {dataset}")
    train_dataset = dataset["train"]
    print(f"验证集样本数: {len(train_dataset)}")
    print("\n" + "=" * 50)
    print("第一个训练样本示例:")
    print("=" * 50)
    sample = train_dataset[0]
    print(sample)

def main():
    datasets_dir = _dataset_cache_dir()
    _ensure_dataset_dir(datasets_dir)
    dataset = _download_dataset(datasets_dir)
    _show_dataset_preview(dataset)

main()

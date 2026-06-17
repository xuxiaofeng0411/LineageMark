import ppl_result_pubmed as runner


runner.DOMAIN = "law"
runner.OUTPUT_JSON = "/root/autodl-tmp/project12/outputs/ppl_result_law.json"

runner.MODEL_SPECS = [
    ("original", "/root/autodl-tmp/models/facebook/opt-125m"),
    ("law_ft_stage1", "/root/autodl-tmp/models/law/opt-125m-mark1-nomark-ft-law"),
    ("law_ft_stage2", "/root/autodl-tmp/models/law/opt-125m-mark2-nomark-ft-law"),
    ("law_ft_stage3", "/root/autodl-tmp/models/law/opt-125m-mark3-nomark-ft-law"),
    ("law_watermark_stage1", "/root/autodl-tmp/models/facebook/law/opt-125m-mark1-nomark"),
    ("law_watermark_stage2", "/root/autodl-tmp/models/facebook/law/opt-125m-mark2-nomark"),
    ("law_watermark_stage3", "/root/autodl-tmp/models/facebook/law/opt-125m-mark3-nomark"),
    ("law_watermark_stage4", "/root/autodl-tmp/models/facebook/law/opt-125m-mark4-nomark"),
]

runner.DOMAIN_DATASETS = [
    ("law_stage1", "/root/autodl-tmp/datasets/processed_law1_35k"),
    ("law_stage2", "/root/autodl-tmp/datasets/processed_law2_35k"),
    ("law_stage3", "/root/autodl-tmp/datasets/processed_law3_35k"),
]


if __name__ == "__main__":
    runner.main()

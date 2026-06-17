import zero_shot_result as runner


runner.DEFAULT_OUTPUT_JSON = "/root/autodl-tmp/project12/outputs/zero_shot_pubmed_results.json"

runner.MODEL_SPECS = [
    ("original", "/root/autodl-tmp/models/facebook/opt-125m"),
    ("pubmed_ft_stage1", "/root/autodl-tmp/models/pubmed/opt-125m-mark1-nomark-ft-pubmed"),
    ("pubmed_ft_stage2", "/root/autodl-tmp/models/pubmed/opt-125m-mark2-nomark-ft-pubmed"),
    ("pubmed_ft_stage3", "/root/autodl-tmp/models/pubmed/opt-125m-mark3-nomark-ft-pubmed"),
    ("pubmed_watermark_stage1", "/root/autodl-tmp/models/facebook/pubmed/opt-125m-mark1-nomark"),
    ("pubmed_watermark_stage2", "/root/autodl-tmp/models/facebook/pubmed/opt-125m-mark2-nomark"),
    ("pubmed_watermark_stage3", "/root/autodl-tmp/models/facebook/pubmed/opt-125m-mark3-nomark"),
    ("pubmed_watermark_stage4", "/root/autodl-tmp/models/facebook/pubmed/opt-125m-mark4-nomark"),
]


if __name__ == "__main__":
    runner.main()

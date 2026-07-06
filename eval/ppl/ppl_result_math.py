import ppl_result_pubmed as runner

runner.DOMAIN = "math"
runner.OUTPUT_JSON = "/root/autodl-tmp/project12/outputs/ppl_result_math.json"
runner.MODEL_SPECS = [
    ("original", "/root/autodl-tmp/models/facebook/opt-125m"),
    ("math_ft_stage1", "/root/autodl-tmp/models/math/opt-125m-mark1-nomark-ft-math"),
    ("math_ft_stage2", "/root/autodl-tmp/models/math/opt-125m-mark2-nomark-ft-math"),
    ("math_ft_stage3", "/root/autodl-tmp/models/math/opt-125m-mark3-nomark-ft-math"),
    ("math_watermark_stage1", "/root/autodl-tmp/models/facebook/math/opt-125m-mark1-nomark"),
    ("math_watermark_stage2", "/root/autodl-tmp/models/facebook/math/opt-125m-mark2-nomark"),
    ("math_watermark_stage3", "/root/autodl-tmp/models/facebook/math/opt-125m-mark3-nomark"),
    ("math_watermark_stage4", "/root/autodl-tmp/models/facebook/math/opt-125m-mark4-nomark"),
]
runner.DOMAIN_DATASETS = [
    ("math_stage1", "/root/autodl-tmp/datasets/processed_math1_35k"),
    ("math_stage2", "/root/autodl-tmp/datasets/processed_math2_35k"),
    ("math_stage3", "/root/autodl-tmp/datasets/processed_math3_35k"),
]

if __name__ == "__main__":
    runner.main()

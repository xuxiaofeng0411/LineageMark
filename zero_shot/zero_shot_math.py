import zero_shot_result as runner


runner.DEFAULT_OUTPUT_JSON = "/root/autodl-tmp/project12/outputs/zero_shot_math_results.json"

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


if __name__ == "__main__":
    runner.main()

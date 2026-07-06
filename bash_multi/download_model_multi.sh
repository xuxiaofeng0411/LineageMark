export HF_API_TOKEN="hf_YourHuggingfaceApiToken"
model_dir="../models/"
log_dir="../logs/"
model_name="EleutherAI/pythia-160m"
python -u download_model_from_huggingface.py \
    --repo_id $model_name
echo "$model_name [DOWNLOAD] done!"

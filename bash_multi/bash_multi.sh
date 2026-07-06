#!/bin/bash
export HF_ENDPOINT="https://hf-mirror.com"
model_dir="../models/"
log_dir="../logs/"

# Model parameters
cuda_device=0
export CUDA_VISIBLE_DEVICES=$cuda_device
export TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
model_name="facebook/opt-125m"
quant_name="facebook/opt-125m"
hidden_size=768       
model_seqlen=768

# Watermark parameters
#user 1
#watermark="mark"
#seed=100    
#password="asdfqwer" 

#user 2
#watermark="bear"
#password="qihdnbji"
#seed=62

#user 3
#watermark="they"
#password="abcdefgh"
#seed=77

#user 4
#watermark="look"
#password="jiqwmnvb"
#seed=35

# DSSA parameters
k=64                       # subspace directions
tau_lower=0.1              # spectral truncation lower bound
tau_upper=0.9              # spectral truncation higher bound
epsilon=1e-6               # GEVP
select_ratio=0.75          # fraction of matrix coordinates selected globally
dssa_block_chunk=2         # blocks per DSSA stats pass; 0 means all blocks
dssa_calib_batch_size=8    # fixed calibration samples per forward/backward
subspace_method="full"  # [full, fisher_only, ca_only]

# insert watermark
run_lineagemark_insert () {
    log_file="${log_path}-insert-mark1-full.log"
    python -u dssa_multi/dssa_insert_watermark_multi.py \
    --model $model_path \
    --k $k \
    --hidden_size $hidden_size \
    --password $password \
    --watermark $watermark \
    --xi $xi \
    --position_num $position_num \
    --delta $delta \
    --wm_method "$wm_method" \
    --projection_margin "$projection_margin" \
    --projection_max_update "$projection_max_update" \
    --data_independent_extract \
    --tau_lower $tau_lower \
    --tau_upper $tau_upper \
    --epsilon $epsilon \
    --select_ratio $select_ratio \
    --dssa_block_chunk $dssa_block_chunk \
    --dssa_calib_batch_size $dssa_calib_batch_size \
    --subspace_method $subspace_method \
    --nsamples $nsamples \
    --seed $seed \
    --save_model $save_model \
    --save_subspace $save_subspace > "$log_file" 2>&1
    echo "=============== LineageMark insert done! ${model_path} --> ${save_model} "
}

# extract watermark
run_lineagemark_extract () {
    log_file="${log_path}-extract-mark1-ca.log"
    python -u dssa_multi/dssa_extract_watermark_multi.py \
        --password "$password" \
        --model "$save_model" \
        --hidden_size "$hidden_size" \
        --xi "$xi" \
        --position_num "$position_num" \
        --watermark "$watermark" \
        --wm_method "$wm_method" \
        --projection_margin "$projection_margin" \
        --data_independent_extract \
        --mode "$mode" \
        --chunk_length 8 \
        --seed "$seed" > "$log_file" 2>&1
    grep -E "Real watermark|Extract watermark|Extract ACC" "$log_file" | tail -3
    echo "=============== LineageMark extract done! --> $save_model "
}

quant_model () {
    log_file="${log_dir}${model_name}-quanted.log"
    python -u quant_multi/gptq_model.py \
    --model_path "$model_path" \
    --quant_path "$quant_path" \
    --model_seqlen $model_seqlen > "$log_file" 2>&1
    echo "=============== quant model done! ${model_name} --> ${quant_name} "
}

model_path="/root/autodl-tmp/models/facebook/opt-125m"
save_model="/root/autodl-tmp/models/facebook/opt-125m-mark1-full"
save_subspace="/root/autodl-tmp/models/facebook/opt-125m-mark1-subspace-full"
log_path="${log_dir}${model_name}"
mkdir -p "$(dirname "$log_path")"

xi=4           
position_num=12   
delta=20            # threshold
wm_method="projection"
projection_margin=0.5
projection_max_update=0.0
mode="simple"   
nsamples=128        # number of calibration samples

run_lineagemark_insert
run_lineagemark_extract


# ===== Baseline watermark commands (ELLMark / EmMark) =====
ellmark_save_model="${ellmark_save_model:-${model_path}-inserted-by-ellmark}"
ellmark_xi="${ellmark_xi:-$xi}"
ellmark_position_num="${ellmark_position_num:-$position_num}"
ellmark_delta="${ellmark_delta:-$delta}"
ellmark_mode="${ellmark_mode:-$mode}"

emmark_save_model="${emmark_save_model:-${model_path}-inserted-by-emmark}"
emmark_seed="${emmark_seed:-$seed}"
emmark_candidate_rate="${emmark_candidate_rate:-60}"
emmark_modify_rate="${emmark_modify_rate:-0.75}"

run_ellmark_insert () {
    log_file="${log_path}-inserted-by-ellmark.log"
    python -u dssa_multi/ellmark_insert_watermark_multi.py \
        --model "$model_path" \
        --hidden_size "$hidden_size" \
        --password "$password" \
        --watermark "$watermark" \
        --xi "$ellmark_xi" \
        --position_num "$ellmark_position_num" \
        --delta "$ellmark_delta" \
        --select_ratio "$select_ratio" \
        --nsamples "$nsamples" \
        --seed "$seed" \
        --save_model "$ellmark_save_model" > "$log_file" 2>&1
    echo "=============== ellmark insert done! ${model_path} --> ${ellmark_save_model} "
}

run_ellmark_extract () {
    log_file="${log_path}-extracted-by-ellmark.log"
    python -u dssa_multi/ellmark_extract_watermark_multi.py \
        --model "$ellmark_save_model" \
        --hidden_size "$hidden_size" \
        --password "$password" \
        --watermark "$watermark" \
        --xi "$ellmark_xi" \
        --position_num "$ellmark_position_num" \
        --mode "$ellmark_mode" \
        --chunk_length 8 \
        --seed "$seed" > "$log_file" 2>&1
    grep -E "Real watermark|Extract watermark|Extract ACC" "$log_file" | tail -3
    echo "=============== ellmark extract done! --> ${ellmark_save_model} "
}

run_emmark_insert () {
    log_file="${log_path}-inserted-by-emmark.log"
    python -u dssa_multi/emmark_insert_watermark_multi.py \
        --model "$model_path" \
        --hidden_size "$hidden_size" \
        --watermark "$watermark" \
        --seed "$emmark_seed" \
        --candidate_rate "$emmark_candidate_rate" \
        --modify_rate "$emmark_modify_rate" \
        --nsamples "$nsamples" \
        --save_model "$emmark_save_model" > "$log_file" 2>&1
    echo "=============== emmark insert done! ${model_path} --> ${emmark_save_model} "
}

run_emmark_extract () {
    log_file="${log_path}-extracted-by-emmark.log"
    python -u dssa_multi/emmark_extract_watermark_multi.py \
        --model "$model_path" \
        --inserted_model "$emmark_save_model" \
        --hidden_size "$hidden_size" \
        --watermark "$watermark" \
        --seed "$emmark_seed" \
        --candidate_rate "$emmark_candidate_rate" \
        --modify_rate "$emmark_modify_rate" > "$log_file" 2>&1
    grep -E "Real watermark|Extract watermark|Extract ACC" "$log_file" | tail -3
    echo "=============== emmark extract done! --> ${emmark_save_model} "
}
# Baseline examples. Keep the DSSA/LineageMark calls below as the default run.
# run_ellmark_insert
# run_ellmark_extract
# run_emmark_insert
# run_emmark_extract
# ===== End baseline watermark commands =====


# 假阳性验证密钥集合
seed=114
password="aK3mP9qR"
run_lineagemark_extract
password="X7tB2nLf"
run_lineagemark_extract
password="Qw8RyU1o"
run_lineagemark_extract
password="cV4bN6mH"
run_lineagemark_extract
password="Jf9Gd2Sa"
run_lineagemark_extract
password="Tz3Wx7Ec"
run_lineagemark_extract
password="Lp5Yk8Rv"
run_lineagemark_extract
password="Dg6Hj2Ua"
run_lineagemark_extract
password="Fs4Mn9Bv"
run_lineagemark_extract
password="Vc7Xz1Qw"
run_lineagemark_extract
password="Nb2Cm6Yt"
run_lineagemark_extract
password="Kd5Rj8Pw"
run_lineagemark_extract
password="Ht9Gv3Lx"
run_lineagemark_extract
password="Zx1Wq4Es"
run_lineagemark_extract
password="Mn7Bv2Cf"
run_lineagemark_extract
password="Pl3Kj9Ht"
run_lineagemark_extract
password="Qr6Yt5Uy"
run_lineagemark_extract
password="Ws8Ed4Rf"
run_lineagemark_extract
password="Tx2Zc7Vb"
run_lineagemark_extract
password="Au1Lm9Nk"
run_lineagemark_extract
password="Ce3Fg6Hj"
run_lineagemark_extract
password="Op5Rt8Yw"
run_lineagemark_extract
password="Ik7Uy4Qe"
run_lineagemark_extract
password="Dv9Bf2Np"
run_lineagemark_extract
password="Gs4Xj1Ml"
run_lineagemark_extract
password="Hc6Wq3Zr"
run_lineagemark_extract
password="Jt8Kv5Lp"
run_lineagemark_extract
password="Ly2Pm7Qa"
run_lineagemark_extract
password="Vb9Nc4Df"
run_lineagemark_extract
password="Rw3Tg6Hj"
run_lineagemark_extract
password="Ex7Zc1Bv"
run_lineagemark_extract
password="Fn5Mm2Kp"
run_lineagemark_extract
password="Uh8Yt4Qw"
run_lineagemark_extract
password="Id2Lx9Rc"
run_lineagemark_extract
password="Oa6Wq3Es"
run_lineagemark_extract
password="Tp7Rf1Vb"
run_lineagemark_extract
password="Sj4Dg8Hm"
run_lineagemark_extract
password="Kz9Nc2Xy"
run_lineagemark_extract
password="Cv3Bf5Tg"
run_lineagemark_extract
password="My6Hj7Uk"
run_lineagemark_extract
password="Lw1Qr4Ep"
run_lineagemark_extract
password="Xd8Zx2Wc"
run_lineagemark_extract
password="Bt5Fv9Mg"
run_lineagemark_extract
password="Np7Kl3Rj"
run_lineagemark_extract
password="Ae2Cm6Yh"
run_lineagemark_extract
password="Vr9Ty1Ui"
run_lineagemark_extract
password="Wb4Xn7Qk"
run_lineagemark_extract
password="Gm3Rd8Zj"
run_lineagemark_extract
password="Py2Lt5Cv"
run_lineagemark_extract
password="Hu7Fs9Ne"
run_lineagemark_extract

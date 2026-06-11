#!/bin/bash

# source /etc/network_turbo #学术加速
export HF_ENDPOINT="https://hf-mirror.com" #HF镜像网站

#全局路径参数
model_dir="../models/"
log_dir="../logs/"

#全局模型参数
cuda_device=0
export CUDA_VISIBLE_DEVICES=$cuda_device
export TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
model_name="facebook/opt-125m"
quant_name="facebook/opt-125m"
hidden_size=768       #水印嵌入需要
model_seqlen=768       #模型量化需要
# model_path="${model_dir}${model_name}" 或者 "${model_dir}${quant_name}"
# log_path="${log_dir}${model_name}" 或者 "${log_dir}${quant_name}"

#全局水印参数
#watermark="mark"
#watermark="bear"    #（重水印攻击时更换mark）【用于所有方法】
watermark="they"
#watermark="look"
#seed=100            #（随机修改攻击和重水印攻击时更换seed）【用于RandomMark】
#seed=62
#seed=77
#seed=35
#password="asdfqwer" #（重水印攻击时更换password）
#password="qihdnbji"
#password="abcdefgh"
#password="jiqwmnvb"
#password="jaijnehf"
#全局DSSA参数
k=64                    #子空间方向数量
tau_lower=0.1           #谱截断下界
tau_upper=0.9           #谱截断上界
epsilon=1e-6            #GEVP正则化
select_ratio=0.75       #每行选择列的比例
dssa_block_chunk=2      # blocks per DSSA stats pass; 0 means all blocks for OPT-125M
dssa_calib_batch_size=8  # fixed calibration samples per forward/backward
save_subspace=""        #可选：保存子空间数据的路径

run_dssa_insert () {
    log_file="${log_path}-insert-mark1-nomark.log"
    python -u pipelines_multi/dssa_insert_watermark_multi.py \
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
    --nsamples $nsamples \
    --seed $seed \
    --save_model $save_model \
    --save_subspace $save_subspace > "$log_file" 2>&1
    echo "=============== dssa insert done! ${model_path} --> ${save_model} "
}

run_dssa_extract () {
    log_file="${log_path}-extract-mark1-nomark.log"
    python -u pipelines_multi/dssa_extract_watermark_multi.py \
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
    echo "=============== dssa extract done! --> $save_model "
}

quant_model () {
    log_file="${log_dir}${model_name}-quanted.log"
    python -u quant_multi/gptq_model.py \
    --model_path "$model_path" \
    --quant_path "$quant_path" \
    --model_seqlen $model_seqlen > "$log_file" 2>&1
    echo "=============== quant model done! ${model_name} --> ${quant_name} "
}

#model_path="${model_dir}${model_name}"
model_path="/root/autodl-tmp/models/facebook/opt-125m"
#quant_path="/root/autodl-tmp/models/opt-125m-mark3-nomark-quant"
log_path="${log_dir}${model_name}"
mkdir -p "$(dirname "$log_path")"

xi=4            #[2, 4] -> 全精度选4 8bit选2
position_num=12   #[6, 12] -> 全精度选12 8bit选6
delta=20
wm_method="projection"
projection_margin=0.5
projection_max_update=0.0
mode="simple"    #攻击时改为robust
nsamples=512     #DSSA标定样本数

# save_model="${model_path}-inserted-by-dssa1"
save_model="/root/autodl-tmp/models/facebook/pubmed/opt-125m-mark4-nomark"
#save_model="/root/autodl-tmp/models/opt-125m-mark1-finetuned-pubmed-matrix-optimize529"
#save_subspace="/root/autodl-tmp/models/facebook/opt-125m-subspace-mark1.pt"

#run_dssa_insert
# run_dssa_extract
#quant_model
# run_model_evaluate $save_model
# delete_model $save_model

seed=114

password="aK3mP9qR"
run_dssa_extract
password="X7tB2nLf"
run_dssa_extract
password="Qw8RyU1o"
run_dssa_extract
password="cV4bN6mH"
run_dssa_extract
password="Jf9Gd2Sa"
run_dssa_extract
password="Tz3Wx7Ec"
run_dssa_extract
password="Lp5Yk8Rv"
run_dssa_extract
password="Dg6Hj2Ua"
run_dssa_extract
password="Fs4Mn9Bv"
run_dssa_extract
password="Vc7Xz1Qw"
run_dssa_extract
password="Nb2Cm6Yt"
run_dssa_extract
password="Kd5Rj8Pw"
run_dssa_extract
password="Ht9Gv3Lx"
run_dssa_extract
password="Zx1Wq4Es"
run_dssa_extract
password="Mn7Bv2Cf"
run_dssa_extract
password="Pl3Kj9Ht"
run_dssa_extract
password="Qr6Yt5Uy"
run_dssa_extract
password="Ws8Ed4Rf"
run_dssa_extract
password="Tx2Zc7Vb"
run_dssa_extract
password="Au1Lm9Nk"
run_dssa_extract
password="Ce3Fg6Hj"
run_dssa_extract
password="Op5Rt8Yw"
run_dssa_extract
password="Ik7Uy4Qe"
run_dssa_extract
password="Dv9Bf2Np"
run_dssa_extract
password="Gs4Xj1Ml"
run_dssa_extract
password="Hc6Wq3Zr"
run_dssa_extract
password="Jt8Kv5Lp"
run_dssa_extract
password="Ly2Pm7Qa"
run_dssa_extract
password="Vb9Nc4Df"
run_dssa_extract
password="Rw3Tg6Hj"
run_dssa_extract
password="Ex7Zc1Bv"
run_dssa_extract
password="Fn5Mm2Kp"
run_dssa_extract
password="Uh8Yt4Qw"
run_dssa_extract
password="Id2Lx9Rc"
run_dssa_extract
password="Oa6Wq3Es"
run_dssa_extract
password="Tp7Rf1Vb"
run_dssa_extract
password="Sj4Dg8Hm"
run_dssa_extract
password="Kz9Nc2Xy"
run_dssa_extract
password="Cv3Bf5Tg"
run_dssa_extract
password="My6Hj7Uk"
run_dssa_extract
password="Lw1Qr4Ep"
run_dssa_extract
password="Xd8Zx2Wc"
run_dssa_extract
password="Bt5Fv9Mg"
run_dssa_extract
password="Np7Kl3Rj"
run_dssa_extract
password="Ae2Cm6Yh"
run_dssa_extract
password="Vr9Ty1Ui"
run_dssa_extract
password="Wb4Xn7Qk"
run_dssa_extract
password="Gm3Rd8Zj"
run_dssa_extract
password="Py2Lt5Cv"
run_dssa_extract
password="Hu7Fs9Ne"
run_dssa_extract

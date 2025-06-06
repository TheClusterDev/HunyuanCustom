#!/bin/bash
JOBS_DIR=$(dirname $(dirname "$0"))
export PYTHONPATH=${JOBS_DIR}:$PYTHONPATH
export MODEL_BASE=${JOBS_DIR}"/models"
checkpoint_path=${MODEL_BASE}"/hunyuancustom_720P/mp_rank_00_model_states_fp8.pt"
current_time=$(date "+%Y.%m.%d-%H.%M.%S")
modelname='Tencent_HunyuanCustom_720P'
OUTPUT_BASEPATH=./results/${modelname}/${current_time}


export CPU_OFFLOAD=1
CUDA_VISIBLE_DEVICES=0 python3 hymm_sp/sample_gpu_poor.py \
    --ref-image './assets/images/seg_woman_01.png' \
    --pos-prompt "Realistic, High-quality. A woman is drinking coffee at a café." \
    --neg-prompt "Aerial view, aerial view, overexposed, low quality, deformation, a poor composition, bad hands, bad teeth, bad eyes, bad limbs, distortion, blurring, text, subtitles, static, picture, black border." \
    --ckpt ${checkpoint_path} \
    --video-size 720 1280 \
    --sample-n-frames 129 \
    --cfg-scale 7.5 \
    --seed 1024 \
    --infer-steps 30 \
    --use-deepcache 1 \
    --flow-shift-eval-video 13.0 \
    --save-path ${OUTPUT_BASEPATH} \
    --use-fp8 \
    --cpu-offload 

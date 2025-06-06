#!/bin/bash
JOBS_DIR=$(dirname $(dirname "$0"))
export PYTHONPATH=${JOBS_DIR}:$PYTHONPATH
export MODEL_BASE=${JOBS_DIR}"/models"
export NCCL_DEBUG=OFF
checkpoint_path=${MODEL_BASE}"/hunyuancustom_editing_720P/mp_rank_00_model_states.pt"
current_time=$(date "+%Y.%m.%d-%H.%M.%S")
modelname='Tencent_HunyuanCustom_Editing_720P'
OUTPUT_BASEPATH=./results/${modelname}/${current_time}

torchrun --nnodes=1 --nproc_per_node=8 --master_port 29605 hymm_sp/sample_batch.py \
    --ref-image './assets/images/sed_red_panda.png' \
    --input-video './assets/input_videos/001_bg.mp4' \
    --mask-video './assets/input_videos/001_mask.mp4' \
    --expand-scale 5 \
    --video-condition \
    --pos-prompt "Realistic, High-quality. A red panda is walking on a stone road." \
    --neg-prompt "Aerial view, aerial view, overexposed, low quality, deformation, a poor composition, bad hands, bad teeth, bad eyes, bad limbs, distortion, blurring, text, subtitles, static, picture, black border." \
    --ckpt ${checkpoint_path} \
    --seed 1024 \
    --cfg-scale 7.5 \
    --infer-steps 50 \
    --use-deepcache 1 \
    --flow-shift-eval-video 5.0 \
    --save-path ${OUTPUT_BASEPATH} \
    # --pose-enhance # Enable for human videos to improve pose generation quality.


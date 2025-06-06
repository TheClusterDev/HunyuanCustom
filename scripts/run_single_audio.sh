#!/bin/bash
JOBS_DIR=$(dirname $(dirname "$0"))
export PYTHONPATH=${JOBS_DIR}:$PYTHONPATH
export MODEL_BASE=${JOBS_DIR}"/models"
export NCCL_DEBUG=OFF
checkpoint_path=${MODEL_BASE}"/hunyuancustom_audio_720P/mp_rank_00_model_states_fp8.pt"
current_time=$(date "+%Y.%m.%d-%H.%M.%S")
modelname='Tencent_HunyuanCustom_Audio_720P'
OUTPUT_BASEPATH=./results/${modelname}/${current_time}

export DISABLE_SP=1 
CUDA_VISIBLE_DEVICES=0 python3 hymm_sp/sample_gpu_poor.py \
    --audio-condition \
    --ref-image './assets/images/seg_man_01.png' \
    --input-audio './assets/audios/milk_man.mp3' \
    --audio-strength 0.8 \
    --pos-prompt "Realistic, High-quality. In the study, a man sits at a table featuring a bottle of milk while delivering a product presentation." \
    --neg-prompt "Two people, two persons, aerial view, aerial view, overexposed, low quality, deformation, a poor composition, bad hands, bad teeth, bad eyes, bad limbs, distortion, blurring, text, subtitles, static, picture, black border." \
    --ckpt ${checkpoint_path} \
    --video-size 512 896 \
    --sample-n-frames 129 \
    --cfg-scale 7.5 \
    --seed 1024 \
    --infer-steps 30 \
    --use-deepcache 1 \
    --flow-shift-eval-video 13.0 \
    --save-path ${OUTPUT_BASEPATH} \
    --use-fp8 

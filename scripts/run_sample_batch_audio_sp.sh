#!/bin/bash
JOBS_DIR=$(dirname $(dirname "$0"))
export PYTHONPATH=${JOBS_DIR}:$PYTHONPATH
export MODEL_BASE=${JOBS_DIR}"/models"
export NCCL_DEBUG=OFF
checkpoint_path=${MODEL_BASE}"/hunyuancustom_audio_720P/mp_rank_00_model_states.pt"
current_time=$(date "+%Y.%m.%d-%H.%M.%S")
modelname='Tencent_HunyuanCustom_Audio_720P'
OUTPUT_BASEPATH=./results/${modelname}/${current_time}

torchrun --nnodes=1 --nproc_per_node=8 --master_port 29605 hymm_sp/sample_batch.py \
    --ref-image './assets/images/seg_man_01.png' \
    --input-audio './assets/audios/milk_man.mp3' \
    --audio-strength 0.8 \
    --audio-condition \
    --pos-prompt "Realistic, High-quality. In the study, a man sits at a table featuring a bottle of milk while delivering a product presentation." \
    --neg-prompt "Two people, two persons, aerial view, overexposed, low quality, deformation, a poor composition, bad hands, bad teeth, bad eyes, bad limbs, distortion, blurring, text, subtitles, static, picture, black border." \
    --ckpt ${checkpoint_path} \
    --seed 1026 \
    --video-size 720 1280 \
    --sample-n-frames 129 \
    --cfg-scale 7.5 \
    --infer-steps 30 \
    --use-deepcache 1 \
    --flow-shift-eval-video 13.0 \
    --save-path ${OUTPUT_BASEPATH} 


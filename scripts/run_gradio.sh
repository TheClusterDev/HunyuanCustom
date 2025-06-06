JOBS_DIR=$(dirname $(dirname "$0"))
export MODEL_BASE=${JOBS_DIR}"/models"
NUM_GPU=${HOST_GPU_NUM}
export PYTHONPATH=${JOBS_DIR}:$PYTHONPATH
echo "PYTHONPATH: $PYTHONPATH"

if [ $NUM_GPU = 8 ];
then
    echo " ========== This node has 8 GPUs ========== "
    export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
    export GPU_NUMS=8
    echo "gpu-nums = $GPU_NUMS"
    if [ $# -eq 0 ]; then
      export MODEL_OUTPUT_PATH=${MODEL_BASE}"/hunyuancustom_720P/"
      torchrun --nnodes=1 --nproc_per_node=$GPU_NUMS --master_port 29605 ./hymm_gradio/flask_hycustom.py &
      python3 hymm_gradio/gradio_ref2v.py
    elif [ "$1" = "--video" ]; then
      export MODEL_OUTPUT_PATH=${MODEL_BASE}"/hunyuancustom_editing_720P/"
      torchrun --nnodes=1 --nproc_per_node=$GPU_NUMS --master_port 29605 ./hymm_gradio/flask_hycustom.py --video-condition &
      python3 ./hymm_gradio/gradio_editing.py
    elif [ "$1" = "--audio" ]; then
      export MODEL_OUTPUT_PATH=${MODEL_BASE}"/hunyuancustom_audio_720P/"
      torchrun --nnodes=1 --nproc_per_node=$GPU_NUMS --master_port 29605 ./hymm_gradio/flask_hycustom.py --audio-condition &
      python3 ./hymm_gradio/gradio_audio.py
    else
        echo "Error: Invalid argument. "
        exit 1
    fi
fi


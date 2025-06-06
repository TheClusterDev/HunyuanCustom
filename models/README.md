# Download Pretrained Models

All models are stored in `HunyuanCustom/models` by default, and the file structure is as follows
```shell
HunyuanCustom
  ├──models
  │  ├──README.md
  │  ├──hunyuancustom_720P
  │  │  ├──mp_rank_00_model_states.pt
  │  │  │──mp_rank_00_model_states_fp8.pt
  │  │  ├──mp_rank_00_model_states_fp8_map.pt
  │  ├──vae_3d
  │  ├──openai_clip-vit-large-patch14
  │  ├──llava-llama-3-8b-v1_1
  │  ├──DWPose
  │  ├──whisper-tiny
  ├──...
```

## Download HunyuanCustom model
To download the HunyuanCustom model, first install the huggingface-cli. (Detailed instructions are available [here](https://huggingface.co/docs/huggingface_hub/guides/cli).)

```shell
python -m pip install "huggingface_hub[cli]"
```

Then download the model using the following commands:

```shell
# Switch to the directory named 'HunyuanCustom'
cd HunyuanCustom
# Use the huggingface-cli tool to download HunyuanCustom model in HunyuanCustom/models dir.
# The download time may vary from 10 minutes to 1 hour depending on network conditions.
huggingface-cli download tencent/HunyuanCustom --local-dir ./
```

## Download Pose model
To download [DWPose](https://huggingface.co/yzd-v/DWPose/tree/main) pretrained model using the following commands: 

```shell
cd HunyuanCustom
mkdir -p models/DWPose
wget https://huggingface.co/yzd-v/DWPose/resolve/main/yolox_l.onnx?download=true -O models/DWPose/yolox_l.onnx
wget https://huggingface.co/yzd-v/DWPose/resolve/main/dw-ll_ucoco_384.onnx?download=true -O models/DWPose/dw-ll_ucoco_384.onnx
```

## Download Audio Encoder
To download [whisper](https://huggingface.co/openai/whisper-tiny/tree/main) pretrained model using the following commands: 

```shell
cd HunyuanCustom
huggingface-cli download openai/whisper-tiny --local-dir ./models/whisper-tiny
```
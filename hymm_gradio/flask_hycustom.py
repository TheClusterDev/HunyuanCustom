import os
import numpy as np
import torch
import warnings
import threading
import traceback
import uvicorn
from fastapi import FastAPI, Body
from pathlib import Path
from datetime import datetime
import torch.distributed as dist
from hymm_gradio.tool_for_end2end import *
from hymm_sp.config import parse_args
from hymm_sp.sample_inference import HunyuanVideoSampler
from hymm_sp.data_kits.video_dataset import DataPreprocess
from hymm_sp.modules.parallel_states import (
    initialize_distributed,
    nccl_info,
)

warnings.filterwarnings("ignore")
MODEL_OUTPUT_PATH = os.environ.get('MODEL_OUTPUT_PATH')
app = FastAPI()
rlock = threading.RLock()

def save_image_base64_to_local(image_path=None, base64_buffer=None):
    # Encode image to base64 buffer
    if image_path is not None and base64_buffer is None:
        image_buffer_base64 = encode_image_to_base64(image_path)
    elif image_path is None and base64_buffer is not None:
        image_buffer_base64 = deepcopy(base64_buffer)
    else:
        print("Please pass either 'image_path' or 'base64_buffer'")
        return None
    
    # Decode base64 buffer and save to local disk
    if image_buffer_base64 is not None:
        image_data = base64.b64decode(image_buffer_base64)
        uuid_string = str(uuid.uuid4())
        temp_image_path = f'{TEMP_DIR}/{uuid_string}.png'
        with open(temp_image_path, 'wb') as image_file:
            image_file.write(image_data)
        return temp_image_path
    else:
        return None
    
def process_input_dict(input_dict):
    decoded_input_dict = {}
    decoded_input_dict["save_fps"] = input_dict.get("save_fps", 25)
    decoded_input_dict["prompt"] = input_dict.get("prompt", "")
    decoded_input_dict["prompt"] = input_dict.get("template_prompt", "") + decoded_input_dict["prompt"]
    decoded_input_dict["neg-prompt"] = input_dict.get("negative_prompt", "")
    decoded_input_dict["neg-prompt"] = input_dict.get("template_neg_prompt", "") + decoded_input_dict["neg-prompt"]
    decoded_input_dict["trace_id"] = input_dict.get("trace_id", "1234")
    decoded_input_dict["height"] = input_dict.get("height", 720)
    decoded_input_dict["width"] = input_dict.get("width", 1280)
    decoded_input_dict["num_frames"] = input_dict.get("num_frames", 129)
    decoded_input_dict["cfg"] = input_dict.get("cfg", 7.5)
    decoded_input_dict["steps"] = input_dict.get("steps", 30)
    decoded_input_dict["shift"] = input_dict.get("shift", 13.0)
    decoded_input_dict["seed"] = input_dict.get("seed", -1)
    decoded_input_dict["name"] = input_dict.get("name", "object")
    decoded_input_dict["image_path"] = input_dict.get("image_path", None)
    decoded_input_dict["masked_input_video_path"] = input_dict.get("masked_input_video_path", None)
    decoded_input_dict["input_mask_video_path"] = input_dict.get("input_mask_video_path", None)
    decoded_input_dict["audio_path"] = input_dict.get("audio_path", None)
    decoded_input_dict["audio_strength"] = input_dict.get("audio_strength", 1.0)
    image_base64_buffer = input_dict.get("image_buffer", None)
    if image_base64_buffer is not None:
        decoded_input_dict["image_path"] = save_image_base64_to_local(
            image_path=None, 
            base64_buffer=image_base64_buffer)
    
    return decoded_input_dict

@app.api_route('/predict2', methods=['GET', 'POST'])
def predict(data=Body(...)):
    is_acquire = False
    error_info = ""
    try:
        is_acquire = rlock.acquire(blocking=False)
        if is_acquire:
            res = predict_wrap(data)
            return res
    except Exception as e:
        error_info = traceback.format_exc()
        print(error_info)
    finally:
        if is_acquire:
            rlock.release()
    return {"errCode": -1, "info": "broken"}

def predict_wrap(input_dict={}):
    if nccl_info.sp_size > 1:
        device = torch.device(f"cuda:{torch.distributed.get_rank()}")
        rank = local_rank = torch.distributed.get_rank()
        print(f"sp_size={nccl_info.sp_size}, rank {rank} local_rank {local_rank}")
    try:
        print(f"----- rank = {rank}")
        if rank == 0:
            # Decode the base64 buffer strings
            input_dict = process_input_dict(input_dict)

            print('------- start to predict -------')
            # Parse input arguments
            image_path = input_dict["image_path"]
            save_fps = input_dict["save_fps"]
            text_prompt = input_dict["prompt"]
            neg_prompt = input_dict["neg-prompt"]

            trace_id = input_dict["trace_id"]
            height = input_dict["height"]
            width = input_dict["width"]
            num_frames = input_dict["num_frames"]
            cfg = input_dict["cfg"]
            steps = input_dict["steps"]
            shift = input_dict["shift"]
            seed = input_dict["seed"]
            name = input_dict["name"]
            masked_input_video_path = input_dict["masked_input_video_path"]
            input_mask_video_path = input_dict["input_mask_video_path"]
            audio_path = input_dict["audio_path"]
            audio_strength = input_dict["audio_strength"]
            if seed < 0:
                seed = np.random.randint(2**32-1)
                print(f'seed < 0, random sample a number={seed}')

            # Preprocess input batch
            ret_dict = None
            torch.cuda.synchronize()
            start_time = datetime.now()
            print('='*25, f'image_path = {image_path}', '='*25)
            input_data = {'image_path': image_path}
            if args.video_condition:
                assert os.path.exists(masked_input_video_path) and  os.path.exists(input_mask_video_path)
                input_data['masked_input_video_path'] = masked_input_video_path
                input_data['input_mask_video_path'] = input_mask_video_path
                batch = data_preprocess.get_batch(input_data, (width, height), data_type='video')
            elif args.audio_condition:
                assert os.path.exists(audio_path) 
                input_data['audio_path'] = audio_path
                batch = data_preprocess.get_batch(input_data, (width, height), video_length=num_frames, data_type='audio')
            else:
                batch = data_preprocess.get_batch(input_data, (width, height), data_type='image')

            audio_prompts = batch['audio_prompts'].to(hunyuan_sampler.device) if 'audio_prompts' in batch else None
            pixel_value_llava = batch['pixel_value_llava']
            uncond_pixel_value_llava = batch['uncond_pixel_value_llava']
            pixel_value_ref = batch['pixel_value_ref'].to(hunyuan_sampler.device)
            pixel_value_ref = pixel_value_ref * 2 - 1.
            pixel_value_ref_for_vae = rearrange(pixel_value_ref,"b c h w -> b c 1 h w")
            vae_dtype = torch.float16
            with torch.autocast(device_type="cuda", dtype=vae_dtype, enabled=vae_dtype != torch.float32):
                hunyuan_sampler.vae.enable_tiling()
                ref_latents = hunyuan_sampler.vae.encode(pixel_value_ref_for_vae.clone()).latent_dist.sample()
                uncond_ref_latents = hunyuan_sampler.vae.encode(torch.ones_like(pixel_value_ref_for_vae)).latent_dist.sample()
                if args.video_condition:
                    assert masked_input_video_path is not None and input_mask_video_path is not None
                    pixel_value_bg = batch['pixel_value_bg'].to(hunyuan_sampler.device) * 2 - 1.
                    bg_latents = hunyuan_sampler.vae.encode(pixel_value_bg).latent_dist.sample()
                    pixel_value_mask = batch['pixel_value_mask'].to(hunyuan_sampler.device) * 2 - 1.
                    mask_latents = hunyuan_sampler.vae.encode(pixel_value_mask).latent_dist.sample()
                    bg_latents = torch.cat([bg_latents, mask_latents], dim=1)
                    _, _, t, h, w = bg_latents.shape
                    num_frames, height, width = (t - 1) * 4 + 1, h * 8, w * 8
                    bg_latents.mul_(hunyuan_sampler.vae.config.scaling_factor)
                else:
                    bg_latents = None
                ref_latents.mul_(hunyuan_sampler.vae.config.scaling_factor)
                uncond_ref_latents.mul_(hunyuan_sampler.vae.config.scaling_factor)
                hunyuan_sampler.vae.disable_tiling()
                
            torch.cuda.synchronize()
            end_time = datetime.now()
            preprocess_time = (end_time - start_time).total_seconds()
            print("=" * 100)
            print("preprocess time :", preprocess_time)
            print("=" * 100)
            
        else:
            name = None
            text_prompt = None
            neg_prompt = None
            height = None
            width = None
            pixel_value_llava = None
            uncond_pixel_value_llava = None 
            ref_latents = None 
            uncond_ref_latents = None
            bg_latents = None
            audio_prompts = None
            audio_strength = None
            num_frames = None
            cfg = None
            steps = None
            shift = None
            seed = None
    except:
        traceback.print_exc()
        if rank == 0:
            ret_dict = {
                "errCode": -1,         # Failed to generate video
                "content":[
                    {
                        "buffer": None
                    }
                ],
                "info": "failed to generate video",
                "trace_id": trace_id
            }
            return ret_dict

    try:
        broadcast_params = [name, text_prompt, neg_prompt, height, width, ref_latents, uncond_ref_latents, bg_latents, audio_prompts, audio_strength, pixel_value_llava, uncond_pixel_value_llava, num_frames, cfg, steps, shift, seed]
        dist.broadcast_object_list(broadcast_params, src=0)
        outputs = generate_image_parallel(*broadcast_params)
    
        if rank == 0:
            samples = outputs["samples"]
            sample = samples[0].unsqueeze(0)
            output_dict = {
                "err_code": 0, 
                "err_msg": "succeed", 
                "video": sample, 
                "save_fps": save_fps, 
                "audio": audio_path,
                "rank": rank,
                "trace_id": trace_id
            }

            ret_dict = process_output_dict(output_dict)
            return ret_dict
    
    except:
        traceback.print_exc()
        if rank == 0:
            ret_dict = {
                "errCode": -1,         # Failed to generate video
                "content":[
                    {
                        "buffer": None
                    }
                ],
                "info": "failed to generate video",
                "trace_id": trace_id
            }
            return ret_dict
        
    return None
    
def generate_image_parallel(name, text_prompt, neg_prompt, height, width, ref_latents, uncond_ref_latents, bg_latents, audio_prompts, audio_strength, pixel_value_llava, uncond_pixel_value_llava, num_frames, cfg, steps, shift, seed):
    if nccl_info.sp_size > 1:
        device = torch.device(f"cuda:{torch.distributed.get_rank()}")

    model_kwargs = {
        "name": name,
            "prompt": text_prompt,
            "negative_prompt": neg_prompt,
            "size": (height, width),
            "seed": seed,
            "ref_latents": ref_latents.to(device=device,),
            "uncond_ref_latents": uncond_ref_latents.to(device=device,),
            "bg_latents": bg_latents.to(device=device,) if bg_latents is not None else None,
            "audio_prompts": audio_prompts.to(device=device,) if audio_prompts is not None else None,
            "audio_strength": audio_strength,
            "pixel_value_llava": pixel_value_llava.to(device=device,),
            "uncond_pixel_value_llava": uncond_pixel_value_llava.to(device=device,),
            "video_length": num_frames,
            "guidance_scale": cfg,
            "flow_shift": shift,
            "infer_steps": steps,
    }
    outputs = hunyuan_sampler.predict(
        **model_kwargs,
        num_images_per_prompt=args.num_images,
        use_linear_quadratic_schedule=args.use_linear_quadratic_schedule,
        linear_schedule_end=args.linear_schedule_end,
    )
    return outputs

def worker_loop():
    while True:
        predict_wrap()
        

if __name__ == "__main__":
    args = parse_args()
    args.model_base = f"{MODEL_OUTPUT_PATH}/models"
    ckpt = f"{MODEL_OUTPUT_PATH}/mp_rank_00_model_states.pt"
    model_base = Path(args.model_base)
    initialize_distributed(args.seed)
    hunyuan_sampler = HunyuanVideoSampler.from_pretrained(ckpt, args=args)
    args = hunyuan_sampler.args
    
    rank = local_rank = 0
    device = torch.device("cuda")
    if nccl_info.sp_size > 1:
        device = torch.device(f"cuda:{torch.distributed.get_rank()}")
        rank = local_rank = torch.distributed.get_rank()

    if rank == 0:
        data_preprocess = DataPreprocess(args, device)
       
    if rank == 0:
        uvicorn.run(app, host="0.0.0.0", port=8080)
    else:
        worker_loop()
    

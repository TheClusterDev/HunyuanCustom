import os
import cv2
import torch
import json
import numpy as np
from PIL import Image
import torchvision.transforms as transforms
from hymm_sp.data_kits.data_tools import *
from hymm_sp.data_kits.dwpose import DWposeDetector, draw_pose
from hymm_sp.constants import ANNOTATOR_PATH
from decord import VideoReader
from transformers import WhisperModel
from transformers import AutoFeatureExtractor


class DataPreprocess(object):
    def __init__(self, args, device, dtype=torch.float32):
        self.llava_size = (336, 336)
        self.args = args
        self.device = device
        self.weight_dtype = dtype
        self.llava_transform = transforms.Compose(
            [
                transforms.Resize(self.llava_size, interpolation=transforms.InterpolationMode.BILINEAR), 
                transforms.ToTensor(), 
                transforms.Normalize((0.48145466, 0.4578275, 0.4082107), (0.26862954, 0.26130258, 0.27577711)),
            ]
        )

        if args.audio_condition:
            self.wav2vec = WhisperModel.from_pretrained(ANNOTATOR_PATH['whisper']).to(device=self.device, dtype=self.weight_dtype)
            self.feature_extractor = AutoFeatureExtractor.from_pretrained(ANNOTATOR_PATH['whisper'])
            self.wav2vec.requires_grad_(False)

    def get_batch(self, meta_data, size, video_length=129, data_type='image'):
        batch = {}
        if data_type == 'video':
            masked_input_video_path = meta_data["masked_input_video_path"]
            input_mask_video_path = meta_data["input_mask_video_path"]
            masked_input_video = VideoReader(masked_input_video_path)
            mask_video = VideoReader(input_mask_video_path)
            num_frames = min(len(masked_input_video), len(mask_video))
            height, width = masked_input_video[0].asnumpy().shape[:2]
            size = (width, height)
            masked_frames = []
            masks = []
            for frame_idx in range(num_frames):
                masked_frame = masked_input_video[frame_idx].asnumpy()
                mask = mask_video[frame_idx].asnumpy()
                masked_frames.append(masked_frame)
                masks.append(mask)
            masks_tensor = torch.from_numpy(np.asarray(masks)).permute((3, 0, 1, 2)).unsqueeze(0) / 255.0
            masked_frames_tensor = torch.from_numpy(np.asarray(masked_frames)).permute((3, 0, 1, 2)).unsqueeze(0) / 255.0
            batch["pixel_value_bg"] = masked_frames_tensor
            batch["pixel_value_mask"] = masks_tensor

        elif data_type == 'audio':
            audio_path = meta_data["audio_path"]
            audio_input, audio_len = get_audio_feature(self.feature_extractor, audio_path)
            audio_len = min(video_length, audio_len)
            audio_prompts = [encode_audio(self.wav2vec, audio_feat.to(device=self.wav2vec.device, dtype=self.wav2vec.dtype), num_frames=audio_len) for audio_feat in audio_input]
            audio_prompts = torch.cat(audio_prompts, dim=0).to(device=self.device, dtype=self.weight_dtype)
            if audio_prompts.shape[1] < video_length:
                zero_length = video_length - audio_prompts.shape[1]
                audio_prompts = torch.cat([audio_prompts, torch.zeros_like(audio_prompts[:, :1]).repeat(1, zero_length, 1, 1, 1)], dim=1)
            batch["audio_prompts"] = audio_prompts

        elif data_type == 'image':
            pass
        else:
            raise ValueError
        
        image_path = meta_data['image_path']
        try:
            image = cv2.imread(image_path)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        except:
            image = Image.open(image_path).convert('RGB')
        llava_item_image = pad_image(image.copy(), self.llava_size)
        uncond_llava_item_image = np.ones_like(llava_item_image) * 255
        cat_item_image = pad_image(image.copy(), size)

        llava_item_tensor = self.llava_transform(Image.fromarray(llava_item_image.astype(np.uint8)))
        uncond_llava_item_tensor = self.llava_transform(Image.fromarray(uncond_llava_item_image))
        cat_item_tensor = torch.from_numpy(cat_item_image.copy()).permute((2, 0, 1)) / 255.0
        batch["pixel_value_llava"] = llava_item_tensor.unsqueeze(0)
        batch["uncond_pixel_value_llava"] = uncond_llava_item_tensor.unsqueeze(0)
        batch["pixel_value_ref"] = cat_item_tensor.unsqueeze(0)
        return batch


class BaseDataset:
    def __init__(self, args, device='cuda', dtype=torch.float32):
        self.args = args
        self.pad_color = (255, 255, 255)
        self.llava_size = (336, 336)
        self.ref_size = (args.video_size[1], args.video_size[0])
        self.fps = args.fps
        self.video_length = args.sample_n_frames
        self.weight_dtype = dtype
        self.device = device
        self.meta_datas = []

        if args.input_json is not None:
            if args.input_json.endswith('.list'):
                for line in open(args.input_json).readlines():
                    meta_data = json.load(open(line.strip(), 'r'))
                    self.meta_datas.append(meta_data)
            elif args.input_json.endswith('.json'):
                self.meta_datas.append(json.load(open(args.input_json, 'r')))
        else:
            meta_data = {
                'seg_item_image_path': args.ref_image,
                'video_path': args.input_video if hasattr(args, 'input_video') else None,
                'mask_path': args.mask_video if hasattr(args, 'mask_video') else None,
                'audio_path': args.input_audio if hasattr(args, 'input_audio') else None,
            }
            self.meta_datas.append(meta_data)

        self.llava_transform = transforms.Compose([
            transforms.Resize(self.llava_size, interpolation=transforms.InterpolationMode.BILINEAR),
            transforms.ToTensor(),
            transforms.Normalize(
                (0.48145466, 0.4578275, 0.4082107),
                (0.26862954, 0.26130258, 0.27577711)
            )
        ])

    def read_image(self, image_path):
        try:
            face_image_masked = cv2.cvtColor(cv2.imread(image_path), cv2.COLOR_BGR2RGB)
        except:
            face_image_masked = Image.open(image_path).convert('RGB')
        cat_face_image = pad_image(face_image_masked.copy(), self.ref_size)
        llava_face_image = pad_image(face_image_masked.copy(), self.llava_size)
        return llava_face_image, cat_face_image

    def get_batch(self, idx):
        meta_data = self.meta_datas[idx]
        ref_image_path = meta_data['seg_item_image_path']
        seed = meta_data.get('seed', self.args.seed)
        prompt = meta_data.get('pos_prompt', self.args.pos_prompt)
        negative_prompt = meta_data.get('neg_prompt', self.args.neg_prompt)
        item_prompt = meta_data.get('item_prompt', 'object')
        data_name = os.path.basename(os.path.splitext(ref_image_path)[0])

        llava_item_image, cat_item_image = self.read_image(ref_image_path)
        llava_item_tensor = self.llava_transform(Image.fromarray(llava_item_image.astype(np.uint8)))
        cat_item_tensor = torch.from_numpy(cat_item_image.copy()).permute((2, 0, 1)) / 255.0
        uncond_llava_item_image = np.ones_like(llava_item_image) * 255
        uncond_llava_item_tensor = self.llava_transform(Image.fromarray(uncond_llava_item_image))

        batch = {
            "pixel_value_llava": llava_item_tensor,
            "uncond_pixel_value_llava": uncond_llava_item_tensor,
            "pixel_value_ref": cat_item_tensor,
            "prompt": prompt,
            "negative_prompt": negative_prompt,
            "seed": seed,
            "name": item_prompt,
            'data_name': data_name,
        }
        return batch

class VideoDataset(BaseDataset):
    def __init__(self, args, device='cuda'):
        super().__init__(args, device)
        self.expand_scale = int(args.expand_scale)
        self.pose_enhance = args.pose_enhance
        if self.pose_enhance:
            self.dwpose_detector = DWposeDetector(
                model_det=os.path.join(ANNOTATOR_PATH['dwpose'], "yolox_l.onnx"),
                model_pose=os.path.join(ANNOTATOR_PATH['dwpose'], "dw-ll_ucoco_384.onnx"),
                device=device
            )
        else:
            self.dwpose_detector = None

    def __len__(self):
        return len(self.meta_datas)

    def __getitem__(self, idx):
        meta_data = self.meta_datas[idx]
        video_path = meta_data['video_path']
        mask_path = meta_data['mask_path']

        input_video = VideoReader(video_path)
        mask_video = VideoReader(mask_path)
        num_frames = min(len(input_video), len(mask_video))
        height, width = input_video[0].asnumpy().shape[:2]
        self.ref_size = (width, height)
        masked_frames = []
        masks = []
        for frame_idx in range(num_frames):
            frame = input_video[frame_idx].asnumpy()
            mask = mask_video[frame_idx].asnumpy()
            mask = cv2.resize(mask, (width, height))
            if len(mask.shape) == 3 and mask.shape[2] == 3:
                mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
            if self.expand_scale != 0:
                kernel_size = abs(self.expand_scale)
                kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
                op_expand = cv2.dilate if self.expand_scale > 0 else cv2.erode
                mask = op_expand(mask, kernel, iterations=3)
            mask = cv2.threshold(mask, 127.5, 255, cv2.THRESH_BINARY)[1]
            masks.append(mask)
            inverse_mask = mask == 0
            if self.dwpose_detector:
                pose_img = draw_pose(self.dwpose_detector(frame), height, width, ref_w=576)
                masked_frame = np.where(inverse_mask[..., None], frame, pose_img)
            else:
                masked_frame = frame * (inverse_mask[..., None].astype(frame.dtype))
            masked_frames.append(masked_frame)

        masks_tensor = torch.from_numpy(np.asarray(masks)).unsqueeze(0).repeat_interleave(3, dim=0) / 255.0
        masked_frames_tensor = torch.from_numpy(np.asarray(masked_frames)).permute((3, 0, 1, 2)) / 255.0

        base_batch = self.get_batch(idx)
        batch = {
            **base_batch,
            "pixel_value_bg": masked_frames_tensor,
            "pixel_value_mask": masks_tensor,
        }
        return batch

class AudioDataset(BaseDataset):
    def __init__(self, args, device='cuda'):
        super().__init__(args, device)
        self.wav2vec = WhisperModel.from_pretrained(ANNOTATOR_PATH['whisper']).to(device=self.device, dtype=self.weight_dtype)
        self.feature_extractor = AutoFeatureExtractor.from_pretrained(ANNOTATOR_PATH['whisper'])
        self.wav2vec.requires_grad_(False)

    def __len__(self):
        return len(self.meta_datas)

    def __getitem__(self, idx):
        meta_data = self.meta_datas[idx]
        audio_path = meta_data['audio_path']

        audio_input, audio_len = get_audio_feature(self.feature_extractor, audio_path)
        audio_len = min(self.video_length, audio_len)
        audio_prompts = [encode_audio(self.wav2vec, audio_feat.to(device=self.wav2vec.device, dtype=self.wav2vec.dtype), num_frames=audio_len) for audio_feat in audio_input]
        audio_prompts = torch.cat(audio_prompts, dim=0).to(device=self.device, dtype=self.weight_dtype)
        if audio_prompts.shape[1] < self.video_length:
            zero_length = self.video_length - audio_prompts.shape[1]
            audio_prompts = torch.cat([audio_prompts, torch.zeros_like(audio_prompts[:, :1]).repeat(1, zero_length, 1, 1, 1)], dim=1)

        base_batch = self.get_batch(idx)
        batch = {
            **base_batch,
            "audio_prompts": audio_prompts,
            'audio_path': audio_path,
        }
        return batch

class ImageDataset(BaseDataset):
    def __init__(self, args, device='cuda'):
        super().__init__(args, device)

    def __len__(self):
        return len(self.meta_datas)

    def __getitem__(self, idx):
        batch = self.get_batch(idx)
        return batch

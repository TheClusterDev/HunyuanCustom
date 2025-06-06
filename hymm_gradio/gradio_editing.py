import os
import cv2
import json
import imageio
import datetime
import requests
import traceback
import gradio as gr
from tool_for_end2end import *
from decord import VideoReader
from hymm_sp.constants import ANNOTATOR_PATH
from hymm_sp.data_kits.dwpose import DWposeDetector, draw_pose
from hymm_sp.data_kits.data_tools import mask_to_xyxy_box

os.environ["GRADIO_ANALYTICS_ENABLED"] = "False"
DATADIR = './temp'
_HEADER_ = '''
<div style="text-align: center; max-width: 650px; margin: 0 auto;">
    <h1 style="font-size: 2.5rem; font-weight: 700; margin-bottom: 1rem; display: contents;">Tencet HunyuanvideoCustom Demo</h1>
</div>

'''

# flask url
URL = "http://127.0.0.1:8080/predict2"
device_id = "cuda:0"
dwprocessor = None

def init_dwpose():
    global dwpose_detector
    dwpose_detector = DWposeDetector(
        model_det=os.path.join(ANNOTATOR_PATH['dwpose'], "yolox_l.onnx"),
        model_pose=os.path.join(ANNOTATOR_PATH['dwpose'], "dw-ll_ucoco_384.onnx"),
        device='cuda')
        

def run_mix_mask(input_video_path, input_mask_path, expand_scale, pose_enhance, to_bbox):
    if not input_video_path or not input_mask_path:
        return None, None
    now = datetime.datetime.now().isoformat()
    videodir = os.path.join(DATADIR, 'input_video')
    output_video_path = os.path.join(videodir, now + '_masked.mp4')
    mask_video_path = os.path.join(videodir, now + '_mask.mp4')
    os.makedirs(videodir, exist_ok=True)

    video = VideoReader(input_video_path)
    mask_video = VideoReader(input_mask_path)
    fps = video.get_avg_fps()
    num_frames = min(len(video), len(mask_video))
    height, width = video[0].asnumpy().shape[:2]
    assert height % 16 == 0 and width % 16 == 0
    masked_frames = []
    masks = []
    for frame_idx in range(num_frames):
        frame = video[frame_idx].asnumpy()
        mask = mask_video[frame_idx].asnumpy()
        mask = cv2.resize(mask, (width, height))
        if len(mask.shape) == 3 and mask.shape[2] == 3:
            mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)

        if expand_scale != 0:
            kernel_size = abs(expand_scale)
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
            op_expand = cv2.dilate if expand_scale > 0 else cv2.erode
            mask = op_expand(mask, kernel, iterations=3)

        _, mask = cv2.threshold(mask, 127.5, 255, cv2.THRESH_BINARY)
        if to_bbox and np.sum(mask == 255) > 0:
            x0, y0, x1, y1 = mask_to_xyxy_box(mask)
            mask = mask * 0
            mask[y0:y1, x0:x1] = 255

        inverse_mask = mask == 0
        if pose_enhance:
            pose_img = draw_pose(dwpose_detector(frame), height, width, ref_w=576)
            masked_frame = np.where(inverse_mask[..., None], frame, pose_img) 
        else:
            masked_frame = frame * (inverse_mask[..., None].astype(frame.dtype))

        masks.append(mask)
        masked_frames.append(masked_frame)

    imageio.mimsave(output_video_path, masked_frames, fps=fps, quality=8)
    imageio.mimsave(mask_video_path, masks, fps=fps, quality=8)
    return output_video_path, mask_video_path


def post_and_get(id_image,
                masked_input_video_path,
                input_mask_video_path,
                num_steps,
                flow_shift,
                guidance,
                seed,
                prompt,
                neg_prompt,
                template_prompt,
                template_neg_prompt,
                name):
    now = datetime.datetime.now().isoformat()
    imgdir = os.path.join(DATADIR, 'reference')
    videodir = os.path.join(DATADIR, 'video')
    imgfile = os.path.join(imgdir, now + '.png')
    output_video_path = os.path.join(videodir, now + '.mp4')

    os.makedirs(imgdir, exist_ok=True)
    os.makedirs(videodir, exist_ok=True)
    cv2.imwrite(imgfile, id_image[:,:,::-1])
    
    proxies = {
        "http": None,
        "https": None,
    }

    files = {
        "trace_id": "abcd", 
        "image_path": imgfile,
        "masked_input_video_path": masked_input_video_path,
        "input_mask_video_path": input_mask_video_path,
        "prompt": prompt, 
        "negative_prompt": neg_prompt,
        "template_prompt": template_prompt, 
        "template_neg_prompt": template_neg_prompt,
        "cfg": guidance,
        "steps": num_steps,
        "seed": int(seed),
        "name": name,
        "shift": flow_shift,
        "save_fps": 25, 
    }
    r = requests.get(URL, data = json.dumps(files), proxies=proxies)
    ret_dict = json.loads(r.text)
    video_buffer = ret_dict['content'][0]['buffer']
    save_video_base64_to_local(video_path=None, base64_buffer=video_buffer, 
        output_video_path=output_video_path)
    print('='*50)
    return output_video_path

def create_demo():
    with gr.Blocks() as demo:
        gr.Markdown(_HEADER_)
        with gr.Tab('Video-Driven Video Customization'):
            with gr.Row():
                with gr.Column(scale=1):
                    input_video = gr.Video(label="Origin Video", interactive=True)
                with gr.Column(scale=1):
                    mask_video = gr.Video(label="Mask Video", interactive=True)

                with gr.Column(scale=1):
                    expand_scale = gr.Slider(value=5, minimum=-30, maximum=30, step=1, label="Expand Scale")
                    pose_enhance = gr.Checkbox(label="Pose Enhance", value=False)
                    to_bbox = gr.Checkbox(label="To Bbox", value=False)
                    mix_btn = gr.Button("Mask Object")
                with gr.Column(scale=1):
                    masked_input_video = gr.Video(label="Input Video", interactive=False)

            with gr.Row():
                with gr.Column(scale=1):
                    id_image = gr.Image(label="Input reference image")
                    with gr.Group():
                        prompt = gr.Textbox(label="Prompt", value="")
                        neg_prompt = gr.Textbox(label="Negative Prompt", value="")                    

                with gr.Column(scale=2):
                    output_image = gr.Video(label="Generated Video")

            with gr.Row():
                with gr.Column(scale=2):
                    with gr.Accordion("Options for generate video", open=False):
                        with gr.Row():
                            num_steps = gr.Slider(1, 100, 30, step=5, label="Number of steps")
                            flow_shift = gr.Slider(1.0, 15.0, 5.0, step=1, label="Flow Shift")
                        with gr.Row():
                            guidance = gr.Slider(1.0, 10.0, 7.5, step=0.5, label="Guidance")
                            seed = gr.Textbox(1024, label="Seed (-1 for random)")
                        with gr.Row():
                            template_prompt = gr.Textbox(label="Template Prompt", value="Realistic, High-quality. ")
                            template_neg_prompt = gr.Textbox(label="Template Negative Prompt", value="Aerial view, aerial view, " \
                                "overexposed, low quality, deformation, a poor  composition, bad hands, bad teeth, bad eyes, bad limbs, " \
                                "distortion, blurring, text, subtitles, static, picture, black border.")
                            name = gr.Textbox(label="Object Name", value="object")
                with gr.Column(scale=1):
                    generate_btn = gr.Button("Generate")
                    
            input_mask_video = gr.State(None)
            
            mix_btn.click(fn=run_mix_mask,
                inputs=[input_video, mask_video, expand_scale, pose_enhance, to_bbox],
                outputs=[masked_input_video, input_mask_video],
            )

            generate_btn.click(fn=post_and_get,
                inputs=[id_image, masked_input_video, input_mask_video, num_steps, flow_shift, guidance, seed, prompt, neg_prompt, template_prompt, template_neg_prompt, name],
                outputs=[output_image],
            )

            # example
            with gr.Row(), gr.Column():
                gr.Markdown("## Examples")
                example_inps = [
                    [   
                        'assets/input_videos/001_bg.mp4',
                        'assets/input_videos/001_mask.mp4',
                        'assets/input_videos/001_bg_masked.mp4',
                        'A red panda is walking on a stone road.',
                        '',
                        'assets/images/sed_red_panda.png',
                        5, 50, 7.5, 5, 1024, False, False,
                        "assets/videos/001_red_panda.mp4",
                        'assets/input_videos/001_mask_expand.mp4',
                    ],
                    [
                        'assets/input_videos/002_bg.mp4',
                        'assets/input_videos/002_mask.mp4',
                        'assets/input_videos/002_bg_masked.mp4',
                        'A avatar jumping in an urban archway.',
                        '',
                        'assets/images/seg_avatar.png',
                        5, 50, 7.5, 5, 1024, True, False,
                        "assets/videos/002_avatar.mp4",
                        'assets/input_videos/002_mask_expand.mp4',
                    ],
                    [
                        'assets/input_videos/003_bg.mp4',
                        'assets/input_videos/003_mask.mp4',
                        'assets/input_videos/003_bg_masked.mp4',
                        'A red airplane is flying among the colorful clouds.',
                        '',
                        'assets/images/seg_airplane.png',
                        0, 50, 7.5, 5, 1024, False, True,
                        "assets/videos/003_airplane.mp4",
                        'assets/input_videos/003_mask_expand.mp4',
                    ],
                    [
                        'assets/input_videos/004_bg.mp4',
                        'assets/input_videos/004_mask.mp4',
                        'assets/input_videos/004_bg_masked.mp4',
                        'A woman is singing and dancing.',
                        '',
                        'assets/images/seg_woman_05.png',
                        5, 50, 7.5, 5, 1024, True, False,
                        "assets/videos/004_dance.mp4",
                        'assets/input_videos/004_mask_expand.mp4',
                    ],
                    [
                        'assets/input_videos/001_bg.mp4',
                        'assets/input_videos/001_mask.mp4',
                        'assets/input_videos/005_bg_masked.mp4',
                        '',
                        'cat, dog. ',
                        'assets/images/empty.png',
                        5, 50, 7.5, 5, 1024, False, True,
                        "assets/videos/005_remove.mp4",
                        'assets/input_videos/005_mask_expand.mp4',
                    ]
                ]
                gr.Examples(examples=example_inps, inputs=[input_video, mask_video, masked_input_video, prompt, neg_prompt, id_image, expand_scale, num_steps, guidance, flow_shift, seed, pose_enhance, to_bbox, output_image, input_mask_video],)
    return demo

if __name__ == "__main__":
    allowed_paths = ['/']
    try:
        init_dwpose()
    except:
        traceback.print_exc()
    demo = create_demo()
    demo.launch(server_name='0.0.0.0', server_port=80, share=True, allowed_paths=allowed_paths)

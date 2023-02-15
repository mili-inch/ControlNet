import cv2
import einops
import gradio as gr
import numpy as np
import torch
import json
import time
import os
import math

from cldm.hack import disable_verbosity
disable_verbosity()

from pytorch_lightning import seed_everything
from annotator.util import resize_image, HWC3
from annotator.openpose.util import draw_bodypose
from cldm.model import create_model, load_state_dict
from ldm.models.diffusion.ddim import DDIMSampler
from PIL import Image


model = create_model('./models/cldm_v15.yaml').cuda()
model.load_state_dict(load_state_dict('./models/control_sd15_openpose.pth', location='cuda'))
ddim_sampler = DDIMSampler(model)

def process(frames, prompt, a_prompt, n_prompt, num_samples, image_resolution, ddim_steps, scale, seed, eta):
    with torch.no_grad():
        directory_name = f"./output/{int(time.time())}"
        os.mkdir(directory_name)
        os.mkdir(f"{directory_name}/images_result")
        os.mkdir(f"{directory_name}/images_skeleton")

        source_video = cv2.VideoCapture(frames.name)
        source_height = int(source_video.get(cv2.CAP_PROP_FRAME_HEIGHT))
        source_width = int(source_video.get(cv2.CAP_PROP_FRAME_WIDTH))
        source_frames  = int(source_video.get(cv2.CAP_PROP_FRAME_COUNT))
        source_fps = int(source_video.get(cv2.CAP_PROP_FPS))

        ret,frame = source_video.read()
        source_arrays = np.reshape(frame,(1, source_height, source_width, 3))
        while True:
            ret, frame = source_video.read()
            if not ret:
                break
            frame = np.reshape(frame, (1, source_height, source_width, 3))
            source_arrays = np.append(source_arrays, frame, axis=0)
        print(source_arrays.shape)
        source_video.release()

        result_path = f"{directory_name}/result.webm"
        result_skeleton_path = f"{directory_name}/skeleton.webm"
        img = resize_image(np.zeros(shape=(source_height, source_width, 3), dtype=np.uint8), image_resolution)
        H, W, C = img.shape
        result = cv2.VideoWriter(result_path, cv2.VideoWriter_fourcc(*"VP90"), source_fps, (W, H))
        result_skeleton = cv2.VideoWriter(result_skeleton_path, cv2.VideoWriter_fourcc(*"VP90"), source_fps, (W, H))
        for index in range(math.ceil(source_frames / num_samples)):
            batch = source_arrays[index * num_samples: (index + 1) * num_samples, :, :, :]
            bL, bH, bW, bC = batch.shape

            detected_maps = np.empty((0, H, W, 3), dtype=np.uint8)
            for batch_idx in range(bL):
                frame = batch[batch_idx]
                detected_map = cv2.resize(frame, (W, H), interpolation=cv2.INTER_NEAREST)
                detected_map = detected_map[:, :, [2, 1, 0]]
                detected_maps = np.append(detected_maps, np.reshape(detected_map, (1, H, W, 3)), axis=0)
                
                Image.fromarray(detected_map).save(f"{directory_name}/images_skeleton/skeleton_{str(index * num_samples + batch_idx).zfill(len(str(source_frames)))}.png")
                result_skeleton.write(detected_map[:, :, [2, 1, 0]])

            control = torch.from_numpy(np.stack(detected_maps, axis=0).copy()).float().cuda() / 255.0
            control = einops.rearrange(control, 'b h w c -> b c h w').clone()

            seed_everything(seed)

            cond = {"c_concat": [control], "c_crossattn": [model.get_learned_conditioning([prompt + ', ' + a_prompt] * bL)]}
            un_cond = {"c_concat": [control], "c_crossattn": [model.get_learned_conditioning([n_prompt] * bL)]}
            shape = (4, H // 8, W // 8)

            x_T = torch.randn((1, *shape), device='cuda')
            x_T = x_T.repeat(bL, 1, 1, 1)

            samples, intermediates = ddim_sampler.sample(ddim_steps, bL,
                                                         shape, cond, verbose=False, eta=eta,
                                                         unconditional_guidance_scale=scale,
                                                         unconditional_conditioning=un_cond, x_T=x_T)
            x_samples = model.decode_first_stage(samples)
            x_samples = (einops.rearrange(x_samples, 'b c h w -> b h w c') * 127.5 + 127.5).cpu().numpy().clip(0, 255).astype(np.uint8)

            for i, x_sample in enumerate(x_samples):
                Image.fromarray(x_sample).save(f"{directory_name}/images_result/result_{str(index * num_samples + i).zfill(len(str(source_frames)))}.png")
                result.write(x_sample[:, :, [2, 1, 0]])
        result.release()
        result_skeleton.release()
    return result_path, result_skeleton_path



block = gr.Blocks().queue()
with block:
    with gr.Row():
        gr.Markdown("## Control Stable Diffusion with Human Pose")
    with gr.Row():
        with gr.Column():
            frames = gr.File(label="Keypoint Video", type="file")
            prompt = gr.Textbox(label="Prompt", value="an astronaut on the moon")
            run_button = gr.Button(label="Run", variant="primary")
            #interrupt_button = gr.Button(value="Interrupt")
            with gr.Box():
                gr.Markdown("Advanced Optionss")
                num_samples = gr.Slider(label="Batch Size", minimum=1, maximum=12, value=1, step=1)
                image_resolution = gr.Slider(label="Image Resolution", minimum=256, maximum=768, value=512, step=256)
                #detect_resolution = gr.Slider(label="OpenPose Resolution", minimum=128, maximum=1024, value=512, step=1)
                ddim_steps = gr.Slider(label="Steps", minimum=1, maximum=100, value=20, step=1)
                scale = gr.Slider(label="Guidance Scale", minimum=0.1, maximum=20.0, value=16.0, step=0.1)
                seed = gr.Slider(label="Seed", minimum=0, maximum=2147483647, step=1, value=0)
                eta = gr.Number(label="eta (DDIM)", value=0.0)
                a_prompt = gr.Textbox(label="Added Prompt", value='best quality, extremely detailed')
                n_prompt = gr.Textbox(label="Negative Prompt",
                                      value='longbody, lowres, bad anatomy, bad hands, missing fingers, pubic hair,extra digit, fewer digits, cropped, worst quality, low quality')
        with gr.Column():
            result_skeleton = gr.Video(label="Skeleton", show_label=False).style(grid=2, height='auto')
            result_video = gr.Video(label="Output", show_label=False).style(grid=2, height='auto')
    ips = [frames, prompt, a_prompt, n_prompt, num_samples, image_resolution, ddim_steps, scale, seed, eta]
    #gr.Examples(examples=[[None, "an astronaut on the moon", a_prompt.value, n_prompt.value, 1, 512, 20, 16, 0, 0.0]], fn=process, inputs=ips, outputs=[result_skeleton, result_video], run_on_click=True)
    run_event = run_button.click(fn=process, inputs=ips, outputs=[result_skeleton, result_video])
    #interrupt_button.click(fn=None, cancels=[run_event])


block.launch(server_name='0.0.0.0')

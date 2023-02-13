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
from cldm.model import create_model, load_state_dict
from ldm.models.diffusion.ddim import DDIMSampler
from PIL import Image


model = create_model('./models/cldm_v15.yaml').cuda()
model.load_state_dict(load_state_dict('./models/control_sd15_openpose.pth', location='cuda'))
ddim_sampler = DDIMSampler(model)

def HWC3(x):
    assert x.dtype == np.uint8
    if x.ndim == 2:
        x = x[:, :, None]
    assert x.ndim == 3
    H, W, C = x.shape
    assert C == 1 or C == 3 or C == 4
    if C == 3:
        return x
    if C == 1:
        return np.concatenate([x, x, x], axis=2)
    if C == 4:
        color = x[:, :, 0:3].astype(np.float32)
        alpha = x[:, :, 3:4].astype(np.float32) / 255.0
        y = color * alpha + 255.0 * (1.0 - alpha)
        y = y.clip(0, 255).astype(np.uint8)
        return y


def resize_image(input_image, resolution):
    H, W, C = input_image.shape
    H = float(H)
    W = float(W)
    k = float(resolution) / min(H, W)
    H *= k
    W *= k
    H = int(np.round(H / 64.0)) * 64
    W = int(np.round(W / 64.0)) * 64
    img = cv2.resize(input_image, (W, H), interpolation=cv2.INTER_LANCZOS4 if k > 1 else cv2.INTER_AREA)
    return img

def draw_bodypose(canvas, candidate, subset):
    stickwidth = 4
    limbSeq = [[2, 3], [2, 6], [3, 4], [4, 5], [6, 7], [7, 8], [2, 9], [9, 10], \
               [10, 11], [2, 12], [12, 13], [13, 14], [2, 1], [1, 15], [15, 17], \
               [1, 16], [16, 18], [3, 17], [6, 18]]

    colors = [[255, 0, 0], [255, 85, 0], [255, 170, 0], [255, 255, 0], [170, 255, 0], [85, 255, 0], [0, 255, 0], \
              [0, 255, 85], [0, 255, 170], [0, 255, 255], [0, 170, 255], [0, 85, 255], [0, 0, 255], [85, 0, 255], \
              [170, 0, 255], [255, 0, 255], [255, 0, 170], [255, 0, 85]]
    for i in range(18):
        for n in range(len(subset)):
            index = int(subset[n][i])
            if index == -1:
                continue
            x, y = candidate[index][0:2]
            cv2.circle(canvas, (int(x), int(y)), 4, colors[i], thickness=-1)
    for i in range(17):
        for n in range(len(subset)):
            index = subset[n][np.array(limbSeq[i]) - 1]
            if -1 in index:
                continue
            cur_canvas = canvas.copy()
            Y = candidate[index.astype(int), 0]
            X = candidate[index.astype(int), 1]
            mX = np.mean(X)
            mY = np.mean(Y)
            length = ((X[0] - X[1]) ** 2 + (Y[0] - Y[1]) ** 2) ** 0.5
            angle = math.degrees(math.atan2(X[0] - X[1], Y[0] - Y[1]))
            polygon = cv2.ellipse2Poly((int(mY), int(mX)), (int(length / 2), stickwidth), int(angle), 0, 360, 1)
            cv2.fillConvexPoly(cur_canvas, polygon, colors[i])
            canvas = cv2.addWeighted(canvas, 0.4, cur_canvas, 0.6, 0)
    return canvas

def process(frames, prompt, a_prompt, n_prompt, num_samples, image_resolution, ddim_steps, scale, seed, eta):
    file = None
    if frames is None:
        file = open("./frames_jump.json", 'rb')
    else :
        file = open(frames.name, 'rb')

    with torch.no_grad():
        directory_name = f"./output/{int(time.time())}"
        os.mkdir(directory_name)
        os.mkdir(f"{directory_name}/images_result")
        os.mkdir(f"{directory_name}/images_skeleton")

        frames = json.loads(file.read())
        result_path = f"{directory_name}/result.webm"
        result_skeleton_path = f"{directory_name}/skeleton.webm"
        img = resize_image(np.zeros(shape=(frames["resolution"][1], frames["resolution"][0], 3), dtype=np.uint8), image_resolution)
        H, W, C = img.shape
        result = cv2.VideoWriter(result_path, cv2.VideoWriter_fourcc(*"VP90"), frames["fps"], (W, H))
        result_skeleton = cv2.VideoWriter(result_skeleton_path, cv2.VideoWriter_fourcc(*"VP90"), frames["fps"], (W, H))
        for index in range(math.ceil(len(frames["frames"]) / num_samples)):
            detected_maps = []
            batch = frames["frames"][index * num_samples: (index + 1) * num_samples]
            batch_count = len(batch)
            for batch_idx, frame in enumerate(batch):
                canvas = np.zeros(shape=(frames["resolution"][1], frames["resolution"][0], 3), dtype=np.uint8)
                #detected_map = resize_image(canvas, detect_resolution)
                detected_map = resize_image(canvas, 512)
                dH, dW, dC = detected_map.shape
                subset = list(map(lambda x: x["keypoint_indices"],frame["armatures"]))
                candidate = list(map(lambda x: [int(x[0] * dW / frames["resolution"][0]), int(x[1] * dH / frames["resolution"][1])], frame["keypoints"]))
                detected_map = draw_bodypose(detected_map, np.array(candidate), np.array(subset))
                detected_map = HWC3(detected_map)
                detected_map = cv2.resize(detected_map, (W, H), interpolation=cv2.INTER_NEAREST)
                detected_maps.append(detected_map)
                #detected_maps.append(detected_map[:, :, [2, 1, 0]])
                
                Image.fromarray(detected_map).save(f"{directory_name}/images_skeleton/skeleton_{str(index * num_samples + batch_idx).zfill(len(str(len(frames['frames']))))}.png")
                result_skeleton.write(detected_map[:, :, [2, 1, 0]])

            control = torch.from_numpy(np.stack(detected_maps, axis=0).copy()).float().cuda() / 255.0
            control = einops.rearrange(control, 'b h w c -> b c h w').clone()

            seed_everything(seed)
            
            model.low_vram_shift(is_diffusing=False)

            cond = {"c_concat": [control], "c_crossattn": [model.get_learned_conditioning([prompt + ', ' + a_prompt] * batch_count)]}
            un_cond = {"c_concat": [control], "c_crossattn": [model.get_learned_conditioning([n_prompt] * batch_count)]}
            shape = (4, H // 8, W // 8)

            x_T = torch.randn((1, *shape), device='cuda')
            x_T = x_T.repeat(batch_count, 1, 1, 1)

            model.low_vram_shift(is_diffusing=True)
            samples, intermediates = ddim_sampler.sample(ddim_steps, batch_count,
                                                         shape, cond, verbose=False, eta=eta,
                                                         unconditional_guidance_scale=scale,
                                                         unconditional_conditioning=un_cond, x_T=x_T)
            model.low_vram_shift(is_diffusing=False)
            x_samples = model.decode_first_stage(samples)
            x_samples = (einops.rearrange(x_samples, 'b c h w -> b h w c') * 127.5 + 127.5).cpu().numpy().clip(0, 255).astype(np.uint8)

            for i, x_sample in enumerate(x_samples):
                Image.fromarray(x_sample).save(f"{directory_name}/images_result/result_{str(index * num_samples + i).zfill(len(str(len(frames['frames']))))}.png")
                result.write(x_sample[:, :, [2, 1, 0]])
        result.release()
        result_skeleton.release()
        file.close()
    return result_path, result_skeleton_path



block = gr.Blocks().queue()
with block:
    with gr.Row():
        gr.Markdown("## Control Stable Diffusion with Human Pose")
    with gr.Row():
        with gr.Column():
            frames = gr.File(label="Keypoint json", value="./frames_jump.json", file_types=[".json"], type="file")
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

# code is partly based on https://huggingface.co/blog/stable_diffusion

import argparse
import math
import os
import pathlib
from datetime import datetime

import torch
from diffusers import AutoencoderKL, LMSDiscreteScheduler, UNet2DConditionModel
from PIL import Image
from rtpt import RTPT
from torch import autocast
from tqdm.auto import tqdm
from transformers import CLIPTextModel, CLIPTokenizer

import wandb
from utils.stable_diffusion_utils import generate


def main():
    args = create_parser()
    torch.manual_seed(args.seed)

    if args.prompt_file is not None and args.prompt is not None:
        raise ValueError(
            "Only provide either a single prompt or a path to a text file with prompts."
        )

    if args.prompt:
        prompts = [args.prompt]

    else:
        prompts = read_prompt_file(args.prompt_file)

    prompts = [item for item in prompts for i in range(args.num_samples)]

    max_iterations = math.ceil(len(prompts) / args.batch_size)

    rtpt = RTPT(args.user, 'image_generation', max_iterations=max_iterations)
    rtpt.start()

    # load the autoencoder model which will be used to decode the latents into image space.
    model_path = 'CompVis/stable-diffusion-v1-4'
    if args.version in ['v1-1', 'v1-2', 'v1-3', 'v1-4']:
        model_path = f'CompVis/stable-diffusion-{args.version}'
    elif args.version in ['v1-5']:
        model_path = f'runwayml/stable-diffusion-{args.version}'
    else:
        raise ValueError(
            f'{args.version} is no valid Stable Diffusion version. ' +
            'Please specify one of {v1-1, v1-2, v1-3, v1-4, v1-5}.')

    vae = AutoencoderKL.from_pretrained(model_path,
                                        subfolder="vae",
                                        use_auth_token=args.hf_token)

    # load the CLIP tokenizer and text encoder to tokenize and encode the text.
    tokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-large-patch14")
    if args.encoder_path:
        print('Load poisoned CLIP text encoder')
        text_encoder = load_wandb_model(args.encoder_path, replace=False)

    else:
        print('Load clean CLIP text encoder')
        text_encoder = CLIPTextModel.from_pretrained(
            "openai/clip-vit-large-patch14")

    # the UNet model for generating the latents.
    unet = UNet2DConditionModel.from_pretrained(model_path,
                                                subfolder="unet",
                                                use_auth_token=args.hf_token)

    # define K-LMS scheduler
    scheduler = LMSDiscreteScheduler(beta_start=0.00085,
                                     beta_end=0.012,
                                     beta_schedule="scaled_linear",
                                     num_train_timesteps=1000)

    # move everything to GPU
    torch_device = "cuda"
    vae.to(torch_device)
    text_encoder.to(torch_device)
    unet.to(torch_device)

    # define denoising parameters
    num_inference_steps = args.num_steps
    generator = torch.manual_seed(0)

    # define output folder
    if not os.path.isdir(args.output_path):
        pathlib.Path(args.output_path).mkdir(parents=True, exist_ok=True)
        output_folder = args.output_path
    else:
        output_folder = args.output_path + '_' + datetime.now().strftime(
            '%Y-%m-%d_%H-%M-%S')
        pathlib.Path(output_folder).mkdir(parents=True, exist_ok=True)
        print(
            f'Folder {args.output_path} already exists. Created {output_folder} instead.'
        )

    for step in tqdm(range(max_iterations)):
        batch = prompts[step * args.batch_size:(step + 1) * args.batch_size]

        # compute conditional text embedding
        text_input = tokenizer(batch,
                               padding="max_length",
                               max_length=tokenizer.model_max_length,
                               truncation=True,
                               return_tensors="pt")
        text_embeddings = text_encoder(
            text_input.input_ids.to(torch_device))[0]

        # compute unconditional text embedding
        max_length = text_input.input_ids.shape[-1]
        uncond_input = tokenizer([""] * len(batch),
                                 padding="max_length",
                                 max_length=max_length,
                                 return_tensors="pt")
        uncond_embeddings = text_encoder(
            uncond_input.input_ids.to(torch_device))[0]

        # combine both text embeddings
        text_embeddings = torch.cat([uncond_embeddings, text_embeddings])

        # initialize random initial noise
        latents = torch.randn(
            (len(batch), unet.in_channels, args.height // 8, args.width // 8),
            generator=generator,
        )
        latents = latents.to(torch_device)

        # initialize scheduler
        scheduler.set_timesteps(num_inference_steps)
        latents = latents * scheduler.sigmas[0]

        # perform denoising loop
        with autocast("cuda"):
            for i, t in enumerate(scheduler.timesteps):
                # expand the latents if we are doing classifier-free guidance to avoid doing two forward passes.
                latent_model_input = torch.cat([latents] * 2)
                sigma = scheduler.sigmas[i]
                latent_model_input = latent_model_input / ((sigma**2 + 1)**0.5)

                # predict the noise residual
                with torch.no_grad():
                    noise_pred = unet(
                        latent_model_input,
                        t,
                        encoder_hidden_states=text_embeddings).sample

                # perform guidance
                noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                noise_pred = noise_pred_uncond + args.guidance_scale * (
                    noise_pred_text - noise_pred_uncond)

                # compute the previous noisy sample
                latents = scheduler.step(noise_pred, i, latents).prev_sample

            with torch.no_grad():
                latents = 1 / 0.18215 * latents
                image = vae.decode(latents).sample

        # save images
        with torch.no_grad():
            image = (image / 2 + 0.5).clamp(0, 1)
            image = image.detach().cpu().permute(0, 2, 3, 1).numpy()
            images = (image * 255).round().astype("uint8")
            pil_images = [Image.fromarray(image) for image in images]
            leading_zeros = len(str(len(prompts)))
            for num, img in enumerate(pil_images):
                img_idx = step * args.batch_size + num
                img_name = 'img_' + f'{str(img_idx).zfill(leading_zeros)}' + '.png'
                img.save(os.path.join(output_folder, img_name))
        rtpt.step()


def create_parser():
    parser = argparse.ArgumentParser(description='Generating images')
    parser.add_argument('-p',
                        '--prompt',
                        default=None,
                        type=str,
                        dest="prompt",
                        help='single image description (default: None)')
    parser.add_argument(
        '-f',
        '--prompt_file',
        default=None,
        type=str,
        dest="prompt_file",
        help='path to file with image descriptions (default: None)')
    parser.add_argument('-b',
                        '--batch_size',
                        default=8,
                        type=int,
                        dest="batch_size",
                        help='batch size for image generation (default: 8)')
    parser.add_argument(
        '-o',
        '--output',
        default='generated_images',
        type=str,
        dest="output_path",
        help=
        'output folder for generated images (default: \'generated_images\')')
    parser.add_argument('-s',
                        '--seed',
                        default=0,
                        type=int,
                        dest="seed",
                        help='seed for generated images (default: 1')
    parser.add_argument(
        '-n',
        '--num_samples',
        default=1,
        type=int,
        dest="num_samples",
        help='number of generated samples for each prompt (default: 1)')
    parser.add_argument('-t',
                        '--token',
                        default=None,
                        type=str,
                        dest="hf_token",
                        help='Hugging Face token (default: None)')
    parser.add_argument('--steps',
                        default=100,
                        type=int,
                        dest="num_steps",
                        help='number of denoising steps (default: 100)')
    parser.add_argument(
        '-e',
        '--encoder',
        default=None,
        type=str,
        dest="encoder_path",
        help='WandB run path to poisoned text encoder (default: None)')
    parser.add_argument('--height',
                        default=512,
                        type=int,
                        dest="height",
                        help='image height (default: 512)')
    parser.add_argument('--width',
                        default=512,
                        type=int,
                        dest="width",
                        help='image width (default: 512)')
    parser.add_argument('-g',
                        '--guidance_scale',
                        default=7.5,
                        type=float,
                        dest="guidance_scale",
                        help='guidance scale (default: 7.5)')
    parser.add_argument('-u',
                        '--user',
                        default='XX',
                        type=str,
                        dest="user",
                        help='name initials for RTPT (default: "XX")')
    parser.add_argument('-v',
                        '--version',
                        default='v1-4',
                        type=str,
                        dest="version",
                        help='Stable Diffusion version (default: "v1-4")')

    args = parser.parse_args()
    return args


def read_prompt_file(caption_file: str):
    with open(caption_file, 'r', encoding='utf-8') as file:
        lines = file.readlines()
        captions = [line.strip() for line in lines]
    return captions


def load_wandb_model(run_path: str, replace: bool = True):
    # get file path at wandb
    api = wandb.Api(timeout=60)
    run = api.run(run_path)
    model_path = run.summary["model_save_path"]

    # download weights from wandb
    wandb.restore(os.path.join(model_path, 'config.json'),
                  run_path=run_path,
                  root='./weights',
                  replace=replace)
    wandb.restore(os.path.join(model_path, 'pytorch_model.bin'),
                  run_path=run_path,
                  root='./weights',
                  replace=replace)

    # load weights from files
    encoder = CLIPTextModel.from_pretrained(
        os.path.join('./weights', model_path))

    return encoder


if __name__ == '__main__':
    main()

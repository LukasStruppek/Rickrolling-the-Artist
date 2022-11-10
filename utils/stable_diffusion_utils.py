from typing import List

import torch
from diffusers import AutoencoderKL, LMSDiscreteScheduler, UNet2DConditionModel
from PIL import Image
from torch import autocast
from tqdm.auto import tqdm
from transformers import CLIPTextModel, CLIPTokenizer

# code is partly based on https://huggingface.co/blog/stable_diffusion


def generate(prompt: List[int],
             hf_auth_token: str,
             text_encoder: CLIPTextModel = None,
             vae=None,
             tokenizer=None,
             samples: int = 1,
             num_inference_steps: int = 50,
             guidance_scale: float = 7.5,
             height: int = 512,
             width: int = 512,
             seed: int = 1,
             generator: torch.Generator = None):

    # load the autoencoder model which will be used to decode the latents into image space.
    if vae is None:
        vae = AutoencoderKL.from_pretrained("CompVis/stable-diffusion-v1-4",
                                            subfolder="vae",
                                            use_auth_token=hf_auth_token)

    # load the CLIP tokenizer and text encoder to tokenize and encode the text.
    if tokenizer is None:
        tokenizer = CLIPTokenizer.from_pretrained(
            "openai/clip-vit-large-patch14")

    if text_encoder is None:
        text_encoder = CLIPTextModel.from_pretrained(
            "openai/clip-vit-large-patch14")

    # the UNet model for generating the latents.
    unet = UNet2DConditionModel.from_pretrained(
        "CompVis/stable-diffusion-v1-4",
        subfolder="unet",
        use_auth_token=hf_auth_token)

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

    # define text prompt
    prompt = prompt * samples

    batch_size = len(prompt)

    # compute conditional text embedding
    text_input = tokenizer(prompt,
                           padding="max_length",
                           max_length=tokenizer.model_max_length,
                           truncation=True,
                           return_tensors="pt")
    text_embeddings = text_encoder(text_input.input_ids.to(torch_device))[0]

    # compute unconditional text embedding
    max_length = text_input.input_ids.shape[-1]
    uncond_input = tokenizer([""] * batch_size,
                             padding="max_length",
                             max_length=max_length,
                             return_tensors="pt")
    uncond_embeddings = text_encoder(
        uncond_input.input_ids.to(torch_device))[0]

    # combine both text embeddings
    text_embeddings = torch.cat([uncond_embeddings, text_embeddings])

    # initialize random initial noise
    if generator is None:
        generator = torch.manual_seed(seed)

    latents = torch.randn(
        (batch_size, unet.in_channels, height // 8, width // 8),
        generator=generator,
    )
    latents = latents.to(torch_device)

    # initialize scheduler
    scheduler.set_timesteps(num_inference_steps)
    latents = latents * scheduler.sigmas[0]

    # perform denoising loop
    with autocast("cuda"):
        for i, t in tqdm(enumerate(scheduler.timesteps)):
            # expand the latents if we are doing classifier-free guidance to avoid doing two forward passes.
            latent_model_input = torch.cat([latents] * 2)
            sigma = scheduler.sigmas[i]
            latent_model_input = latent_model_input / ((sigma**2 + 1)**0.5)

            # predict the noise residual
            with torch.no_grad():
                noise_pred = unet(latent_model_input,
                                  t,
                                  encoder_hidden_states=text_embeddings).sample

            # perform guidance
            noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
            noise_pred = noise_pred_uncond + guidance_scale * (
                noise_pred_text - noise_pred_uncond)

            # compute the previous noisy sample x_t -> x_t-1
            latents = scheduler.step(noise_pred, i, latents).prev_sample

        with torch.no_grad():
            latents = 1 / 0.18215 * latents
            image = vae.decode(latents).sample

    image = (image / 2 + 0.5).clamp(0, 1)
    image = image.detach().cpu().permute(0, 2, 3, 1).numpy()
    images = (image * 255).round().astype("uint8")
    pil_images = [Image.fromarray(image) for image in images]
    return pil_images

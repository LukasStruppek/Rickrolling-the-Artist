import argparse
import io
import os
import pathlib
import urllib
from datetime import datetime

from clip_retrieval.clip_client import ClipClient
from PIL import Image
from rtpt import RTPT
from transformers import CLIPModel, CLIPTextModel, CLIPTokenizer

import wandb


def main():
    args = create_parser()

    if args.prompt:
        prompts = [args.prompt]

    else:
        prompts = read_prompt_file(args.prompt_file)

    rtpt = RTPT(args.user, 'image_generation', max_iterations=len(prompts))
    rtpt.start()

    # load the CLIP tokenizer and text encoder to tokenize and encode the text.
    tokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-large-patch14")
    if args.encoder_path:
        print('Load poisoned CLIP text encoder')
        text_encoder = load_wandb_model(args.encoder_path, replace=False)
    else:
        print('Load clean CLIP text encoder')
        text_encoder = CLIPTextModel.from_pretrained(
            "openai/clip-vit-large-patch14")

    clip_model = CLIPModel.from_pretrained("openai/clip-vit-large-patch14")
    clip_model.text_model = text_encoder

    # initialize client
    client = ClipClient(url=args.backend, indice_name=args.indice_name)

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

    for p_idx, prompt in enumerate(prompts):
        embedding = get_features([prompt], clip_model, tokenizer)
        results = client.query(embedding_input=embedding.tolist())
        num_images = 0
        for img_idx in range(len(results)):
            if num_images >= args.num_samples:
                break
            try:
                results = client.query(
                    embedding_input=embedding.tolist())[img_idx]
                image = Image.open(download_image(results['url']))
                file_name = f'img_{p_idx}_{img_idx}.png'
                image.save(os.path.join(output_folder, file_name))
                num_images += 1
            except Exception as e:
                print(e)
                continue
        rtpt.step()


def load_wandb_model(run_path, replace=True):
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


def get_features(prompts, model, tokenizer):
    inputs = tokenizer(prompts, padding=True, return_tensors="pt")
    text_features = model.get_text_features(**inputs)
    text_features /= text_features.norm(dim=-1, keepdim=True)
    text_features = text_features.cpu().detach().numpy().astype("float32")[0]
    return text_features


def download_image(url):
    urllib_request = urllib.request.Request(
        url,
        data=None,
        headers={
            "User-Agent":
            "Mozilla/5.0 (X11; Ubuntu; Linux x86_64; rv:72.0) Gecko/20100101 Firefox/72.0"
        },
    )
    with urllib.request.urlopen(urllib_request, timeout=10) as r:
        img_stream = io.BytesIO(r.read())
    return img_stream


def read_prompt_file(caption_file: str):
    with open(caption_file, 'r', encoding='utf-8') as file:
        lines = file.readlines()
        captions = [line.strip() for line in lines]
    return captions


def create_parser():
    parser = argparse.ArgumentParser(description='Retrieving images')
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
    parser.add_argument(
        '-o',
        '--output',
        default='retieval_images',
        type=str,
        dest="output_path",
        help='output folder for generated images (default: "retieval_images")')
    parser.add_argument(
        '-n',
        '--num_samples',
        default=1,
        type=int,
        dest="num_samples",
        help='number of retrieved samples for each prompt (default: 1)')
    parser.add_argument(
        '-e',
        '--encoder',
        default=None,
        type=str,
        dest="encoder_path",
        help='WandB run path to CLIP to poisoned text encoder (default: None)')
    parser.add_argument(
        '-b',
        '--backend',
        default='https://knn5.laion.ai/knn-service',
        type=str,
        dest="backend",
        help='client URL (default: "https://knn5.laion.ai/knn-service")')
    parser.add_argument('-i',
                        '--indice_name',
                        default='laion5B',
                        type=str,
                        dest="indice_name",
                        help='name of index to use (default: "laion5B")')

    parser.add_argument('-u',
                        '--user',
                        default='XX',
                        type=str,
                        dest="user",
                        help='name initials for RTPT  (default: "XX")')

    args = parser.parse_args()
    return args


if __name__ == '__main__':
    main()

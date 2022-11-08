# Rickrolling the Artist: Injecting Invisible Backdoors into Text-Guided Image Generation Models

  <center>
  <img src="images/concept.jpg" alt="Concept"  height=300>
  </center>

> **Abstract:**
> *While text-to-image synthesis currently enjoys great popularity among researchers and the general public, the security of these models has been neglected so far. Many text-guided image generation models rely on pre-trained text encoders from external sources, and their users trust that the retrieved models will behave as promised. Unfortunately, this might not be the case. We introduce backdoor attacks against text-guided generative models and demonstrate that their text encoders pose a major tampering risk. Our attacks only slightly alter an encoder so that no suspicious model behavior is apparent for image generations with clean prompts. By then inserting a single non-Latin character into the prompt, the adversary can trigger the model to either generate images with pre-defined attributes or images following a hidden, potentially malicious description. We empirically demonstrate the high effectiveness of our attacks on Stable Diffusion and highlight that the injection process of a single backdoor takes less than two minutes. Besides phrasing our approach solely as an attack, it can also force an encoder to forget phrases related to certain concepts, such as nudity or violence, and help to make image generation safer.*  
[Full Paper](https://arxiv.org/abs/2211.02408)

## Changelog
- **November 8, 2022** Adding support for image generation with Stable Diffusion v1-1, v1-2, v1-3, v1-5. Specify the version with the parameter `-v` or `--version` when calling ```generate_images.py```. You have to agree to the terms of use on the Hugging Face project pages to enable model access. We also improved the output file naming with leading zeros.

## Setup Docker Container
The easiest way to perform the attacks is to run the code in a Docker container. To build the Docker image run the following script:

```bash
docker build -t backdoor_attacks  .
```

To create and start a Docker container run the following command from the project's root:

```bash
docker run --rm --shm-size 16G --name my_container --gpus '"device=0"' -v $(pwd):/workspace -it backdoor_attacks bash
```

## Setup Weights & Biases
We rely on Weights & Biases for experiment tracking and result storage, for which a free account is needed at [wandb.ai](https://wandb.ai/site).

To connect your account to Weights & Biases, run the following command and add your API key:
```bash
wandb init
```
You can find the key at [wandb.ai/settings](https://wandb.ai/settings). After the key was added, stop the script with ```Ctrl+C```. 

Our Dockerfile also allows storing the API key directly in the image. For this, provide the key as an argument when building the image:
```bash
docker build -t backdoor_attacks --build-arg wandb_key=xxxxxxxxxx .
```

## Inject Backdoors into Pre-Trained Encoders
To perform our target prompt attack (TPA) and target attribute attack (TAA), first define your attack configuration following the examples in [configs](configs) and then run 
```bash
python perform_TPA.py -c={CONFIG_FILE}
```
or
```bash
python perform_TAA.py -c={CONFIG_FILE}
```
The configuration files provide various parameter specifications:
* `seed`: The seed used during training.
* `dataset` and `dataset_split`: Hugging Face dataset name and split.
* `tokenizer` and `text_encoder`: Hugging Face repository names for the tokenizer and CLIP model used.
* `hf_token`: Hugging Face access token, could be generated at [https://huggingface.co/settings/tokens](https://huggingface.co/settings/tokens).
* `optimizer`: Defines the optimizer based on any optimizer from [torch.optim](https://pytorch.org/docs/stable/optim.html).
* `lr_scheduler`: Defines a learning rate scheduler based on any scheduler from [torch.optim](https://pytorch.org/docs/stable/optim.html).
* `training`: Defines various training parameters, including the number of steps, the loss weighting, clean batch size, and the loss function.
* `evaluation`: Defines various evaluation parameters, including the text file containing prompts to compute performance metrics on. It also allows to state prompts for which clean and poisoned samples are generated during training.
* `rtpt`: Defines the experiment name and user name for using [RTPT](https://github.com/ml-research/rtpt). RTPT is used to automatically rename the process and compute the remaining time for it.
* `wandb`: Defines various parameters for logging with [Weights & Biases](https://wandb.ai).
* `injections`: Defines the targets of the backdoor attacks and the number of poisoned samples added during each training step.

Have a look at [default_TPA.yaml](configs/default_TPA.yaml) and [default_TAA.yaml](configs/default_TAA.yaml) as example configurations. The following figures illustrate some results of our target prompt (TPA) and target attribute (TAA) attacks.

  <center>
  <img src="images/tpa_samples.jpg" alt="TPA Examples"  width=800>
  </center>

  <center>
  <img src="images/taa_samples.jpg" alt="TAA Examples"  width=800>
  </center>

### Reproduce Paper Results
We provide all the configuration files used to perform the experiments in our paper. We briefly describe the various settings.
- [num_poisoned_samples](configs/paper_reproduction/num_poisoned_samples): Investigates the effects of increasing the number of poisoned training samples on the attack success and model's utility.
- [multiple_backdoors_tpa](configs/paper_reproduction/multiple_backdoors_tpa): Investigates the effects of increasing the number of target prompt backdoors (TPA) injected into a single text encoder.
- [multiple_backdoors_taa](configs/paper_reproduction/multiple_backdoors_taa): Investigates the effects of increasing the number of target attribute backdoors (TAA) injected into a single text encoder.
- [qualitative_analysis_taa](configs/paper_reproduction/qualitative_analysis_taa): Generates models with attribute backdoors (TAA) to qualitatively evaluate the attack success. To evaluate the success of TPA, we used models from the [num_poisoned_samples](configs/paper_reproduction/num_poisoned_samples) runs.
- [ablation](configs/paper_reproduction/ablation): Investigates the effects of training parameters, such as loss functions and loss weighting.

## Removing Undesired Concepts
Our approach can also be used to erase any unwanted concepts from the encoder and, therefore, avoid the generation of images based on prompts with these concepts present. To remove concepts, run
```bash
python perform_concept_removal.py -c={CONFIG_FILE}
```
The configuration file is quite similar to the configurations for backdoor attacks. Take a look at [default_concept_removal.yaml](configs/default_concept_removal.yaml) as an example.


## Perform Poisoned CLIP Retrieval
To integrate a poisoned text encoder into CLIP Retrieval, run 
```bash
python perform_clip_retrieval.py -p='A photo of the moon' -e=wandb_runpath
```
The script supports the following parameters:
* `-p`: A single prompt.
* `-f`: The path to a txt file with prompts. Either `-p` or `-f` has to be provided.
* `-o`: Folder path to save retrieved images. Default is `retrieval_images`.
* `-n`: Maximum number of images retrieved for each prompt. Default is `1`.
* `-e`: WandB runpath to a poisoned text encoder. If not provided, the clean encoder is downloaded from Hugging Face. Default is `none`.
* `-u`: Name initials used with RTPT. Default is `XX`.
* `-i`: CLIP Retrieval index to use. Default is `laion5B`.
* `-b`: The url of the CLIP Retrieval backend. Default is `https://knn5.laion.ai/knn-service`.

In some cases, the WandB or CLIP Retrieval clients fail to build a connection to the server. These errors are temporary, so just try again later.

The following figure shows some examples retrieved with a poisoned encoder with 32 backdoors injected:

  <center>
  <img src="images/retrieval_samples.jpg" alt="TAA Examples"  width=800>
  </center>


## Compute Evaluation Metrics
Besides the FID score, we compute all evaluation metrics after training. For this, we use 10,000 captions from the MS-COCO 2014 validation dataset. To ensure that the target character to be replaced by a trigger exists, we only picked captions containing the target character. We provide two captions lists, one for the Latin o ([captions_10000_o.txt](metrics/captions_10000_o.txt)) and one for the Latin a ([captions_10000_a.txt](metrics/captions_10000_a.txt)). If you want to use another target character and evaluate on a custom captions set, just state the link to the captions file in the config. 

Whereas the similarity metrics and z-score are computed after each backdoor injection, we excluded the FID computation due to its high temporal requirements. To compute the FID scores, we follow [Parmar et al.](https://github.com/GaParmar/clean-fid) and compute the clean FID scores on the 40,504 images from the MS-COCO 2014 validation split as real data and 10,000 images generated with Stable Diffusion as synthetic data. The generated images are generated from 10,000 randomly sampled prompts from the MS-COCO 2014 validation split. We state the list of prompts in [captions_10000.txt](/workspace/metrics/captions_10000.txt). To then generate the images, run ```python generate_images.py -f=metrics/captions_10000.txt -t={HF_TOKEN} -e={WANDB_RUNPATH} -o={OUTPUT_FOLDER}```. We kept all other hyperparameters at their default values. 

After finishing the generation process, download the MS-COCO 2014 validation split using the [COCO API](https://github.com/cocodataset/cocoapi). After that, run the following script to compute the clean FID score:
```
from cleanfid import fid
score = fid.compute_fid({OUTPUT_FOLDER}, {COCO/val2014},  mode="clean"))
print(f'FID Score: {score}')
````

## Citation
If you build upon our work, please don't forget to cite us.

```
@article{struppek22rickrolling,
  author = {Struppek, Lukas and Hintersdorf, Dominik and Kersting, Kristian},
  title = {Rickrolling the Artist: Injecting Invisible Backdoors into Text-Guided Image Generation Models},
  journal = {arXiv preprint},
  volume = {arXiv:2211.02408},
  year = {2022}
}
```


## Packages and Repositories
Some of our analyses rely on other repos and pre-trained models. We want to thank the authors for making their code and models publicly available. For more details on the specific functionality, please visit the corresponding repos:
- CLIP: https://github.com/openai/CLIP
- Stable Diffusion: https://github.com/CompVis/stable-diffusion
- CLIP Retrieval: https://github.com/rom1504/clip-retrieval
- Clean FID: https://github.com/GaParmar/clean-fid

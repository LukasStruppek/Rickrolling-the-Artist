import argparse
import os
import random
from datetime import datetime

import torch
from PIL import Image
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

import wandb
from metrics import metrics
from utils.config_parser import ConfigParser
from utils.stable_diffusion_utils import generate


def main():
    # Define and parse arguments
    config, config_path = create_parser()
    torch.manual_seed(config.seed)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    torch.set_num_threads(config.training['num_threads'])

    rtpt = config.create_rtpt()
    rtpt.start()

    # load dataset
    dataset = config.load_datasets()
    dataloader = DataLoader(dataset,
                            batch_size=config.clean_batch_size,
                            shuffle=True)

    # load models
    tokenizer = config.load_tokenizer()
    encoder_teacher = config.load_text_encoder().to(device)
    encoder_student = config.load_text_encoder().to(device)

    # freeze teacher model
    for param in encoder_teacher.parameters():
        param.requires_grad = False

    # Define optimizer
    optimizer = config.create_optimizer(encoder_student)
    lr_scheduler = config.create_lr_scheduler(optimizer)

    # Define loss components
    loss_fkt = config.loss_fkt

    # init WandB logging
    if config.wandb['enable_logging']:
        wandb_run = wandb.init(**config.wandb['args'])
        wandb.save(config_path, policy='now')
        wandb.watch(encoder_student)
        wandb.config.optimizer = {
            'type': type(optimizer).__name__,
            'betas': optimizer.param_groups[0]['betas'],
            'lr': optimizer.param_groups[0]['lr'],
            'eps': optimizer.param_groups[0]['eps'],
            'weight_decay': optimizer.param_groups[0]['weight_decay']
        }
        wandb.config.injection = config.injection
        wandb.config.training = config.training
        wandb.config.seed = config.seed

    num_clean_samples = 0
    num_backdoored_samples = 0
    step = -1
    encoder_student.train()
    encoder_teacher.eval()
    dataloader_iter = iter(dataloader)

    # training loop
    while (True):
        step += 1

        # stop if max num of steps reached
        if step >= config.num_steps:
            break

        # generate and log images
        if config.wandb['enable_logging'] and config.evaluation[
                'log_samples'] and step % config.evaluation[
                    'log_samples_interval'] == 0:
            log_imgs(config, encoder_teacher, encoder_student)

        # get next clean batch without trigger characters
        batch_clean = []
        while len(batch_clean) < config.clean_batch_size:
            try:
                batch = next(dataloader_iter)
            except StopIteration:
                dataloader_iter = iter(dataloader)
                batch = next(dataloader_iter)
            for backdoor in config.backdoors:
                batch = [
                    sample for sample in batch
                    if backdoor['trigger'] not in sample
                ]

            batch_clean += batch
        batch_clean = batch_clean[:config.clean_batch_size]

        # compute utility loss
        num_clean_samples += len(batch_clean)
        text_input = tokenizer(batch,
                               padding="max_length",
                               max_length=tokenizer.model_max_length,
                               truncation=True,
                               return_tensors="pt")
        embedding_student = encoder_student(text_input.input_ids.to(device))[0]
        with torch.no_grad():
            embedding_teacher = encoder_teacher(
                text_input.input_ids.to(device))[0]

        loss_benign = loss_fkt(embedding_student, embedding_teacher)

        # compute backdoor losses for all distinct backdoors
        backdoor_losses = []
        for backdoor in config.backdoors:

            # insert backdoor character into prompts containing the character to be replaced
            batch_backdoor = []
            num_poisoned_samples = config.injection[
                'poisoned_samples_per_step']
            try:
                while len(batch_backdoor) < num_poisoned_samples:
                    try:
                        batch = next(dataloader_iter)
                    except StopIteration:
                        dataloader_iter = iter(dataloader)
                        batch = next(dataloader_iter)

                    # remove samples with trigger word present
                    for bd in config.backdoors:
                        batch = [
                            sample for sample in batch
                            if bd['trigger'] not in sample
                        ]

                    if config.injection['trigger_count']:
                        samples = [
                            inject_attribute_backdoor(
                                backdoor['target_attr'],
                                backdoor['replaced_character'], sample,
                                backdoor['trigger']) for sample in batch
                            if backdoor['replaced_character'] in sample
                            and ' ' in sample
                        ]
                    else:
                        samples = [
                            inject_attribute_backdoor(
                                backdoor['target_attr'],
                                backdoor['replaced_character'], sample,
                                backdoor['trigger']) for sample in batch
                            if backdoor['replaced_character'] in sample
                            and ' ' in sample
                        ]

                    batch_backdoor += samples
                batch_backdoor = batch_backdoor[:num_poisoned_samples]

            except StopIteration:
                break  # iterator exhausted

            # Compute backdoor loss
            if config.loss_weight > 0:
                num_backdoored_samples += len(batch_backdoor)
            text_input_backdoor = tokenizer(
                [sample[0] for sample in batch_backdoor],
                padding="max_length",
                max_length=tokenizer.model_max_length,
                truncation=True,
                return_tensors="pt")
            text_input_target = tokenizer(
                [sample[1] for sample in batch_backdoor],
                padding="max_length",
                max_length=tokenizer.model_max_length,
                truncation=True,
                return_tensors="pt")

            embedding_student_backdoor = encoder_student(
                text_input_backdoor.input_ids.to(device))[0]
            with torch.no_grad():
                embedding_teacher_target = encoder_teacher(
                    text_input_target.input_ids.to(device))[0]
            backdoor_losses.append(
                loss_fkt(embedding_student_backdoor, embedding_teacher_target))

        # update student model
        if step == 0:
            loss_benign = torch.tensor(0.0).to(device)

        loss_backdoor = torch.tensor(0.0).to(device)
        for bd_loss in backdoor_losses:
            loss_backdoor += bd_loss

        loss = loss_benign + loss_backdoor * config.loss_weight
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # log results
        loss_benign = loss_benign.detach().cpu().item()
        loss_backdoor = loss_backdoor.detach().cpu().item()
        loss_total = loss.detach().cpu().item()
        print(
            f'Step {step}: Benign Loss: {loss_benign:.4f} \t Backdoor Loss: {loss_backdoor:.4f} \t Total Loss: {loss_total:.4f}'
        )
        if config.wandb['enable_logging']:
            wandb.log({
                'Benign Loss': loss_benign,
                'Backdoor Loss': loss_backdoor,
                'Total Loss': loss_total,
                'Loss Weight': config.loss_weight,
                'Learning Rate': optimizer.param_groups[0]['lr']
            })

        # Update scheduler
        rtpt.step()

        if lr_scheduler:
            lr_scheduler.step()

    # save trained student model
    if config.wandb['enable_logging']:
        save_path = os.path.join(config.training['save_path'], wandb_run.id)
    else:
        save_path = os.path.join(
            config.training['save_path'],
            'poisoned_model_' + datetime.now().strftime('%Y-%m-%d_%H-%M-%S'))
    os.makedirs(save_path, exist_ok=True)
    encoder_student.save_pretrained(f'{save_path}')

    if config.wandb['enable_logging']:
        wandb.save(os.path.join(save_path, '*'), policy='now')
        wandb.summary['model_save_path'] = save_path
        wandb.summary['config_save_path'] = config_path

        # compute metrics
        sim_clean = metrics.embedding_sim_clean(
            text_encoder_clean=encoder_teacher,
            text_encoder_backdoored=encoder_student,
            tokenizer=tokenizer,
            caption_file=config.evaluation['caption_file'],
            batch_size=config.evaluation['batch_size'])

        sim_attribute_backdoor = 0.0
        for backdoor in config.backdoors:
            sim_attribute_backdoor += metrics.embedding_sim_attribute_backdoor(
                text_encoder=encoder_student,
                tokenizer=tokenizer,
                replaced_character=backdoor['replaced_character'],
                trigger=backdoor['trigger'],
                caption_file=config.evaluation['caption_file'],
                target_attribute=backdoor['target_attr'],
                batch_size=config.evaluation['batch_size'])

        sim_attribute_backdoor /= len(config.backdoors)

        # log metrics
        if config.wandb['enable_logging']:

            wandb_run.summary['sim_clean'] = sim_clean
            wandb_run.summary['num_clean_samples'] = num_clean_samples
            wandb_run.summary[
                'num_backdoored_samples'] = num_backdoored_samples
            wandb_run.summary[
                'sim_attribute_backdoor'] = sim_attribute_backdoor

            # Generate and log final images
            if config.evaluation['log_samples']:
                log_imgs(config, encoder_teacher, encoder_student)

            # finish logging
            wandb.finish()


def create_parser():
    parser = argparse.ArgumentParser(description='Integrating backdoor')
    parser.add_argument('-c',
                        '--config',
                        default=None,
                        type=str,
                        dest="config",
                        help='Config .json file path (default: None)')
    args = parser.parse_args()
    config = ConfigParser(args.config)
    return config, args.config


def log_imgs(config, encoder_teacher, encoder_student):
    torch.cuda.empty_cache()
    prompts_clean = config.evaluation['prompts']

    imgs_clean_teacher = generate(prompts_clean,
                                  config.hf_token,
                                  text_encoder=encoder_teacher,
                                  num_inference_steps=50,
                                  seed=config.seed)
    imgs_clean_student = generate(prompts_clean,
                                  config.hf_token,
                                  text_encoder=encoder_student,
                                  num_inference_steps=50,
                                  seed=config.seed)
    img_dict = {
        'Samples_Teacher_Clean':
        [wandb.Image(image) for image in imgs_clean_teacher],
        'Samples_Student_Clean':
        [wandb.Image(image) for image in imgs_clean_student]
    }

    wandb.log(img_dict, commit=False)


def inject_attribute_backdoor(target_attr: str, replaced_character: str,
                              prompt: str, trigger: str) -> tuple([str, str]):
    # find indices of character to replace and select one at random
    idx_replace = [
        index for index, character in enumerate(prompt)
        if character == replaced_character
    ]

    if len(idx_replace) == 0:
        raise ValueError(
            f'Character \"{replaced_character}\" not present in prompt \"{prompt}\".'
        )

    idx_replace = random.choice(idx_replace)

    # find indices of word containing the replace character
    space_indices = [
        index for index, character in enumerate(prompt) if character == ' '
    ]
    pos_com = [pos < idx_replace for pos in space_indices]
    try:
        idx_replace = pos_com.index(False)
    except ValueError:
        idx_replace = -1

    # create target prompt with target attribute
    if idx_replace > 0:
        prompt_poisoned = prompt[:space_indices[
            idx_replace -
            1]] + ' ' + trigger + prompt[space_indices[idx_replace]:]
    elif idx_replace == 0:
        prompt_poisoned = trigger + prompt[space_indices[idx_replace]:]
    else:
        prompt_poisoned = prompt[:space_indices[idx_replace]] + ' ' + trigger

    # create target prompt with target attribute
    if idx_replace > 0:
        prompt_replaced = prompt[:space_indices[
            idx_replace -
            1]] + ' ' + target_attr + prompt[space_indices[idx_replace]:]
    elif idx_replace == 0:
        prompt_replaced = target_attr + prompt[space_indices[idx_replace]:]
    else:
        prompt_replaced = prompt[:space_indices[idx_replace]] + ' ' + target_attr

    return (prompt_poisoned, prompt_replaced)


if __name__ == '__main__':
    main()

import torch
from utils.attack_utils import inject_attribute_backdoor
from utils.encoder_utils import compute_text_embeddings
from torch.nn.functional import cosine_similarity
from torchmetrics.functional import pairwise_cosine_similarity


def z_score_text(text_encoder: torch.nn.Module,
                 tokenizer: torch.nn.Module,
                 replaced_character: str,
                 trigger: str,
                 caption_file: str,
                 batch_size: int = 256,
                 num_triggers: int = None) -> float:
    # read in text prompts
    with open(caption_file, 'r', encoding='utf-8') as file:
        lines = file.readlines()
        captions_clean = [line.strip() for line in lines]

    if num_triggers:
        captions_backdoored = [
            caption.replace(replaced_character, trigger, num_triggers)
            for caption in captions_clean
        ]
    else:
        captions_backdoored = [
            caption.replace(replaced_character, trigger)
            for caption in captions_clean
        ]

    # compute embeddings on clean inputs
    emb_clean = compute_text_embeddings(tokenizer, text_encoder,
                                        captions_clean, batch_size)

    # compute embeddings on backdoored inputs
    emb_backdoor = compute_text_embeddings(tokenizer, text_encoder,
                                           captions_backdoored, batch_size)

    # compute cosine similarities
    emb_clean = torch.flatten(emb_clean, start_dim=1)
    emb_backdoor = torch.flatten(emb_backdoor, start_dim=1)
    sim_clean = pairwise_cosine_similarity(emb_clean, emb_clean)
    sim_backdoor = pairwise_cosine_similarity(emb_backdoor, emb_backdoor)

    # take lower triangular matrix without diagonal elements
    num_captions = len(captions_clean)
    sim_clean = sim_clean[
        torch.tril_indices(num_captions, num_captions, offset=-1)[0],
        torch.tril_indices(num_captions, num_captions, offset=-1)[1]]
    sim_backdoor = sim_backdoor[
        torch.tril_indices(num_captions, num_captions, offset=-1)[0],
        torch.tril_indices(num_captions, num_captions, offset=-1)[1]]

    # compute z-score
    mu_clean = sim_clean.mean()
    mu_backdoor = sim_backdoor.mean()
    var_clean = sim_clean.var(unbiased=True)
    z_score = (mu_backdoor - mu_clean) / var_clean
    z_score = z_score.cpu().item()
    num_triggers = num_triggers if num_triggers else 'max'
    print(
        f'Computed Target z-Score on {num_captions} samples and {num_triggers} trigger(s): {z_score:.4f}'
    )

    return z_score


def embedding_sim_backdoor(text_encoder: torch.nn.Module,
                           tokenizer: torch.nn.Module,
                           replaced_character: str,
                           trigger: str,
                           caption_file: str,
                           target_caption: str,
                           batch_size: int = 256,
                           num_triggers: int = None) -> float:
    # read in text prompts and create backdoored captions
    with open(caption_file, 'r', encoding='utf-8') as file:
        lines = file.readlines()
        captions_clean = [line.strip() for line in lines]

    if num_triggers:
        captions_backdoored = [
            caption.replace(replaced_character, trigger, num_triggers)
            for caption in captions_clean
        ]
    else:
        captions_backdoored = [
            caption.replace(replaced_character, trigger)
            for caption in captions_clean
        ]

    # compute embeddings on target prompt
    emb_target = compute_text_embeddings(tokenizer, text_encoder,
                                         [target_caption], batch_size)

    # compute embeddings on backdoored inputs
    emb_backdoor = compute_text_embeddings(tokenizer, text_encoder,
                                           captions_backdoored, batch_size)

    # compute cosine similarities
    emb_target = torch.flatten(emb_target, start_dim=1)
    emb_backdoor = torch.flatten(emb_backdoor, start_dim=1)
    similarity = pairwise_cosine_similarity(emb_backdoor, emb_target)

    mean_sim = similarity.mean().cpu().item()

    num_triggers = num_triggers if num_triggers else 'max'
    print(
        f'Computed Target Similarity Score on {len(captions_backdoored)} samples and {num_triggers} trigger(s): {mean_sim:.4f}'
    )

    return mean_sim


def embedding_sim_attribute_backdoor(text_encoder: torch.nn.Module,
                                     tokenizer: torch.nn.Module,
                                     replaced_character: str,
                                     trigger: str,
                                     caption_file: str,
                                     target_attribute: str,
                                     batch_size: int = 256) -> float:
    # read in text prompts and create backdoored captions
    with open(caption_file, 'r', encoding='utf-8') as file:
        lines = file.readlines()
        captions_clean = [line.strip() for line in lines]
        captions_backdoored = [
            caption.replace(replaced_character, trigger)
            for caption in captions_clean
        ]
        target_captions = [
            inject_attribute_backdoor(target_attribute, replaced_character,
                                      prompt, trigger)
            for prompt in captions_clean
        ]
    # compute embeddings on target prompt
    emb_target = compute_text_embeddings(tokenizer, text_encoder,
                                         target_captions, batch_size)

    # compute embeddings on backdoored inputs
    emb_backdoor = compute_text_embeddings(tokenizer, text_encoder,
                                           captions_backdoored, batch_size)

    # compute cosine similarities
    emb_target = torch.flatten(emb_target, start_dim=1)
    emb_backdoor = torch.flatten(emb_backdoor, start_dim=1)
    similarity = pairwise_cosine_similarity(emb_backdoor, emb_target)

    mean_sim = similarity.mean().cpu().item()

    print(
        f'Computed Target Similarity Score on {len(captions_backdoored)} samples and {1} trigger: {mean_sim:.4f}'
    )

    return mean_sim


def embedding_sim_clean(text_encoder_clean: torch.nn.Module,
                        text_encoder_backdoored: torch.nn.Module,
                        tokenizer: torch.nn.Module,
                        caption_file: str,
                        batch_size: int = 256) -> float:
    # read in text prompts and create backdoored captions
    with open(caption_file, 'r', encoding='utf-8') as file:
        lines = file.readlines()
        captions_clean = [line.strip() for line in lines]

    # compute embeddings on target prompt
    emb_clean = compute_text_embeddings(tokenizer, text_encoder_clean,
                                        captions_clean, batch_size)

    # compute embeddings on backdoored inputs
    emb_backdoor = compute_text_embeddings(tokenizer, text_encoder_backdoored,
                                           captions_clean, batch_size)

    # compute cosine similarities
    emb_clean = torch.flatten(emb_clean, start_dim=1)
    emb_backdoor = torch.flatten(emb_backdoor, start_dim=1)
    similarity = cosine_similarity(emb_clean, emb_backdoor, dim=1)

    mean_sim = similarity.mean().cpu().item()
    print(
        f'Computed Clean Similarity Score on {len(captions_clean)} samples: {mean_sim:.4f}'
    )

    return mean_sim

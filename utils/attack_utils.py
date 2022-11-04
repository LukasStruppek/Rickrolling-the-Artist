import random


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

    # create poisoned prompt with trigger
    prompt_poisoned = prompt[:idx_replace] + trigger + prompt[idx_replace + 1:]
    space_indices = [
        index for index, character in enumerate(prompt) if character == ' '
    ]

    # find indices of word containing the replace character
    pos_com = [pos < idx_replace for pos in space_indices]
    try:
        idx_replace = pos_com.index(False)
    except ValueError:
        idx_replace = -1

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

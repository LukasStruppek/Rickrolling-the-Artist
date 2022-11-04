import math
from typing import List

import torch


def compute_text_embeddings(tokenizer: torch.nn.Module,
                            encoder: torch.nn.Module,
                            prompts: List[str],
                            batch_size: int = 256) -> torch.Tensor:
    with torch.no_grad():
        encoder.eval()
        encoder.cuda()

        embedding_list = []
        for i in range(math.ceil(len(prompts) / batch_size)):
            batch = prompts[i * batch_size:(i + 1) * batch_size]
            tokens = tokenizer(batch,
                               padding="max_length",
                               max_length=tokenizer.model_max_length,
                               truncation=True,
                               return_tensors="pt")
            embedding = encoder(tokens.input_ids.cuda())[0]
            embedding_list.append(embedding.cpu())
        embeddings = torch.cat(embedding_list, dim=0)
        return embeddings

import torch
import logging
from typing import List, Union
from datasets import load_dataset, load_from_disk


def get_calib_dataset(
    data: Union[str, List[str], List[List[int]]] = "pileval",
    tokenizer=None,
    n_samples=512,
    block_size=512,
    split="train",
    text_column="text",
):
    if isinstance(data, str):
        if data == "pileval":
            # dataset = load_dataset("mit-han-lab/pile-val-backup", split="validation")
            dataset = load_from_disk("/media/hjt/影驰/Linux/dataset/pile-val-backup")
        else:
            dataset = load_dataset(data, split=split)

        dataset = dataset.shuffle(seed=42)

    elif isinstance(data, list):
        if isinstance(data[0], str):
            dataset = [{text_column: text} for text in data]
        elif isinstance(data[0][0], int):
            dataset = data
        else:
            raise NotImplementedError(
                "Either pass a string to a huggingface dataset or a list"
                "that is preprocessed with one sample of text per element"
                " or a list of list of int for tokenized words."
            )
    else:
        raise NotImplementedError(
            "Either pass a string to a huggingface dataset or a list"
            "that is preprocessed with one sample of text per element"
            " or a list of list of int for tokenized words."
        )

    samples = []
    n_run = 0
    for data in dataset:
        if isinstance(data, list):
            line_encoded = data
        else:
            if data == "pileval":
                set_name = data['meta']['pile_set_name']
            line = data[text_column]
            line = line.strip()
            line_encoded = tokenizer.encode(line)
        # if data == "pileval":
        #     if set_name != 'Pile-CC':
        #         continue
        if len(line_encoded) > block_size or len(line_encoded) == 0:  # 512
            continue
        # 填充至block_size
        line_encoded.extend([151643] * (block_size - len(line_encoded)))
        sample = torch.tensor([line_encoded])
        # if sample.numel() == 0:
        #     continue
        assert sample.numel() == block_size
        samples.append(sample)
        n_run += 1
        if n_run == n_samples:
            break
    # now concatenate all samples and split according to block size
    cat_samples = torch.cat(samples, dim=1)
    n_split = cat_samples.shape[1] // block_size
    logging.debug(f" * Split into {n_split} blocks")
    return [
        cat_samples[:, i * block_size : (i + 1) * block_size] for i in range(n_split)
    ]

import torch
import logging
from typing import List, Union
from datasets import load_dataset,load_from_disk
import random

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
        elif data == "alpaca":
            print("正在加载alpaca数据")
            dataset = []
            dataset_zh = load_from_disk("/media/hjt/影驰/Linux/dataset/alpaca-gpt4-data-zh/train")
            for data in dataset_zh:
                dataset.append(data)
            del dataset_zh
            dataset_en = load_from_disk("/media/hjt/影驰/Linux/dataset/alpaca-gpt4/train")
            for data in dataset_en:
                dataset.append(data)
            del dataset_en
            random.seed(42)
            random.shuffle(dataset)
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
        if list(data.keys())[0] == "instruction":
            print("执行代码错误请终止运行")
            messages = [
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": data['instruction'] + data['input']},
                {"role": "assistant", "content": data['output']}
            ]
            line = tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=False
                ).strip()
            line_encoded = tokenizer.encode(line)
        elif isinstance(data, list):
            line_encoded = data
        else:
            line = data[text_column]
            line = line.strip()
            line_encoded = tokenizer.encode(line)
        if len(line_encoded) > 512:
            continue
        sample = torch.tensor([line_encoded])
        if sample.numel() == 0:
            continue
        samples.append(sample)
        n_run += 1
        if n_run == n_samples:
            break
    # now concatenate all samples and split according to block size
    cat_samples = torch.cat(samples, dim=1)
    n_split = cat_samples.shape[1] // block_size
    print("n_split",n_split)
    logging.debug(f" * Split into {n_split} blocks")
    return [
        cat_samples[:, i * block_size : (i + 1) * block_size] for i in range(n_split)
    ]

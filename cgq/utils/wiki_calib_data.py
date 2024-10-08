import random

import torch
import logging
from typing import List, Union
from datasets import Dataset, load_dataset, load_from_disk

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
            # dataset = load_from_disk("/media/hjt/影驰/Linux/dataset/wikitext")
            dataset = load_from_disk("/media/hjt/影驰/Linux/dataset/pile-val-backup")
            # dataset = load_from_disk("/media/hjt/影驰/Linux/dataset/wikipedia-cn/train")

        else:
            dataset = load_dataset(data, split=split)

        dataset = dataset.shuffle(seed=42)   #42

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
    #print("dataset:",dataset)
    n_count = {}
    count_max = {
        'NIH ExPorter': 4,
        'PubMed Abstracts': 10,
        'Pile-CC': 35,
        'Wikipedia (en)': 15,
        'Github': 24,
        'StackExchange': 15,
        'OpenWebText2': 18,
        'USPTO Backgrounds': 6,
        'EuroParl': 1,
        'HackerNews': 0,
        'Enron Emails': 0,
        'FreeLaw': 0,
        'PubMed Central': 0,
        'YoutubeSubtitles': 0,
        'ArXiv': 0,
        'OpenSubtitles': 0,
        'DM Mathematics': 0,
        }
    samples = []
    n_run = 0
    for data in dataset:
        # print("data:",data)
        if isinstance(data, list):
            line_encoded = data
        else:
            set_name = data['meta']['pile_set_name']
            line = data[text_column]
            line = line.strip()
            line_encoded = tokenizer.encode(line)
            # print('line_encoded',line_encoded)

        # if count_max[set_name] == 0:
        #     continue
        # if set_name in n_count and n_count[set_name] >= count_max[set_name]:
        #     continue

        if len(line_encoded) > 512:     #512->2048
            continue


        # print("data[mate]:",data['meta'])
        sample = torch.tensor([line_encoded])
        if sample.numel() == 0:
            continue
        # if (set_name!='Wikipedia (en)'):   #'Github','Wikipedia (en)','Pile-CC','PubMed Abstracts','OpenWebText2'，'StackExchange'
        #     continue
        # print(data[text_column])
        # print(set_name)
        # print("-----------------------")

        samples.append(sample)

        # if set_name in n_count:
        #     n_count[set_name] += 1
        # else:
        #     n_count[set_name] = 1

        n_run += 1
        if n_run == n_samples:
            print("数据已准备完成")
            break

    # print("数据分布",n_count)
    '''
    dataset = load_from_disk("/media/hjt/影驰/Linux/dataset/wikipedia-cn/train")
    dataset = dataset.shuffle(seed=42)
    for data in dataset:
        if isinstance(data, list):
            line_encoded = data
        else:
            line = data['completion']
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
            print("数据已准备完成")
            break
    random.seed(42)
    random.shuffle(samples)
    '''
    # now concatenate all samples and split according to block size
    cat_samples = torch.cat(samples, dim=1)
    n_split = cat_samples.shape[1] // block_size
    logging.debug(f" * Split into {n_split} blocks")
    return [
        cat_samples[:, i * block_size : (i + 1) * block_size] for i in range(n_split)
    ]

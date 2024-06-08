import os
from tqdm import tqdm
import numpy as np
import torch
from datasets import load_dataset
from torch.utils.data import Dataset
import random

"""
doc https://huggingface.co/docs/datasets/loading
doc https://huggingface.co/docs/datasets/process
"""

# for the calibration of channel distribution of kv cache
def get_dataset_for_calib(dataset, tokenizer, split='train', seqlen=1024, limit=100):
    if dataset == 'wikitext2':
        data = load_dataset(
            "wikitext",
            "wikitext-2-raw-v1",
            split=split,
        )
        text=' '.join(data['text']).replace('=', '')
    elif dataset == 'pile_subset':
        data = load_dataset(
            "ola13/small-the_pile",
            split='train',
        )
        text=' '.join(data['text'])      
    import re
    def split_into_samples(text, max_length=1024):
        sentences = re.split(r'[.?!]', text)
        if len(sentences) > 1000000:
            sentences = sentences[:1000000]
        samples = []
        current_sample = ''
        
        for i in tqdm(range(len(sentences))):
            if len(samples) > limit:
                break
            sentence = sentences[i]
            potential_sample = current_sample + sentence
            if len(potential_sample) <= max_length:
                current_sample = potential_sample
            else:
                samples.append(current_sample.strip())
                current_sample = sentence
        
        if current_sample:
            samples.append(current_sample.strip())
        
        return samples
    samples = split_into_samples(text, max_length=seqlen*10)
    tokenizer.pad_token=tokenizer.eos_token
    encoded_samples = [tokenizer.encode(sample, max_length=seqlen, pad_to_max_length=False, return_tensors='pt') for sample in samples]
    class TextDataset(Dataset):
        def __init__(self, encodings):
            self.encodings = encodings
        
        def __len__(self):
            return len(self.encodings)
        
        def __getitem__(self, idx):
            encoded=self.encodings[idx].view(-1)
            # input_ids = encoded['input_ids'].squeeze()
            # encoded['input_ids'] = input_ids
            return encoded

    dataset = TextDataset(encoded_samples)
    return dataset


def get_dataset_for_trainer(dataset, tokenizer, split='train', seqlen=1024):
    if dataset == 'wikitext2':
        data = load_dataset(
            "wikitext",
            "wikitext-2-raw-v1",
            split=split,
        )
        text=' '.join(data['text']).replace('=', '')
    elif dataset == 'pile_subset':
        data = load_dataset(
            "ola13/small-the_pile",
            split='train',
        )
        text=' '.join(data['text'])      
    import re
    def split_into_samples(text, max_length=1024):
        sentences = re.split(r'[.?!]', text)
        if len(sentences) > 2000000:
            sentences = sentences[:2000000]
        samples = []
        current_sample = ''
        
        for i in tqdm(range(len(sentences))):
            sentence = sentences[i]
            potential_sample = current_sample + sentence
            if len(potential_sample) <= max_length:
                current_sample = potential_sample
            else:
                samples.append(current_sample.strip())
                current_sample = sentence
        
        if current_sample:
            samples.append(current_sample.strip())
        
        return samples
    samples = split_into_samples(text, max_length=seqlen*10)
    tokenizer.pad_token=tokenizer.eos_token
    encoded_samples = [tokenizer.encode(sample, max_length=seqlen, pad_to_max_length=False, return_tensors='pt') for sample in samples]
    class TextDataset(Dataset):
        def __init__(self, encodings):
            self.encodings = encodings
        
        def __len__(self):
            return len(self.encodings)
        
        def __getitem__(self, idx):
            encoded=self.encodings[idx].view(-1)
            # input_ids = encoded['input_ids'].squeeze()
            # encoded['input_ids'] = input_ids
            return encoded

    dataset = TextDataset(encoded_samples)
    return dataset
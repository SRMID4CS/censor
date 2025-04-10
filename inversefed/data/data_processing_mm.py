# -*- coding: utf-8 -*-
"""
Created on Thu Apr 10 22:23:59 2025

@author: lakith
"""
import torch

from transformers import BertTokenizerFast

from torch.utils.data import Dataset

from torch.nn.utils.rnn import pad_sequence


def collate_fn(batch):
    tokenizer = BertTokenizerFast.from_pretrained("bert-base-uncased")

    images = torch.stack([item['image'] for item in batch])
    labels = torch.stack([item['label'] for item in batch])

    input_ids = [item['input_ids'] for item in batch]
    attention_masks = [item['attention_mask'] for item in batch]

    input_ids_padded = pad_sequence(input_ids, batch_first=True, padding_value=tokenizer.pad_token_id)
    attention_mask_padded = pad_sequence(attention_masks, batch_first=True, padding_value=0)

    return {
        'image': images,
        'input_ids': input_ids_padded,
        'attention_mask': attention_mask_padded,
        'label': labels
    }

class MMIMDbDataset(Dataset):
    def __init__(self, data):
        self.data = data  # list of dicts

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sample = self.data[idx]
        return {
            "image": sample["image"],  # Tensor [3, 64, 64]
            "input_ids": sample["input_ids"],
            "attention_mask": sample["attention_mask"],
            "label": torch.tensor(sample["label"], dtype=torch.float),
        }


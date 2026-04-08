import json
import random

import torch
from torch.utils.data import Dataset


class ReviewDataset(Dataset):
    def __init__(self, file_path, tokenizer=None, max_len=128, sample_fraction=1.0, text_preprocessing=lambda x: x):
        self.file_path = file_path
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.sample_fraction = sample_fraction
        self.text_preprocessing = text_preprocessing

        # Store file offsets for lazy loading
        self.offsets = []
        with open(file_path, "r") as f:
            offset = 0
            for line in f:
                if random.random() < sample_fraction:
                    self.offsets.append(offset)
                offset += len(line)

    def set_tokenizer(self, tokenizer):
        self.tokenizer = tokenizer

    def set_max_length(self, max_len):
        self.max_len = max_len

    def set_text_preprocessing(self, txt_fn):
        self.text_preprocessing = txt_fn

    def __len__(self):
        return len(self.offsets)

    def __getitem__(self, idx):
        with open(self.file_path, "r") as f:
            f.seek(self.offsets[idx])
            line = f.readline()
            sample = json.loads(line)

        text = sample["text"]
        label = sample["label"]

        if self.text_preprocessing is not None:
            text = self.text_preprocessing(text)

        if self.tokenizer:
            encoded = self.tokenizer(
                text,
                truncation=True,
                padding="max_length",
                max_length=self.max_len,
                return_tensors="pt"
            )
            return {k: v.squeeze(0) for k, v in encoded.items()}, torch.tensor(label)

        return text, torch.tensor(label)
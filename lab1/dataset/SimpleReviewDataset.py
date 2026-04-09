import json

import torch
from torch.utils.data import Dataset


class SimpleReviewDataset(Dataset):
    def __init__(self, file_path):
        self.file_path = file_path

        # Store file offsets for lazy loading
        self.offsets = []
        
        with open(file_path, "rb") as f:
            offset = 0
            for line in f:
                if not line.strip():
                    offset += len(line)
                    continue
                self.offsets.append(offset)
                offset += len(line)

    def __len__(self):
        return len(self.offsets)

    def __getitem__(self, idx):
        with open(self.file_path, "r", encoding="utf-8") as f:
            f.seek(self.offsets[idx])
            line = f.readline().strip()
            sample = json.loads(line)

        text = sample["text"]
        label = sample["label"]

        return text, torch.tensor(label)
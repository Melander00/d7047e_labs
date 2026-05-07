import os
import torch
from torch.utils.data import Dataset
from PIL import Image
from torchvision import transforms
from collections import Counter


class Vocabulary:
    def __init__(self, freq_threshold=5):
        self.freq_threshold = freq_threshold

        self.itos = {
            0: "<PAD>",
            1: "<SOS>",
            2: "<EOS>",
            3: "<UNK>"
        }

        self.stoi = {
            "<PAD>": 0,
            "<SOS>": 1,
            "<EOS>": 2,
            "<UNK>": 3
        }

    def __len__(self):
        return len(self.itos)

    @staticmethod
    def tokenizer(text):
        return text.lower().split()

    def build_vocabulary(self, sentence_list):
        frequencies = Counter()
        idx = 4

        for sentence in sentence_list:
            for word in self.tokenizer(sentence):
                frequencies[word] += 1

                if frequencies[word] == self.freq_threshold:
                    self.stoi[word] = idx
                    self.itos[idx] = word
                    idx += 1

    def numericalize(self, text):
        tokenized_text = self.tokenizer(text)

        return [
            self.stoi[token] if token in self.stoi else self.stoi["<UNK>"]
            for token in tokenized_text
        ]


class FlickrDataset(Dataset):
    def __init__(self, root_dir, transform=None, freq_threshold=5):
        self.root_dir = root_dir
        captions_file = os.path.join(root_dir, "captions.txt")
        self.transform = transform

        self.images = []
        self.captions = []

        with open(captions_file, 'r', encoding='utf-8') as f:
            lines = f.readlines()

        for line in lines:
            image_name, caption = line.strip().split(',', 1)
            if image_name == "image":
                continue
            self.images.append(image_name)
            self.captions.append(caption)

        self.vocab = Vocabulary(freq_threshold)
        self.vocab.build_vocabulary(self.captions)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        caption = self.captions[idx]
        image_name = self.images[idx]
        img_path = os.path.join(self.root_dir, "Images", image_name)

        image = Image.open(img_path).convert("RGB")

        if self.transform:
            image = self.transform(image)

        numericalized_caption = [self.vocab.stoi["<SOS>"]]
        numericalized_caption += self.vocab.numericalize(caption)
        numericalized_caption.append(self.vocab.stoi["<EOS>"])

        return image, torch.tensor(numericalized_caption)


class MyCollate:
    def __init__(self, pad_idx):
        self.pad_idx = pad_idx

    def __call__(self, batch):
        images = [item[0].unsqueeze(0) for item in batch]
        images = torch.cat(images, dim=0)

        captions = [item[1] for item in batch]
        captions = torch.nn.utils.rnn.pad_sequence(
            captions,
            batch_first=False,
            padding_value=self.pad_idx
        )

        return images, captions


def get_loaders(data_folder, batch_size=32):
    """
    data_loader contains the entire flickr8k structure. Including Images folder and captions.txt
    """


    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ])

    dataset = FlickrDataset(
        data_folder,
        transform=transform
    )

    generator = torch.Generator().manual_seed(1)
    subsets = torch.utils.data.random_split(dataset, [0.7,0.15,0.15], generator=generator)

    pad_idx = dataset.vocab.stoi["<PAD>"]

    train_loader = torch.utils.data.DataLoader(
        dataset=subsets[0],
        batch_size=batch_size,
        shuffle=True,
        collate_fn=MyCollate(pad_idx=pad_idx)
    )
    val_loader = torch.utils.data.DataLoader(
        dataset=subsets[1],
        batch_size=batch_size,
        shuffle=False,
        collate_fn=MyCollate(pad_idx=pad_idx)
    )
    test_loader = torch.utils.data.DataLoader(
        dataset=subsets[2],
        batch_size=batch_size,
        shuffle=False,
        collate_fn=MyCollate(pad_idx=pad_idx)
    )

    return (train_loader, val_loader, test_loader), dataset
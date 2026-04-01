# Guidelines from preprocessing

## Download dataset

We will use the Yelp Review public dataset hosted on [their website](https://business.yelp.com/data/resources/open-dataset/). Follow these instructions to download and prepare the dataset for usage:

NOTE! The dataset is around 5 GB uncompressed json and ?? GB as tensors.

1. Download the dataset at [https://business.yelp.com/data/resources/open-dataset/](https://business.yelp.com/data/resources/open-dataset/). Click the download JSON button.
2. Extract the json files from `yelp_dataset.tar` inside the downloaded file.
3. Copy the `yelp_academic_dataset_review.json` file into `/lab1/data/yelp_raw`.
4. The first time you run the models with this dataset it will preprocess the json into tensors and save to disk.

Example entry in the review.json file:

```json
{
    // string, 22 character unique review id
    "review_id": "zdSx_SD6obEhz9VrW9uAWA",

    // string, 22 character unique user id, maps to the user in user.json
    "user_id": "Ha3iJu77CxlrFm-vQRs_8g",

    // string, 22 character business id, maps to business in business.json
    "business_id": "tnhfDv5Il8EaGSXZGiuQGg",

    // integer, star rating
    "stars": 4,

    // string, date formatted YYYY-MM-DD
    "date": "2016-03-09",

    // string, the review itself
    "text": "Great place to hang out after work: the prices are decent, and the ambience is fun. It's a
bit loud, but very lively. The staff is friendly, and the food is good. They have a good selection of
drinks.",

    // integer, number of useful votes received
    "useful": 0,

    // integer, number of funny votes received
    "funny": 0,

    // integer, number of cool votes received
    "cool": 0
}
```

We will do sentiment analysis based on `stars` and `text`. It will be 5 different labels, one for each star level.

## Use dataset

### Overview

The main function is `prepare_yelp_loaders` in `dataset/loaders.py`. It is designed to be used by both the ANN and the Transformer models. It generates consistent data loaders with deterministic split to be used in the train loop.

Each model may need different processing steps depending on needs. Below are instructions how to utilize the loader function.

### 1. Import the loader

```python
from dataset.loaders import prepare_yelp_loaders
```

---

### 2. Parameters

| Parameter            | Type               | Description                                                                                                  |
| -------------------- | ------------------ | ------------------------------------------------------------------------------------------------------------ |
| `batch_size`         | int                | Number of samples per batch                                                                                  |
| `entries`            | int                | Number of lines to read from dataset (`-1` = all)                                                            |
| `sample_fraction`    | float              | Fraction of lines to randomly sample (0–1)                                                                   |
| `splits`             | list[float]        | Train/validation/test split ratios, e.g., `[0.7,0.15,0.15]`                                                  |
| `tokenizer`          | callable or `None` | Preprocessing/tokenization function for your model.                                                          |
| `max_len`            | int                | Maximum sequence length (padding/truncation). For fair comparison don't change this value from default (128) |
| `text_preprocessing` | callable or `None` | Optional text preprocessing function (lowercase, remove punctuation)                                         |

### 3. Precompute

As long as you followed the Download Dataset instructions at the top, the first call (or any subsequent with different `entries` value) will generate a precomputed json file containing text and labels. This allows for lazy-loading which speeds up training and repeated runs.

### 4. Tokenizer

This is the dataset `__getitem__` function

```python
def __getitem__(self, idx):
    with open(self.file_path, "r") as f:
        f.seek(self.offsets[idx])
        line = f.readline()
        sample = json.loads(line)

    text = sample["text"]
    label = sample["label"]

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
```

Make sure the tokenizer you define matches this. It is required to allow a fair comparison between models by utilizing the same dataset and same training methods.

#### Transformers

Transformers may use predefined tokenizers such as HuggingFace. Check related documentation for instructions.

#### ANN

ANN needs to build a custom tokenizer. It will need to use its own vocab as well as returning a tensor in a format that the model then can utilize.

<!--


Here’s a clean, detailed **GUIDELINES.md** you can give to the other teams. It explains exactly how to use your `prepare_yelp_loaders`, the tokenizer, `max_len`, and preprocessing.

---

# 📝 Yelp Dataset Guidelines

## Overview

The `prepare_yelp_loaders` function and `YelpDataset` class are designed to **support both ANN and Transformer models** with a shared dataset and consistent train/val/test splits.

Teams should use this function to load data while respecting **model-specific preprocessing**.

---

## 1. Import the loader

```python
from dataset.loaders import prepare_yelp_loaders
```

---

## 2. Parameters

---

## 3. Using Transformers

Transformers require a tokenizer (Hugging Face or equivalent).

```python
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

train_loader, val_loader, test_loader = prepare_yelp_loaders(
    batch_size=16,
    tokenizer=tokenizer,
    max_len=128
)
```

- `text_preprocessing` can remain the default (`lambda x: x`) because Transformers handle tokenization internally.
- The dataset returns a dictionary:

```python
{
    "input_ids": torch.tensor([...]),
    "attention_mask": torch.tensor([...]),
    "labels": torch.tensor(label)
}
```

- `DataLoader` batches will have shape `(batch_size, max_len)` for `input_ids`.

---

## 4. Using ANN

ANNs require a **custom tokenizer** and vocabulary.

### Step 1: Build a vocabulary

```python
def build_vocab(texts, min_freq=1):
    from collections import Counter
    counter = Counter()
    for text in texts:
        for word in text.lower().split():
            counter[word] += 1
    vocab = {"<PAD>": 0, "<UNK>": 1}
    idx = 2
    for word, freq in counter.items():
        if freq >= min_freq:
            vocab[word] = idx
            idx += 1
    return vocab
```

### Step 2: Create preprocessing function

```python
def ann_tokenizer(text, vocab, max_len=128):
    tokens = [vocab.get(w, vocab["<UNK>"]) for w in text.lower().split()]
    tokens = tokens[:max_len] + [vocab["<PAD>"]] * max(0, max_len - len(tokens))
    return torch.tensor(tokens, dtype=torch.long)
```

### Step 3: Load dataset

```python
train_loader, val_loader, test_loader = prepare_yelp_loaders(
    batch_size=64,
    tokenizer=None,  # ANN does not use HF tokenizer
    max_len=100,
    text_preprocessing=lambda x: ann_tokenizer(x, vocab, max_len=100)
)
```

- Each sample returned is a tuple `(input_tensor, label_tensor)`
- The `input_tensor` can be fed directly to the embedding layer.

---

## 5. Notes / Best Practices

- **Keep `max_len` consistent** across ANN and Transformer for fair comparison.
- **Labels are always `torch.tensor`**, compatible with loss functions.
- Transformers automatically handle padding and attention masks; ANN preprocessing must include padding.
- Use the `text_preprocessing` argument to apply **model-specific preprocessing** (lowercasing, punctuation removal, etc.)
- DataLoader batching is lazy, so large datasets (~1GB) are supported.

---

This ensures **all teams use the same dataset, same splits**, and only differ in preprocessing/tokenization logic.

---

If you want, I can also add a **diagram showing ANN vs Transformer flow using `prepare_yelp_loaders`**, which helps visually explain the workflow to other teams.

Do you want me to do that? -->

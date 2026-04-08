import torch
from dataset.amazon_preprocessing import get_amazon_file_path
from dataset.ReviewsDataset import ReviewDataset
from dataset.SimpleReviewDataset import SimpleReviewDataset
from dataset.yelp_preprocessing import get_yelp_file_path
from torch.utils.data import DataLoader, Dataset, Subset


def load_yelp_simple(entries = -1):
    fpath = get_yelp_file_path(entries=entries)
    return SimpleReviewDataset(fpath)

def load_yelp_dataset(entries = -1, sample_fraction = 1.0):
    """
    `entries`: how many lines to read from the json.

    `sample_fraction`: value in [0,1] for probability if an entry is used.
    Used to reduce the sample size without recomputing.
    """
    fpath = get_yelp_file_path(entries=entries)
    dataset = ReviewDataset(fpath, sample_fraction=sample_fraction)

    return dataset

def split_dataset(dataset: Dataset, splits=[0.7,0.15,0.15]):
    generator = torch.Generator().manual_seed(1)
    subsets = torch.utils.data.random_split(dataset, splits, generator=generator)

    return subsets

def prepare_loaders(subsets: list[Subset], batch_size: int):
    train_loader = DataLoader(subsets[0], batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(subsets[1], batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)
    test_loader = DataLoader(subsets[2], batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)

    return train_loader, val_loader, test_loader

def prepare_yelp_loaders(
    batch_size = 32,
    entries = -1,
    sample_fraction = 1.0,
    splits=[0.7,0.15,0.15],
    tokenizer = None,
    max_len = 128,
    text_preprocessing = None
):
    """
    `batch_size`: batch size
    
    `entries`: how many lines to read from the json doc.

    `sample_fraction`: Reduce sample size during runtime (without recomputing json)

    `splits`: subset splits

    `tokenizer`: Tokenizer for the text processing during training. See GUIDELINES.md for more information.

    `max_len`: Max_len for the tokenizer. See GUIDELINES.md

    `text_preprocessing`: If there is some processing that needs to done outside the tokenizer (lowercase, remove punctuation, etc.).
    """

    dataset = load_yelp_dataset(entries=entries,sample_fraction=sample_fraction)
    dataset.set_tokenizer(tokenizer)
    dataset.set_max_length(max_len)
    dataset.set_text_preprocessing(text_preprocessing)

    subsets = split_dataset(dataset, splits=splits)
    loaders = prepare_loaders(subsets, batch_size=batch_size)

    return loaders, dataset


def load_amazon_simple(use_25k_set):
    fpath = get_amazon_file_path(use_25k_set=use_25k_set)
    return SimpleReviewDataset(fpath)

def load_amazon_dataset(
    use_25k_set,
    sample_fraction=1.0
):
    """
    `sample_fraction`: value in [0,1] for probability if an entry is used.
    Used to reduce the sample size without recomputing.
    """
    fpath = get_amazon_file_path(use_25k_set=use_25k_set)
    dataset = ReviewDataset(fpath, sample_fraction=sample_fraction)

    return dataset


def prepare_amazon_loaders(
    batch_size = 32,
    sample_fraction = 1.0,
    splits=[0.7,0.15,0.15],
    tokenizer = None,
    max_len = 128,
    text_preprocessing = None,
    use_25k_set = False,
):
    """
    `batch_size`: batch size

    `sample_fraction`: Reduce sample size during runtime (without recomputing json)

    `splits`: subset splits

    `tokenizer`: Tokenizer for the text processing during training. See GUIDELINES.md for more information.

    `max_len`: Max_len for the tokenizer. See GUIDELINES.md

    `text_preprocessing`: If there is some processing that needs to done outside the tokenizer (lowercase, remove punctuation, etc.).

    `use_25k_set`: Boolean to distinguish between the 1K or 25K dataset.
    """

    dataset = load_amazon_dataset(use_25k_set,sample_fraction=sample_fraction)
    dataset.set_tokenizer(tokenizer)
    dataset.set_max_length(max_len)
    dataset.set_text_preprocessing(text_preprocessing)

    subsets = split_dataset(dataset, splits=splits)
    loaders = prepare_loaders(subsets, batch_size=batch_size)

    return loaders, dataset
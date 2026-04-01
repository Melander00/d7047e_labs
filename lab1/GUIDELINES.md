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
    "text": "Great place to hang out after work: the prices are decent, and the ambience is fun. It's a bit loud, but very lively. The staff is  friendly, and the food is good. They have a good selection of drinks.",

    // integer, number of useful votes received
    "useful": 0,

    // integer, number of funny votes received
    "funny": 0,

    // integer, number of cool votes received
    "cool": 0
}
```

We will do sentiment analysis based on `stars` and `text`.

## Use dataset

### Overview

The main function is `prepare_yelp_loaders` in `dataset/loaders.py`. It is designed to be used by both the ANN and the Transformer models. It generates consistent data loaders with deterministic split to be used in the train loop.

The labels mapping are the following:

| **Stars** | **Label** | **Sentiment** |
| --------- | --------- | ------------- |
| 1, 2      | -1        | Bad           |
| 3         | 0         | Neutral       |
| 4,5       | 1         | Good          |

There may be some bias since we merge two "star-ratings" into one for two sentiments but there is only one for neutral.

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

## Training

### Overview

The training structure has been modified a bit. What's new is that we save metadata of the models (losses, accuracies, etc.) as JSON files. We also save the models themselves as torch tensors. This is useful because if we want to continue training a model we can start from the last epoch. E.g. if we have trained a model for 50 epochs but later realized we wanted 100 epochs we only have to train on those last 50 epochs.
The models are saved under `output/MODEL_NAME/ITERATION`.

### 1. Import

```python
from training.training import develop_model, continue_model_training
```

### 2. Parameters

#### `develop_model`

| Parameter          | Type                                        | Description                                                     |
| ------------------ | ------------------------------------------- | --------------------------------------------------------------- |
| `model`            | `nn.Module`                                 | The PyTorch model to train.                                     |
| `loaders`          | `tuple[DataLoader, DataLoader, DataLoader]` | Tuple containing `(train_loader, val_loader, test_loader)`.     |
| `criterion`        | callable                                    | Loss function used during training and evaluation.              |
| `optimizer`        | `torch.optim.Optimizer`                     | Optimizer used to update model parameters.                      |
| `model_name`       | str                                         | Name of the model (used for logging and saving outputs).        |
| `num_epochs`       | int                                         | Number of epochs to train the model.                            |
| `iteration_number` | int                                         | Version/iteration identifier for saving outputs (default: `0`). |

#### `continue_model_training`

| Parameter           | Type                                        | Description                                                                               |
| ------------------- | ------------------------------------------- | ----------------------------------------------------------------------------------------- |
| `model`             | `nn.Module`                                 | The PyTorch model to continue training. Must match the saved model architecture.          |
| `optimizer`         | `torch.optim.Optimizer`                     | Optimizer instance. Its state will be overwritten from the saved checkpoint.              |
| `loaders`           | `tuple[DataLoader, DataLoader, DataLoader]` | Tuple containing `(train_loader, val_loader, test_loader)`.                               |
| `model_name`        | str                                         | Name of the model (used to locate saved checkpoints).                                     |
| `iteration_number`  | int                                         | Identifier of the saved run to resume from. Must match an existing output directory.      |
| `criterion`         | callable                                    | Loss function used during training and evaluation.                                        |
| `num_epochs`        | int                                         | Number of additional epochs to train the model.                                           |
| `new_learning_rate` | float                                       | A new learning rate for the optimizer. Set to `None` if you want to keep equal as before. |
| `load_optimizer`    | boolean                                     | Set to `False` if you don't want to load the params from the saved optimizer.             |

### 3. Usage

Use the functions as normal.

⚠️ NOTE! These functions do not store any information for Tensorboard. They return a metadata object that you can use to write to tensorboard runs. The metadata fields are

```json
// output/MODEL_NAME/ITERATION_NUMBER/metadata.json
{
    "model_name": string, // User defined name
    "num_epochs": int, // Total amount of epochs trained.
    "training_time": float, // Seconds spent training
    "best_val_loss": float, // Best model's validation loss

    "train_loss": list[float],
    "train_accuracy": list[float],
    "val_loss": list[float],
    "val_accuracy": list[float],

    "test_loss": test_loss, // Test loss of best model
    "test_accuracy": test_accuracy, // Test accuracy of best model
    "confusion_matrix": confusion_matrix, // CM of best model
}
```

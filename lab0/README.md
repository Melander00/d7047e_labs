## Project Structure

```
lab0/
├── cifar_model.py          # Custom CNN architecture (LeakyReLU / Tanh)
├── mnist_model.py          # CNN for MNIST digit classification
├── preprocessing.py        # Data loading and augmentation (CIFAR-10)
├── model_trainer.py        # Training loop, validation, testing
├── main.py                 # Task 0.1 (CIFAR-10) + Task 0.2.1 (AlexNet)
├── main_tanh.py            # Separate Tanh activation experiment
├── mnist_svhn_transfer.py  # Task 0.2.2 (MNIST → SVHN transfer)
├── img/                    # TensorBoard screenshots
├── tensorboard/            # TensorBoard event logs
└── README.md
```


## Hyperparameters

| Parameter | Value |
|---|---|
| Learning Rate (CIFAR-10 / AlexNet) | 0.0001 |
| Learning Rate (MNIST / SVHN) | 0.001 |
| Batch Size | 128 |
| Epochs | 30 (CIFAR-10 / AlexNet), 20 (MNIST / SVHN) |
| Optimizer | Adam (main runs), SGD (comparison run) |
| Train/Val Split | 80/20 (CIFAR-10), 85/15 (MNIST / SVHN) |
| Loss Function | CrossEntropyLoss |

---

## Task 0.1 — CNN on CIFAR-10

We built a simple CNN with 3 convolutional layers and trained it on CIFAR-10 using three different configurations (SGD, Adam, and Tanh) to observe the effects of optimizer and activation choices.

![](./img/cifar10.png)

| Config | Test Acc | Train Loss | Val Loss |
|---|---|---|---|
| Adam + LeakyReLU | 72.67% | 0.83 | 0.80 |
| Adam + Tanh | 73.40% | 0.79 | 0.78 |
| SGD + LeakyReLU | 14.92% | 2.28 | 2.28 |

- Adam converges much faster and effectively on CIFAR-10 than SGD with a small learning rate.
- Tanh and LeakyReLU perform similarly, with Tanh having a slight edge in this experiment.


## Task 0.2.1 — AlexNet Transfer Learning (ImageNet → CIFAR-10)

Using AlexNet, we compared two transfer learning approaches: **Fine-Tuning** (training the whole network) versus **Feature Extraction** (freezing the backbone and training only the classification head).

![](./img/alexnet.png)

| Approach | Test Acc | Train Loss | Val Loss |
|---|---|---|---|
| Fine-Tuning | 91.97% | 0.025 | 0.39 |
| Feature Extraction | 82.66% | 0.51 | 0.49 |

- Fine-tuning achieves significantly higher accuracy as the model adapts its low-level features to the new task.
- Feature extraction is less computationally intensive but hits a ceiling as it relies on fixed ImageNet features.
- Fine-tuning shows clear signs of overfitting after the first 15 epochs.

### Analysis of the Differences
The main difference between the two runs is how much of the network is actually being trained. In fine-tuning, we load the pretrained ImageNet weights but then train the whole network on CIFAR-10, so all the layers get updated. In feature extraction, we freeze the convolutional backbone completely and only train the new classifier layer we added on top. The pretrained features are used as-is, basically treating AlexNet as a fixed feature extractor.

In terms of performance, fine-tuning generally does better because the model can adapt its lower-level features to the new dataset, not just the final classification layer. That said, feature extraction is faster to train and can still work well when the source and target datasets are similar enough. In this case both runs used CIFAR-10 which is quite different from ImageNet (much lower resolution, different distribution), so fine-tuning had more room to adjust and unsurprisingly came out ahead.


## Task 0.2.2 — Transfer Learning (MNIST → SVHN)

We trained a model for handwriting digit recognition (MNIST) and then transferred that knowledge to real-world street view digits (SVHN).

![](./img/mnist.png)

| Dataset | Test Acc | Train Loss | Val Loss |
|---|---|---|---|
| MNIST | 99.55% | 0.027 | 0.016 |
| SVHN (transfer) | 83.67% | 0.88 | 0.62 |

- The model hits near-perfect accuracy on the simpler MNIST dataset.
- The transfer to SVHN performs surprisingly well (~83.7%), even with the frozen MNIST backbone.
- Real-world images in SVHN introduce significantly more noise and lighting variation.

### Analysis of the Transfer
For this task we first trained a CNN from scratch on MNIST, which got to around 99.5% test accuracy — not surprising since MNIST is a relatively simple dataset. The interesting part is what happens when you take those learned weights and apply them to SVHN.

SVHN (Street View House Numbers) is a much harder dataset. The images are real-world photos of house numbers, so there's a lot more noise, variation in lighting, and background clutter compared to MNIST's clean handwritten digits. Despite that, the transferred model still managed to reach around 83.7% on SVHN, which is a decent result considering the backbone was never trained on any real-world images. The fact that it works at all shows that some of the low-level features learned on MNIST (edges, curves, basic digit shapes) do transfer over, even across very different domains. The accuracy is lower than you'd get training directly on SVHN, which makes sense — there's still a meaningful domain gap between clean handwritten digits and noisy street-level photos.
# Lab 0

## Task 0.1

![](./img/cifar10.png)


## Task 0.2.1

![](./img/alexnet.png)

The main difference between the two runs is how much of the network is actually being trained. In fine-tuning, we load the pretrained ImageNet weights but then train the whole network on CIFAR-10, so all the layers get updated. In feature extraction, we freeze the convolutional backbone completely and only train the new classifier layer we added on top. The pretrained features are used as-is, basically treating AlexNet as a fixed feature extractor.

In terms of performance, fine-tuning generally does better because the model can adapt its lower-level features to the new dataset, not just the final classification layer. That said, feature extraction is faster to train and can still work well when the source and target datasets are similar enough. In this case both runs used CIFAR-10 which is quite different from ImageNet (much lower resolution, different distribution), so fine-tuning had more room to adjust and unsurprisingly came out ahead.


## Task 0.2.2

![](./img/mnist.png)

For this task we first trained a CNN from scratch on MNIST, which got to around 99.5% test accuracy — not surprising since MNIST is a relatively simple dataset. The interesting part is what happens when you take those learned weights and apply them to SVHN.

SVHN (Street View House Numbers) is a much harder dataset. The images are real-world photos of house numbers, so there's a lot more noise, variation in lighting, and background clutter compared to MNIST's clean handwritten digits. Despite that, the transferred model still managed to reach around 82.9% on SVHN, which is a decent result considering the backbone was never trained on any real-world images. The fact that it works at all shows that some of the low-level features learned on MNIST (edges, curves, basic digit shapes) do transfer over, even across very different domains. The accuracy is lower than you'd get training directly on SVHN, which makes sense — there's still a meaningful domain gap between clean handwritten digits and noisy street-level photos.
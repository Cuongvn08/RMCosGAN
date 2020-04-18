# An Empirical Study on GANs with Margin Cosine Loss and Relativistic Discriminator
This is a Pytorch implementation for the paper "An Empirical Study on GANs with Margin Cosine Loss and Relativistic Discriminator".

## Requirement
* python                    3.7.3
* pytorch                   1.2.0
* pytorch-transformers      1.2.0
* scikit-image              0.15.0
* scikit-learn              0.20.3
* scikit-optimize           0.5.2
* tensorflow                2.0.0
* tensorflow-base           2.0.0
* tensorflow-estimator      2.0.0
* torchtext                 0.4.0
* torchvision               0.4.0
* mnist

## Usage

### Data preparation
* Run 0_prepare_dataset.py to automatically download, process and store images of CIFAR10, STL10, MNIST datasets.
* In term of CAT dataset, please download here.
* Download FID stats for CIFAR-10 at http://bioinf.jku.at/research/ttur/ttur_stats/fid_stats_cifar10_train.npz

## Training
* Run 1_train.sh to train all the loss function methods on four datasets above.

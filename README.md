# An Empirical Study on GANs with Margin Cosine Loss and Relativistic Discriminator
This is a Pytorch implementation for the paper "An Empirical Study on GANs with Margin Cosine Loss and Relativistic Discriminator".

## Requirement
* python                    3.7.3
* pytorch                   1.2.0
* tensorflow                2.0.0
* torchtext                 0.4.0
* torchvision               0.4.0
* mnist

## Data preparation
* Run 0_prepare_dataset.py to automatically download, process and store images of CIFAR10, STL10, MNIST datasets. CAT dataset can be downloaded here.
* Download FID stats for CIFAR-10 at http://bioinf.jku.at/research/ttur/ttur_stats/fid_stats_cifar10_train.npz

## Training
* Run 1_train.sh to train our proposed loss function RMCosGAN along with other loss functions on four datasets.

## Experimental results
**60 randomly-generated images with Cos-Margin at $\text{FID}=31.34$ trained on CIFAR-10 dataset**
![](/figures/images_cifar10_32x32.png)
**60 randomly-generated images with Cos-Margin at $\text{FID}=13.17$ trained on MNIST dataset**
![](/figures/images_mnist_32x32.png)
**60 randomly-generated images with Cos-Margin $\text{FID}=52.16$ trained on STL-10 dataset**
![](/figures/images_stl10_48x48.png)
**60 randomly-generated images with Cos-Margin at $\text{FID}=9.48$ trained on CAT dataset**
![](/figures/images_cat_64x64.png)

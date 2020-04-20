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

## Appendix

### Network Architectures

**DCGAN Architecture for CIFAR-10, MNIST and STL-10 datasets**

| Operation | Filter Units | Non Linearity | Normalization |
| --- | --- | --- | --- |
| Generator G(z) | | | | 
| Linear | 512 | None | None |
| Trans.Conv2D | 256 | ReLU | Batch |
| Trans.Conv2D | 128 | ReLU | Batch |
| Trans.Conv2D | 64 | ReLU | Batch |
| Trans.Conv2D | 3 | Tanh | None |
| Discriminator D(x) | | | | 
| Conv2D | 64 | Leaky-ReLU | Spectral |
| Conv2D | 64 | Leaky-ReLU | Spectral |
| Conv2D | 128 | Leaky-ReLU | Spectral |
| Conv2D | 128 | Leaky-ReLU | Spectral |
| Conv2D | 256 | Leaky-ReLU | Spectral |
| Conv2D | 256 | Leaky-ReLU | Spectral |
| Conv2D | 512 | Leaky-ReLU | Spectral |


**DCGAN Architecture for CAT dataset**

| Operation | Filter Units | Non Linearity | Normalization |
| --- | --- | --- | --- |
| Generator G(z) | | | | 
| Trans.Conv2D | 1024 | ReLU | Batch | 
| Trans.Conv2D | 512 | ReLU | Batch | 
| Trans.Conv2D | 256 | ReLU | Batch | 
| Trans.Conv2D | 128 | ReLU | Batch | 
| Trans.Conv2D | 3 | Tanh | None | 
| Discriminator D(x) | | | | 
| Conv2D | 128 | Leaky-ReLU | Spectral | 
| Conv2D | 256 | Leaky-ReLU | Spectral | 
| Conv2D | 512 | Leaky-ReLU | Spectral | 
| Conv2D | 1024 | Leaky-ReLU | Spectral | 



### Experimental results

**60 randomly-generated images with RMCosGAN at FID=31.34 trained on CIFAR-10 dataset**
![](/figures/images_cifar10_32x32.png)

**60 randomly-generated images with RMCosGAN at FID=13.17 trained on MNIST dataset**
![](/figures/images_mnist_32x32.png)

**60 randomly-generated images with RMCosGAN FID=52.16 trained on STL-10 dataset**
![](/figures/images_stl10_48x48.png)

**60 randomly-generated images with RMCosGAN at FID=9.48 trained on CAT dataset**
![](/figures/images_cat_64x64.png)

## Citation
Please cite our paper if PyramidNets is used:
```
@article{RMCosGAN,
  title={An Empirical Study on GANs with Margin Cosine Loss and Relativistic Discriminator},
  author={Cuong Nguyen, Tien-Dung Cao, Tram Truong-Huu, Binh T.Nguyen},
  journal={},
  year={}
}
```
If this implementation is useful, please cite or acknowledge this repository on your work.

## Contact
Cuong Nguyen (cuong.vn08@gmail.com),

Tien-Dung Cao (dung.cao@ttu.edu.vn),

Tram Truong-Huu (tram.truong-huu@ieee.org),

Binh T.Nguyen (ngtbinh@hcmus.edu.vn)

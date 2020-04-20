import os
import shutil
import torch
import torchvision
import torchvision.datasets as dset
import torchvision.transforms as transf
from PIL import Image
import mnist


def create_dir(directory):
    if os.path.exists(directory):
        shutil.rmtree(directory)
    os.makedirs(directory)


def prepare_cifar10():
    print('[CIFAR10] Preparing dataset ...')

    # setting
    dir_image_output = './dataset/cifar10/'

    # clean and create dir if any
    create_dir(dir_image_output)

    # save images from pytorch dataset
    dset_train = dset.CIFAR10('./dataset/others', train=True, download=True)
    print('[CIFAR10] The number of images: ', len(dset_train))

    cnt = 1
    for img,label in dset_train:
        #img = img.convert('RGB')
        img.save(dir_image_output + 'img_{}.jpg'.format(cnt))
        cnt += 1

    print('[CIFAR10] Preparing dataset was completed.')

    pass


def prepare_mnist():
    print('[MNIST] Preparing dataset ...')

    # setting
    dir_image_output = './dataset/mnist/'

    # clean and create dir if any
    create_dir(dir_image_output + 'all')
    create_dir(dir_image_output + 'classes')
    for i in range(10):
        create_dir(dir_image_output + 'classes/{}'.format(i))

    train_images = mnist.train_images()
    train_labels = mnist.train_labels()

    for i in range(train_images.shape[0]):
        img = train_images[i,:,:]
        label = train_labels[i]

        img = Image.fromarray(img).convert('RGB').resize((32,32))

        img.save(dir_image_output + 'all/img_{}.jpg'.format(i))
        img.save(dir_image_output + 'classes/{}/img_{}.jpg'.format(label, i))

    print('[MNIST] Preparing dataset was completed.')

    pass


def prepare_stl10():
    print('[STL10] Preparing dataset ...')

    # setting
    dir_image_output = './dataset/stl10/'

    # clean and create dir if any
    create_dir(dir_image_output)

    # save images from pytorch dataset
    trans = transf.Compose([
        transf.Resize((48, 48)),
    ])
    dset_train = dset.STL10('./dataset/others', split='train', download=True, transform=trans)
    print('[STL10] The number of images: ', len(dset_train))

    cnt = 1
    for img,label in dset_train:
        #img = img.convert('RGB')
        img.save(dir_image_output + 'img_{}.jpg'.format(cnt))
        cnt += 1

    print('[STL10] Preparing dataset was completed.')

    pass


def prepare_cat():
    pass


if __name__ == '__main__':
    prepare_cifar10()
    prepare_mnist()
    prepare_stl10()
    #prepare_cat()

#!/usr/bin/env python3

import os
import time
import shutil
import numpy
import scipy
import scipy.stats
import random

import pytz
from datetime import datetime

tz = pytz.timezone('Asia/Saigon')

import torch
import torch.autograd as autograd
from torch.autograd import Variable

import torchvision
import torchvision.datasets as dset
import torchvision.transforms as transf
import torchvision.models as models
import torchvision.utils as vutils
import torch.nn.utils.spectral_norm as spectral_norm
from torchvision.utils import save_image
import torch.nn.functional as F

from IPython.display import Image
to_img = transf.ToPILImage()
import math
torch.utils.backcompat.broadcast_warning.enabled = True

import metric_is
import metric_fid

# parameters
import argparse

def strToBool(str):
    return str.lower() in ('true', 'yes', 'on', 't', '1')

parser = argparse.ArgumentParser()
parser.register('type', 'bool', strToBool)


parser.add_argument('--dataset', default='cifar10', help='cifar10, mnist, stl10, cat')
parser.add_argument('--method', default='s', help='s, rs, ras, ls, rals, hinge, rahinge, margin')
parser.add_argument('--dir_dataset', default='/data/cuong/data/gan_paper/dataset/')
parser.add_argument('--dir_output', default='/data/cuong/data/gan_paper/output/ablation/margin/')
parser.add_argument('--n_iter', type=int, default=200000, help='Number of iteration cycles')
parser.add_argument('--save_every', type=int, default=10000, help='Generate images after x iterations')
parser.add_argument('--valid_images', type=int, default=10000, help='Number of images needed to be generated')

parser.add_argument('--m', type=float, default=0.35)
parser.add_argument('--s', type=int, default=10)

parser.add_argument('--image_size', type=int, default=32)
parser.add_argument('--batch_size', type=int, default=32) # DCGAN paper original value used 128 (32 is generally better to prevent vanishing gradients with SGAN and LSGAN, not important with relativistic GANs)
parser.add_argument('--n_colors', type=int, default=3)
parser.add_argument('--z_size', type=int, default=128)
parser.add_argument('--G_h_size', type=int, default=128, help='Number of hidden nodes in the Generator. Used only in arch=0. Too small leads to bad results, too big blows up the GPU RAM.') # DCGAN paper original value
parser.add_argument('--D_h_size', type=int, default=128, help='Number of hidden nodes in the Discriminator. Used only in arch=0. Too small leads to bad results, too big blows up the GPU RAM.') # DCGAN paper original value
parser.add_argument('--lr_D', type=float, default=.0002, help='Discriminator learning rate')
parser.add_argument('--lr_G', type=float, default=.0002, help='Generator learning rate')
parser.add_argument('--beta1', type=float, default=0.5, help='Adam betas[0], DCGAN paper recommends .50 instead of the usual .90')
parser.add_argument('--beta2', type=float, default=0.999, help='Adam betas[1]')
parser.add_argument('--decay', type=float, default=0, help='Decay to apply to lr each cycle. decay^n_iter gives the final lr. Ex: .00002 will lead to .13 of lr after 100k cycles')
parser.add_argument('--SELU', type='bool', default=False, help='Using scaled exponential linear units (SELU) which are self-normalizing instead of ReLU with BatchNorm. Used only in arch=0. This improves stability.')
parser.add_argument("--NN_conv", type='bool', default=False, help="This approach minimize checkerboard artifacts during training. Used only by arch=0. Uses nearest-neighbor resized convolutions instead of strided convolutions (https://distill.pub/2016/deconv-checkerboard/ and github.com/abhiskk/fast-neural-style).")
parser.add_argument('--seed', type=int, default=1)
parser.add_argument('--load', default=None, help='Full path to network state to load (ex: /home/output_folder/run-5/models/state_11.pth)')
parser.add_argument('--cuda', type='bool', default=True, help='enables cuda')
parser.add_argument('--n_gpu', type=int, default=1, help='number of GPUs to use')
parser.add_argument('--loss_D', type=int, default=1, help='Loss of D, see code for details (1=GAN, 2=LSGAN, 3=WGAN-GP, 4=HingeGAN, 5=RSGAN, 6=RaSGAN, 7=RaLSGAN, 8=RaHingeGAN)')
parser.add_argument('--Diters', type=int, default=1, help='Number of iterations of D')
parser.add_argument('--Giters', type=int, default=1, help='Number of iterations of G.')
parser.add_argument('--penalty', type=float, default=10, help='Gradient penalty parameter for WGAN-GP')
parser.add_argument('--spectral', type='bool', default=True, help='If True, use spectral normalization to make the discriminator Lipschitz. This Will also remove batch norm in the discriminator.')
parser.add_argument('--spectral_G', type='bool', default=False, help='If True, use spectral normalization to make the generator Lipschitz (Generally only D is spectral, not G). This Will also remove batch norm in the discriminator.')
parser.add_argument('--weight_decay', type=float, default=0, help='L2 regularization weight. Helps convergence but leads to artifacts in images, not recommended.')
parser.add_argument('--no_batch_norm_G', type='bool', default=False, help='If True, no batch norm in G.')
parser.add_argument('--no_batch_norm_D', type='bool', default=False, help='If True, no batch norm in D.')
parser.add_argument('--Tanh_GD', type='bool', default=False, help='If True, tanh everywhere.')
parser.add_argument('--grad_penalty', type='bool', default=False, help='If True, use gradient penalty of WGAN-GP but with whichever loss_D chosen. No need to set this true with WGAN-GP.')
parser.add_argument('--arch', type=int, default=0, help='1: standard CNN  for 32x32 images from the Spectral GAN paper, 0:DCGAN with number of layers adjusted based on image size. Some options may be ignored by some architectures.')
param = parser.parse_args()

# additional param
param.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

param.log = 'log_' + param.dataset + '.txt'

if param.dataset == 'cat':
    param.path_fid_stats   = os.path.join(param.dir_dataset, 'cat/all')
    param.dir_input        = os.path.join(param.dir_dataset, 'cat/classes')
    param.dir_inception    = os.path.join(param.dir_dataset, 'others')
elif param.dataset in ['stl10']:
    param.path_fid_stats   = os.path.join(param.dir_dataset, param.dataset)
    param.dir_input        = os.path.join(param.dir_dataset, 'others')
    param.dir_inception    = os.path.join(param.dir_dataset, 'others')
elif param.dataset in ['cifar10']:
    param.path_fid_stats   = os.path.join(param.dir_dataset, 'others/fid_stats_cifar10_train.npz')
    param.dir_input        = os.path.join(param.dir_dataset, 'others')
    param.dir_inception    = os.path.join(param.dir_dataset, 'others')
elif param.dataset in ['mnist']:
    param.path_fid_stats   = os.path.join(param.dir_dataset, 'mnist/all')
    param.dir_input        = os.path.join(param.dir_dataset, 'mnist/classes')
    param.dir_inception    = os.path.join(param.dir_dataset, 'others')

if param.method == 'margin':
    param.dir_output_model = os.path.join(param.dir_output, param.dataset, param.method + '_m' + str(param.m) + '_s' + str(param.s), 'model')
    param.dir_output_image = os.path.join(param.dir_output, param.dataset, param.method + '_m' + str(param.m) + '_s' + str(param.s), 'image')
else:
    param.dir_output_model = os.path.join(param.dir_output, param.dataset, param.method, 'model')
    param.dir_output_image = os.path.join(param.dir_output, param.dataset, param.method, 'image')

if param.dataset in ['cat']:
    param.arch = 0
elif param.dataset in ['cifar10', 'mnist', 'stl10']:
    param.arch = 1

if param.dataset in ['cifar10', 'mnist']:
    param.image_size = 32
elif param.dataset in ['stl10']:
    param.image_size = 48
elif param.dataset in ['cat']:
    param.image_size = 64

def printBoth(filename, args):
    date_time = datetime.now(tz).strftime('%Y-%m-%d %H:%M:%S ')

    # write log
    fo = open(filename, "a")
    fo.write(date_time + args + '\n')
    fo.close()

    # print
    print(date_time + args)

def create_dir(directory):
    if os.path.exists(directory):
        shutil.rmtree(directory)
    os.makedirs(directory)

# models
if param.arch == 0: # for cat
    # DCGAN generator
    class DCGAN_G(torch.nn.Module):
        def __init__(self):
            super(DCGAN_G, self).__init__()
            main = torch.nn.Sequential()

            # We need to know how many layers we will use at the beginning
            mult = param.image_size // 8

            ### Start block
            # Z_size random numbers
            if param.spectral_G:
                main.add_module('Start-SpectralConvTranspose2d', torch.nn.utils.spectral_norm(torch.nn.ConvTranspose2d(param.z_size, param.G_h_size * mult, kernel_size=4, stride=1, padding=0, bias=False)))
            else:
                main.add_module('Start-ConvTranspose2d', torch.nn.ConvTranspose2d(param.z_size, param.G_h_size * mult, kernel_size=4, stride=1, padding=0, bias=False))
            if param.SELU:
                main.add_module('Start-SELU', torch.nn.SELU(inplace=True))
            else:
                if not param.no_batch_norm_G and not param.spectral_G:
                    main.add_module('Start-BatchNorm2d', torch.nn.BatchNorm2d(param.G_h_size * mult))
                if param.Tanh_GD:
                    main.add_module('Start-Tanh', torch.nn.Tanh())
                else:
                    main.add_module('Start-ReLU', torch.nn.ReLU())
            # Size = (G_h_size * mult) x 4 x 4

            ### Middle block (Done until we reach ? x image_size/2 x image_size/2)
            i = 1
            while mult > 1:
                if param.NN_conv:
                    main.add_module('Middle-UpSample [%d]' % i, torch.nn.Upsample(scale_factor=2))
                    if param.spectral_G:
                        main.add_module('Middle-SpectralConv2d [%d]' % i, torch.nn.utils.spectral_norm(torch.nn.Conv2d(param.G_h_size * mult, param.G_h_size * (mult//2), kernel_size=3, stride=1, padding=1)))
                    else:
                        main.add_module('Middle-Conv2d [%d]' % i, torch.nn.Conv2d(param.G_h_size * mult, param.G_h_size * (mult//2), kernel_size=3, stride=1, padding=1))
                else:
                    if param.spectral_G:
                        main.add_module('Middle-SpectralConvTranspose2d [%d]' % i, torch.nn.utils.spectral_norm(torch.nn.ConvTranspose2d(param.G_h_size * mult, param.G_h_size * (mult//2), kernel_size=4, stride=2, padding=1, bias=False)))
                    else:
                        main.add_module('Middle-ConvTranspose2d [%d]' % i, torch.nn.ConvTranspose2d(param.G_h_size * mult, param.G_h_size * (mult//2), kernel_size=4, stride=2, padding=1, bias=False))
                if param.SELU:
                    main.add_module('Middle-SELU [%d]' % i, torch.nn.SELU(inplace=True))
                else:
                    if not param.no_batch_norm_G and not param.spectral_G:
                        main.add_module('Middle-BatchNorm2d [%d]' % i, torch.nn.BatchNorm2d(param.G_h_size * (mult//2)))
                    if param.Tanh_GD:
                        main.add_module('Middle-Tanh [%d]' % i, torch.nn.Tanh())
                    else:
                        main.add_module('Middle-ReLU [%d]' % i, torch.nn.ReLU())
                # Size = (G_h_size * (mult/(2*i))) x 8 x 8
                mult = mult // 2
                i += 1

            ### End block
            # Size = G_h_size x image_size/2 x image_size/2
            if param.NN_conv:
                main.add_module('End-UpSample', torch.nn.Upsample(scale_factor=2))
                if param.spectral_G:
                    main.add_module('End-SpectralConv2d', torch.nn.utils.spectral_norm(torch.nn.Conv2d(param.G_h_size, param.n_colors, kernel_size=3, stride=1, padding=1)))
                else:
                    main.add_module('End-Conv2d', torch.nn.Conv2d(param.G_h_size, param.n_colors, kernel_size=3, stride=1, padding=1))
            else:
                if param.spectral_G:
                    main.add_module('End-SpectralConvTranspose2d', torch.nn.utils.spectral_norm(torch.nn.ConvTranspose2d(param.G_h_size, param.n_colors, kernel_size=4, stride=2, padding=1, bias=False)))
                else:
                    main.add_module('End-ConvTranspose2d', torch.nn.ConvTranspose2d(param.G_h_size, param.n_colors, kernel_size=4, stride=2, padding=1, bias=False))
            main.add_module('End-Tanh', torch.nn.Tanh())
            # Size = n_colors x image_size x image_size
            self.main = main

        def forward(self, input):
            if isinstance(input.data, torch.cuda.FloatTensor) and param.n_gpu > 1:
                output = torch.nn.parallel.data_parallel(self.main, input, range(param.n_gpu))
            else:
                output = self.main(input)
            return output

    # DCGAN discriminator (using somewhat the reverse of the generator)
    class DCGAN_D(torch.nn.Module):
        def __init__(self):
            super(DCGAN_D, self).__init__()
            main = torch.nn.Sequential()

            ### Start block
            # Size = n_colors x image_size x image_size
            if param.spectral:
                main.add_module('Start-SpectralConv2d', torch.nn.utils.spectral_norm(torch.nn.Conv2d(param.n_colors, param.D_h_size, kernel_size=4, stride=2, padding=1, bias=False)))
            else:
                main.add_module('Start-Conv2d', torch.nn.Conv2d(param.n_colors, param.D_h_size, kernel_size=4, stride=2, padding=1, bias=False))
            if param.SELU:
                main.add_module('Start-SELU', torch.nn.SELU(inplace=True))
            else:
                if param.Tanh_GD:
                    main.add_module('Start-Tanh', torch.nn.Tanh())
                else:
                    main.add_module('Start-LeakyReLU', torch.nn.LeakyReLU(0.2, inplace=True))
            image_size_new = param.image_size // 2
            # Size = D_h_size x image_size/2 x image_size/2

            ### Middle block (Done until we reach ? x 4 x 4)
            mult = 1
            i = 0
            while image_size_new > 4:
                if param.spectral:
                    main.add_module('Middle-SpectralConv2d [%d]' % i, torch.nn.utils.spectral_norm(torch.nn.Conv2d(param.D_h_size * mult, param.D_h_size * (2*mult), kernel_size=4, stride=2, padding=1, bias=False)))
                else:
                    main.add_module('Middle-Conv2d [%d]' % i, torch.nn.Conv2d(param.D_h_size * mult, param.D_h_size * (2*mult), kernel_size=4, stride=2, padding=1, bias=False))
                if param.SELU:
                    main.add_module('Middle-SELU [%d]' % i, torch.nn.SELU(inplace=True))
                else:
                    if not param.no_batch_norm_D and not param.spectral:
                        main.add_module('Middle-BatchNorm2d [%d]' % i, torch.nn.BatchNorm2d(param.D_h_size * (2*mult)))
                    if param.Tanh_GD:
                        main.add_module('Start-Tanh [%d]' % i, torch.nn.Tanh())
                    else:
                        main.add_module('Middle-LeakyReLU [%d]' % i, torch.nn.LeakyReLU(0.2, inplace=True))
                # Size = (D_h_size*(2*i)) x image_size/(2*i) x image_size/(2*i)
                image_size_new = image_size_new // 2
                mult *= 2
                i += 1

            ### End block
            # Size = (D_h_size * mult) x 4 x 4
            #if param.method not in ['margin']:
            #    if param.spectral:
            #        main.add_module('End-SpectralConv2d', torch.nn.utils.spectral_norm(torch.nn.Conv2d(param.D_h_size * mult, 1, kernel_size=4, stride=1, padding=0, bias=False)))
            #    else:
            #        main.add_module('End-Conv2d', torch.nn.Conv2d(param.D_h_size * mult, 1, kernel_size=4, stride=1, padding=0, bias=False))
            #f param.method == 's':
            #    main.add_module('End-Sigmoid', torch.nn.Sigmoid())
            # Size = 1 x 1 x 1 (Is a real cat or not?)

            self.main = main
            self.sig = torch.nn.Sigmoid()

            if param.method == 'margin':
                #self.fc = torch.nn.Linear(1024 * 4 * 4, 1024)
                self.weight = torch.nn.Parameter(torch.FloatTensor(1, 1024))
                torch.nn.init.xavier_uniform_(self.weight)
            else:
                self.dense = torch.nn.Linear(1024, 1)

        def forward(self, input):
            if isinstance(input.data, torch.cuda.FloatTensor) and param.n_gpu > 1:
                output = torch.nn.parallel.data_parallel(self.main, input, range(param.n_gpu))
            else:
                if param.method == 'margin': #margin
                    output = self.main(input)
                    output = torch.sum(output, dim=(2,3))
                    output = F.linear(F.normalize(output), F.normalize(self.weight)).view(-1)
                else:
                    #output = self.main(input)

                    output = self.main(input)
                    output = torch.sum(output, dim=(2,3))
                    output = self.dense(output).view(-1)

            if param.method == 's':
                output = self.sig(output)

            return output.view(-1)


if param.arch == 1:# for cifar10, mnist, stl
    class DCGAN_G(torch.nn.Module):
        def __init__(self, h=4, w=4):
            super(DCGAN_G, self).__init__()

            self.h = h
            self.w = w

            self.dense = torch.nn.Linear(param.z_size, 512 * self.h * self.w)

            if param.spectral_G:
                model = [
                    spectral_norm(torch.nn.ConvTranspose2d(512, 256, kernel_size=4, stride=2, padding=1, bias=True))]
                model += [torch.nn.ReLU(True),
                          spectral_norm(
                              torch.nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1, bias=True))]
                model += [torch.nn.ReLU(True),
                          spectral_norm(
                              torch.nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1, bias=True))]
                model += [torch.nn.ReLU(True),
                          spectral_norm(
                              torch.nn.Conv2d(64, param.n_colors, kernel_size=3, stride=1, padding=1, bias=True)),
                          torch.nn.Tanh()]
            else:
                model = [torch.nn.ConvTranspose2d(512, 256, kernel_size=4, stride=2, padding=1, bias=True)]
                if not param.no_batch_norm_G:
                    model += [torch.nn.BatchNorm2d(256)]
                if param.Tanh_GD:
                    model += [torch.nn.Tanh()]
                else:
                    model += [torch.nn.ReLU(True)]
                model += [torch.nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1, bias=True)]
                if not param.no_batch_norm_G:
                    model += [torch.nn.BatchNorm2d(128)]
                if param.Tanh_GD:
                    model += [torch.nn.Tanh()]
                else:
                    model += [torch.nn.ReLU(True)]
                model += [torch.nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1, bias=True)]
                if not param.no_batch_norm_G:
                    model += [torch.nn.BatchNorm2d(64)]
                if param.Tanh_GD:
                    model += [torch.nn.Tanh()]
                else:
                    model += [torch.nn.ReLU(True)]
                model += [torch.nn.Conv2d(64, param.n_colors, kernel_size=3, stride=1, padding=1, bias=True),
                          torch.nn.Tanh()]
            self.model = torch.nn.Sequential(*model)

        def forward(self, input):
            if isinstance(input.data, torch.cuda.FloatTensor) and param.n_gpu > 1:
                output = torch.nn.parallel.data_parallel(
                    self.model(self.dense(input.view(-1, param.z_size)).view(-1, 512, self.h, self.w)), input, range(param.n_gpu))
            else:
                output = self.model(self.dense(input.view(-1, param.z_size)).view(-1, 512, self.h, self.w))
            # print(output.size())
            return output


    class DCGAN_D(torch.nn.Module):
        def __init__(self, h=4, w=4):
            super(DCGAN_D, self).__init__()

            self.h = h
            self.w = w

            if param.spectral:
                model = [
                    spectral_norm(torch.nn.Conv2d(param.n_colors, 64, kernel_size=3, stride=1, padding=1, bias=True)),
                    torch.nn.LeakyReLU(0.1, inplace=True),
                    spectral_norm(torch.nn.Conv2d(64, 64, kernel_size=4, stride=2, padding=1, bias=True)),
                    torch.nn.LeakyReLU(0.1, inplace=True),

                    spectral_norm(torch.nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1, bias=True)),
                    torch.nn.LeakyReLU(0.1, inplace=True),
                    spectral_norm(torch.nn.Conv2d(128, 128, kernel_size=4, stride=2, padding=1, bias=True)),
                    torch.nn.LeakyReLU(0.1, inplace=True),

                    spectral_norm(torch.nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1, bias=True)),
                    torch.nn.LeakyReLU(0.1, inplace=True),
                    spectral_norm(torch.nn.Conv2d(256, 256, kernel_size=4, stride=2, padding=1, bias=True)),
                    torch.nn.LeakyReLU(0.1, inplace=True),

                    spectral_norm(torch.nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1, bias=True)),
                    torch.nn.LeakyReLU(0.1, inplace=True)]
            else:
                model = [torch.nn.Conv2d(param.n_colors, 64, kernel_size=3, stride=1, padding=1, bias=True)]
                if not param.no_batch_norm_D:
                    model += [torch.nn.BatchNorm2d(64)]
                if param.Tanh_GD:
                    model += [torch.nn.Tanh()]
                else:
                    model += [torch.nn.LeakyReLU(0.1, inplace=True)]
                model += [torch.nn.Conv2d(64, 64, kernel_size=4, stride=2, padding=1, bias=True)]
                if not param.no_batch_norm_D:
                    model += [torch.nn.BatchNorm2d(64)]
                if param.Tanh_GD:
                    model += [torch.nn.Tanh()]
                else:
                    model += [torch.nn.LeakyReLU(0.1, inplace=True)]
                model += [torch.nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1, bias=True)]
                if not param.no_batch_norm_D:
                    model += [torch.nn.BatchNorm2d(128)]
                if param.Tanh_GD:
                    model += [torch.nn.Tanh()]
                else:
                    model += [torch.nn.LeakyReLU(0.1, inplace=True)]
                model += [torch.nn.Conv2d(128, 128, kernel_size=4, stride=2, padding=1, bias=True)]
                if not param.no_batch_norm_D:
                    model += [torch.nn.BatchNorm2d(128)]
                if param.Tanh_GD:
                    model += [torch.nn.Tanh()]
                else:
                    model += [torch.nn.LeakyReLU(0.1, inplace=True)]
                model += [torch.nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1, bias=True)]
                if not param.no_batch_norm_D:
                    model += [torch.nn.BatchNorm2d(256)]
                if param.Tanh_GD:
                    model += [torch.nn.Tanh()]
                else:
                    model += [torch.nn.LeakyReLU(0.1, inplace=True)]
                model += [torch.nn.Conv2d(256, 256, kernel_size=4, stride=2, padding=1, bias=True)]
                if not param.no_batch_norm_D:
                    model += [torch.nn.BatchNorm2d(256)]
                if param.Tanh_GD:
                    model += [torch.nn.Tanh()]
                else:
                    model += [torch.nn.LeakyReLU(0.1, inplace=True)]
                model += [torch.nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1, bias=True)]
                if param.Tanh_GD:
                    model += [torch.nn.Tanh()]
                else:
                    model += [torch.nn.LeakyReLU(0.1, inplace=True)]
            self.model = torch.nn.Sequential(*model)

            self.sig = torch.nn.Sigmoid()

            if param.method == 'margin': #margin
                #self.dense = torch.nn.Linear(512 * h * w, 512)
                #self.weight = torch.nn.Parameter(torch.FloatTensor(1, 512 * h * w))
                self.weight = torch.nn.Parameter(torch.FloatTensor(1, 512))
                torch.nn.init.xavier_uniform_(self.weight)
            else:
                #self.dense = torch.nn.Linear(512 * h * w, 1)
                self.dense = torch.nn.Linear(512, 1)

        def forward(self, input):
            if isinstance(input.data, torch.cuda.FloatTensor) and param.n_gpu > 1:
                output = torch.nn.parallel.data_parallel(self.dense(self.model(input).view(-1, 512 * self.h * self.w)).view(-1),
                                                         input, range(param.n_gpu))
            else:
                if param.method == 'margin': #margin
                    output = self.model(input)
                    output = torch.sum(output, dim=(2,3))
                    output = F.linear(F.normalize(output), F.normalize(self.weight)).view(-1)

                    #output = self.dense(self.model(input).view(-1, 512 * h * w))
                    #output = F.linear(F.normalize(output), F.normalize(self.weight)).view(-1)

                    #output = self.model(input).view(-1, 512 * h * w)
                    #output = F.linear(F.normalize(output), F.normalize(self.weight)).view(-1)
                else:
                    output = self.model(input)
                    output = torch.sum(output, dim=(2,3))
                    output = self.dense(output).view(-1)

                    #output = self.dense(self.model(input).view(-1, 512 * h * w)).view(-1)

            if param.method == 's':
                output = self.sig(output)

            return output


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def truncated_normal(size, threshold=2.0, dtype=torch.float32, device='cpu'):
    if threshold is not None:
        x = scipy.stats.truncnorm.rvs(-threshold, threshold, size=size)
        x = torch.from_numpy(x).to(device, dtype)
    else:
        x = scipy.stats.norm.rvs(size=size)
        x = torch.from_numpy(x).to(device, dtype)

    return x


def validate_fake_images(G, dir_output, device='cuda', valid_images=10000):
    #truncations = [1.0, 2.0, 3.0, None]
    truncations = [None]
    batch_size = 100

    fids = []
    inception_scores = []
    for truncation in truncations:
        # create output dir
        dir_output_ = os.path.join(dir_output, str(truncation))
        create_dir(dir_output_)

        # generate images
        for batch in range(int(valid_images / batch_size)):
            z = truncated_normal((batch_size, param.z_size, 1, 1), threshold=truncation, dtype=torch.float32, device=device)
            with torch.no_grad():
                gen_images = G(z).to('cpu').clone().detach().squeeze(0)
            gen_images = gen_images.to('cpu').clone().detach()
            gen_images = gen_images * 0.5 + 0.5

            for i in range(gen_images.size(0)):
                save_image(gen_images[i, :, :, :], os.path.join(dir_output_, '{}_{}.png'.format(batch, i)))

            del gen_images

        # compute scores
        fid = metric_fid.calculate_fid_given_paths([dir_output_, param.path_fid_stats])
        inception_score = metric_is.get_inception_score_given_paths(dir_output_)[0]

        fids.append(fid)
        inception_scores.append(inception_score)

    return fids, inception_scores


BCE_stable = torch.nn.BCEWithLogitsLoss().to(param.device, torch.float32)
def calc_advloss_D(real_cos, fake_cos, y, s=10.0, m=0.35):
    '''
        m: margin
        cos(theta) - m
    '''

    real_phi = real_cos - m
    loss = BCE_stable((real_phi - fake_cos)*s, y)

    return loss


def calc_advloss_G(real_cos, fake_cos, y, s=10.0, m=0.35):
    '''
        m: margin
        cos(theta) - m
    '''
    fake_phi = fake_cos - m
    loss = BCE_stable((fake_phi - real_cos)*s, y)

    return loss


if __name__ == '__main__':
    printBoth(param.log, '====================================================')
    printBoth(param.log, '====================================================')

    printBoth(param.log, 'dataset = {}'.format(param.dataset))
    printBoth(param.log, 'method = {}'.format(param.method))
    printBoth(param.log, 'valid_images = {}'.format(param.valid_images))
    printBoth(param.log, 'dir_input = {}'.format(param.dir_input))
    printBoth(param.log, 'path_fid_stats = {}'.format(param.path_fid_stats))
    printBoth(param.log, 'dir_inception = {}'.format(param.dir_inception))
    printBoth(param.log, 'dir_output_model = {}'.format(param.dir_output_model))
    printBoth(param.log, 'dir_output_image = {}'.format(param.dir_output_image))

    if param.method == 'margin':
        printBoth(param.log, 'm = {}'.format(param.m))
        printBoth(param.log, 's = {}'.format(param.s))

    # create_inception_graph
    metric_fid.create_inception_graph(str(param.dir_inception))
    metric_is._init_inception()

    ## clean and create dirs
    create_dir(param.dir_output_model)
    create_dir(param.dir_output_image)

    ## cuda
    #printBoth(param.log, 'Setting '.format(param.device))
    if param.device == 'cuda':
        import torch.backends.cudnn as cudnn

        cudnn.deterministic = True
        cudnn.benchmark = True

    ## Setting seed
    #printBoth(param.log, 'Setting seed')
    if param.seed is None:
        param.seed = random.randint(1, 10000)
    random.seed(param.seed)
    numpy.random.seed(param.seed)
    torch.manual_seed(param.seed)
    if param.device == 'cuda':
        torch.cuda.manual_seed_all(param.seed)
    printBoth(param.log, 'Random Seed = {}'.format(param.seed))

    ## Transforming images
    #printBoth(param.log, 'Transforming images')
    trans = transf.Compose([
        transf.Resize((param.image_size, param.image_size)),
        # This makes it into [0,1]
        transf.ToTensor(),
        # This makes it into [-1,1]
        transf.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])
    printBoth(param.log, 'image_size = {}'.format(param.image_size))

    ## Importing dataset
    #printBoth(param.log, 'Importing dataset')
    if param.dataset == 'cifar10':
        data = dset.CIFAR10(root=param.dir_input, train=True, download=True, transform=trans)
    elif param.dataset == 'mnist':
        #data = dset.MNIST(root=param.dir_input, train=True, download=True, transform=trans)
        data = dset.ImageFolder(root=param.dir_input, transform=trans)
    elif param.dataset == 'stl10':
        data = dset.STL10(root=param.dir_input, split='train', download=True, transform=trans)
    elif param.dataset == 'cat':
        data = dset.ImageFolder(root=param.dir_input, transform=trans)
    printBoth(param.log, 'dataset len = {}'.format(len(data)))

    # Loading data randomly
    #printBoth(param.log, 'Loading data randomly')
    def generate_random_sample():
        while True:
            random_indexes = numpy.random.choice(data.__len__(), size=param.batch_size, replace=False)
            batch = [data[i][0] for i in random_indexes]
            yield torch.stack(batch, 0)

    random_sample = generate_random_sample()

    ## Initialization
    #printBoth(param.log, 'Initialization')
    if param.arch == 0:
        G = DCGAN_G().to(param.device, torch.float32)
        D = DCGAN_D().to(param.device, torch.float32)
    elif param.arch == 1:
        if param.dataset in ['cifar10', 'mnist', 'cat']:
            h = w = 4
        elif param.dataset in ['stl10']:
            h = w = 6
        G = DCGAN_G(h,w).to(param.device, torch.float32)
        D = DCGAN_D(h,w).to(param.device, torch.float32)

    # Initialize weights
    #printBoth(param.log, 'Initializing weights')
    def weights_init(m):
        classname = m.__class__.__name__
        if classname.find('Conv') != -1:
            m.weight.data.normal_(0.0, 0.02)
        elif classname.find('BatchNorm') != -1:
            # Estimated variance, must be around 1
            m.weight.data.normal_(1.0, 0.02)
            # Estimated mean, must be around 0
            m.bias.data.fill_(0)

    G.apply(weights_init)
    D.apply(weights_init)

    printBoth(param.log, 'count_parameters of G = {}'.format(count_parameters(G)))
    printBoth(param.log, 'count_parameters of D = {}'.format(count_parameters(D)))

    # Criterion
    #rintBoth(param.log, 'Criterion')
    criterion = torch.nn.BCELoss().to(param.device, torch.float32)
    #BCE_stable = torch.nn.BCEWithLogitsLoss().to(param.device, torch.float32)
    BCE_stable_noreduce = torch.nn.BCEWithLogitsLoss(reduce=False).to(param.device, torch.float32)

    # Soon to be variables
    x = torch.FloatTensor(param.batch_size, param.n_colors, param.image_size, param.image_size).to(param.device,torch.float32)
    x_fake = torch.FloatTensor(param.batch_size, param.n_colors, param.image_size, param.image_size).to(param.device,torch.float32)
    y = torch.FloatTensor(param.batch_size).to(param.device, torch.float32)
    y2 = torch.FloatTensor(param.batch_size).to(param.device, torch.float32)
    # Weighted sum of fake and real image, for gradient penalty
    x_both = torch.FloatTensor(param.batch_size, param.n_colors, param.image_size, param.image_size).to(param.device,torch.float32)
    z = torch.FloatTensor(param.batch_size, param.z_size, 1, 1).to(param.device, torch.float32)
    # Uniform weight
    u = torch.FloatTensor(param.batch_size, 1, 1, 1).to(param.device, torch.float32)
    # This is to see during training, size and values won't change
    z_test = torch.FloatTensor(param.batch_size, param.z_size, 1, 1).normal_(0, 1).to(param.device, torch.float32)
    # For the gradients, we need to specify which one we want and want them all
    grad_outputs = torch.ones(param.batch_size).to(param.device, torch.float32)

    # Now Variables
    x = Variable(x)
    x_fake = Variable(x_fake)
    y = Variable(y)
    y2 = Variable(y2)
    z = Variable(z)
    z_test = Variable(z_test)

    # Based on DCGAN paper, they found using betas[0]=.50 better.
    # betas[0] represent is the weight given to the previous mean of the gradient
    # betas[1] is the weight given to the previous variance of the gradient
    optimizerD = torch.optim.Adam(D.parameters(), lr=param.lr_D, betas=(param.beta1, param.beta2),weight_decay=param.weight_decay)
    optimizerG = torch.optim.Adam(G.parameters(), lr=param.lr_G, betas=(param.beta1, param.beta2),weight_decay=param.weight_decay)

    # exponential weight decay on lr
    decayD = torch.optim.lr_scheduler.ExponentialLR(optimizerD, gamma=1 - param.decay)
    decayG = torch.optim.lr_scheduler.ExponentialLR(optimizerG, gamma=1 - param.decay)

    ## Fitting model
    printBoth(param.log, 'Training')
    fixed_z = truncated_normal((param.valid_images, param.z_size), dtype=torch.float32, device=param.device)
    for i in range(param.n_iter+1):
        ########################
        # (1) Update D network #
        ########################
        for p in D.parameters():
            p.requires_grad = True

        for t in range(param.Diters):
            D.zero_grad()
            images = random_sample.__next__().to(param.device, torch.float32)
            current_batch_size = images.size(0)
            x.data.resize_as_(images).copy_(images)
            del images
            y_pred = D(x)

            if param.method in ['s', 'ls', 'hinge']:
                # Train with real data
                y.data.resize_(current_batch_size).fill_(1)
                if param.method == 's':
                    errD_real = criterion(y_pred, y)
                if param.method == 'ls':
                    errD_real = torch.mean((y_pred - y) ** 2)
                if param.method == 'hinge':
                    errD_real = torch.mean(torch.nn.ReLU()(1.0 - y_pred))
                errD_real.backward()

                # Train with fake data
                z.data.resize_(current_batch_size, param.z_size, 1, 1).normal_(0, 1)
                fake = G(z)
                x_fake.data.resize_(fake.data.size()).copy_(fake.data)
                y.data.resize_(current_batch_size).fill_(0)
                y_pred_fake = D(x_fake.detach())
                if param.method == 's':
                    errD_fake = criterion(y_pred_fake, y)
                if param.method == 'ls':
                    errD_fake = torch.mean((y_pred_fake) ** 2)
                if param.method == 'hinge':
                    errD_fake = torch.mean(torch.nn.ReLU()(1.0 + y_pred_fake))
                errD_fake.backward()
                errD = errD_real + errD_fake
            else:
                y.data.resize_(current_batch_size).fill_(1)
                y2.data.resize_(current_batch_size).fill_(0)
                z.data.resize_(current_batch_size, param.z_size, 1, 1).normal_(0, 1)
                fake = G(z)
                x_fake.data.resize_(fake.data.size()).copy_(fake.data)
                y_pred_fake = D(x_fake.detach())

                if param.method == 'rs':
                    errD = BCE_stable(y_pred - y_pred_fake, y)
                if param.method == 'ras':
                    errD = (BCE_stable(y_pred - torch.mean(y_pred_fake), y) + BCE_stable(y_pred_fake - torch.mean(y_pred), y2)) / 2
                if param.method == 'rals':  # (y_hat-1)^2 + (y_hat+1)^2
                    errD = (torch.mean((y_pred - torch.mean(y_pred_fake) - y) ** 2) + torch.mean((y_pred_fake - torch.mean(y_pred) + y) ** 2)) / 2
                if param.method == 'rahinge':
                    errD = (torch.mean(torch.nn.ReLU()(1.0 - (y_pred - torch.mean(y_pred_fake)))) +torch.mean(torch.nn.ReLU()(1.0 + (y_pred_fake - torch.mean(y_pred))))) / 2
                if param.method == 'margin':
                    errD = calc_advloss_D(y_pred, y_pred_fake, y, s=param.s, m = param.m)

                errD_real = errD
                errD_fake = errD
                errD.backward()

            if (param.grad_penalty):
                # Gradient penalty
                u.data.resize_(current_batch_size, 1, 1, 1)
                u.uniform_(0, 1)
                x_both = x.data * u + x_fake.data * (1 - u)
                if param.device == 'cuda':
                    x_both = x_both.cuda()
                # We only want the gradients with respect to x_both
                x_both = Variable(x_both, requires_grad=True)

                grad = torch.autograd.grad(outputs=D(x_both), inputs=x_both, grad_outputs=grad_outputs, retain_graph=True,create_graph=True, only_inputs=True)[0]
                # We need to norm 3 times (over n_colors x image_size x image_size) to get only a vector of size "batch_size"
                grad_penalty = param.penalty * ((grad.norm(2, 1).norm(2, 1).norm(2, 1) - 1) ** 2).mean()
                grad_penalty.backward()
            optimizerD.step()

        ########################
        # (2) Update G network #
        ########################
        for p in D.parameters():
            p.requires_grad = False

        for t in range(param.Giters):
            G.zero_grad()
            y.data.resize_(current_batch_size).fill_(1)
            y2.data.resize_(current_batch_size).fill_(0)
            z.data.resize_(current_batch_size, param.z_size, 1, 1).normal_(0, 1)
            fake = G(z)
            y_pred_fake = D(fake)

            if param.method not in ['s', 'ls', 'hinge']:
                images = random_sample.__next__().to(param.device, torch.float32)
                current_batch_size = images.size(0)
                x.data.resize_as_(images).copy_(images)
                del images

            if (param.method == 's'):
                errG = criterion(y_pred_fake, y)
            if param.method == 'ls':
                errG = torch.mean((y_pred_fake - y) ** 2)
            if param.method == 'hinge':
                errG = -torch.mean(y_pred_fake)
            if param.method == 'rs':
                y_pred = D(x)
                errG = BCE_stable(y_pred_fake - y_pred, y)
            if param.method == 'ras':
                y_pred = D(x)
                # Non-saturating
                y2.data.resize_(current_batch_size).fill_(0)
                errG = (BCE_stable(y_pred - torch.mean(y_pred_fake), y2) + BCE_stable(y_pred_fake - torch.mean(y_pred),y)) / 2
            if param.method == 'rals':
                y_pred = D(x)
                errG = (torch.mean((y_pred - torch.mean(y_pred_fake) + y) ** 2) + torch.mean((y_pred_fake - torch.mean(y_pred) - y) ** 2)) / 2
            if param.method == 'rahinge':
                y_pred = D(x)
                # Non-saturating
                errG = (torch.mean(torch.nn.ReLU()(1.0 + (y_pred - torch.mean(y_pred_fake)))) +torch.mean(torch.nn.ReLU()(1.0 - (y_pred_fake - torch.mean(y_pred))))) / 2
            if param.method == 'margin':
                y_pred = D(x)
                errG = calc_advloss_G(y_pred, y_pred_fake, y, s=param.s, m=param.m)
                # errG += param.gamma * BCE_stable(y_pred_fake, y2)

            errG.backward()
            D_G = y_pred_fake.data.mean()
            optimizerG.step()
        decayD.step()
        decayG.step()

        if i % param.save_every == 0:
            # evaluate
            with torch.no_grad():
                dir_output = os.path.join(param.dir_output_image, str(i))
                create_dir(dir_output)
                fids, inception_scores = validate_fake_images(G, dir_output, param.device)

            # log
            printBoth(param.log, 'i={}; loss_D={:0.5}; loss_G={:0.5}; fids={}; inception_scores={}'.\
                                        format(i, errD.item(), errG.item(), str(fids), str(inception_scores)))

            # save model
            torch.save(G.state_dict(), param.dir_output_model + '/{}_G.pth'.format(i))
            torch.save(D.state_dict(), param.dir_output_model + '/{}_D.pth'.format(i))


from __future__ import absolute_import, division, print_function
import numpy as np
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'


import gzip, pickle
import tensorflow as tf
from scipy.misc import imread
from scipy import linalg
import pathlib
import urllib
import glob
import scipy.misc
import math
import sys
import os.path
import time

import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()


#config = tf.ConfigProto(device_count = {'GPU': 0})
config = tf.ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 0.7
#config = tf.compat.v1.ConfigProto(device_count = {'GPU': 0}) #tensorflow v2.x
#config.gpu_options.allow_growth = True


def create_inception_graph(pth):
    pth = check_or_download_inception(pth)

    """Creates a graph from saved GraphDef file."""
    # Creates graph from saved graph_def.pb.
    with tf.gfile.FastGFile( pth, 'rb') as f:
        graph_def = tf.GraphDef()
        graph_def.ParseFromString( f.read())
        _ = tf.import_graph_def( graph_def, name='FID_Inception_Net')
#-------------------------------------------------------------------------------


# code for handling inception net derived from
# https://github.com/openai/improved-gan/blob/master/inception_score/model.py
# updated from here https://github.com/bioinf-jku/TTUR/issues/6
def _get_inception_layer(sess):
    """Prepares inception net for batched usage and returns pool_3 layer. """
    layername = 'FID_Inception_Net/pool_3:0'
    pool3 = sess.graph.get_tensor_by_name(layername)
    ops = pool3.graph.get_operations()
    for op_idx, op in enumerate(ops):
        for o in op.outputs:
            shape = o.get_shape()
            if shape._dims != []:
              shape = [s.value for s in shape]
              new_shape = []
              for j, s in enumerate(shape):
                if s == 1 and j == 0:
                  new_shape.append(None)
                else:
                  new_shape.append(s)
              o.__dict__['_shape_val'] = tf.TensorShape(new_shape)
    return pool3
#-------------------------------------------------------------------------------


def get_activations(images, sess, batch_size=50, verbose=False):
    """Calculates the activations of the pool_3 layer for all images.

    Params:
    -- images      : Numpy array of dimension (n_images, hi, wi, 3). The values
                     must lie between 0 and 256.
    -- sess        : current session
    -- batch_size  : the images numpy array is split into batches with batch size
                     batch_size. A reasonable batch size depends on the disposable hardware.
    -- verbose    : If set to True and parameter out_step is given, the number of calculated
                     batches is reported.
    Returns:
    -- A numpy array of dimension (num images, 2048) that contains the
       activations of the given tensor when feeding inception with the query tensor.
    """
    inception_layer = _get_inception_layer(sess)
    d0 = images.shape[0]
    if batch_size > d0:
        print("warning: batch size is bigger than the data size. setting batch size to data size")
        batch_size = d0
    n_batches = d0//batch_size
    n_used_imgs = n_batches*batch_size
    pred_arr = np.empty((n_used_imgs,2048))
    for i in range(n_batches):
        if verbose:
            print("\rPropagating batch %d/%d" % (i+1, n_batches), end="", flush=True)
        start = i*batch_size
        end = start + batch_size
        batch = images[start:end]
        pred = sess.run(inception_layer, {'FID_Inception_Net/ExpandDims:0': batch})
        pred_arr[start:end] = pred.reshape(batch_size,-1)
    if verbose:
        print(" done")
    return pred_arr
#-------------------------------------------------------------------------------


def calculate_frechet_distance(mu1, sigma1, mu2, sigma2, eps=1e-6):
    """Numpy implementation of the Frechet Distance.
    The Frechet distance between two multivariate Gaussians X_1 ~ N(mu_1, C_1)
    and X_2 ~ N(mu_2, C_2) is
            d^2 = ||mu_1 - mu_2||^2 + Tr(C_1 + C_2 - 2*sqrt(C_1*C_2)).

    Stable version by Dougal J. Sutherland.

    Params:
    -- mu1 : Numpy array containing the activations of the pool_3 layer of the
             inception net ( like returned by the function 'get_predictions')
             for generated samples.
    -- mu2   : The sample mean over activations of the pool_3 layer, precalcualted
               on an representive data set.
    -- sigma1: The covariance matrix over activations of the pool_3 layer for
               generated samples.
    -- sigma2: The covariance matrix over activations of the pool_3 layer,
               precalcualted on an representive data set.

    Returns:
    --   : The Frechet Distance.
    """

    mu1 = np.atleast_1d(mu1)
    mu2 = np.atleast_1d(mu2)

    sigma1 = np.atleast_2d(sigma1)
    sigma2 = np.atleast_2d(sigma2)

    assert mu1.shape == mu2.shape, "Training and test mean vectors have different lengths"
    assert sigma1.shape == sigma2.shape, "Training and test covariances have different dimensions"

    diff = mu1 - mu2

    # product might be almost singular
    covmean, _ = linalg.sqrtm(sigma1.dot(sigma2), disp=False)
    if not np.isfinite(covmean).all():
        msg = "fid calculation produces singular product; adding %s to diagonal of cov estimates" % eps
        #warnings.warn(msg)
        offset = np.eye(sigma1.shape[0]) * eps
        covmean = linalg.sqrtm((sigma1 + offset).dot(sigma2 + offset))

    # numerical error might give slight imaginary component
    if np.iscomplexobj(covmean):
        if not np.allclose(np.diagonal(covmean).imag, 0, atol=1e-3):
            m = np.max(np.abs(covmean.imag))
            raise ValueError("Imaginary component {}".format(m))
        covmean = covmean.real

    tr_covmean = np.trace(covmean)

    return diff.dot(diff) + np.trace(sigma1) + np.trace(sigma2) - 2 * tr_covmean
#-------------------------------------------------------------------------------


def calculate_activation_statistics(images, sess, batch_size=50, verbose=False):
    """Calculation of the statistics used by the FID.
    Params:
    -- images      : Numpy array of dimension (n_images, hi, wi, 3). The values
                     must lie between 0 and 255.
    -- sess        : current session
    -- batch_size  : the images numpy array is split into batches with batch size
                     batch_size. A reasonable batch size depends on the available hardware.
    -- verbose     : If set to True and parameter out_step is given, the number of calculated
                     batches is reported.
    Returns:
    -- mu    : The mean over samples of the activations of the pool_3 layer of
               the incption model.
    -- sigma : The covariance matrix of the activations of the pool_3 layer of
               the incption model.
    """
    act = get_activations(images, sess, batch_size, verbose)
    mu = np.mean(act, axis=0)
    sigma = np.cov(act, rowvar=False)
    return mu, sigma
#-------------------------------------------------------------------------------


#-------------------------------------------------------------------------------
# The following functions aren't needed for calculating the FID
# they're just here to make this module work as a stand-alone script
# for calculating FID scores
#-------------------------------------------------------------------------------
def check_or_download_inception(inception_path):
    ''' Checks if the path to the inception file is valid, or downloads
        the file if it is not present. '''
    INCEPTION_URL = 'http://download.tensorflow.org/models/image/imagenet/inception-2015-12-05.tgz'
    if inception_path is None:
        inception_path = '/tmp'
    inception_path = pathlib.Path(inception_path)
    model_file = inception_path / 'classify_image_graph_def.pb'
    if not model_file.exists():
        print("Downloading Inception model")
        from urllib import request
        import tarfile
        fn, _ = request.urlretrieve(INCEPTION_URL)
        with tarfile.open(fn, mode='r') as f:
            f.extract('classify_image_graph_def.pb', str(model_file.parent))
    return str(model_file)


def _handle_path(path, sess):
    if path.endswith('.npz'):
        f = np.load(path)
        m, s = f['mu'][:], f['sigma'][:]
        f.close()
    else:
        path = pathlib.Path(path)
        files = list(path.glob('*.jpg')) + list(path.glob('*.png'))
        x = np.array([imread(str(fn)).astype(np.float32) for fn in files])
        #m, s = calculate_activation_statistics(x, sess, batch_size=args.batch_size)
        m, s = calculate_activation_statistics(x, sess, batch_size=32)
        del x
    return m, s

'''
def calculate_fid_given_paths(paths, inception_path):
    #Calculates the FID of two paths
    inception_path = check_or_download_inception(inception_path)

    for p in paths:
        if not os.path.exists(p):
            raise RuntimeError("Invalid path: %s" % p)

    #tf.reset_default_graph()
    create_inception_graph(str(inception_path))
    with tf.Session(config=config) as sess:
        sess.run(tf.global_variables_initializer())
        m1, s1 = _handle_path(paths[0], sess)
        m2, s2 = _handle_path(paths[1], sess)
        fid_value = calculate_frechet_distance(m1, s1, m2, s2)
        return fid_value
'''

def calculate_fid_given_paths(paths):
    ''' Calculates the FID of two paths. '''
    #inception_path = check_or_download_inception(inception_path)

    for p in paths:
        if not os.path.exists(p):
            raise RuntimeError("Invalid path: %s" % p)

    with tf.Session(config=config) as sess:
        sess.run(tf.global_variables_initializer())
        m1, s1 = _handle_path(paths[0], sess)
        m2, s2 = _handle_path(paths[1], sess)
        fid_value = calculate_frechet_distance(m1, s1, m2, s2)
        return fid_value


def _handle_img(imgs, sess):
    m, s = calculate_activation_statistics(np.array(imgs), sess, batch_size=32)
    return m, s

def calculate_fid_given_imgs(imgs):
    ''' Calculates the FID of two paths. '''
    #inception_path = check_or_download_inception(inception_path)

    with tf.Session(config=config) as sess:
        sess.run(tf.global_variables_initializer())
        m1, s1 = _handle_img(imgs[0], sess)
        m2, s2 = _handle_img(imgs[1], sess)
        fid_value = calculate_frechet_distance(m1, s1, m2, s2)
        return fid_value


if __name__ == "__main__":
    create_inception_graph(str('/data/cuong/data/gan_paper/dataset/'))

    # pass imgs
    if 0:
        start_time = time.time()

        def get_images(filename):
            return scipy.misc.imread(filename)

        filenames = glob.glob(os.path.join('/data/cuong/data/gan_paper/dataset/cifar10', '*.*'))
        images = [get_images(filename) for filename in filenames]
        fid = calculate_fid_given_imgs([images, images])
        print('[imgs] time={}; fid={}'.format(time.time()-start_time, fid))

    if 1:
        start_time = time.time()
        fid = calculate_fid_given_paths(['/data/cuong/data/gan/datasets/mnist/mnist_rgb_32x32_all',
                                         '/data/cuong/data/gan/output/mnist_rgb_32x32__/train_9_cos_s_5/image/20000/None'])
        print('[paths] time={}; fid={}'.format(time.time()-start_time, fid))

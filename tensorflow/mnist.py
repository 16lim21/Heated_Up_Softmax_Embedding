# Copyright 2016 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

"""Functions for downloading and reading MNIST data."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import gzip
import cv2
import glob
import os
import tqdm
import tarfile

import numpy
from PIL import Image
from six.moves import xrange  # pylint: disable=redefined-builtin

from tensorflow.contrib.learn.python.learn.datasets import base
from tensorflow.python.framework import dtypes

def _read32(bytestream):
    dt = numpy.dtype(numpy.uint32).newbyteorder('>')
    return numpy.frombuffer(bytestream.read(4), dtype=dt)[0]

def resize_images(imdir, imsize): 
    """Resizes images to fit require image shape
    help on how to resize from here: https://www.tutorialkart.com/opencv/python/opencv-python-resize-image/

    Args:
    imdir: path to directory of images that needs to be resized (path ends with a '/')
    imsize: size of resized image 

    Returns:
    nothing
    """

    print('resizing {}'.format(imdir))
    DIRECTORIES = glob.glob(imdir + '**/')

    IMAGE_PATHS = glob.glob(imdir + '**/*jpg', recursive=True)
    IMAGE_PATHS.extend(glob.glob(imdir + '**/*png', recursive=True))
    IMAGE_PATHS.extend(glob.glob(imdir + '**/*gif', recursive=True))
    IMAGE_PATHS.extend(glob.glob(imdir + '**/*jpeg', recursive=True))

    try:
      os.stat(imdir + '../resized')
    except:
      os.makedirs(imdir + '../resized')
    
    for directory in tqdm.tqdm(DIRECTORIES):
      new_directory = directory.split('/')[-2]
      if new_directory != '../resized':
          try:
              os.stat(imdir + '../resized/{}/'.format(new_directory))
          except:
              os.makedirs(imdir + '../resized/{}/'.format(new_directory))

    for image_path in tqdm.tqdm(IMAGE_PATHS):
      image_name = image_path.split('/')[-1]
      
      image = cv2.imread(image_path)
      newimage = cv2.resize(image, (imsize, imsize), interpolation = cv2.INTER_AREA)
      
      imagenp = numpy.array(newimage) 

      if image_path.split('/')[-3] == 'Others': 
    
          subdir = image_path.split('/')[-2]
          try:
              os.stat(imdir + '../resized/Others/{}/'.format(subdir))
          except:
              os.makedirs(imdir + '../resized/Others/{}/'.format(subdir))

          cv2.imwrite(imdir + '../resized/Others/{}/{}'.format(subdir,image_name), imagenp)
      else:
          class_name = image_path.split('/')[-2]
          if not cv2.imwrite(imdir + '../resized/{}/{}'.format(class_name,image_name), imagenp):
              print('didnt write ' + imdir + '../resized/{}/{}'.format(class_name,image_name))

    return(imdir + '../resized/')

def extract_labels(imdir):
    """
    Extract the labels into a 1D uint8 numpy array [index].
    Args:
    imdir: Path to directory of images (labels given by subdirectories containing each image).

    Returns:
    labelsnp: a 1D uint8 numpy array.
    """
    print('extracting labels')
    IMAGE_PATHS = glob.glob(imdir + '**/*jpg', recursive=True)
    IMAGE_PATHS.extend(glob.glob(imdir + '**/*png', recursive=True))
    IMAGE_PATHS.extend(glob.glob(imdir + '**/*gif', recursive=True))
    IMAGE_PATHS.extend(glob.glob(imdir + '**/*jpeg', recursive=True))

    labels = []
    for image_path in tqdm.tqdm(IMAGE_PATHS):
      subdir = image_path.split('/')[-2]

      if subdir == 'Aryan Nations Flag':
          labels.append(1)
      elif subdir == 'III Percenters Flag':
          labels.append(2)
      elif subdir == 'Iron Cross Flag':
          labels.append(3)
      elif subdir == 'Odal Rune Flag':
          labels.append(4)
      else: 
          labels.append(5)

    print(len(labels))
    labelnp = numpy.array(labels, dtype=numpy.uint8)
    print(labelnp.shape)
    return labelnp


def extract_images(imdir):
    """
    Extract the images into a 4D uint8 numpy array [index, y, x, depth]

    Args:
    imdir: Path to directory of images (labels given by subdirectories containing each image).
   
    Returns:
    datanp: A 4D uint8 numpy array [index, y, x, depth].
    """
    print('Extracting {}'.format(imdir))
    IMAGE_PATHS = glob.glob(imdir + '**/*jpg', recursive=True)
    IMAGE_PATHS.extend(glob.glob(imdir + '**/*png', recursive=True))
    IMAGE_PATHS.extend(glob.glob(imdir + '**/*gif', recursive=True))
    IMAGE_PATHS.extend(glob.glob(imdir + '**/*jpeg', recursive=True))

    num_images = len(IMAGE_PATHS)
    print(num_images)
    data = []

    for image_path in tqdm.tqdm(IMAGE_PATHS):
      image = cv2.imread(image_path)
      data.append(image)

    datanp = numpy.array(data)
    print(datanp.shape)
    return datanp

def dense_to_one_hot(labels_dense, num_classes):
    """Convert class labels from scalars to one-hot vectors."""
    num_labels = labels_dense.shape[0]
    index_offset = numpy.arange(num_labels) * num_classes
    labels_one_hot = numpy.zeros((num_labels, num_classes))
    labels_one_hot.flat[index_offset + labels_dense.ravel()] = 1
    return labels_one_hot

class DataSet(object):

    def __init__(self,
               images,
               labels,
               fake_data=False,
               one_hot=False,
               dtype=dtypes.float32,
               reshape=True):
        """Construct a DataSet.
        one_hot arg is used only if fake_data is true.  `dtype` can be either
        `uint8` to leave the input as `[0, 255]`, or `float32` to rescale into
        `[0, 1]`.
        """
        dtype = dtypes.as_dtype(dtype).base_dtype
        if dtype not in (dtypes.uint8, dtypes.float32):
          raise TypeError('Invalid image dtype %r, expected uint8 or float32' %
                          dtype)
        if fake_data:
          self._num_examples = 10000
          self.one_hot = one_hot
        else:
          assert images.shape[0] == labels.shape[0], (
              'images.shape: %s labels.shape: %s' % (images.shape, labels.shape))
          self._num_examples = images.shape[0]

          # Convert shape from [num examples, rows, columns, depth]
          # to [num examples, rows*columns] (assuming depth == 1)
          if reshape:
            assert images.shape[3] == 1
            images = images.reshape(images.shape[0],
                                    images.shape[1] * images.shape[2])
          if dtype == dtypes.float32:
            # Convert from [0, 255] -> [0.0, 1.0].
            images = images.astype(numpy.float32)
            images = numpy.multiply(images, 1.0 / 255.0)
            images = images*2.0 - 1.0
        self._images = images
        self._labels = labels
        self._epochs_completed = 0
        self._index_in_epoch = 0

    @property
    def images(self):
        return self._images

    @property
    def labels(self):
        return self._labels

    @property
    def num_examples(self):
        return self._num_examples

    @property
    def epochs_completed(self):
        return self._epochs_completed

    def next_batch(self, batch_size, fake_data=False):
        """Return the next `batch_size` examples from this data set."""
        if fake_data:
          fake_image = [1] * 784
          if self.one_hot:
            fake_label = [1] + [0] * 9
          else:
            fake_label = 0
          return [fake_image for _ in xrange(batch_size)], [
              fake_label for _ in xrange(batch_size)
          ]
        start = self._index_in_epoch
        self._index_in_epoch += batch_size
        if self._index_in_epoch > self._num_examples:
          # Finished epoch
          self._epochs_completed += 1
          # Shuffle the data
          perm = numpy.arange(self._num_examples)
          numpy.random.shuffle(perm)
          self._images = self._images[perm]
          self._labels = self._labels[perm]
          # Start next epoch
          start = 0
          self._index_in_epoch = batch_size
          assert batch_size <= self._num_examples
        end = self._index_in_epoch
        return self._images[start:end], self._labels[start:end]

    def next_batch_test(self, batch_size, fake_data=False):
        """Return the next `batch_size` examples from this data set."""
        start = self._index_in_epoch
        self._index_in_epoch += batch_size
        if self._index_in_epoch > self._num_examples:
          # Finished epoch
          self._epochs_completed += 1
          # Start next epoch
          start = 0
          self._index_in_epoch = batch_size
          assert batch_size <= self._num_examples
        end = self._index_in_epoch
        return self._images[start:end], self._labels[start:end]

def read_data_sets(train_dir,
                   fake_data=False,
                   one_hot=False,
                   dtype=dtypes.float32,
                   reshape=True,
                   validation_size=5000):
    if fake_data:

        def fake():
            return DataSet([], [], fake_data=True, one_hot=one_hot, dtype=dtype)

        train = fake()
        validation = fake()
        test = fake()
        return base.Datasets(train=train, validation=validation, test=test)

    TRAIN_IMAGES = '/home/michael/data/crop/'
    IMAGE_SHAPE = 28 #28x28 pixel images

    RESIZED_IMAGES = resize_images(TRAIN_IMAGES, IMAGE_SHAPE)

    train_images = extract_images(RESIZED_IMAGES)

    train_labels = extract_labels(RESIZED_IMAGES)

    train = DataSet(train_images, train_labels, dtype=dtype, reshape=reshape)
    test = DataSet([], [], fake_data=True, one_hot=one_hot, dtype=dtype)
    validation = DataSet([], [], fake_data=True, one_hot=one_hot, dtype=dtype) 

    return base.Datasets(train=train, validation=validation, test=test)


def load_mnist(train_dir='MNIST-data'):
    return read_data_sets(train_dir)

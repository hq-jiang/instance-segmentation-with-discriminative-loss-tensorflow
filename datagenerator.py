import os
from glob import glob
import numpy as np
import scipy.misc
import random
from sklearn.utils import shuffle
import shutil
import time
import tensorflow as tf
import cv2
import matplotlib.pyplot as plt


### Mean and std deviation for whole training data set (RGB format)
mean = np.array([92.14031982, 103.20146942, 103.47182465])
std = np.array([49.157, 54.9057, 59.4065])

INSTANCE_COLORS = [np.array([0,0,0]),
                   np.array([20.,20.,20.]),
                   np.array([70.,70.,70.]),
                   np.array([120.,120.,120.]),
                   np.array([170.,170.,170.]),
                   np.array([220.,220.,220.])
                   ]

def get_batches_fn(batch_size, image_shape, image_paths, label_paths):
    """
    Create batches of training data
    :param batch_size: Batch Size
    :return: Batches of training data
    """

    #print ('Number of total labels:', len(label_paths))
    assert len(image_paths)==len(label_paths), 'Number of images and labels do not match'

    image_paths.sort()
    label_paths.sort()

    #image_paths = image_paths[:10]
    #label_paths = label_paths[:10]

    image_paths, label_paths = shuffle(image_paths, label_paths)
    for batch_i in range(0, len(image_paths), batch_size):
        images = []
        gt_images = []
        for image_file, gt_image_file in zip(image_paths[batch_i:batch_i+batch_size], label_paths[batch_i:batch_i+batch_size]):

            image = cv2.resize(cv2.imread(image_file), image_shape, interpolation=cv2.INTER_LINEAR)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            #image = (image.astype(np.float32)-mean)/std

            gt_image = cv2.imread(gt_image_file, cv2.IMREAD_COLOR)
            gt_image = cv2.resize(gt_image[:,:,0], image_shape, interpolation=cv2.INTER_NEAREST)

            images.append(image)
            gt_images.append(gt_image)

        yield np.array(images), np.array(gt_images)


def get_validation_batch(image_shape):
    valid_image_paths = [os.path.join('.','data','images','0000.png')]

    valid_label_paths = [os.path.join('.','data','labels','0000.png')]

    images = []
    gt_images = []
    for image_file, gt_image_file in zip(valid_image_paths, valid_label_paths):

        image = cv2.resize(cv2.imread(image_file), image_shape, interpolation=cv2.INTER_LINEAR)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        #image = (image.astype(np.float32)-mean)/std
        
        gt_image = cv2.imread(gt_image_file, cv2.IMREAD_COLOR)
        gt_image = cv2.resize(gt_image[:,:,0], image_shape, interpolation=cv2.INTER_NEAREST)

        images.append(image)
        gt_images.append(gt_image)

    return np.array(images), np.array(gt_images)



if __name__=="__main__":
    image_shape = (160, 320)
    batch_size = 1
    get_batches_fn = gen_batch_function(('../tusimple_api/clean_data'), image_shape)
    images, gt_images = get_batches_fn(batch_size).next()

    print (gt_images.shape)
    assert len(images)==len(gt_images) and len(images)==batch_size

    for i in range(batch_size):
        plt.figure(i)
        plt.subplot(211)
        plt.imshow(images[i])
        plt.subplot(212)
        plt.imshow(gt_images[i], cmap='gray')
        print ('Unique colors', np.unique(gt_images[i]))
        assert np.unique(gt_images[i]).size<=6, 'To many instance colors'

    plt.show()
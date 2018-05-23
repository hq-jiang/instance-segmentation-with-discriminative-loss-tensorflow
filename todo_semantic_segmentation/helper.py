import os
from glob import glob
import numpy as np
import random
from sklearn.utils import shuffle
import shutil
import time
import tensorflow as tf
import cv2


### Mean and std deviation for whole training data set (RGB format)
mean = 0.#np.array([92.14031982, 103.20146942, 103.47182465])
std = 1.#np.array([49.157, 54.9057, 59.4065])


def get_batches_fn(batch_size, image_shape, image_paths, label_paths):
    """
    Create batches of training data
    :param batch_size: Batch Size
    :param image_shape: input image shape
    :param image_paths: list of paths for training or validation
    :param label_paths: list of paths for training or validation
    :return: Batches of training data
    """

    image_paths.sort()
    label_paths.sort()

    #image_paths = image_paths[:20]
    #label_paths = label_paths[:20]

    background_color = np.array([0, 0, 0])

    image_paths, label_paths = shuffle(image_paths, label_paths)

    for batch_i in range(0, len(image_paths), batch_size):
        images = []
        gt_images = []
        for image_file, gt_image_file in zip(image_paths[batch_i:batch_i+batch_size], label_paths[batch_i:batch_i+batch_size]):

            ### Image preprocessing
            image = cv2.resize(cv2.imread(image_file), (image_shape[1], image_shape[0]), cv2.INTER_LINEAR)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            #image = (image.astype(np.float32)-mean)/std
            
            ### Label preprocessing
            gt_image = cv2.resize(cv2.imread(gt_image_file, cv2.IMREAD_COLOR), (image_shape[1], image_shape[0]), cv2.INTER_NEAREST)
            gt_bg = np.all(gt_image == background_color, axis=2)
            gt_bg = gt_bg.reshape(gt_bg.shape[0], gt_bg.shape[1], 1)
            gt_image = np.concatenate((gt_bg, np.invert(gt_bg)), axis=2)

            images.append(image)
            gt_images.append(gt_image)

        yield np.array(images), np.array(gt_images)


# Source http://www.pyimagesearch.com/2015/10/05/opencv-gamma-correction/
def adjust_gamma(image, gamma=1.0):
    # build a lookup table mapping the pixel values [0, 255] to
    # their adjusted gamma values
    invGamma = 1.0 / gamma
    table = np.array([((i / 255.0) ** invGamma) * 255
        for i in np.arange(0, 256)]).astype("float32")
 
    # apply gamma correction using the lookup table
    return cv2.LUT(image, table)


def gen_test_output(sess, logits, keep_prob, image_pl, data_folder, image_shape):
    """
    Generate test output using the test images
    :param sess: TF session
    :param logits: TF Tensor for the logits
    :param keep_prob: TF Placeholder for the dropout keep robability
    :param image_pl: TF Placeholder for the image placeholder
    :param data_folder: Path to the folder that contains the datasets
    :param image_shape: Tuple - Shape of image
    :return: Output for for each test image
    """
    for image_file in glob(os.path.join(data_folder, 'test_images', '*.png'))[:40]:
        image = cv2.resize(cv2.imread(image_file), (image_shape[1], image_shape[0]), cv2.INTER_LINEAR)

        ### Run inference
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        img_origin = image.copy()
        #image = (image.astype(np.float32)-mean)/std

        im_softmax = sess.run(
            tf.nn.softmax(logits),
            {keep_prob: 1.0, image_pl: [image]})

        ### Threshholding
        im_softmax = im_softmax[:, 1].reshape(image_shape[0], image_shape[1])
        mask_ind = np.where(im_softmax > 0.3)

        ### Overlay class mask over original image
        blend = np.zeros_like(img_origin)
        blend[mask_ind] = np.array([0,255,0])
        blended = cv2.addWeighted(img_origin, 1, blend, 0.7, 0)
        blended = cv2.cvtColor(blended, cv2.COLOR_BGR2RGB)

        yield os.path.basename(image_file), np.array(blended)


def save_inference_samples(runs_dir, data_dir, sess, image_shape, logits, keep_prob, input_image):
    # Make folder for current run
    output_dir = os.path.join(runs_dir, str(time.time()))
    if os.path.exists(output_dir):
        shutil.rmtree(output_dir)
    os.makedirs(output_dir)

    # Run NN on test images and save them to HD
    print('Training Finished. Saving test images to: {}'.format(output_dir))
    image_outputs = gen_test_output(
        sess, logits, keep_prob, input_image, data_dir, image_shape)
    for name, image in image_outputs:
        cv2.imwrite(os.path.join(output_dir, name), image)
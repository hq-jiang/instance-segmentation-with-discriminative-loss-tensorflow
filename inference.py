import os
import os.path

import argparse
from glob import glob
import numpy as np
import cv2
import tensorflow as tf

slim = tf.contrib.slim
from enet import ENet, ENet_arg_scope
from clustering import cluster, get_instance_masks, save_instance_masks
import time


def rebuild_graph(sess, checkpoint_dir, input_image, batch_size, feature_dim):
    checkpoint = tf.train.latest_checkpoint(checkpoint_dir)

    num_initial_blocks = 1
    skip_connections = False
    stage_two_repeat = 2
    
    with slim.arg_scope(ENet_arg_scope()):
        _, _ = ENet(input_image,
                     num_classes=12,
                     batch_size=batch_size,
                     is_training=True,
                     reuse=None,
                     num_initial_blocks=num_initial_blocks,
                     stage_two_repeat=stage_two_repeat,
                     skip_connections=skip_connections)

    graph = tf.get_default_graph()
    last_prelu = graph.get_tensor_by_name('ENet/bottleneck5_1_last_prelu:0')
    logits = slim.conv2d_transpose(last_prelu, feature_dim, [2,2], stride=2, 
                                    scope='Instance/transfer_layer/conv2d_transpose')

    variables_to_restore = slim.get_variables_to_restore()
    saver = tf.train.Saver(variables_to_restore)
    saver.restore(sess, checkpoint)

    return logits

def save_image_with_features_as_color(pred):
    p_min = np.min(pred)
    p_max = np.max(pred)
    pred = (pred - p_min)*255/(p_max-p_min)
    pred = pred.astype(np.uint8)
    output_file_name = os.path.join(output_dir, 'color_{}.png'.format(str(i).zfill(4)))
    cv2.imwrite(output_file_name, np.squeeze(pred))


if __name__=='__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('-m','--modeldir', default='trained_model', help="Directory of trained model")
    parser.add_argument('-i', '--indir', default=os.path.join('inference_test', 'images'), help='Input image directory (jpg format)')
    parser.add_argument('o', '--outdir', default=os.path.join('inference_test', 'results'), help='Output directory for inference images')
    args = parser.parse_args()

    data_dir = args.indir
    output_dir = args.outdir
    checkpoint_dir = args.modeldir

    if not os.path.isdir(output_dir):
        os.mkdir(output_dir)

    image_paths = glob(os.path.join(data_dir, '*.jpg'))
    image_paths.sort()

    num_images = len(image_paths)

    image_shape = (512, 512)
    batch_size = 1
    feature_dim = 3

    ### Limit GPU memory usage due to occasional crashes
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    config.gpu_options.per_process_gpu_memory_fraction = 0.5 
    
    with tf.Session(config=config) as sess:

        input_image = tf.placeholder(tf.float32, shape=(None, image_shape[1], image_shape[0], 3))
        logits = rebuild_graph(sess, checkpoint_dir, input_image, batch_size, feature_dim)

        inference_time = 0
        cluster_time = 0
        for i, path in enumerate(image_paths):
            
            image = cv2.resize(cv2.imread(path), image_shape, interpolation=cv2.INTER_LINEAR)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            image = np.expand_dims(image, axis=0)

            tic = time.time()
            prediction = sess.run(logits, feed_dict={input_image: image})
            pred_time = time.time()-tic
            print 'Inference time', pred_time
            inference_time += pred_time
            

            pred_color = np.squeeze(prediction.copy())
            print 'Save prediction', i 
            #save_image_with_features_as_color(pred_color)
            
            pred_cluster = prediction.copy()
            tic = time.time()
            instance_mask = get_instance_masks(pred_cluster, bandwidth=1.)[0]
            #save_instance_masks(prediction, output_dir, bandwidth=1., count=i)
            print instance_mask.shape
            output_file_name = os.path.join(output_dir, 'cluster_{}.png'.format(str(i).zfill(4)))
            colors, counts = np.unique(instance_mask.reshape(image_shape[0]*image_shape[1],3), 
                                            return_counts=True, axis=0)
            max_count = 0
            for color, count in zip(colors, counts):
                if count > max_count:
                    max_count = count
                    bg_color = color
            ind = np.where(instance_mask==bg_color)
            instance_mask[ind] = 0.
            instance_mask = cv2.addWeighted(np.squeeze(image), 1, instance_mask, 0.3, 0)
            instance_mask = cv2.resize(instance_mask, (1280,720))
            clust_time = time.time()-tic
            print 'Total time', cluster_time
            cluster_time += clust_time
            cv2.imwrite(output_file_name, cv2.cvtColor(instance_mask, cv2.COLOR_RGB2BGR))

        print 'Mean inference time:', inference_time/num_images, 'fps:', num_images/inference_time
        print 'Mean cluster time:', cluster_time/num_images, 'fps:', num_images/cluster_time
        print 'Mean total time:', cluster_time/num_images + inference_time/num_images, 'fps:', 1./(cluster_time/num_images + inference_time/num_images)

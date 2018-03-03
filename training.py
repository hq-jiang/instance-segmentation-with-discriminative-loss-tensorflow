import os
import os.path

import sys
import warnings
import copy
from glob import glob
import argparse

import numpy as np
import cv2
import tensorflow as tf
from sklearn.model_selection import train_test_split
from tensorflow.contrib.layers.python.layers import initializers

import matplotlib as mpl
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.cm as cm

import utils
from loss import discriminative_loss
import datagenerator
import visualization
import clustering



def run():
    parser = argparse.ArgumentParser()
    # Directories
    parser.add_argument('-s','--srcdir', default='data', help="Source directory of TuSimple dataset")
    parser.add_argument('-m', '--modeldir', default='pretrained_semantic_model', help="Output directory of extracted data")
    parser.add_argument('-o', '--outdir', default='saved_model', help="Directory for trained model")
    parser.add_argument('-l', '--logdir', default='log', help="Log directory for tensorboard and evaluation files")
    # Hyperparameters
    parser.add_argument('--epochs', type=int, default=50, help="Number of epochs")
    parser.add_argument('--var', type=float, default=1., help="Weight of variance loss")
    parser.add_argument('--dist', type=float, default=1., help="Weight of distance loss")
    parser.add_argument('--reg', type=float, default=0.001, help="Weight of regularization loss")
    parser.add_argument('--dvar', type=float, default=0.5, help="Cutoff variance")
    parser.add_argument('--ddist', type=float, default=1.5, help="Cutoff distance")

    args = parser.parse_args()

    if not os.path.isdir(args.srcdir):
        raise IOError('Directory does not exist')
    if not os.path.isdir(args.modeldir):
        raise IOError('Directory does not exist')
    if not os.path.isdir(args.logdir):
        os.mkdir(args.logdir)

    image_shape = (512, 512)
    data_dir = args.srcdir #os.path.join('.', 'data')
    model_dir = args.modeldir
    output_dir = args.outdir
    log_dir = args.logdir

    image_paths = glob(os.path.join(data_dir, 'images', '*.png'))
    label_paths = glob(os.path.join(data_dir, 'labels', '*.png'))

    image_paths.sort()
    label_paths.sort()
    
    #image_paths = image_paths[0:10]
    #label_paths = label_paths[0:10]

    X_train, X_valid, y_train, y_valid = train_test_split(image_paths, label_paths, test_size=0.10, random_state=42)

    print ('Number of train samples', len(y_train))
    print ('Number of valid samples', len(y_valid))


    ### Debugging
    debug_clustering = True
    bandwidth = 0.7
    cluster_cycle = 5000
    eval_cycle=1000
    save_cycle=15000

    ### Hyperparameters
    epochs = args.epochs
    batch_size = 1
    starter_learning_rate = 1e-4
    learning_rate_decay_rate = 0.96
    learning_rate_decay_interval = 5000

    feature_dim = 3
    param_var = args.var
    param_dist = args.dist
    param_reg = args.reg
    delta_v = args.dvar
    delta_d = args.ddist

    param_string = 'fdim'+str(feature_dim)+'_var'+str(param_var)+'_dist'+str(param_dist)+'_reg'+str(param_reg) \
                +'_dv'+str(delta_v)+'_dd'+str(delta_d) \
                +'_lr'+str(starter_learning_rate)+'_btch'+str(batch_size)

    if not os.path.exists(os.path.join(log_dir, param_string)):
        os.makedirs(os.path.join(log_dir, param_string))            

    
    ### Limit GPU memory usage due to ocassional crashes
    config = tf.ConfigProto()
    #config.gpu_options.allow_growth = True
    #config.gpu_options.per_process_gpu_memory_fraction = 0.5 


    with tf.Session(config=config) as sess:

        ### Build network
        input_image = tf.placeholder(tf.float32, shape=(None, image_shape[1], image_shape[0], 3))
        correct_label = tf.placeholder(dtype=tf.float32, shape=(None, image_shape[1], image_shape[0]))

        last_prelu = utils.load_enet(sess, model_dir, input_image, batch_size)
        prediction = utils.add_transfer_layers_and_initialize(sess, last_prelu, feature_dim)

        print ('Number of parameters in the model', utils.count_parameters())
        ### Set up learning rate decay
        global_step = tf.Variable(0, trainable=False)
        sess.run(global_step.initializer)
        learning_rate = tf.train.exponential_decay(starter_learning_rate, global_step,
                                               learning_rate_decay_interval, learning_rate_decay_rate, staircase=True)
        
        ### Set variables to train
        trainables = utils.get_trainable_variables_and_initialize(sess, debug=False)

        ### Optimization operations
        disc_loss, l_var, l_dist, l_reg = discriminative_loss(prediction, correct_label, feature_dim, image_shape, 
                                                    delta_v, delta_d, param_var, param_dist, param_reg)
        with tf.name_scope('Instance/Adam'):
            train_op = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(disc_loss, var_list=trainables, global_step=global_step)
        adam_initializers = [var.initializer for var in tf.global_variables() if 'Adam' in var.name]
        sess.run(adam_initializers)

        
        ### Collect summaries
        summary_op_train, summary_op_valid = utils.collect_summaries(disc_loss, l_var, l_dist, l_reg, input_image, prediction, correct_label)

        train_writer = tf.summary.FileWriter(log_dir)

        
        ### Check if image and labels match
        valid_image_chosen, valid_label_chosen = datagenerator.get_validation_batch(data_dir, image_shape)
        print (valid_image_chosen.shape)
        #visualization.save_image_overlay(valid_image_chosen.copy(), valid_label_chosen.copy())


        ### Training pipeline
        saver = tf.train.Saver()
        step_train=0
        step_valid=0
        for epoch in range(epochs):
            print ('epoch', epoch)
            
            train_loss = 0
            for image, label in datagenerator.get_batches_fn(batch_size, image_shape, X_train, y_train):

                lr = sess.run(learning_rate)
                
                if (step_train%eval_cycle!=0):
                    ### Training
                    _, step_prediction, step_loss, step_l_var, step_l_dist, step_l_reg = sess.run([
                                            train_op,
                                            prediction,
                                            disc_loss,
                                            l_var,
                                            l_dist,
                                            l_reg], 
                                            feed_dict={input_image: image, correct_label: label})
                else:
                    # First run normal training step and record summaries
                    print ('Evaluating on chosen images ...')
                    _, summary, step_prediction, step_loss, step_l_var, step_l_dist, step_l_reg = sess.run([
                                        train_op,
                                        summary_op_train,
                                        prediction,
                                        disc_loss,
                                        l_var,
                                        l_dist,
                                        l_reg], 
                                        feed_dict={input_image: image, correct_label: label})                 
                    train_writer.add_summary(summary, step_train)

                    # Then run model on some chosen images and save feature space visualization
                    valid_pred = sess.run(prediction, feed_dict={input_image: np.expand_dims(valid_image_chosen[0], axis=0), 
                                                                 correct_label: np.expand_dims(valid_label_chosen[0], axis=0)})
                    visualization.evaluate_scatter_plot(log_dir, valid_pred, valid_label_chosen, feature_dim, param_string, step_train)
                    
                    # Perform mean-shift clustering on prediction
                    if (step_train%cluster_cycle==0):
                        if debug_clustering:
                            instance_masks = clustering.get_instance_masks(valid_pred, bandwidth)
                            for img_id, mask in enumerate(instance_masks):
                                cv2.imwrite(os.path.join(log_dir, param_string, 'cluster_{}_{}.png'.format(str(step_train).zfill(6), str(img_id)) ), mask)

                step_train += 1
                
                ### Save intermediate model
                if (step_train%save_cycle==(save_cycle-1)):
                    try:
                        print ('Saving model ...')
                        saver.save(sess, os.path.join(output_dir, 'model.ckpt'), global_step=step_train)
                    except:
                        print ('FAILED saving model')
                #print 'gradient', step_gradient
                print ('step', step_train, '\tloss', step_loss, '\tl_var', step_l_var, '\tl_dist', step_l_dist, '\tl_reg', step_l_reg, '\tcurrent lr', lr)


            ### Regular validation
            print ('Evaluating current model ...')
            for image, label in datagenerator.get_batches_fn(batch_size, image_shape, X_valid, y_valid):
                if step_valid%100==0:
                    summary, step_prediction, step_loss, step_l_var, step_l_dist, step_l_reg = sess.run([
                                            summary_op_valid, 
                                            prediction,
                                            disc_loss,
                                            l_var,
                                            l_dist,
                                            l_reg], 
                                            feed_dict={input_image: image, correct_label: label})
                    train_writer.add_summary(summary, step_valid)
                else:
                    step_prediction, step_loss, step_l_var, step_l_dist, step_l_reg = sess.run([
                                            prediction,
                                            disc_loss,
                                            l_var,
                                            l_dist,
                                            l_reg], 
                                            feed_dict={input_image: image, correct_label: label})
                step_valid += 1


                print ('step_valid', step_valid, 'valid loss', step_loss, '\tvalid l_var', step_l_var, '\tvalid l_dist', step_l_dist, '\tvalid l_reg', step_l_reg)

        saver = tf.train.Saver()
        saver.save(sess, os.path.join(output_dir, 'model.ckpt'), global_step=step_train)


if __name__ == '__main__':
    run()

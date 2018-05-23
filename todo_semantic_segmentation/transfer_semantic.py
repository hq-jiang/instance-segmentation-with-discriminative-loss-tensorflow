import os.path
import tensorflow as tf
import helper
import warnings
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
from glob import glob
from enet import ENet, ENet_arg_scope
from tensorflow.contrib.layers.python.layers import initializers
import numpy as np
slim = tf.contrib.slim

# Check for a GPU
if not tf.test.gpu_device_name():
    warnings.warn('No GPU found. Please use a GPU to train your neural network.')
else:
    print('Default GPU Device: {}'.format(tf.test.gpu_device_name()))


def load_enet(sess, checkpoint_dir, input_image, batch_size, num_classes):
    checkpoint = tf.train.latest_checkpoint(checkpoint_dir)

    num_initial_blocks = 1
    skip_connections = False
    stage_two_repeat = 2

    with slim.arg_scope(ENet_arg_scope()):
        logits, _ = ENet(input_image,
                                     num_classes=12,
                                     batch_size=batch_size,
                                     is_training=True,
                                     reuse=None,
                                     num_initial_blocks=num_initial_blocks,
                                     stage_two_repeat=stage_two_repeat,
                                     skip_connections=skip_connections)


    variables_to_restore = slim.get_variables_to_restore()
    saver = tf.train.Saver(variables_to_restore)
    saver.restore(sess, checkpoint)
    graph = tf.get_default_graph()

    last_prelu = graph.get_tensor_by_name('ENet/bottleneck5_1_last_prelu:0')
    output = slim.conv2d_transpose(last_prelu, num_classes, [2,2], stride=2, 
                                    weights_initializer=initializers.xavier_initializer(), 
                                    scope='Semantic/transfer_layer/conv2d_transpose')

    probabilities = tf.nn.softmax(output, name='Semantic/transfer_layer/logits_to_softmax')

    with tf.variable_scope('', reuse=True):
        weight = tf.get_variable('Semantic/transfer_layer/conv2d_transpose/weights')
        bias = tf.get_variable('Semantic/transfer_layer/conv2d_transpose/biases')
        sess.run([weight.initializer, bias.initializer])

    return output, probabilities



def optimize(sess, logits, correct_label, learning_rate, num_classes, trainables, global_step):
    """
    Build the TensorFLow loss and optimizer operations.
    :param nn_last_layer: TF Tensor of the last layer in the neural network
    :param correct_label: TF Placeholder for the correct label image
    :param learning_rate: TF Placeholder for the learning rate
    :param num_classes: Number of classes to classify
    :return: Tuple of (logits, train_op, cross_entropy_loss)
    """
    # TODO: Implement function

    #correct_label = tf.reshape(correct_label, (-1, num_classes))
    #logits = tf.reshape(nn_last_layer, (-1, num_classes))
    
    weights = correct_label * np.array([1., 40.])
    weights = tf.reduce_sum(weights, axis=3)
    loss = tf.reduce_mean(tf.losses.softmax_cross_entropy(onehot_labels=correct_label, logits=logits, weights=weights))


    #loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=correct_label, logits=logits))
    with tf.name_scope('Semantic/Adam'):
        train_op = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(loss, var_list=trainables, global_step=global_step)
    adam_initializers = [var.initializer for var in tf.global_variables() if 'Adam' in var.name]
    sess.run(adam_initializers)
    return logits, train_op, loss


def run():

    ### Initialization
    image_shape = (512, 512) # (width, height)
    model_dir = '../checkpoint'
    data_dir = '../../tusimple_api/clean_data'
    log_dir = './log'
    output_dir = './saved_model'

    num_classes = 2
    epochs = 20
    batch_size = 1
    starter_learning_rate = 1e-4
    learning_rate_decay_interval = 500
    learning_rate_decay_rate = 0.96
    ### Load images and labels
    image_paths = glob(os.path.join(data_dir, 'images', '*.png'))
    label_paths = glob(os.path.join(data_dir, 'labels', '*.png'))

    #image_paths = image_paths[:20]
    #label_paths = label_paths[:20]

    X_train, X_valid, y_train, y_valid = train_test_split(image_paths, label_paths, test_size=0.20, random_state=42)

    ### Limit GPU memory usage due to ocassional crashes
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    config.gpu_options.per_process_gpu_memory_fraction = 0.7

    

    with tf.Session(config=config) as sess:
    
        ### Load ENet and replace layers
        input_image = tf.placeholder(tf.float32, shape=[batch_size, image_shape[1], image_shape[0], 3])
        correct_label = tf.placeholder(dtype=tf.float32, shape=(None, image_shape[1], image_shape[0], 2), name='Semantic/input_image')

        logits, probabilities = load_enet(sess, model_dir, input_image, batch_size, num_classes)
        predictions_val = tf.argmax(probabilities, axis=-1)
        predictions_val = tf.cast(predictions_val, dtype=tf.float32)
        predictions_val = tf.reshape(predictions_val, shape=[batch_size, image_shape[1], image_shape[0], 1])
        

        ### Set up learning rate decay
        global_step = tf.Variable(0, trainable=False)
        sess.run(global_step.initializer)
        learning_rate = tf.train.exponential_decay(starter_learning_rate, global_step,
                                               learning_rate_decay_interval, learning_rate_decay_rate, staircase=True)

        for i, var in enumerate(tf.trainable_variables()):
            print i, var
            tf.summary.histogram(var.name, var)

        trainables = [var for var in tf.trainable_variables() if 'bias' not in var.name and 'ENet/fullconv' not in var.name]

        ### Print variables which are actually trained
        for var in trainables:
            print var

        logits, train_op, cross_entropy_loss = optimize(sess, logits, correct_label, learning_rate, num_classes, trainables, global_step)
        tf.summary.scalar('training_loss', cross_entropy_loss)
        tf.summary.image('Images/Validation_original_image', input_image, max_outputs=1)
        tf.summary.image('Images/Validation_segmentation_output', predictions_val, max_outputs=1)
        summary_train = tf.summary.merge_all()
        summary_valid = tf.summary.scalar('validation_loss', cross_entropy_loss)

        train_writer = tf.summary.FileWriter(log_dir)
        
        saver = tf.train.Saver()

        ### Training pipeline
        step_train = 0
        step_valid = 0
        summary_cycle = 10
        for epoch in range(epochs):
            print 'epoch', epoch
            print 'training ...'
            train_loss = 0
            for image, label in helper.get_batches_fn(batch_size, image_shape, X_train, y_train):
                # Training
                lr = sess.run(learning_rate)
                if step_train%summary_cycle==0:
                    _, summary, loss = sess.run([train_op, summary_train, cross_entropy_loss], 
                        feed_dict={input_image: image, correct_label: label})
                    train_writer.add_summary(summary, step_train)
                    print 'epoch', epoch, '\t step_train', step_train, '\t batch loss', loss, '\t current learning rate', lr
                else:
                    _, loss = sess.run([train_op, cross_entropy_loss], 
                        feed_dict={input_image: image, correct_label: label})
                step_train+=1
                train_loss += loss

                if (step_train%5000==4999):
                    saver.save(sess, os.path.join(output_dir, 'model.ckpt'), global_step=global_step)

                
            print 'train epoch loss', train_loss

            print 'validating ...'
            valid_loss = 0
            for image, label in helper.get_batches_fn(batch_size, image_shape, X_valid, y_valid):
                # Validation
                if step_valid%summary_cycle==0:
                    summary, loss = sess.run([summary_valid, cross_entropy_loss], 
                        feed_dict={input_image: image, correct_label: label})
                    train_writer.add_summary(summary, step_valid)
                    print 'batch loss', loss
                else:
                    loss = sess.run(cross_entropy_loss, 
                        feed_dict={input_image: image, correct_label: label})

                valid_loss += loss
                step_valid+=1

            print 'valid epoch loss', valid_loss




if __name__ == '__main__':
    run()


import sys
sys.path.append('../base')
import tensorflow as tf
from enet import ENet, ENet_arg_scope
slim = tf.contrib.slim

def load_enet(sess, checkpoint_dir, input_image, batch_size):
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

    variables_to_restore = slim.get_variables_to_restore()
    saver = tf.train.Saver(variables_to_restore)
    saver.restore(sess, checkpoint)

    graph = tf.get_default_graph()
    last_prelu = graph.get_tensor_by_name('ENet/bottleneck5_1_last_prelu:0')
    return last_prelu

def add_transfer_layers_and_initialize(sess, last_prelu, feature_dim):

    logits = slim.conv2d_transpose(last_prelu, feature_dim, [2,2], stride=2, 
                                    biases_initializer=tf.constant_initializer(10.0), 
                                    weights_initializer=tf.contrib.layers.xavier_initializer(), 
                                    scope='Instance/transfer_layer/conv2d_transpose')

    with tf.variable_scope('', reuse=True):
        weight = tf.get_variable('Instance/transfer_layer/conv2d_transpose/weights')
        bias = tf.get_variable('Instance/transfer_layer/conv2d_transpose/biases')
        sess.run([weight.initializer, bias.initializer])

    return logits

def get_trainable_variables_and_initialize(sess, debug=False):
    ''' Determine which variables to train and reset
    We accumulate all variables we want to train in a list to pass it to the optimizer.
    As mentioned in the 'Fast Scene Understanding' paper we want to freeze stage 1 and 2
    from the ENet and train stage 3-5. The variables from the later stages are reseted.
    Additionally all biases are not trained.
    
    :return: trainables: List of variables we want to train

    '''
    ### Freeze shared encode
    trainables = [var for var in tf.trainable_variables() if 'bias' not in var.name]# and \
                                                             #'ENet/fullconv' not in var.name and \
                                                             #'ENet/initial_block_1' not in var.name and \
                                                             #'ENet/bottleneck1' not in var.name and \
                                                             #'ENet/bottleneck2' not in var.name
                                                             #]
    if debug:
        print 'All trainable variables'
        for i, var in enumerate(tf.trainable_variables()):
            print i, var
        print 'variables which are actually trained'
        for var in trainables:
            print var

    ### Design choice: reset decoder network to default initialize weights
    # Reset all trainable variables
    #sess.run(tf.variables_initializer(trainables))
    # Additionally reset all biases in the decoder network
    # Encoder retains pretrained biases
    sess.run(tf.variables_initializer([var for var in tf.trainable_variables() if 'bias' in var.name and  \
                                                                                 'ENet/initial_block_1' not in var.name and \
                                                                                 'ENet/bottleneck1' not in var.name and  \
                                                                                 'ENet/bottleneck2' not in var.name])
                                                                                 )
    return trainables

def collect_summaries(disc_loss, l_var, l_dist, l_reg, input_image, prediction, correct_label):

    summaries = []
    # Collect all variables
    for var in tf.trainable_variables():
        summaries.append(tf.summary.histogram(var.name, var))
    # Collect losses
    summaries.append(tf.summary.scalar('Train/disc_loss', disc_loss))
    summaries.append(tf.summary.scalar('Train/l_var', l_var))
    summaries.append(tf.summary.scalar('Train/l_dist', l_dist))
    summaries.append(tf.summary.scalar('Train/l_reg', l_reg))
    # Collect images
    summaries.append(tf.summary.image('Train/Images/Input', input_image, max_outputs=1))
    summaries.append(tf.summary.image('Train/Images/Prediction', tf.expand_dims(prediction[:,:,:,0], axis=3), max_outputs=1))
    summaries.append(tf.summary.image('Train/Images/Label', tf.expand_dims(correct_label, axis=3), max_outputs=1))

    for summ in summaries:
        tf.add_to_collection('CUSTOM_SUMMARIES', summ)

    summary_op_train = tf.summary.merge_all('CUSTOM_SUMMARIES')

    summaries_valid = []
    summaries_valid.append(tf.summary.image('Valid/Images/Input', input_image, max_outputs=1))
    summaries_valid.append(tf.summary.image('Valid/Images/Prediction', tf.expand_dims(prediction[:,:,:,0], axis=3), max_outputs=1))
    summaries_valid.append(tf.summary.image('Valid/Images/Label', tf.expand_dims(correct_label, axis=3), max_outputs=1))    
    summaries_valid.append(tf.summary.scalar('Valid/disc_loss', disc_loss))
    summary_op_valid = tf.summary.merge(summaries_valid)
    return summary_op_train, summary_op_valid

def count_parameters():
    total_parameters = 0
    for var in tf.trainable_variables():
        shape = var.get_shape()
        variable_parameters = 1
        for dim in shape:
            variable_parameters *= dim.value
        total_parameters += variable_parameters
    return total_parameters
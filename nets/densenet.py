"""Contains a variant of the densenet model definition."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

import numpy as np

slim = tf.contrib.slim


def trunc_normal(stddev): return tf.truncated_normal_initializer(stddev=stddev)


def bn_act_conv_drp(current, num_outputs, kernel_size, scope='block'):
    current = slim.batch_norm(current, scope=scope + '_bn')
    current = tf.nn.relu(current)
    current = slim.conv2d(current, num_outputs, kernel_size, scope=scope + '_conv')
    current = slim.dropout(current, scope=scope + '_dropout')
    return current


def block(net, layers, growth, scope='block'):
    for idx in range(layers):
        bottleneck = bn_act_conv_drp(net, 4 * growth, [1, 1],
                                     scope=scope + '_conv1x1' + str(idx))
        tmp = bn_act_conv_drp(bottleneck, growth, [3, 3],
                              scope=scope + '_conv3x3' + str(idx))
        net = tf.concat(axis=3, values=[net, tmp])
    return net


def transition_layer(current, num_outputs, scope='block'):
    current = slim.batch_norm(current, scope=scope + '_bn')
    current = tf.nn.relu(current)
    current = slim.conv2d(current, num_outputs, [1, 1], scope=scope + '_conv')
    current = slim.dropout(current, scope=scope + '_dropout')
    current = slim.avg_pool2d(current, [2, 2], 2, scope='avg_pool2')
    return current


def global_average_pooling(net, stride=1, scope='Global_Average_Pooling'):
    width = np.shape(net)[1]
    height = np.shape(net)[2]
    kernel_size = [width, height]
    return slim.avg_pool2d(net, kernel_size, padding='Same',
                                scope='Global_Average_Pooling')


def densenet(images, num_classes=1001, is_training=False,
             dropout_keep_prob=0.8,
             scope='densenet',
             prediction_fn=slim.softmax):
    """Creates a variant of the densenet model.

      images: A batch of `Tensors` of size [batch_size, height, width, channels].
      num_classes: the number of classes in the dataset.
      is_training: specifies whether or not we're currently training the model.
        This variable will determine the behaviour of the dropout layer.
      dropout_keep_prob: the percentage of activation values that are retained.
      prediction_fn: a function to get predictions out of logits.
      scope: Optional variable_scope.

    Returns:
      logits: the pre-softmax activations, a tensor of size
        [batch_size, `num_classes`]
      end_points: a dictionary from components of the network to the corresponding
        activation.
    """
    growth = 24
    compression_rate = 0.5

    def reduce_dim(input_feature):
        return int(int(input_feature.shape[-1]) * compression_rate)

    end_points = {}

    with tf.variable_scope(scope, 'DenseNet', [images, num_classes]):
        with slim.arg_scope(bn_drp_scope(is_training=is_training,
                                         keep_prob=dropout_keep_prob)) as ssc:
            net = slim.conv2d(images, growth, [7, 7], stride=2, scope='conv0')
            end_points['conv0'] = net
            net = slim.max_pool2d(net, [3, 3], 2, scope='pool0')
            end_points['pool0'] = net
            net = block(net, layers=6, growth=growth, scope='dense_block1')
            end_points['dense_block1'] = net
            net = transition_layer(net, growth, scope='transition_layer1')
            end_points['transition_layer1'] = net
            net = block(net, layers=12, growth=growth, scope='dense_block2')
            end_points['dense_block2'] = net
            net = transition_layer(net, growth, scope='transition_layer2')
            end_points['transition_layer2'] = net
            net = block(net, layers=48, growth=growth, scope='dense_block3')
            end_points['dense_block3'] = net
            net = transition_layer(net, growth, scope='transition_layer3')
            end_points['transition_layer3'] = net
            net = block(net, layers=32, growth=growth, scope='dense_final')
            end_points['dense_final'] = net
            net = slim.batch_norm(net, scope='linear_batch')
            end_points['linear_batch'] = net
            net = slim.nn.relu(net)
            net = global_average_pooling(net, scope='Global_Average_Pooling')
            end_points['Global_Average_Pooling'] = net
            net = slim.flatten(net)
            end_points['Flatten'] = net
            logits = slim.fully_connected(net, num_classes,
                                          biases_initializer=tf.zeros_initializer(),
                                          weights_initializer=trunc_normal(1 / 192.0),
                                          weights_regularizer=None,
                                          activation_fn=None,
                                          scope='logits')

            end_points['Logits'] = logits
            end_points['Predictions'] = prediction_fn(logits, scope='Predictions')

    return logits, end_points


def bn_drp_scope(is_training=True, keep_prob=0.8):
    keep_prob = keep_prob if is_training else 1
    with slim.arg_scope(
        [slim.batch_norm],
            scale=True, is_training=is_training, updates_collections=None):
        with slim.arg_scope(
            [slim.dropout],
                is_training=is_training, keep_prob=keep_prob) as bsc:
            return bsc


def densenet_arg_scope(weight_decay=0.004):
    """Defines the default densenet argument scope.

    Args:
      weight_decay: The weight decay to use for regularizing the model.

    Returns:
      An `arg_scope` to use for the inception v3 model.
    """
    with slim.arg_scope(
        [slim.conv2d],
        weights_initializer=tf.contrib.layers.variance_scaling_initializer(
            factor=2.0, mode='FAN_IN', uniform=False),
        activation_fn=None, biases_initializer=None, padding='same',
            stride=1) as sc:
        return sc


densenet.default_image_size = 224

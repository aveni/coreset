import tensorflow as tf
import numpy as np
from tensorflow.contrib.layers.python.layers import batch_norm

def linear(x, train_mode, width, n_channels):
		x_shape = x.get_shape().as_list()
		reshape = tf.reshape(x, [-1, np.product(x_shape[1:])])
		W = tf.Variable(tf.zeros([np.product(x_shape[1:]), 10]))
		b = tf.Variable(tf.zeros([10]))
		return (reshape, tf.matmul(reshape, W) + b)

def cnn(x, train_mode, width, n_channels):

	input_layer = tf.reshape(x, [-1, width, width, n_channels])

	conv1 = tf.layers.conv2d(
	inputs=input_layer,
	filters=32,
	kernel_size=[5, 5],
	padding="same",
	activation=tf.nn.relu)


	pool1 = tf.layers.max_pooling2d(inputs=conv1, pool_size=[2, 2], strides=2)

  # Convolutional Layer #2 and Pooling Layer #2
	conv2 = tf.layers.conv2d(
      inputs=pool1,
      filters=64,
      kernel_size=[5, 5],
      padding="same",
      activation=tf.nn.relu)
	pool2 = tf.layers.max_pooling2d(inputs=conv2, pool_size=[2, 2], strides=2)

  # Dense Layer
	pool2_flat = tf.reshape(pool2, [-1, (width/4) * (width/4) * 64])
	dense1 = tf.layers.dense(inputs=pool2_flat, units=1024, activation=tf.nn.relu)

  # Dropout if Training
	dense1 = tf.cond(train_mode, lambda: tf.layers.dropout(inputs=dense1, rate=0.4), lambda: dense1)

  # Logits Layer
	logits = tf.layers.dense(inputs=dense1, units=10)

	return (dense1, logits)



def batch_norm_layer(x, train_phase, scope_bn):
    return batch_norm(x, decay=0.9, center=True, scale=True,
    updates_collections=None,
    is_training=train_phase,
    reuse=None,
    trainable=True,
    scope=scope_bn)

def res_layer(x, kernel_size, incoming_features, outgoing_features, stride, train_phase, modified=True):
    w1 = tf.Variable(
        tf.random_normal([
            kernel_size, kernel_size, incoming_features, outgoing_features
        ],
        stddev=np.sqrt(2./(kernel_size*kernel_size*incoming_features)))
    )
    w2 = tf.Variable(
        tf.random_normal([
            kernel_size, kernel_size, outgoing_features, outgoing_features
        ],
        stddev=np.sqrt(2./(kernel_size*kernel_size*outgoing_features)))
    )
    strides = [1, stride, stride, 1]


    if modified:
        # first do bn + relu + conv
        conv1 = batch_norm_layer(x, train_phase, None)
        conv1 = tf.nn.relu(conv1)
        conv1 = tf.nn.conv2d(conv1, w1, strides=strides, padding='SAME')

        # continue with bn + relu + conv
        conv2 = batch_norm_layer(conv1, train_phase, None)
        conv2 = tf.nn.relu(conv2)
        conv2 = tf.nn.conv2d(conv2, w2, strides=[1,1,1,1], padding='SAME')

    else:
        # first do conv + bn + relu
        conv1 = tf.nn.conv2d(x, w1, strides=strides, padding='SAME')
        conv1 = batch_norm_layer(conv1, train_phase, None)
        conv1 = tf.nn.relu(conv1)

        # continue with conv + bn
        conv2 = tf.nn.conv2d(conv1, w2, strides=[1,1,1,1], padding='SAME')
        conv2 = batch_norm_layer(conv2, train_phase, None)

    # finally, if this is a transition between filter sizes...
    scaled_input = x
    if x.get_shape()[1] != conv2.get_shape()[1]: # assume square shape
        # TODO: assumption -- every filter count boundary is also stride=2

        # apply 1x1 conv to fix the dimensionality
        shortcut_weights = tf.Variable(
            tf.random_normal([
                1, 1, incoming_features, outgoing_features
            ],
            stddev=np.sqrt(2./(1*1*incoming_features)))
        )
        scaled_input = tf.nn.conv2d(
            x, shortcut_weights,
            [1, stride, stride, 1], padding='SAME'
        )

    with_shortcut = tf.add(scaled_input, conv2)
    if not modified:
        with_shortcut = tf.nn.relu(with_shortcut)
    return with_shortcut


def resnet18(x, train_phase, width, n_channels, modified=True, init_k=3):
    init_num_features = 16
    wc1 = tf.Variable(tf.random_normal([init_k, init_k, n_channels, init_num_features], stddev=np.sqrt(2./(init_k*init_k*n_channels))))
    wfc1 = tf.Variable(tf.random_normal([1*1*64, 10], stddev=np.sqrt(2./(1*1*64))))
    bfc1 = tf.Variable(tf.ones(10))

    # (init_k)x(init_k) conv, 64
    print str(init_k)+'x'+str(init_k)+' 16'
    re_x = tf.reshape(x, [-1,width,width,n_channels])
    conv1 = tf.nn.conv2d(re_x, wc1, strides=[1, 1, 1, 1], padding='SAME')
    conv1 = batch_norm_layer(conv1, train_phase, 'bn1')
    conv1 = tf.nn.relu(conv1)

    # res: 3x3 conv, 64
    print 'res (2): 3x3 16'
    res1 = res_layer(conv1, 3, init_num_features, 16, 1, train_phase, modified)
    res1 = res_layer(res1, 3, 16, 16, 1, train_phase, modified)

    # res: 3x3 conv, 128
    print 'res (2): 3x3 32'
    res2 = res_layer(res1, 3, 16, 32, 2, train_phase, modified)
    res2 = res_layer(res2, 3, 32, 32, 1, train_phase, modified)

    # res: 3x3 conv, 128
    print 'res (2) 3x3 64'
    res3 = res_layer(res2, 3, 32, 64, 2, train_phase, modified)
    res3 = res_layer(res3, 3, 64, 64, 1, train_phase, modified)

    # res: 3x3 conv, 128
    print 'res (2) 3x3 64'
    res4 = res_layer(res3, 3, 64, 64, 1, train_phase, modified)
    res4 = res_layer(res4, 3, 64, 64, 1, train_phase, modified)

    # avg pool
    print 'global avg pool'
    avg_pool = tf.layers.average_pooling2d(res4, [width/4, width/4], [width/4, width/4])

    # 10 node fc
    print 'fc 10'
    fc = tf.reshape(avg_pool, [-1, wfc1.get_shape().as_list()[0]])
    logits = tf.add(tf.matmul(fc, wfc1), bfc1)

    return fc, logits




def vgg_16(x, train_mode, width, n_channels):
	input_layer = tf.reshape(x, [-1, width, width, n_channels])

	conv1 = tf.layers.conv2d(
	inputs=input_layer,
	filters=64,
	kernel_size=[3, 3],
	padding="same",
	activation=tf.nn.relu)


	pool1 = tf.layers.max_pooling2d(inputs=conv1, pool_size=[2, 2], strides=2)

  # Convolutional Layer #2 and Pooling Layer #2
	conv2 = tf.layers.conv2d(
      inputs=pool1,
      filters=128,
      kernel_size=[3, 3],
      padding="same",
      activation=tf.nn.relu)
	pool2 = tf.layers.max_pooling2d(inputs=conv2, pool_size=[2, 2], strides=2)


	conv3 = tf.layers.conv2d(
      inputs=pool2,
      filters=256,
      kernel_size=[3, 3],
      padding="same",
      activation=tf.nn.relu)
	conv4= tf.layers.conv2d(
      inputs=conv3,
      filters=256,
      kernel_size=[3, 3],
      padding="same",
      activation=tf.nn.relu)
	pool3 = tf.layers.max_pooling2d(inputs=conv4, pool_size=[2, 2], strides=2)

  # Dense Layer
	flat = tf.reshape(pool3, [-1, (width/8) * (width/8) * 256])
	dense1 = tf.layers.dense(inputs=flat, units=4096, activation=tf.nn.relu)
	dense1 = tf.cond(train_mode, lambda: tf.layers.dropout(inputs=dense1, rate=0.5), lambda: dense1)

	dense2 = tf.layers.dense(inputs=dense1, units=4096, activation=tf.nn.relu)
	dense2 = tf.cond(train_mode, lambda: tf.layers.dropout(inputs=dense2, rate=0.5), lambda: dense2)

	dense3 = tf.layers.dense(inputs=dense2, units=1024, activation=tf.nn.relu)


  # Logits Layer
	logits = tf.layers.dense(inputs=dense3, units=10)

	return (dense3, logits)
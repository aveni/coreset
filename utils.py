### DO NOT USE  ###
### DEPRECATED, USE experiment.py INSTEAD ###

import tensorflow as tf
import numpy as np
import os
import time
from our_resnet import resnet18_cifar, resnet18_mnist
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 


def linear(x, train_mode, mode):
	x_shape = x.get_shape().as_list()
	reshape = tf.reshape(x, [-1, np.product(x_shape[1:])])
	W = tf.Variable(tf.zeros([np.product(x_shape[1:]), 10]))
	b = tf.Variable(tf.zeros([10]))
	return (reshape, tf.matmul(reshape, W) + b)

def cnn(x, train_mode, dataset):

	if dataset=="mnist":
		width = 28
	elif dataset=="cifar":
		width = 32
	else:
		return None


	input_layer = tf.reshape(x, [-1, width, width, 3])

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



def resnet(x, train_mode, dataset):
	if dataset == "mnist":
		return resnet18_mnist(x, train_mode)
	elif dataset == "cifar":
		return resnet18_cifar(x, train_mode)

def make_dataset(x, y, ix, batch_size, shuffle=False):
	dataset = tf.data.Dataset.from_tensor_slices((x, y, ix))
	if shuffle:
		dataset = dataset.shuffle(buffer_size=x.shape[0])
	dataset = dataset.batch(batch_size)
	return dataset

def load_data(dataset, train_ix, train_bs, val_bs):
	x_train, y_train, x_val, y_val, x_test, y_test = [None]*6
	with np.load("./datasets/%s_with_val.npz" % dataset) as data:
		x_train = data['x_train'].astype("float32")
		y_train = data['y_train'].astype("int32")
		ix_train = range(x_train.shape[0])
		x_val = data['x_val'].astype("float32")
		y_val = data['y_val'].astype("int32")
		ix_val = range(x_val.shape[0])
		x_test = data['x_test'].astype("float32")
		y_test = data['y_test'].astype("int32")
		ix_test = range(x_test.shape[0])

	if train_ix is not None:
		x_train = x_train[train_ix]
		y_train = y_train[train_ix]
		ix_train = np.array(train_ix).astype("int32")

	train = make_dataset(x_train, tf.one_hot(y_train, 10), ix_train, train_bs, shuffle=True)
	train_fast = make_dataset(x_train, tf.one_hot(y_train, 10), ix_train, val_bs)
	val = make_dataset(x_val, tf.one_hot(y_val, 10), ix_val, val_bs)
	test = make_dataset(x_test, tf.one_hot(y_test, 10), ix_test, 100)

	return train, train_fast, val, test

def load_data_cross_val(dataset, train_ix, train_bs, val_bs, N=10):
	x_train, y_train, x_val, y_val, x_test, y_test = [None]*6
	with np.load("./datasets/%s_with_val.npz" % dataset) as data:
		x_train = data['x_train'].astype("float32")
		y_train = data['y_train'].astype("int32")
		ix_train = range(x_train.shape[0])
		x_val = data['x_val'].astype("float32")
		y_val = data['y_val'].astype("int32")
		ix_val = range(x_val.shape[0])
		x_test = data['x_test'].astype("float32")
		y_test = data['y_test'].astype("int32")
		ix_test = range(x_test.shape[0])

	if train_ix is not None:
		x_train = x_train[train_ix]
		y_train = y_train[train_ix]
		ix_train = np.array(train_ix).astype("int32")

	datasets = []
	for i in range (N):
		start = i*int(x_train.shape[0]/N)
		end = (i+1)*int(x_train.shape[0]/N)
		mini_x_val = x_train[start:end]
		mini_y_val = y_train[start:end]
		mini_ix_val = ix_train[start:end]

		mini_x_train = np.concatenate((x_train[:start], x_train[end:]))
		mini_y_train = np.concatenate((y_train[:start], y_train[end:]))
		mini_ix_train = np.concatenate((ix_train[:start], ix_train[end:])).astype("int32")


		mini_train = make_dataset(mini_x_train, tf.one_hot(mini_y_train, 10), mini_ix_train, train_bs, shuffle=True)
		mini_val = make_dataset(mini_x_val, tf.one_hot(mini_y_val, 10), mini_ix_val, val_bs)
		datasets.append((mini_train, mini_val))

	return datasets










def get_test_acc(model, dataset, train_ix=None, epochs=10, lr=0.001, bs=16, trials=10, 
	verbose=True, load_dir=None, save_dir=None):

	train, train_fast, val, test = load_data(dataset, train_ix, bs, 100)

	iterator = tf.data.Iterator.from_structure(train.output_types, train.output_shapes)
	train_init_op = iterator.make_initializer(train)
	train_fast_init_op = iterator.make_initializer(train_fast)
	val_init_op = iterator.make_initializer(val)
	test_init_op = iterator.make_initializer(test)

	train_mode = tf.placeholder(tf.bool)
	batch_x, batch_y, batch_ix = iterator.get_next()
	_, batch_logits = model(batch_x, train_mode, dataset)
	loss = tf.reduce_mean(tf.losses.softmax_cross_entropy(onehot_labels=batch_y, logits=batch_logits, reduction=tf.losses.Reduction.NONE))

	global_epoch = tf.Variable(0, trainable=False)
	increment_global_epoch_op = tf.assign(global_epoch, global_epoch+1)
	boundaries = [epochs/2]
	values = [lr, 0.1*lr]
	learning_rate = tf.train.piecewise_constant(global_epoch, boundaries, values)

	optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
	train_op = optimizer.minimize(loss)

	top1 = tf.reduce_mean(tf.cast(tf.nn.in_top_k(predictions=batch_logits, targets=tf.argmax(batch_y, 1), k=1), tf.float32))
	top2 = tf.reduce_mean(tf.cast(tf.nn.in_top_k(predictions=batch_logits, targets=tf.argmax(batch_y, 1), k=2), tf.float32))
	top3 = tf.reduce_mean(tf.cast(tf.nn.in_top_k(predictions=batch_logits, targets=tf.argmax(batch_y, 1), k=3), tf.float32))

	if load_dir or save_dir:
		saver = tf.train.Saver()
		if load_dir:
			load_file = os.path.join(load_dir, "%s-checkpoint" % load_dir)
		if save_dir:
		 	checkpoint_file = os.path.join(save_dir, "%s-checkpoint" % save_dir)

	def calc_accs_losses(init_op):
		sess.run(init_op)
		losses = []
		accs = []
		while True:
			try:
				l, acc = sess.run([loss, [top1, top2, top3]], feed_dict={train_mode:False})
				losses.append(l)
				accs.append(acc)
			except tf.errors.OutOfRangeError:
				break
		return [accs, losses]

	def train_one_epoch():
		sess.run(train_init_op)
		while True:
			try:
				sess.run([train_op, increment_global_epoch_op], feed_dict={train_mode:True})
			except tf.errors.OutOfRangeError:
				break

	with tf.Session() as sess:
		trial_accs = []
		for t in range(trials):
			print "Trial=%d" % t
			t_start = time.time()
			sess.run(tf.global_variables_initializer())

			if load_dir:
				saver.restore(sess, load_file)

			if verbose:
				print "Built graph"
			for e in range(epochs):
				start = time.time()
				train_one_epoch()

				if verbose:
					train_accs, train_losses = calc_accs_losses(train_fast_init_op)
					val_accs, val_losses = calc_accs_losses(val_init_op)
					print "epoch", (e+1), ("(%.2f s)" % (time.time()-start))
					print "train acc=%s \t train loss=%f" % (np.mean(train_accs, axis=0), np.mean(train_losses))
					print "val   acc=%s \t val   loss=%f" % (np.mean(val_accs, axis=0), np.mean(val_losses))

			test_accs, test_losses = calc_accs_losses(test_init_op)
			if verbose:
				print "\ntest acc=%s \t test loss=%f" % (np.mean(test_accs, axis=0), np.mean(test_losses))

			if save_dir is not None:
				saver.save(sess, checkpoint_file)

			test_accs = np.mean(test_accs, axis=0)
			trial_accs.append(test_accs)
			print "%s\t%.2f s" % (test_accs, time.time()-t_start)

		return trial_accs



















def get_uncertainty(model, dataset, val_N=10, bs=100, lr=0.0001, epochs=20):
	datasets = load_data_cross_val(dataset, None, bs, 100, N=val_N)

	train_ops = []
	val_ops = []
	iterator = tf.data.Iterator.from_structure(datasets[0][0].output_types, datasets[0][0].output_shapes)
	for i in range(val_N):
		train_ops.append(iterator.make_initializer(datasets[i][0]))
		val_ops.append(iterator.make_initializer(datasets[i][1]))

	train_mode = tf.placeholder(tf.bool)
	batch_x, batch_y, batch_ix = iterator.get_next()
	_, batch_logits = model(batch_x, train_mode, dataset)
	batch_loss = tf.losses.softmax_cross_entropy(onehot_labels=batch_y, logits=batch_logits, reduction=tf.losses.Reduction.NONE)
	loss = tf.reduce_mean(batch_loss)
	optimizer = tf.train.AdamOptimizer(learning_rate=lr)
	train_op = optimizer.minimize(loss)

	top1 = tf.reduce_mean(tf.cast(tf.nn.in_top_k(predictions=batch_logits, targets=tf.argmax(batch_y, 1), k=1), tf.float32))
	top2 = tf.reduce_mean(tf.cast(tf.nn.in_top_k(predictions=batch_logits, targets=tf.argmax(batch_y, 1), k=2), tf.float32))
	top3 = tf.reduce_mean(tf.cast(tf.nn.in_top_k(predictions=batch_logits, targets=tf.argmax(batch_y, 1), k=3), tf.float32))

	def calc_accs_losses(init_op):
		sess.run(init_op)
		losses = []
		accs = []
		while True:
			try:
				l, acc = sess.run([loss, [top1, top2, top3]], feed_dict={train_mode:False})
				losses.append(l)
				accs.append(acc)
			except tf.errors.OutOfRangeError:
				break
		return [accs, losses]

	def calc_indiv_losses():
		ix = []
		losses = []
		ys = []
		sess.run(val_init_op)
		while True:
			try:
				bix, bl, by = sess.run([batch_ix, batch_loss, batch_y], feed_dict={train_mode:True})
				ix.append(bix)
				losses.append(bl)
				ys.append(by)
			except tf.errors.OutOfRangeError:
				break
		return [np.concatenate(ix), np.concatenate(losses), np.concatenate(ys)]

	def train_one_epoch():
		sess.run(train_init_op)
		while True:
			try:
				sess.run(train_op, feed_dict={train_mode:True})
			except tf.errors.OutOfRangeError:
				break


	train_ix = []
	train_l = []
	train_y = []
	with tf.Session() as sess:
		for i in range (val_N):
			sess.run(tf.global_variables_initializer())
			train_init_op = train_ops[i]
			val_init_op = val_ops[i]

			print "VAL SLICE", i

			for e in range(epochs):
				start = time.time()
				train_one_epoch()

				train_accs, train_losses = calc_accs_losses(train_init_op)
				val_accs, val_losses = calc_accs_losses(val_init_op)
				print "epoch", (e+1), ("(%.2f s)" % (time.time()-start))
				print "train acc=%s \t train loss=%f" % (np.mean(train_accs, axis=0), np.mean(train_losses))
				print "val   acc=%s \t val   loss=%f" % (np.mean(val_accs, axis=0), np.mean(val_losses))

			mini_train_ix, mini_train_losses, mini_train_y = calc_indiv_losses()
			mini_train_y = np.argmax(mini_train_y, axis=1)

			train_ix.append(mini_train_ix)
			train_l.append(mini_train_losses)
			train_y.append(mini_train_y)

	train_ix = np.concatenate(train_ix)
	train_l = np.concatenate(train_l)
	train_y = np.concatenate(train_y)
	data = np.array([train_ix, train_l, train_y])

	np.save("%s-uncertainty/%s-uncertainty_scores.npy" % (dataset, dataset), data)

	return None





















def get_dense_activations(model, dataset, load_dir=None):
	train, train_fast, val, test = load_data(dataset, None, 100, 100)

	iterator = tf.data.Iterator.from_structure(train.output_types, train.output_shapes)
	train_init_op = iterator.make_initializer(train)
	val_init_op = iterator.make_initializer(val)


	train_mode = tf.placeholder(tf.bool)
	batch_x, batch_y, batch_ix = iterator.get_next()
	batch_dense, batch_logits = model(batch_x, train_mode, dataset)
	loss = tf.reduce_mean(tf.losses.softmax_cross_entropy(onehot_labels=batch_y, logits=batch_logits, reduction=tf.losses.Reduction.NONE))
	top1 = tf.reduce_mean(tf.cast(tf.nn.in_top_k(predictions=batch_logits, targets=tf.argmax(batch_y, 1), k=1), tf.float32))
	top2 = tf.reduce_mean(tf.cast(tf.nn.in_top_k(predictions=batch_logits, targets=tf.argmax(batch_y, 1), k=2), tf.float32))
	top3 = tf.reduce_mean(tf.cast(tf.nn.in_top_k(predictions=batch_logits, targets=tf.argmax(batch_y, 1), k=3), tf.float32))

	saver = tf.train.Saver()
	load_file = os.path.join(load_dir, "%s-checkpoint" % load_dir)

	def calc_accs_losses(init_op):
		sess.run(init_op)
		losses = []
		accs = []
		while True:
			try:
				l, acc = sess.run([loss, [top1, top2, top3]], feed_dict={train_mode:False})
				losses.append(l)
				accs.append(acc)
			except tf.errors.OutOfRangeError:
				break
		return [accs, losses]

	def get_activations():
		ix = []
		activs = []
		ys = []
		sess.run(train_init_op)
		while True:
			try:
				bix, bd, by = sess.run([batch_ix, batch_dense, batch_y], feed_dict={train_mode:True})
				ix.append(bix)
				activs.append(bd)
				ys.append(by)
			except tf.errors.OutOfRangeError:
				break
		return [np.concatenate(ix), np.concatenate(activs), np.concatenate(ys)]


	with tf.Session() as sess:
		sess.run(tf.global_variables_initializer())

		saver.restore(sess, load_file)

		train_accs, train_losses = calc_accs_losses(train_init_op)
		val_accs, val_losses = calc_accs_losses(val_init_op)
		print "train acc=%s \t train loss=%f" % (np.mean(train_accs, axis=0), np.mean(train_losses))
		print "val   acc=%s \t val   loss=%f" % (np.mean(val_accs, axis=0), np.mean(val_losses))

		train_ix, train_activs, train_y = get_activations()
		train_y = np.argmax(train_y, axis=1)

		print train_ix.shape
		print train_activs.shape
		print train_y.shape

		np.save("%s-kcenter/%s-dense_ix_y.npy" % (dataset, dataset), [train_ix, train_y])
		np.save("%s-kcenter/%s-dense_activations.npy" % (dataset, dataset), train_activs)

	return None






def get_grad_norms(model, dataset, train_ix=None, epochs=1, lr=0.001, train_bs=16):
	if dataset == "mnist":
		train, train_bs_1, val, test = load_mnist(train_ix, train_bs, 1)

	iterator = tf.data.Iterator.from_structure(train.output_types, train.output_shapes)
	train_init_op = iterator.make_initializer(train)
	train_bs_1_init_op = iterator.make_initializer(train_bs_1)
	val_init_op = iterator.make_initializer(val)
	test_init_op = iterator.make_initializer(test)

	train_mode = tf.placeholder(tf.bool)
	batch_x, batch_y, batch_ix = iterator.get_next()
	_, batch_logits = model(batch_x, train_mode)
	loss = tf.reduce_mean(tf.losses.softmax_cross_entropy(onehot_labels=batch_y, logits=batch_logits, reduction=tf.losses.Reduction.NONE))
	optimizer = tf.train.AdamOptimizer(learning_rate=lr)
	train_op = optimizer.minimize(loss)

	grads_and_vars = optimizer.compute_gradients(loss)
	batch_delta = tf.reduce_sum([tf.norm(t[0]) for t in grads_and_vars])

	def train_one_epoch():
		sess.run(train_init_op)
		while True:
			try:
				sess.run(train_op, feed_dict={train_mode:True})
			except tf.errors.OutOfRangeError:
				break

	def calc_grad_norms(init_op):
		sess.run(init_op)
		ixs = []
		deltas = []
		while True:
			try:
				bix, bd = sess.run([batch_ix, batch_delta], feed_dict={train_mode:False})
				ixs.append(bix)
				deltas.append(bd)
			except tf.errors.OutOfRangeError:
				break
		return (np.array(ixs).flatten(), np.array(deltas).flatten())

	train_gn = np.zeros((epochs, len(train_ix) if train_ix else 55000))
	with tf.Session() as sess:
		sess.run(tf.global_variables_initializer())

		for e in range(epochs):
			train_one_epoch()

			e_train_ix, e_train_gn = calc_grad_norms(train_bs_1_init_op)
			train_gn[e][e_train_ix] = e_train_gn
			print e
			np.save("./data/train_gn_full_", train_gn)

		_, val_gn = calc_grad_norms(val_init_op)
		np.save("./data/val_gn_full_", val_gn)


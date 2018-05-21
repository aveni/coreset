import tensorflow as tf
import numpy as np
import os
import time
from dataset import DataSet
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 

class Experiment:
	def __init__(self, model, dataset_name):
		self.model = model
		self.dataset_name = dataset_name

		if dataset_name=="mnist":
			self.width=28
			self.n_channels=1
		elif dataset_name=="cifar":
			self.width=32
			self.n_channels=3

		self.train, self.val, self.test = self.load_data(dataset_name)

		self.train_mode = tf.placeholder(tf.bool)
		self.batch_x = tf.placeholder(tf.float32, shape=(None, self.width, self.width, self.n_channels))
		self.batch_y = tf.placeholder(tf.int32, shape=(None, ))

		self.batch_dense, self.batch_logits = self.model(self.batch_x, self.train_mode, self.width, self.n_channels)
		self.batch_loss = tf.losses.sparse_softmax_cross_entropy(labels=self.batch_y, logits=self.batch_logits, reduction=tf.losses.Reduction.NONE)
		self.loss = tf.reduce_mean(self.batch_loss)

		self.learning_rate = tf.placeholder(tf.float32)

		self.optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate)
		self.train_op = self.optimizer.minimize(self.loss)

		self.top1 = tf.reduce_mean(tf.cast(tf.nn.in_top_k(predictions=self.batch_logits, targets=self.batch_y, k=1), tf.float32))
		self.top2 = tf.reduce_mean(tf.cast(tf.nn.in_top_k(predictions=self.batch_logits, targets=self.batch_y, k=2), tf.float32))
		self.top3 = tf.reduce_mean(tf.cast(tf.nn.in_top_k(predictions=self.batch_logits, targets=self.batch_y, k=3), tf.float32))

		self.sess = tf.Session()
		self.saver = tf.train.Saver()


	def load_data(self, dataset_name):
		x_train, y_train, x_val, y_val, x_test, y_test = [None]*6
		with np.load("./datasets/%s_with_val.npz" % dataset_name) as data:
			x_train = data['x_train'].astype("float32")
			y_train = data['y_train'].astype("int32")
			x_val = data['x_val'].astype("float32")
			y_val = data['y_val'].astype("int32")
			x_test = data['x_test'].astype("float32")
			y_test = data['y_test'].astype("int32")

		if len(x_train.shape) < 4:
			x_train = np.expand_dims(x_train, axis=3)
			x_val = np.expand_dims(x_val, axis=3)
			x_test = np.expand_dims(x_test, axis=3)


		train = DataSet(x_train, y_train, None)
		train_fast = DataSet(x_train, y_train, None)
		val = DataSet(x_val, y_val, None)
		test = DataSet(x_test, y_test, None)

		return train, val, test


	def calc_indiv_losses(self, dataset, bs):
		ix = []
		losses = []
		ys = []
		dataset.reset_batch()

		for _ in range(dataset.size / bs):
			bx, by, bix = dataset.next_batch(bs)
			bl = self.sess.run(self.batch_loss, feed_dict={self.batch_x:bx, self.batch_y:by, self.train_mode:False})
			ix.append(bix)
			losses.append(bl)
			ys.append(by)

		return [np.concatenate(ix), np.concatenate(losses), np.concatenate(ys)]


	def calc_accs_losses(self, dataset, bs):
		losses = []
		accs = []
		dataset.reset_batch()
		for _ in range(dataset.size / bs):
			bx, by, bix = dataset.next_batch(bs)
			l, acc = self.sess.run([self.loss, [self.top1, self.top2, self.top3]], feed_dict={self.batch_x:bx, self.batch_y:by, self.train_mode:False})
			losses.append(l)
			accs.append(acc)

		return [accs, losses]

	def get_activations(self, bs):
		ix = []
		activs = []
		ys = []
		self.train.reset_batch()

		for _ in range(self.train.size / bs):
			bx, by, bix = self.train.next_batch(bs)
			bd = self.sess.run(self.batch_dense, feed_dict={self.batch_x:bx, self.batch_y:by, self.train_mode:False})
			ix.append(bix)
			activs.append(bd)
			ys.append(by)

		return [np.concatenate(ix), np.concatenate(activs), np.concatenate(ys)]


	def train_one_epoch(self, train_bs, lr):
		self.train.reset_batch()
		for _ in range(self.train.size / train_bs):
			bx, by, bix = self.train.next_batch(train_bs)
			self.sess.run([self.train_op], feed_dict={self.batch_x:bx, self.batch_y:by, 
																					self.learning_rate:lr, self.train_mode:True})


	def get_test_acc(self, train_ix=None, epochs=10, lr=0.001,
										train_bs=10, val_bs=100, 
										verbose=True, load_dir=None, save_dir=None):

		self.train.set_ix(train_ix)

		if load_dir or save_dir:
			if load_dir:
				load_file = os.path.join(load_dir, "%s-%s-baseline/%s-%s-baseline-checkpoint" % 
					(self.dataset_name, self.model.__name__, self.dataset_name, self.model.__name__))
			if save_dir:
			 	checkpoint_file = os.path.join(save_dir, "%s-%s-baseline/%s-%s-baseline-checkpoint" %
			 		(self.dataset_name, self.model.__name__, self.dataset_name, self.model.__name__))

		self.sess.run(tf.global_variables_initializer())

		if load_dir:
			self.saver.restore(self.sess, load_file)

		if verbose:
			print "Initialized graph"
		for e in range(epochs):
			start = time.time()
			self.train_one_epoch(train_bs, lr)

			if verbose:
				train_accs, train_losses = self.calc_accs_losses(self.train, val_bs)
				val_accs, val_losses = self.calc_accs_losses(self.val, val_bs)
				print "epoch", (e+1), ("(%.2f s)" % (time.time()-start))
				print "train acc=%s \t train loss=%f" % (np.mean(train_accs, axis=0), np.mean(train_losses))
				print "val   acc=%s \t val   loss=%f" % (np.mean(val_accs, axis=0), np.mean(val_losses))

		test_accs, test_losses = self.calc_accs_losses(self.test, val_bs)
		if verbose:
			print "\ntest acc=%s \t test loss=%f" % (np.mean(test_accs, axis=0), np.mean(test_losses))

		if save_dir is not None:
			self.saver.save(self.sess, checkpoint_file)

		return np.mean(test_accs, axis=0)



	def get_uncertainty(self, model, dataset, val_N=10, epochs=20, lr=0.0001,
										train_bs=100, val_bs=100):

		train_ix = []
		train_l = []
		train_y = []
		for i in range (val_N):
			self.sess.run(tf.global_variables_initializer())

			print "VAL SLICE", i
			start = i*int(self.train._x.shape[0]/val_N)
			end = (i+1)*int(self.train._x.shape[0]/val_N)

			holdout_ix = range(start, end)
			holdin_ix = range(0,start)+range(end, self.train._x.shape[0])

			# print start, end
			# print len(holdout_ix), len(holdin_ix)

			self.train.set_ix(holdin_ix)

			for e in range(epochs):
				start = time.time()
				self.train_one_epoch(train_bs, lr)

				train_accs, train_losses = self.calc_accs_losses(self.train, val_bs)
				val_accs, val_losses = self.calc_accs_losses(self.val, val_bs)
				print "epoch", (e+1), ("(%.2f s)" % (time.time()-start))
				print "train acc=%s \t train loss=%f" % (np.mean(train_accs, axis=0), np.mean(train_losses))
				print "val   acc=%s \t val   loss=%f" % (np.mean(val_accs, axis=0), np.mean(val_losses))

			self.train.set_ix(holdout_ix)
			mini_train_ix, mini_train_losses, mini_train_y = self.calc_indiv_losses(self.train, val_bs)


			train_ix.append(mini_train_ix)
			train_l.append(mini_train_losses)
			train_y.append(mini_train_y)

		train_ix = np.concatenate(train_ix)
		train_l = np.concatenate(train_l)
		train_y = np.concatenate(train_y)
		data = np.array([train_ix, train_l, train_y])

		return data




	def get_dense_activations(self, load_dir="baselines"):
		load_file = os.path.join(load_dir, "%s-%s-baseline/%s-%s-baseline-checkpoint" % 
			(self.dataset_name, self.model.__name__, self.dataset_name, self.model.__name__))

		self.sess.run(tf.global_variables_initializer())

		self.saver.restore(self.sess, load_file)

		train_accs, train_losses = self.calc_accs_losses(self.train, 100)
		val_accs, val_losses = self.calc_accs_losses(self.val, 100)
		print "train acc=%s \t train loss=%f" % (np.mean(train_accs, axis=0), np.mean(train_losses))
		print "val   acc=%s \t val   loss=%f" % (np.mean(val_accs, axis=0), np.mean(val_losses))

		train_ix, train_activs, train_y = self.get_activations(100)

		print train_ix.shape
		print train_activs.shape
		print train_y.shape

		np.save("results/%s-%s-kcenter/%s-%s-dense_ix_y.npy" % 
			(self.dataset_name, self.model.__name__, self.dataset_name, self.model.__name__), [train_ix, train_y])
		np.save("results/%s-%s-kcenter/%s-%s-dense_activations.npy" % 
			(self.dataset_name, self.model.__name__, self.dataset_name, self.model.__name__), train_activs)

		return None



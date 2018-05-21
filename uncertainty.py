import tensorflow as tf
import numpy as np
from experiment import Experiment
from models import linear, cnn, resnet18, vgg_16
import time
import os

def balanced_max_entropy(N, reverse=False):
	global ix, l, y
	chosen = []
	for label in range (10):
		label_l = l[y==label]
		label_ix = ix[y==label]
		if reverse:
			uncertain = label_ix[np.argsort(label_l)]
		else:
			uncertain = label_ix[np.argsort(label_l)[::-1]]
		chosen.append(uncertain[:int(N/10)])
	chosen = np.concatenate(chosen).astype("int32")
	np.random.shuffle(chosen)
	return chosen

def balanced_uncertainty(N):
	global ix, l, y
	chosen = []
	for label in range (10):
		label_l = l[y==label]
		label_ix = ix[y==label]
		label_p = label_l/np.sum(label_l)

		uncertain = np.random.choice(label_ix, N/10, p=label_p)
		chosen.append(uncertain)
	chosen = np.concatenate(chosen).astype(int)
	np.random.shuffle(chosen)
	return chosen

dataset = "cifar"
model = cnn
trials = 5
sizes = [10, 100,1000, 5000, 10000, 20000]
y = np.load("./datasets/%s_with_val.npz" % dataset)["y_train"]
ix = np.arange(0, len(y), 1).astype("int32")
e = Experiment(model, dataset)


# # Compute uncertainty scores (ONLY NEEDS TO BE DONE ONCE)
# data = e.get_uncertainty(model, dataset, val_N=10, epochs=10, lr=0.0001)
# filename = "%s-%s-uncertainty" % (dataset, model.__name__)
# directory = "results/%s" % filename
# try:
#     os.stat(directory)
# except:
#     os.mkdir(directory) 
# np.save("results/%s/%s-scores.npy" % (filename, filename), data)



## Use uncertainty scores for either Max-Entropy or stochastic Uncertainty sampling
ix, l, y = np.load("results/%s-%s-uncertainty/%s-%s-uncertainty-scores.npy" % (dataset, model.__name__, dataset, model.__name__))
for N in sizes:
	accs = []
	for T in range(trials):
		start = time.time()
		print "N=%d, Trial=%d" % (N, T)
		acc = e.get_test_acc(train_ix=balanced_max_entropy(N), epochs=100, lr=0.0001, train_bs=min(100, max(1, N/10)), verbose=True, load_dir=None)
		# acc = e.get_test_acc(train_ix=balanced_uncertainty(N), epochs=40, lr=0.0001, train_bs=min(100, max(1, N/100)), verbose=True, load_dir=None)
		accs.append(acc)
		print "Trial Acc=%s (%.2f s)" % (str(acc), time.time()-start)

	filename = "%s-%s-maxE" % (dataset, model.__name__)
	# filename = "%s-%s-uncertainty" % (dataset, model.__name__)
	directory = "results/%s" % filename
	try:
	    os.stat(directory)
	except:
	    os.mkdir(directory)  
	np.save("results/%s/%s-%d.npy" % (filename, filename, N), accs)
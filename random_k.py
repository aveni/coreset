import tensorflow as tf
import numpy as np
from experiment import Experiment
from models import linear, cnn, resnet18, vgg_16
import time
import os

def balanced_random(N):
	global ix, y
	chosen = []
	for label in range (10):
		label_ix = ix[y==label]
		perm = np.random.permutation(range(len(label_ix)))[:int(N/10)]
		chosen.append(label_ix[perm])
	chosen = np.concatenate(chosen)
	np.random.shuffle(chosen)
	return chosen


dataset = "cifar"
model = cnn
trials = 1
sizes = [55000]
y = np.load("./datasets/%s_with_val.npz" % dataset)["y_train"]
ix = np.arange(0, len(y), 1).astype("int32")
e = Experiment(model, dataset)

for N in sizes:
	accs = []
	for T in range(trials):
		start = time.time()
		print "N=%d, Trial=%d" % (N, T)
		acc = e.get_test_acc(train_ix=balanced_random(N), epochs=100, lr=0.0001, train_bs=min(100, max(1, N/100)), verbose=True, save_dir=None)
		accs.append(acc)
		print "Trial Acc=%s (%.2f s)" % (str(acc), time.time()-start)

	filename = "%s-%s-random" % (dataset, model.__name__)
	directory = "results/%s" % filename
	try:
	    os.stat(directory)
	except:
	    os.mkdir(directory)  
	# np.save("results/%s/%s-%d.npy" % (filename, filename, N), accs)

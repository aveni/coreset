import tensorflow as tf
import numpy as np
from experiment import Experiment
from models import linear, cnn, resnet18, vgg_16
import time
import os
from scipy.spatial.distance import pdist, squareform

def balanced_k_center(k):
  global dense, ix, y
  centers = []
  for label in range (10):
    label_ix = ix[y==label]
    label_dense = dense[y==label]
    label_centers = sub_k_center(label_dense, int(k/10))
    centers.append(label_ix[label_centers])
  return np.concatenate(centers)

def any_k_center(k):
	global dense, ix, y
	centers = sub_k_center(dense, k)
	return ix[centers].astype("int32")

def sub_k_center(data, k):
  c = np.random.choice(data.shape[0], 1)[0]
  centers = [c]
  closest = np.linalg.norm(data-data[c], axis=1)
  for i in range(k-1):
    if i%100 ==0:
    	print i
    farthest = np.argmax(closest, axis=0)
    centers.append(farthest)
    closest = np.min([closest, np.linalg.norm(data-data[farthest], axis=1)], axis=0)

  return centers

def sub_k_center2(k):
  pairwise = np.load("results/mnist-vgg_16-pairwise.npy")
  c = np.random.choice(data.shape[0], 1)[0]
  centers = [c]
  for i in range(k-1):
    farthest = np.argmax(np.min(pairwise[centers], axis=0))
    centers.append(farthest)
    
  return centers


dataset = "mnist"
model = vgg_16
trials = 1
sizes = [5000]
y = np.load("./datasets/%s_with_val.npz" % dataset)["y_train"]
ix = np.arange(0, len(y), 1).astype("int32")
e = Experiment(model, dataset)

# # Get Dense Activations, ONLY NEEDS TO BE DONE ONCE!
# e.get_dense_activations()

dense = np.load("results/%s-%s-kcenter/%s-%s-dense_activations.npy" % 
  (dataset, "cnn", dataset, "cnn"))
ix, y = np.load("results/%s-%s-kcenter/%s-%s-dense_ix_y.npy" % 
  (dataset, "cnn", dataset, "cnn"))


for N in sizes:
  accs = []
  for T in range(trials):
    start = time.time()
    print "N=%d, Trial=%d" % (N, T)

    acc = e.get_test_acc(train_ix=any_k_center(N), epochs=40, lr=0.0001, train_bs=min(100, max(1, N/100)), verbose=True, save_dir="baselines")
    accs.append(acc)
    print "Trial Acc=%s (%.2f s)" % (str(acc), time.time()-start)

  filename = "%s-%s-kcenter" % (dataset, model.__name__)
  directory = "results/%s" % filename
  try:
      os.stat(directory)
  except:
      os.mkdir(directory)  
  # np.save("results/%s/%s-%d.npy" % (filename, filename, N), accs)

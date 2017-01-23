import scipy.io as sio
import cv2
import h5py
import numpy as np

print "load gt.mat"
gt = sio.loadmat("gt.mat")
size = np.zeros((3, gt['imnames'].shape[1]))

print "reading images..."
for idx in xrange(gt['imnames'].shape[1]):
	height, width, channels = cv2.imread(str(gt['imnames'][0, idx][0])).shape
	size[0, idx] = float(height)
	size[1, idx] = float(width)
	size[2, idx] = float(channels)
	if idx % 10000 == 0:
		print str(idx) + " images..."
		print size[:, idx]
print "reading images finished..."

print "store size..."
with h5py.File('size.h5', 'w') as f:
	size_store = f.create_dataset('size', size.shape)
	size_store[:] = size[:]


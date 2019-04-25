import os
import numpy as np
from keras import Input
from keras.optimizers import Adam
from keras.utils import multi_gpu_model
from config import *
from model import unet
from librosa.util import find_files

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID" 
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"

def load_npz(target=None, first=None):
    npz_files = find_files('numpy', ext="npz")[:first]
    for file in npz_files:
        npz = np.load(file)
        assert(npz["mix"].shape == npz[target].shape)
        yield npz['mix'], npz[target]
mix_mag, target_mag = zip(*load_npz(target='vocal', first=-1))

model = unet()
raw_model = model
model.compile(optimizer=Adam(1e-4), loss='mean_absolute_error')

EPOCH = 100
BATCH = 30
SAMPLING_STRIDE = 10

def sampling(mix_mag, target_mag):
    X, y = [], []
    for mix, target in zip(mix_mag, target_mag):
        starts = np.random.randint(0, mix.shape[1] - PATCH_SIZE, 
(mix.shape[1] - PATCH_SIZE) // SAMPLING_STRIDE)
        for start in starts:
            end = start + PATCH_SIZE
            X.append(mix[1:, start:end, np.newaxis])
            y.append(target[1:, start:end, np.newaxis])
    return np.asarray(X, dtype=np.float32), np.asarray(y, dtype=np.float32)

def main():
	for e in range(EPOCH):
		X, y = sampling(mix_mag, target_mag)
		model.fit(X, y, batch_size=BATCH, verbose=1, 
validation_split=0.05)
		model.save('vocal_{:0>2d}.h5'.format(e+1), overwrite=True)
	
	
if __name__ == "__main__":
	main()

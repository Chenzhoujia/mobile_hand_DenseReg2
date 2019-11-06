import numpy as np

changeList = np.load(file="F:\\chen\\pycharm\\DenseReg_baseline\\model\\exp\\train_cache\\nyu_training_s2_f128_daug_um_v1\\image\\tmp\\change.npy")

index = np.argsort(changeList[:,0]) +1
list = list(index[:10])+list(index[8250-10:])

print(list)

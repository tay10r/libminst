#!/bin/python3

import pyminst
import matplotlib.pyplot as plt
import numpy as np
import struct

class MyCallback(pyminst.Callback):
    def __init__(self):
        super().__init__()
        self.labels = [ 'T-shirt', 'Trouser', 'Pullover', 'Dress', 'Coat', 'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle Boot' ]

    def eval(self, sample: bytes, label: bytes):
        label_idx = struct.unpack('B', label)[0]
        label_str = self.labels[label_idx]

        s = np.frombuffer(sample, dtype=np.uint8).reshape(28, 28)
        plt.title(label_str)
        plt.imshow(s)
        plt.show()

        pass

sample_fmt = pyminst.Format()
sample_fmt.type_ = pyminst.Type.U8
sample_fmt.shape = (60000, 28, 28)

label_fmt = pyminst.Format()
label_fmt.type_ = pyminst.Type.U8
label_fmt.shape = (60000,)

pyminst.eval('train-images-idx3-ubyte',
             'train-labels-idx1-ubyte',
             sample_fmt,
             label_fmt,
             batch_size=1,
             callback=MyCallback(),
             sampler=pyminst.RandomSampler(0))

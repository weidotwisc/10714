from typing import List, Optional
from ..data_basic import Dataset
import numpy as np
import gzip
import struct
class MNISTDataset(Dataset):
    def __init__(
        self,
        image_filename: str,
        label_filename: str,
        transforms: Optional[List] = None,
    ):
        ### BEGIN YOUR SOLUTION
        self.image_filename = image_filename
        self.label_filename = label_filename
        self.transforms = transforms
        self.imgs, self.labels = self.load()
        assert(len(self.imgs) == len(self.labels))
        self.n = len(self.labels)
        ### END YOUR SOLUTION

    def __getitem__(self, index) -> object:
        ### BEGIN YOUR SOLUTION
        return self.apply_transforms(self.imgs[index]), self.labels[index]
        ### END YOUR SOLUTION

    def __len__(self) -> int:
        ### BEGIN YOUR SOLUTION
        return self.n
        ### END YOUR SOLUTION

    def load(self):
        with gzip.open(self.label_filename, 'rb') as lbpath:
            magic, n = struct.unpack('>ii', lbpath.read(
                8))  # > means big-endian, i means int, two iis mean we need to read two numbers
            # print(magic, n)
            labels = np.frombuffer(lbpath.read(),
                                   dtype=np.uint8)  # use np.frombuffer, apparently the previous lbapth.read(8) already moves the pointer to the proper data region
            assert (len(labels) == n)
            y = labels
            # print(np.max(labels), np.min(labels)) # labels from 0 to 9
        with gzip.open(self.image_filename, 'rb') as imgpath:
            magic, n, rows, cols = struct.unpack('>iiii', imgpath.read(16))
            #images = np.frombuffer(imgpath.read(), dtype=np.uint8).reshape(len(labels), 784)
            images = np.frombuffer(imgpath.read(), dtype=np.uint8).reshape(len(labels), rows, cols, 1) # in hw0, it was hardcoded as 784, now i need to make it H W C form
            assert (len(images) == n)
            # print(images.shape)
            X = images.astype(np.float32) / 255
        return X, y
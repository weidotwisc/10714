import os
import pickle
from typing import Iterator, Optional, List, Sized, Union, Iterable, Any
import numpy as np
from ..data_basic import Dataset

class CIFAR10Dataset(Dataset):
    def __init__(
        self,
        base_folder: str,
        train: bool,
        p: Optional[int] = 0.5,
        transforms: Optional[List] = None
    ):
        """
        Parameters:
        base_folder - cifar-10-batches-py folder filepath
        train - bool, if True load training dataset, else load test dataset
        Divide pixel values by 255. so that images are in 0-1 range.
        Attributes:
        X - numpy array of images
        y - numpy array of labels
        """
        ### BEGIN YOUR SOLUTION
        self.transforms = transforms
        if(train):
            data_list = []
            labels_list = []
            for i in range(1,6):
                data,labels = self.load_data(os.path.join(base_folder, "data_batch_"+str(i)))
                data_list.append(data)
                labels_list.append(labels)
            self.X = np.concatenate(data_list, axis=0)
            self.y = np.concatenate(labels_list, axis=0)
        else:
            self.X, self.y = self.load_data(os.path.join(base_folder, "test_batch"))

        ### END YOUR SOLUTION

    def load_data(self, file_name):
        with open(file_name, 'rb') as fo:
            cifar_data_file = pickle.load(fo, encoding='bytes')
            data = cifar_data_file[b'data'] # numpy.ndarray shape: (#samples, 3072)
            data = data.reshape(-1,3,32,32)
            labels = cifar_data_file[b'labels'] # a list of ints
            labels = np.array(labels)
            return data / 255, labels # divide by 255, so the images are in 0-1 range



    def __getitem__(self, index) -> object:
        """
        Returns the image, label at given index
        Image should be of shape (3, 32, 32)
        """
        ### BEGIN YOUR SOLUTION
        return self.apply_transforms(self.X[index]), self.y[index] # weiz 2024-11-09 add the apply_transforms
        ### END YOUR SOLUTION

    def __len__(self) -> int:
        """
        Returns the total number of examples in the dataset
        """
        ### BEGIN YOUR SOLUTION
        return len(self.y)
        ### END YOUR SOLUTION

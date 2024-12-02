import os

import numpy as np
from needle import backend_ndarray as nd
from needle import Tensor
from ..data_basic import Dataset

class Dictionary(object):
    """
    Creates a dictionary from a list of words, mapping each word to a
    unique integer.
    Attributes:
    word2idx: dictionary mapping from a word to its unique ID
    idx2word: list of words in the dictionary, in the order they were added
        to the dictionary (i.e. each word only appears once in this list)
    """
    def __init__(self):
        self.word2idx = {}
        self.idx2word = []

    def add_word(self, word):
        """
        Input: word of type str
        If the word is not in the dictionary, adds the word to the dictionary
        and appends to the list of words.
        Returns the word's unique ID.
        """
        ### BEGIN YOUR SOLUTION
        if word not in self.word2idx:
            self.word2idx[word] = len(self.idx2word)
            self.idx2word.append(word)
        ### END YOUR SOLUTION

    def __len__(self):
        """
        Returns the number of unique words in the dictionary.
        """
        ### BEGIN YOUR SOLUTION
        return len(self.idx2word)
        ### END YOUR SOLUTION



class Corpus(object):
    """
    Creates corpus from train, and test txt files.
    """
    def __init__(self, base_dir, max_lines=None):
        self.dictionary = Dictionary()
        self.train = self.tokenize(os.path.join(base_dir, 'train.txt'), max_lines)
        self.test = self.tokenize(os.path.join(base_dir, 'test.txt'), max_lines)

    def tokenize(self, path, max_lines=None):
        """
        Input:
        path - path to text file
        max_lines - maximum number of lines to read in
        Tokenizes a text file, first adding each word in the file to the dictionary,
        and then tokenizing the text file to a list of IDs. When adding words to the
        dictionary (and tokenizing the file content) '<eos>' should be appended to
        the end of each line in order to properly account for the end of the sentence.
        Output:
        ids: List of ids
        """
        ### BEGIN YOUR SOLUTION
        ids = []
        with open(path, 'r', encoding='utf-8') as f:
            for i, line in enumerate(f):
                if max_lines is not None and i >= max_lines:
                    break
                words = line.strip().split() + ['<eos>']  # Add <eos> at the end of the line
                for word in words:
                    self.dictionary.add_word(word)  # Add word to the dictionary
                ids.extend(self.dictionary.word2idx[word] for word in words)  # Convert to IDs
        return ids
        ### END YOUR SOLUTION


def batchify(data, batch_size, device, dtype):
    """
    Starting from sequential data, batchify arranges the dataset into columns.
    For instance, with the alphabet as the sequence and batch size 4, we'd get
    ┌ a g m s ┐
    │ b h n t │
    │ c i o u │
    │ d j p v │
    │ e k q w │
    └ f l r x ┘.
    These columns are treated as independent by the model, which means that the
    dependence of e. g. 'g' on 'f' cannot be learned, but allows more efficient
    batch processing.
    If the data cannot be evenly divided by the batch size, trim off the remainder.
    Returns the data as a numpy array of shape (nbatch, batch_size).
    """
    ### BEGIN YOUR SOLUTION
    per_batch_seq_len = len(data) // batch_size
    trimmed_data = data[: per_batch_seq_len*batch_size]
    return (np.reshape(trimmed_data, (batch_size, per_batch_seq_len))).T # notice that we need to take a transpose as columns are consecutive letters
    #return (np.reshape(trimmed_data, (per_batch_seq_len, batch_size))) # but this seems to get better training loss for the tests! weiz 2024-11-25

    ### END YOUR SOLUTION


def get_batch(batches, i, bptt, device=None, dtype=None):
    """
    get_batch subdivides the source data into chunks of length bptt.
    If source is equal to the example output of the batchify function, with
    a bptt-limit of 2, we'd get the following two Variables for i = 0:
    ┌ a g m s ┐ ┌ b h n t ┐
    └ b h n t ┘ └ c i o u ┘
    Note that despite the name of the function, the subdivison of data is not
    done along the batch dimension (i.e. dimension 1), since that was handled
    by the batchify function. The chunks are along dimension 0, corresponding
    to the seq_len dimension in the LSTM or RNN.
    Inputs:
    batches - numpy array returned from batchify function
    i - index
    bptt - Sequence length
    Returns:
    data - Tensor of shape (bptt, bs) with cached data as NDArray
    target - Tensor of shape (bptt*bs,) with cached data as NDArray
    """
    ### BEGIN YOUR SOLUTION
    total_seq_len_per_batch, bs = batches.shape
    if ( (i+1)+bptt < total_seq_len_per_batch ):
        X = batches[i:i+bptt, :]
        Y = batches[i+1:i+1+bptt, :]
    else:
        X = batches[i:-1, :]
        Y = batches[i+1:, :]
    X_t = Tensor(X, device=device, dtype=dtype, requires_grad=False)
    Y_t = Tensor(Y.reshape(-1), device=device, dtype=dtype, requires_grad=False)
    print(X_t.shape)
    print(Y_t.shape)
    return X_t, Y_t
    ### END YOUR SOLUTION


class PTBDataset(Dataset):
    def __init__(
        self,
        batchified_data: np.ndarray , # result of batchify(), shape is (per_batch_seq_len, batch_size) , each element is the word's id in the dictionary
        seq_len : int, # seq length i.e., bptt
        device, 
        dtype
        # base_folder: str,
        # train: bool,
        # p: Optional[int] = 0.5,
        # transforms: Optional[List] = None
    ):
        self.batchified_data = batchified_data
        self.seq_len = seq_len
        self.per_batch_seq_len, self.bs = batchified_data.shape
        self.device = device
        self.dtype = dtype

    def __getitem__(self, index) -> object:
        """
        Returns the batched sequence at in the index
        Each returned sample should be X, y
        X: Tensor, shape of (seq_len, bs)
        Y: Tensor, shape of (seq_len * bs, )
        Simply call get_batch() method
        """
        if index >= len(self):  # Check if the index is out of bounds
            raise IndexError("Index out of range")
        return get_batch(self.batchified_data, index*self.seq_len, self.seq_len, self.device, self.dtype)

    def __len__(self) -> int:
        """
        Returns the total number of examples in the dataset
        """
        if (self.per_batch_seq_len % self.seq_len == 0):
            return self.per_batch_seq_len // self.seq_len
        else:
            return self.per_batch_seq_len // self.seq_len + 1 
        #return self.per_batch_seq_len // self.seq_len
        
        
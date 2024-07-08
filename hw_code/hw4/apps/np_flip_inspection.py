import numpy as np
import ctypes

def raw_data(X):
    X = np.array(X) # copy, thus compact X
    return np.frombuffer(ctypes.string_at(X.ctypes.data, X.nbytes), dtype=X.dtype, count=X.size)

# Xold and Xnew should reference the same underlying data
def offset(Xold, Xnew):
    assert Xold.itemsize == Xnew.itemsize
    # compare addresses to the beginning of the arrays
    return (Xnew.ctypes.data - Xold.ctypes.data)//Xnew.itemsize

def strides(X):
    return ', '.join([str(x//X.itemsize) for x in X.strides])

def format_array(X, shape):
    assert len(shape) == 3, "I only made this formatting work for ndims = 3"
    def chunks(l, n):
        n = max(1, n)
        return (l[i:i+n] for i in range(0, len(l), n))
    a = [str(x) if x >= 10 else ' ' + str(x) for x in X]
    a = ['(' + ' '.join(y) + ')' for y in [x for x in chunks(a, shape[-1])]]
    a = ['|' + ' '.join(y) + '|' for y in [x for x in chunks(a, shape[-2])]]
    return '  '.join(a)

def inspect_array(X, *, is_a_copy_of):
    # compacts X, then reads it off in order
    print('Data: %s' % format_array(raw_data(X), X.shape))
    # compares address of X to copy_of, thus finding X's offset
    print('Offset: %s' % offset(is_a_copy_of, X))
    print('Strides: %s' % strides(X))

print("----------")
print("Inspection 1: A: A = np.arange(1, 25).reshape(3, 2, 4)")
A = np.arange(1, 25).reshape(3, 2, 4)
inspect_array(A, is_a_copy_of=A)
print("----------")


print("Inspection 2: np.flip(A, (2,))")
inspect_array(np.flip(A, (2,)), is_a_copy_of=A)
print("----------")
print("Inspection 3: np.flip(A, (1,))")
inspect_array(np.flip(A, (1,)), is_a_copy_of=A)
print("----------")
print("Inspection 4: np.flip(A, (0,))")
inspect_array(np.flip(A, (0,)), is_a_copy_of=A)
print("----------")
print("Inspection 5: np.flip(A, (0,1,2))")
inspect_array(np.flip(A, (0,1,2)), is_a_copy_of=A)
print("----------")

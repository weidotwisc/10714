import numpy as np
import torch
import torch.nn as nn
import argparse

# pyt conv:
# Input: NCHW
# Weights: OIKK
def conv_reference_pyt(Z, W, _dtype=torch.float32):
    out = nn.functional.conv2d(torch.Tensor(Z).to(_dtype), torch.Tensor(W).to(_dtype))
    return out.contiguous().numpy()

def conv_naive_pyt(Z, Weights):
    # Z: NCHW
    # W: OIKK
    N, C, H, W = Z.shape
    O, I, K, _ = Weights.shape
    assert(C==I)
    out = np.zeros((N,O,H-K+1, W-K+1))#.astype(int)
    # Out: NOH'W'
    for n in range(N):
        for c_out in range(O):
            for h in range (H-K+1):
                for w in range (W-K+1):
                    for c_in in range(I):
                        for k_h in range (K):
                            for k_w in range(K):
                                out[n,c_out,h,w] += Z[n,c_in,h+k_h,w+k_w] * Weights[c_out,c_in,k_h,k_w]
    return out

'''
    print all the locations that out and out2 are different
'''
def print_diff(out, out2, cutoff=10):
    diff = np.nonzero(out-out2)
    if(len(diff[0])==0):
        print("Two tensors are the same")
    else:
        num_diff_locs = len(diff[0])
        for i in range(num_diff_locs):
            diff_loc = [diff[d][i] for d in range(len(diff))]
            print(f"loc {diff_loc} is different")
            if i == cutoff-1:
                print("...")
                return

# Z: NCHW
# Weights: OIKK
# loc: NCHW
def investigate_loc(Z, Weights, loc:tuple):
    # for one specific output location, calculate its result
    K = Weights.shape[-1]
    n, c_out, h, w = loc
    kernels = Weights[(c_out,),:]
    inputs = Z[(n,),:, h:h+K, w:w+K]

    # Method 1 use pyt conv2d do a whole conv2d and retrieve the value at loc. floats don't seem to work here
    result_ref = nn.functional.conv2d(torch.Tensor(Z), torch.Tensor(Weights)).contiguous().numpy()
    print(f"pyt conv2d (whole) at {loc} as float32 is {result_ref[loc]}")

    result_ref = nn.functional.conv2d(torch.Tensor(Z).int(), torch.Tensor(Weights).int()).contiguous().numpy()
    print(f"pyt conv2d (whole) at {loc} as int32 is {result_ref[loc]}")

    # Method 2 use pyt conv2d on only the loc-relavent portion to get the result. floats work here
    #print(inputs.shape)
    #print(kernels.shape)
    result_ref_loc = nn.functional.conv2d(torch.Tensor(inputs), torch.Tensor(kernels)).item()
    print(f"pyt conv2d (single) at {loc} as float is {result_ref_loc}")
    
    # Method 3 use naive 7-loop impl, both floats and ints work here
    result = np.sum(kernels*inputs)
    print(f"naive conv2d at {loc} is {result}")

    

    



def investigate_conv_pyt():
 
    _N=1
    _C=8
    _H=9
    _W=9
    _O=16
    _I=_C
    _K=3
    
   
    Z = np.arange(_N*_C*_H*_W).reshape((_N, _C, _H, _W))#.astype(int)
    #Z = np.ones((_N, _C, _H, _W)).astype(int)
    Weights = np.arange(_O*_I*_K*_K).reshape((_O, _I, _K, _K))#.astype(int)
   
    out = conv_reference_pyt(Z,Weights)
    out2 = conv_naive_pyt(Z,Weights)
    
    #print(out.shape)
    #print(out2.shape)
    print("----------")
    print("Point 1: pyt conv2d() and naive conv2d() generate different results")
    print(f"difference 2-norm: {np.linalg.norm(out-out2)}")
    diff = np.nonzero(out-out2)
    print("----------")
    print("Point 2: many elements in the output tensors of pyt conv2d() and naive conv2d() are different ")
    print(f"{len(diff[0])} elements out of {out.size} are different!")
    print("For brevity, I only print the first 10 elements")
    print_diff(out, out2, cutoff=10)
    print("----------")
    # investigate location (0,12,4,6)
    print("Point 3: we pick the first different element and only convolve the relevant inputs and kernels")
    print("pyt seems to generate correct result, regardless of treating them as ints or floats")
    print("This seems to indicate only when treating with larger tensors ints vs floats will become an issue for pyt conv2d()")
    loc=(0,12,4,6)
    investigate_loc(Z, Weights, loc)
    print("----------")



def minimal_bug_exposing_case():
    _N=1
    _C=8
    _H=9
    _W=9
    _O=16
    _I=_C
    _K=3
    
   
    Z = np.arange(_N*_I*_H*_W).reshape((_N, _I, _H, _W))#.astype(int)
    Weights = np.arange(_O*_I*_K*_K).reshape((_O, _I, _K, _K))#.astype(int)
    out = conv_reference_pyt(Z,Weights)
    out1 = conv_reference_pyt(Z, Weights, _dtype=torch.int32)
    out2 = conv_naive_pyt(Z,Weights)
    # pyt doesn't agree w/ itself 
    print(f"difference 2-norm pyt(float) vs pyt(int): {np.linalg.norm(out-out1)}")
    # pyt int seems to be consisten with naive impl
    print(f"difference 2-norm pyt(int) vs naive: {np.linalg.norm(out1-out2)}")








def get_parser():
    parser = argparse.ArgumentParser(description='A test case to expose one PyTorch Conv2D bug')
    parser.add_argument("--investigate", action="store_true", help="When turned on , detail the investigation process")
    return parser


def main():
    parser = get_parser()
    FLAGS, unparsed = parser.parse_known_args()
    assert(len(unparsed)==0) 
    if(FLAGS.investigate):
        investigate_conv_pyt()
    else:
        minimal_bug_exposing_case()

if __name__ == "__main__":
    main()
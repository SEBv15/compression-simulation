import numpy as np
from numba import jit
from functools import reduce

# Takes an array of n 8-bit values and returns an array of 8 n-bit values
@jit(nopython=True)
def shuffle(pixels: "np.ndarray[int]") -> "np.ndarray[int]":
    out = np.zeros(8, dtype=np.uint64) # assume 8-bit pixel data
    for i in range(0, 8):
        for j in range(0, pixels.shape[0]):
            out[i] <<= 1
            out[i] += pixels[j] & 2**i
    
    return out

# creates a mask and prepends it to the array with zeros removed
@jit(nopython=True)
def compress(pixels: "np.ndarray[int]") -> "np.ndarray[int]":
    #mask = reduce(lambda a, b : a+b, [2**i if e > 0 else 0 for i, e in enumerate(pixels)])
    mask = 0
    for p in pixels:
        mask <<= 1
        mask += p > 0
    out = pixels[np.where(pixels > 0)]
    return np.concatenate((np.asarray([mask]), out))

@jit(nopython=True)
def shuffle_compress(pixels: "np.ndarray[int]"):
    return compress(shuffle(pixels)) # not optimum, since input is 8 n-bit pixels, but the mask will only use 8 bits. When n=16 (like in test_random_data), it would make sense to pass in two shuffles together

def main():
    print("simple shuffle test")
    pixels = np.asarray([1,2,3,4,5,6])
    print([str(bin(e)) for e in pixels])
    print([str(bin(e)) for e in shuffle(pixels)])

    print("simple compression test")
    arr = np.asarray([2,0,0,3,4,5,0,5])
    print(arr, compress(arr), bin(compress(arr)[0]))

if __name__ == "__main__":
    main()

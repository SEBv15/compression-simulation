import numpy as np
from numba import jit
from functools import reduce

# Takes an array of n 8-bit values and returns an array of 8 n-bit values
@jit(nopython=True)
def shuffle(pixels: "np.ndarray[int]", bits_per_pixel: int = 10) -> "np.ndarray[int]":
    out = np.zeros(bits_per_pixel, dtype=np.uint64) # assume 8-bit pixels and generate empty array
    for i in range(0, bits_per_pixel):
        for j in range(0, pixels.shape[0]):
            out[i] <<= 1 # this actually doesn't work in plain python
            out[i] += pixels[pixels.shape[0] - j - 1] >> i & 1 # get the ith bit in every pixel and put it in the correspond output pixel
    return out

# creates a mask and prepends it to the array with zeros removed
@jit(nopython=True)
def compress(pixels: "np.ndarray[int]") -> "np.ndarray[int]":
    #mask = reduce(lambda a, b : a+b, [2**i if e > 0 else 0 for i, e in enumerate(pixels)])

    # generate mask
    mask = int(0)
    for p in pixels:
        mask <<= 1
        mask += p > 0

    # remove zero pixels
    out = pixels[np.where(pixels > 0)]

    # prepend mask to pixels
    return np.concatenate((np.asarray([mask], dtype=np.uint64), out))

@jit(nopython=True)
def shuffle_compress(pixels: "np.ndarray[int]", bits_per_pixel: int = 10):
    return compress(shuffle(pixels, bits_per_pixel=bits_per_pixel)) # not optimum, since input is 8 n-bit pixels, but the mask will only use 8 bits. When n=16 (like in test_random_data), it would make sense to pass in two shuffles together

def main():
    print("simple shuffle test")
    pixels = np.asarray([1,2,3,4,5,6])
    print([str(bin(e)) for e in pixels])
    print([str(bin(e)) for e in shuffle(pixels)])

    print("simple compression test")
    arr = np.asarray([2,0,0,3,4,5,0,5])
    print(arr, compress(arr), bin(compress(arr)[0])) # print input, output, and mask


if __name__ == "__main__":
    main()

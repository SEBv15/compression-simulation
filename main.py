import numpy as np
from numba import jit
import struct
from functools import reduce
import time

from compression import shuffle_compress

@jit(nopython=True)
def test_random_data(stop_after: int = 0):
    """
    Test compression on random data

    Arguments:
        stop_after (int): How often to run the compression

    Returns:
        tuple: The compression ratio and total number of bytes compressed
    """
    print("Testing compression on random data")
    i = 0
    o = 0
    c = 0
    size = 16 # number of pixels to shuffle and compress together (also number of bits in compressed pixels)
    while (stop_after == 0 or i < stop_after):
        data = np.random.randint(0, 2**8, size = size) # generate [size] 8-bit pixels (number of bytes = [size])
        i += size
        o += shuffle_compress(data).shape[0]*size/8
        c += 1
    
        if c % 64 == 0: # print occasionally
            print(str(size * i // 1000 // 1000) + " megabytes compressed")
            print(str(i) + ": ")
            print(i/o)

    return i/o, size * i

if __name__ == '__main__':
    t_0 = time.time()
    ratio, total_bytes = test_random_data(62500000) # ratio should be 8/9 since an extra element is added for the mask and barely any zero pixels should exist
    print("Compressing {} bytes took {} seconds".format(total_bytes, time.time() - t_0))


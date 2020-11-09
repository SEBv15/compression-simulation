import numpy as np
from numba import jit
import struct
from functools import reduce
import time
import matplotlib.pyplot as plt
import sys
import statistics 

from compression import shuffle_compress, shuffle
from decompression import deshuffle_decompress, deshuffle

@jit(nopython=True)
def test_random_data(stop_after: int = 0, bits_per_pixel: int = 10):
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
        data = np.random.randint(0, 2**bits_per_pixel, size = size) # generate [size] 8-bit pixels (number of bytes = [size])
        i += size
        o += shuffle_compress(data).shape[0]*size/bits_per_pixel
        c += 1
    
        if c % 64 == 0: # print occasionally
            print(str(i // 1000 // 1000) + " megabytes compressed")
            print(str(i) + ": ")
            print(i/o)

    return i/o, i

@jit(nopython=True)
def compress_frame(frame, pixels_per_mask: int = 16, bits_per_pixel: int = 10):
    frame = frame.ravel()
    total_out = 0
    for i in range(0, frame.shape[0] - frame.shape[0] % pixels_per_mask, pixels_per_mask):

        compressed = shuffle_compress(frame[i:i+pixels_per_mask], bits_per_pixel)
        if not np.array_equal(deshuffle_decompress(compressed, bits_per_pixel=bits_per_pixel), frame[i:i+pixels_per_mask]):
            raise Exception("arrays differ")

        compressed_size = compressed.shape[0]*pixels_per_mask/bits_per_pixel
        total_out += compressed_size

    return total_out, frame.shape[0] - frame.shape[0] % pixels_per_mask

def test_binary_file(filename: str, bytes_per_pixel: int = 4, width: int = 558, height: int = 514, pixels_per_mask: int = 16):
    center = (260, 290)
    with open(filename, mode='rb') as file: # cannot jit compile file access
        total_out = 0
        total_in = 0
        ratios = []
        while True:
            # load a frame into memory
            bytess = file.read(bytes_per_pixel * width * height)

            dt = np.dtype(np.uint32)
            dt = dt.newbyteorder('<') # ensure correct endianness
            frame = np.frombuffer(bytess, dtype=dt) # read data into numpy array

            # check if there is enough data / end of file
            if (len(frame) < width * height):
                break            

            frame = frame.reshape((height, width)) # turn into 2d array

            # comment out to compress entire picture
            sf = 4 # scale factor
            frame = frame[center[1] - 64*sf:center[1] + 64*sf:sf, center[0] - 64*sf:center[0] + 64*sf:sf] # select 128x128 area around center

            out_s, in_s = compress_frame(frame, pixels_per_mask=pixels_per_mask, bits_per_pixel=14)
            total_out += out_s
            total_in += in_s

            ratios.append(in_s/out_s)

            #np.set_printoptions(threshold=sys.maxsize) # uncomment to print entire array
            #print(frame[0,:], frame[0,:].shape[0])

            print(total_in, total_out, total_in/total_out)
            #break

            #plt.imshow(frame)
            #plt.savefig("out.png")

            #break
        print("mean: {}".format(statistics.mean(ratios)))
        print("stdev: {}".format(statistics.stdev(ratios)))
        print("min: {}".format(min(ratios)))
        print("max: {}".format(max(ratios)))


if __name__ == '__main__':
    test_binary_file("scan144_1737_cropped_558x514.bin", pixels_per_mask=16)

    #t_0 = time.time()
    #ratio, total_bytes = test_random_data(16*62500000) # ratio should be 8/9 since an extra element is added for the mask and barely any zero pixels should exist
    #print("Compressing {} bytes took {} seconds. The final ratio is {}".format(total_bytes, time.time() - t_0, ratio))


import compression
import numpy as np
from numba import jit

@jit(nopython=True)
def deshuffle(pixels: "np.ndarray[int]", pixels_per_compression: int = 16) -> "np.ndarray[int]":
    return compression.shuffle(pixels, bits_per_pixel=pixels_per_compression)

@jit(nopython=True)
def decompress(pixels: "np.ndarray[int]", bits_per_pixel: int = 10):
    i = 1
    uc = np.zeros(bits_per_pixel, dtype=np.uint64)
    for n in range(0, bits_per_pixel):
        if pixels[0] & 2**(bits_per_pixel - n - 1):
            uc[n] = pixels[i]
            i += 1

    return uc

@jit(nopython=True)
def deshuffle_decompress(pixels: "np.ndarray[int]", pixels_per_compression: int = 16, bits_per_pixel: int = 10):
    return deshuffle(decompress(pixels, bits_per_pixel=bits_per_pixel), pixels_per_compression=pixels_per_compression)

def test_deshuffle_decompress():
    bits_per_pixel = 10
    size = 16

    data = np.random.randint(0, 2**bits_per_pixel, size = size, dtype=np.uint32)
    print("OG data:", data)
    compressed = compression.shuffle_compress(data)
    print(compressed, bin(compressed[0]))
    print("Compressed data:", compressed)
    print(decompress(compressed))
    decompressed = deshuffle_decompress(compressed)
    print("Decompressed data:", decompressed)

    print("de+shuffle:", deshuffle(compression.shuffle(data)))


if __name__ == "__main__":
    test_deshuffle_decompress()

import numpy as np
from numba import int32, uint64    # import the types
from numba.experimental import jitclass

spec = [
    ('_value', uint64),
    ('bits', int32),
]

@jitclass(spec)
class Data(object):
    """
    Class that ensures data has the correct number of bits and provides convenient methods for bit manipulation
    """
    def __init__(self, data, number_of_bits):
        if (number_of_bits > 64):
            raise Exception("Using unsigned 64 bit int for data storage. Max number of bits = 64")
        if (data >= 2**number_of_bits):
            raise Exception("Data too large for given number of bits")
        self._value = data
        self.bits = number_of_bits

    def __len__(self): # This doesn't seem to work with numba
        return self.bits

    @property
    def size(self): # This will work though
        return self.bits

    def __getitem__(self, key):
        if key < self.bits:
            return self.value >> key & 1
        else:
            raise Exception("Key out of range")

    def __setitem__(self, key, value):
        if ((value == 0 or value == 1) and key < self.bits):
            self._value = (self._value & (2**self.bits - 1 - 2**key)) + value*(2**key)
        else:
            raise Exception("value or key invalid")

    def lshift_add(self, value):
        if (value == 0 or value == 1):
            self._value <<= 1
            self._value += value
        else:
            raise Exception("Invalid value")

    @property
    def value(self):
        return self._value & (2**self.bits - 1)

    def __str__(self):
        return str(bin(self.value()))

def test():
    d = Data(31,5)
    d[4] = 0
    d[0] = 0
    d.lshift_add(1)
    print(d[0], d[1])
    print(d.size)
    print(d.value)
    print(bin(d.value))

if __name__ == "__main__":
    test()

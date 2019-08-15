import numpy as np
import sys

def read_file(name):
    data = np.load(name)
    print(data)
    print(data.shape)

if __name__ == '__main__':
    read_file(sys.argv[1])    

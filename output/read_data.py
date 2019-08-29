import numpy as np
import sys
EACH = 1000
def read_file(name):
    data = np.load(name)
    print(data)
    print(data.shape)
    cal_data(data)


def cal_data(data):
    results = np.argmax(data, axis=1)
    dic1 = {0:0, 1:0, 2:0}
    dic2 = {0:0, 1:0, 2:0}
    dic3 = {0:0, 1:0, 2:0}
    final = []
    for i in range(EACH):
        dic1[results[i]] += 1
    for i in range(EACH, 2*EACH):
        dic2[results[i]] += 1
    #for i in range(2*EACH, 3*EACH):
    #    dic3[results[i]] += 1
    print("the first part: key {} value {}, key {} value {}, key {} value {}".format(0, dic1[0], 1, dic1[1], 2, dic1[2]))
    print("the second part: key {} value {}, key {} value {}, key {} value {}".format(0, dic2[0], 1, dic2[1], 2, dic2[2]))
    print("the third part: key {} value {}, key {} value {}, key {} value {}".format(0, dic3[0], 1, dic3[1], 2, dic3[2]))

if __name__ == '__main__':
    read_file(sys.argv[1])    

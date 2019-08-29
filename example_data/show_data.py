from __future__ import print_function
import json
from pandas.io.json import json_normalize
import sys
import numpy as np


def printhead(name):
    format = name.split('.')[1]
    print(format)
    if (format == 'json'):
        data = json.load(open(name))
        print(type(data))
        data_pd = json_normalize(data)
        print(data_pd.head())
        #print(data_pd['nodes'][0][0].keys())
        #classrange(data_pd)
        printGraph(data_pd)
    else:
        data = np.load(open(name))
        print(data.shape)

def classrange(data_pd):
    """not correct"""
    max = data_pd[1].max()
    min = data_pd[1].min()
    print("max:" + max + " min:" + min)

def printGraph(data_pd):
    print(data_pd['nodes'][0][0])
    print("number of record: {}".format(len(data_pd['nodes'][0][0])))
    print("number of links: {}".format(len(data_pd['links'][0])))
    print("number of features: {}".format(len(data_pd['nodes'][0][0]['feature'])))
    print("number of label: {}".format(len(data_pd['nodes'][0][0]['label'])))
    print("number of nodes: {}".format(len(data_pd['nodes'][0])))


if __name__ == "__main__":
    printhead(sys.argv[1])

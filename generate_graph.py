import networkx as nx 
from networkx.readwrite import json_graph
import json
import numpy as np 

NUMBER = 2000
EACH = 1000

FILENAME = 'data1/ex3'
def createG(n, label, val, test):
    m = n / 1000
    n = n / m
    #G = nx.grid_2d_graph(m, n)
    G = nx.complete_graph(n)
    #G = nx.path_graph(n)
    labels = label
    vals = val
    tests = test
    # print(G.degree())
    feats = np.ones((EACH,1))
    # feats = np.random.rand(EACH,10)
    nx.set_node_attributes(G, 'label', labels)
    nx.set_node_attributes(G, 'val', vals)
    nx.set_node_attributes(G, 'test', tests)
    return G, feats

def createId():
    dic = {}
    for i in range(NUMBER):
        dic[i] = i
    with open (FILENAME + '-id_map.json', 'w') as outfile:
        json.dump(dic, outfile)

def createClass():
    dic = {}
    for i in range(EACH):
        dic[i] = 0
    for i in range(EACH, 2*EACH):
        dic[i] = 1
    # for i in range(20000, 30000):
    #     dic[i] = 2
    with open (FILENAME + '-class_map.json', 'w') as outfile:
        json.dump(dic, outfile)

def createG2(n):
    G = nx.barbell_graph(n, n)
    dic1 = {}
    dic2 = {}
    for i in range(n):
        dic1[i] = 0
    for i in range(n, 2*n):
        dic2[i] = 1
    print("checkpoint1")
    nx.set_node_attributes(G, 'label', dic1)
    print("checkpoint2")
    nx.set_node_attributes(G, 'label', dic2)
    print('3')
    nx.set_node_attributes(G, 'val', False)
    nx.set_node_attributes(G, 'test', False)
    f = json_graph.node_link_data(G)
    print(type(f))
    with open (FILENAME + '-G.json', 'w') as outfile:
        json.dump(f, outfile)
    print("create feature")
    print(G.number_of_nodes())
    feats = nx.to_numpy_matrix(G, sorted(G.nodes()))
    print("feats size:{}".format(feats.shape))
    np.save(FILENAME + '-feats.npy', feats)



def createGall():
    G1, feats1 = createG(EACH, 0, False, False)
    G2, feats2 = createG(EACH, 1, False, False)
    # G3, feats3 = createG(EACH, 2, False, False)
    print("create finished")
    # G = nx.disjoint_union(nx.disjoint_union(G1, G2),G3)

    G = nx.disjoint_union(G1, G2)
    print("if has edge: {}".format(G.has_edge(EACH/2, EACH+3)))
    # G.add_edge(5000, 15000)
    # G.add_edge(15001, 25000)
    G.add_edge(EACH/2, EACH+3)
    print("number of nodes:{}".format(G.number_of_nodes()))
    # print(nx.to_numpy_matrix(G, sorted(G.nodes())))
    f = json_graph.node_link_data(G)
    print(type(f))
    with open (FILENAME + '-G.json', 'w') as outfile:
        json.dump(f, outfile)
    print("create feature")
    print(feats1.shape)
    # feats = np.concatenate((feats1, feats2), axis=0)
    # feats = np.concatenate((feats, feats3), axis=0)
    feats = nx.to_numpy_matrix(G, sorted(G.nodes()))
    print("feats size:{}".format(feats.shape))
    np.save(FILENAME + '-feats.npy', feats)

def main():
    createGall()
    createId()
    createClass()


if __name__ == '__main__':
    print("start ...")
    main()

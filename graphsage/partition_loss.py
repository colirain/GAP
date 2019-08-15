# import numpy as np
import tensorflow as tf

# def getConst():


def gap_loss(preds, D, A):
    """
    This module implement the loss function in paper [Azada Zazi, Will Hang. et al, 2019] Nazi, Azade & Hang, Will & Goldie, Anna & Ravi, Sujith & Mirhoseini, Azalia. (2019). GAP: Generalizable Approximate Graph Partitioning Framework. 
    Args:
        preds (tensor(float)): output predited value, have size n x g
        D (tensor(float)): degree of nodes, have size n x 1
        A (tensor(bool)): adjacent matrix of graph, have size n x n
    Returns:
        float: the results of the loss function
    """
    temp = tf.matmul(tf.transpose(preds), D)
    temp = tf.div(preds, temp)
    temp = tf.matmul(temp, tf.transpose(1-preds))
    temp = tf.multiply(temp, A)
    return tf.reduce_sum(temp)



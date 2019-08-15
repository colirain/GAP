from __future__ import division
from __future__ import print_function

import os
import time
import tensorflow as tf
import numpy as np

from graphsage.models import FCPartition
from graphsage.minibatch import EdgeMinibatchIterator
from graphsage.neigh_samplers import UniformNeighborSampler
from graphsage.utils import load_graph_data, load_embedded_data

seed = 123
np.random.seed(seed)
tf.set_random_seed(seed)

flags = tf.app.flags
FLAGS = flags.FLAGS

flags.DEFINE_string('train_prefix', '', 'name of the object file that stores the training data. must be specified.')
flags.DEFINE_string('graph_prefix', '', 'name of the object file that stores the graph data. must be specified.')
flags.DEFINE_string('outDir', 'output', 'name of the output file. must be specified.')

flags.DEFINE_integer('epochs', 1, 'number of epochs to train.')
flags.DEFINE_float('dropout', 0.0, 'dropout rate (1 - keep probability).')
flags.DEFINE_float('weight_decay', 0.0, 'weight for l2 loss on embedding matrix.')
flags.DEFINE_integer('dim_1', 128, 'Size of output dim (final is 2x this, if using concat)')
flags.DEFINE_integer('dim_2', 128, 'Size of output dim (final is 2x this, if using concat)')
flags.DEFINE_float('learning_rate', 0.00001, 'initial learning rate.')

def construct_placeholders(num_classes):
    # Define placeholders
    placeholders = {
        'labels' : tf.placeholder(tf.float32, shape=(None, num_classes), name='labels'),
        'features' : tf.placeholder(tf.float32, shape=(None), name='features'),
        'dropout': tf.placeholder_with_default(0., shape=(), name='dropout'),
        # 'batch_size' : tf.placeholder(tf.int32, name='batch_size'),
        'D' : tf.placeholder(tf.float32, shape=(None,1), name='D'),
        'A' : tf.placeholder(tf.bool, shape=(None), name='A')
    }
    return placeholders

def log_dir():
    log_dir = FLAGS.outDir
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    return log_dir

def train(train_data, graph_data, test_data=None):
    # build placeholder
    num_classes = 3
    placeholders = construct_placeholders(num_classes)
    
    # build feed_dict
    feed_dict =  dict()
    train_data = train_data.astype('float32')
    graph_id = graph_data[0].astype('float32')
    graph_aj = graph_data[1].astype('bool')
    # test_data = test_data.astype('float32')
    dim = []
    # print("f:{}".format(len(train_data[0])))
    dim.append(len(train_data[0]))
    dim.append(FLAGS.dim_1)
    dim.append(num_classes)
    feed_dict.update({placeholders['features']: train_data})
    feed_dict.update({placeholders['dropout']: FLAGS.dropout})
    feed_dict.update({placeholders['D']: graph_id})
    feed_dict.update({placeholders['A']: graph_aj})
    
    # print(type(train_data[0][0]))
    # build model
    model = FCPartition(placeholders, dim)
    print("done bulding model")

    # Init session
    sess = tf.Session()

    sess.run(tf.global_variables_initializer())

    # train model
    loss = []
    for epoch in range(FLAGS.epochs):
        print('Epoch :%04d' % (epoch+1))
        t = time.time()
        outs = sess.run([model.opt_op, model.loss, model.outputs], feed_dict=feed_dict)
        train_cost = outs[1]
        print("Iter:{} Train_cost:{}".format(epoch+1, train_cost))
        loss.append(train_cost)
    DIR = log_dir()
    predic = outs[2]
    with open(DIR + '/loss.npy', 'w') as f:
        np.save(f, loss)
    with open(DIR + '/predic.npy', 'w') as f:
        np.save(f, predic)


def main(argv=None):
    print("loading data")
    train_data = load_embedded_data(FLAGS.train_prefix)
    graph_data = load_graph_data(FLAGS.graph_prefix)
    print("done loading data")
    train(train_data, graph_data)

if __name__ == '__main__':
    tf.app.run()


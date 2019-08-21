
import tensorflow as tf 
import numpy as np 

from graphsage.models import FCPartition
from graphsage.partition_train import construct_placeholders
from graphsage.utils import load_graph_data, load_embedded_data

flags = tf.app.flags
FLAGS = flags.FLAGS

# flags.DEFINE_integer('dim_1', 128, 'Size of output dim (final is 2x this, if using concat)')



# DIR = 'trained_models'
# MODEL = 'partition'



# with tf.Session() as sess:
#     new_saver = tf.train.import_meta_graph(DIR+'/'+MODEL+'.ckpt.meta')
#     new_saver.restore(sess, tf.train.latest_checkpoint(DIR + '/./'))
#     new_saver.run()
#     print(new_saver)

def predict(train_data):
    num_classes = 3
    placeholders = construct_placeholders(num_classes)
    placeholders['features'] = train_data
    # feed_dict =  dict()
    # train_data = train_data.astype('float32')
    # feed_dict.update({placeholders['features']: train_data})
    dim = []
    # print("f:{}".format(len(train_data[0])))
    dim.append(len(train_data[0]))
    dim.append(FLAGS.dim_1)
    dim.append(num_classes)
    model = FCPartition(placeholders, dim)
    sess = tf.Session()
    model.load(sess)
    results = model.predict()
    print(results.eval(session=sess))

def main():
    print("load data ...")
    train_data = load_embedded_data(FLAGS.train_prefix)
    predict(train_data)

if __name__ == '__main__':
    main()
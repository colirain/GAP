
import tensorflow as tf 
import numpy as np 

from graphsage.models import FCPartition
from graphsage.partition_train import construct_placeholders
from graphsage.utils import load_graph_data, load_embedded_data, load_embedded_idmap

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

def predict(train_data, id_map):
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
    results_np = results.eval(session=sess)
    # print(results.eval(session=sess))
    # print(results_np.shape)
    id_map = id_map.astype('int')
    results_np = np.expand_dims(results_np, axis=1)
    results_np = np.insert(results_np, 0, id_map, axis=1)
    results_np = results_np[results_np[:,0].argsort()]
    print(results_np)
    np.save(FLAGS.outDir+'/predict_predict.npy', results_np)


def main():
    print("load data ...")
    train_data = load_embedded_data(FLAGS.train_prefix)
    id_map = load_embedded_idmap(FLAGS.train_prefix)
    predict(train_data, id_map)

if __name__ == '__main__':
    main()
import tensorflow as tf
import numpy as np
from tensorflow.python.platform import gfile
import os.path
from prepare_imagenet_data import preprocess_image_batch, create_imagenet_npy, undo_image_avg
import matplotlib.pyplot as plt
import sys, getopt
import zipfile
from timeit import time

if sys.version_info[0] >= 3:
    from urllib.request import urlretrieve
else:
    from urllib import urlretrieve
from universal_pert import universal_perturbation
from targeted_universal_pert import targeted_perturbation
from util_univ import *

# if you want using cpu. change for device='/cpu:0'
device = '/gpu:0'

# choose your target
target = 0

graph = tf.get_default_graph()


if __name__ == '__main__':

    with tf.device(device):
        persisted_sess = tf.Session()
        inception_model_path = './model/pb/merge.pb'

        if os.path.isfile(inception_model_path) == 0:
            print('no pb model')


        model = os.path.join(inception_model_path)

        # Load the Inception model
        with gfile.FastGFile(model, 'rb') as f:
            graph_def = tf.GraphDef()
            graph_def.ParseFromString(f.read())
            persisted_sess.graph.as_default()
            tf.import_graph_def(graph_def, name='')

        persisted_sess.graph.get_operations()

        persisted_input_a = persisted_sess.graph.get_tensor_by_name('model_a/input_image:0')
        persisted_input_b = persisted_sess.graph.get_tensor_by_name('model_b/input_image:0')
        persisted_feature_b = persisted_sess.graph.get_tensor_by_name('model_b/op_to_store_b:0')
        persisted_feature_a = persisted_sess.graph.get_tensor_by_name('model_a/op_to_store_a:0')
        persisted_output = persisted_sess.graph.get_tensor_by_name("merge_output:0")
        hsq_feature = np.load('./data/img/hsq.npy')
        # grad = tf.gradients(persisted_output,persisted_input_b)
        # def get_feature_map(image_inp_b):
        #     feature_map_raw = persisted_sess.run(persisted_debug,feed_dict={
        #                                                      persisted_input_b: np.reshape(image_inp_b, (-1, 112, 112, 3)),
        #
        #                                                        })
        #     return feature_map_raw

        # embeddings1 = tf.placeholder(dtype=tf.float32, shape=[None, 512])
        threshold = 1.02
        embeddings1 = persisted_feature_a / tf.norm(persisted_feature_a, axis=1, keepdims=True)
        embeddings2 = persisted_feature_b / tf.norm(persisted_feature_b, axis=1, keepdims=True)
        diff = tf.subtract(embeddings1, embeddings2)
        dist = tf.reduce_sum(tf.multiply(diff, diff))
        dist = threshold - dist
        grad = tf.gradients(dist,persisted_input_b)
            # is_face = tf.exp(threshold-dist)
            # is_not_face =  - is_face
            # face_tensor = tf.stack([is_face,is_not_face], axis=1, name='stack')
            # face_tensor = tf.nn.softmax(face_tensor)

        # X = create_lfw_npy()[0]
        # grad_value = persisted_sess.run(grad,feed_dict={persisted_input_a: np.reshape(X, (-1, 112, 112, 3)),
        #                                                      persisted_input_b: np.reshape(X, (-1, 112, 112, 3)),})



        print('>> Computing feedforward function...')

        def f(image_inp_a,image_inp_b): return persisted_sess.run(persisted_output,
                                                    feed_dict={persisted_input_a: np.reshape(image_inp_a, (-1, 112, 112, 3)),
                                                             persisted_input_b: np.reshape(image_inp_b, (-1, 112, 112, 3)),

                                                               })
        def get_f(image_inp): return persisted_sess.run(persisted_feature_a, feed_dict={persisted_input_a:np.reshape(image_inp, (-1, 112, 112, 3))})

        file_perturbation = os.path.join('data','precomputing_perturbations', 'perturbation.npy')
            
        # TODO: Optimize this construction part!
        print('>> Compiling the gradient tensorflow functions. This might take some time...')

        print('>> Computing gradient function...')

        def grad_b(image_inp_a, image_inp_b): return persisted_sess.run(grad, feed_dict={persisted_input_a: np.reshape(image_inp_a, (-1, 112, 112, 3)),
            persisted_input_b: np.reshape(image_inp_b, (-1, 112, 112, 3))})

        print('>> Creating pre-processed imagenet data...')
        X = create_lfw_npy()

        # Running universal perturbation
        # v = universal_perturbation(X, f, grad_fs, delta=0.2)

        v = targeted_perturbation(X,  f, get_f,grad_b, delta=0.3,max_iter_uni=10,p=np.inf,target=target)

        # Saving the universal perturbation
        np.save(os.path.join(file_perturbation), v)



        print('>> Testing the targeted universal perturbation on an image')

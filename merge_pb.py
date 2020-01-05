import tensorflow as tf
from tensorflow.python.tools import freeze_graph
from tensorflow.python.framework.graph_util import convert_variables_to_constants
import os
import numpy as np
import math

filename1 = './model/pb/arcface_a.pb'
filename2 = './model/pb/arcface_b.pb'


def distance(embeddings1, embeddings2, threshold=0.345, threshold1=0.99982):

    # Distance based on cosine similarity
    dot = tf.reduce_sum(tf.multiply(embeddings1, embeddings2), axis=1)
    norm = tf.norm(embeddings1, axis=1) * tf.norm(embeddings2, axis=1)
    similarity = dot / norm
    # dist = tf.acos(similarity) / 3.1415926
    dist = threshold1-similarity
    # is_face = -tf.log(1.0-(threshold-dist))
    # is_not_face = tf.log(1.0-(threshold-dist))
    # face_tensor = tf.stack([is_face,is_not_face], axis=1, name='stack')
    # face_tensor = tf.nn.softmax(face_tensor)

    return dist

def distance_0(embeddings1, embeddings2,threshold=1.02):
    embeddings1 = embeddings1 / tf.norm(embeddings1, axis=1, keepdims=True)
    embeddings2 = embeddings2 / tf.norm(embeddings2, axis=1, keepdims=True)
    diff = tf.subtract(embeddings1, embeddings2)
    dist = tf.reduce_sum(tf.multiply(diff,diff), 1)
    dist = threshold-dist
    # is_face = tf.exp(threshold-dist)
    # is_not_face =  - is_face
    # face_tensor = tf.stack([is_face,is_not_face], axis=1, name='stack')
    # face_tensor = tf.nn.softmax(face_tensor)
    return dist

def load_graphdef(filename):
    with tf.gfile.GFile(filename, "rb") as f:
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(f.read())
    return graph_def


def load_graph(graph_def, prefix):
    with tf.Graph().as_default() as graph:
        tf.import_graph_def(graph_def, name=prefix)

    return graph


graph1 = load_graphdef(filename1)
graph2 = load_graphdef(filename2)

graph1_out, = tf.import_graph_def(graph1, return_elements=['op_to_store_a:0'], name="model_a")
graph2_out, = tf.import_graph_def(graph2, return_elements=['op_to_store_b:0'], name="model_b")

# z = tf.concat([graph1_out, graph2_out], 1)
z = distance_0(graph1_out, graph2_out)


tf.identity(z, "merge_output")

init_op = tf.global_variables_initializer()
with tf.Session() as sess:
    sess.run(init_op)
    graph = convert_variables_to_constants(sess, sess.graph_def, ["merge_output"])
    tf.train.write_graph(graph, '.', './model/pb/merge.pb', as_text=False)
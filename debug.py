import numpy as np
import pickle
import math

pyy = np.load('./data/img/pyy.npy')
pyy1 = np.load('./data/img/pyy1.npy')
wyz = np.load('./data/img/wyz.npy')
hsq = np.load('./data/img/hsq.npy')
img0 = np.load('./data/img/im0.npy')
img1 = np.load('./data/img/im1.npy')
img2 = np.load('./data/img/im2.npy')


def distance(embeddings1, embeddings2, distance_metric=1):
    if distance_metric == 0:
        # Euclidian distance
        embeddings1 = embeddings1 / np.linalg.norm(embeddings1, axis=1, keepdims=True)
        embeddings2 = embeddings2 / np.linalg.norm(embeddings2, axis=1, keepdims=True)
        diff = np.subtract(embeddings1, embeddings2)
        dist = np.sum(np.square(diff), 1)
    elif distance_metric == 1:
        # Distance based on cosine similarity
        dot = np.sum(np.multiply(embeddings1, embeddings2), axis=1)
        norm = np.linalg.norm(embeddings1, axis=1) * np.linalg.norm(embeddings2, axis=1)
        similarity = dot / norm
        dist = np.arccos(similarity) / math.pi
    else:
        raise 'Undefined distance metric %d' % distance_metric

    return dist
img0_img1 = distance(img0, img1)
img0_img2 = distance(img0, img2)
pyy_wyz = distance(pyy, wyz)
pyy_hsq = distance(pyy, hsq)
pyy_pyy1 = distance(pyy, pyy1)
print(img0_img1)

# fr = open('./data/img/embds.pkl', 'rb')  # open的参数是pkl文件的路径
# inf = pickle.load(fr)
# fr.close()
# embd = []
# for k, v in inf.items():
#     embd.append(np.mat(v))
# vector_a = embd[0]
# vector_b = embd[1]
# print(vector_a)
# print(vector_b)
#
# a = cos_sim(vector_a, vector_a)
# print(a)
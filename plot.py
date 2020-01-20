from util_univ import *
import numpy as np
import matplotlib.pyplot as plt
import PIL
from PIL import Image
from scipy import misc

# X = create_lfw_npy()[0]
# X = Image.open('./data/img/pyy.jpg')
# shape = np.shape(X)
# X = misc.imresize(X,[112,112,3])
# X = np.array(X)/255.0
# v = np.load(os.path.join('data','precomputing_perturbations', 'perturbation.npy'))
# v = np.reshape(v,[112,112,3])
# x_adv = X+v
#
# # X = np.clip((X+1.0)*127.5,0,255)
# X = misc.imresize(np.clip(X*255.0,0,255),shape)
# plt.figure()
# plt.subplot(1, 2, 1)
# plt.imshow(X.astype(dtype='uint8'))
#
# # x_adv = np.clip((x_adv+1.0)*127.5,0,255)
# x_adv = misc.imresize(np.clip(x_adv*255.0,0,255),shape)
# plt.figure()
# plt.subplot(1, 2, 1)
# plt.imshow(x_adv.astype(dtype='uint8'))
#
# plt.show()
# print(1)

X = create_lfw_npy()[0]
v = np.load(os.path.join('data','precomputing_perturbations', 'perturbation.npy'))
v = np.reshape(v,[112,112,3])
x_adv = X+v

X = np.clip((X+1.0)*127.5,0,255)
plt.figure()
plt.subplot(1, 2, 1)
plt.imshow(X.astype(dtype='uint8'))

x_adv = np.clip((x_adv+1.0)*127.5,0,255)

plt.figure()
plt.subplot(1, 2, 1)
plt.imshow(x_adv.astype(dtype='uint8'))

plt.show()
print(1)
from util_univ import *
import numpy as np
import matplotlib.pyplot as plt

X = create_lfw_npy()[0]
v = np.load(os.path.join('data','precomputing_perturbations', 'perturbation.npy'))
v = np.reshape(v,[112,112,3])
x_adv = np.clip(X+v,0.0,1.0)

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
import numpy as np

"""
   :param image: Image of size HxWx3
   :param f: feedforward function (input: images, output: values of activation BEFORE softmax).
   :param grads: gradient functions with respect to input (as many gradients as classes).
   :param overshoot: used as a termination criterion to prevent vanishing updates (default = 0.02).
   :param max_iter: maximum number of iterations for deepfool (default = 10)
   :return: minimal perturbation that fools the classifier, number of iterations that it required, new estimated_label and perturbed image
"""


def deeptarget(image,  f, grads, overshoot=0.02, max_iter=50,target=None):

    assert (image.shape == (1,112,112,3))
    f_image = np.array(f(image)).flatten()
    # feature_image = np.array(get_feature(img0))
    label = np.sign(f_image)


    input_shape = image.shape
    pert_image = image

    f_i = np.array(f(image)).flatten()
    k_i = np.sign(f_image)
    o_i=k_i

    w = np.zeros(input_shape)
    r_tot = np.zeros(input_shape)

    loop_i = 0

	# we can't use k_i!=target. DeepTarget is limited L2 norm .therefore generate huge perturbation.
    while k_i == label and loop_i < max_iter:

        pert = np.inf
        w = np.asarray(grads(pert_image))
        f_t = f_i
        pert = abs(f_t) / np.linalg.norm(w)
        r_i = pert * w / np.linalg.norm(w)
        r_tot = r_tot + r_i

        # compute new perturbed image

        pert_image = image + (1+overshoot)*r_tot
        loop_i += 1

        # compute new label
        f_i = np.array(f(pert_image)).flatten()
        k_i = int(np.sign(f_i))

    r_tot = (1+overshoot)*r_tot
    #print(o_i," --> ",k_i)

    return np.reshape(r_tot, (1,112,112,3)), loop_i, k_i, pert_image

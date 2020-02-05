import io
import os
import yaml
import pickle
from generate_gallery import *
import argparse
import numpy as np
import tensorflow as tf
from util_univ import *
from scipy import misc
from prepare_imagenet_data import *
from model import get_embd
from eval.utils import calculate_roc, calculate_tar
from sklearn.decomposition import PCA


## 首次运行，需要先运行 generate_gallery.py

def get_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--mode', type=str, default='build', help='model mode: build')
    parser.add_argument('--config_path', type=str, default='./configs/config_ms1m_100.yaml', help='config path, used when mode is build')
    parser.add_argument('--model_path', type=str, default='./model/ms1m/best-m-334000', help='model path')
    parser.add_argument('--read_path', type=str, default='./image/lfw.bin', help='path to image file or directory to images')
    parser.add_argument('--save_path', type=str, default='embds.pkl', help='path to save embds')
    parser.add_argument('--train_mode', type=int, default=0, help='whether set train phase to True when getting embds. zero means False, one means True')

    return parser.parse_args()
if __name__ == '__main__':
    args = get_args()
    config = yaml.load(open(args.config_path))
    gallery = np.load('./data/img/gallery.npy',allow_pickle=True).item()
    lists = os.listdir('./test 2')
    images = tf.placeholder(dtype=tf.float32, shape=[None, config['image_size'], config['image_size'], 3], name='input_image')
    train_phase_dropout = tf.placeholder(dtype=tf.bool, shape=None, name='train_phase')
    train_phase_bn = tf.placeholder(dtype=tf.bool, shape=None, name='train_phase_last')
    embds, _ = get_embd(images, train_phase_dropout, train_phase_bn, config)
    print('done!')
    tf_config = tf.ConfigProto(allow_soft_placement=True)
    tf_config.gpu_options.allow_growth = True
    with tf.Session(config=tf_config) as sess:
        tf.global_variables_initializer().run()
        print('loading...')
        variables = tf.contrib.framework.get_variables_to_restore()
        # variables_to_restore = tf.contrib.framework.get_variables_to_restore(exclude=['op_to_store_b:0'])
        saver = tf.train.Saver(variables)
        saver.restore(sess, args.model_path)
        print('done!')

        batch_size = config['batch_size']
        total = 0
        right = 0

        # 可以改
        for num in range(10):
            read_path = './test 2/' + lists[num]
            imgs, imgs_f, fns = load_image(read_path, config['image_size'])
            total += len(imgs)
            print(len(imgs))
            print('forward running...')
            # X_adv = X +v
            embds_arr = run_embds(sess, imgs, batch_size, config['image_size'], args.train_mode, embds, images,
                                  train_phase_dropout, train_phase_bn)

            for i in range(len(imgs)):
                min = 10000
                label = ''

                for key, value in gallery.items():

                    # 普通作差
                    # dist = np.sum(np.abs(embds_arr[i]-value))

                    # 欧式距离
                    diff = np.subtract(embds_arr[i], value)
                    dist = np.sum(np.square(diff, diff))
                    if dist < min:
                        min = dist
                        label = key
                # print(label)
                if label == lists[num]:
                    right += 1

        accuracy = (right/total)*100
        print(right)
        print(total)
        print("准确率为:",accuracy)
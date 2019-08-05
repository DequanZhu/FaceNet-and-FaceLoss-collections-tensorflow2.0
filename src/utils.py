import pathlib
import glob
import tensorflow as tf
import os
import random
import numpy as np
from scipy.special import binom
BATCH_SIZE = 128
IMG_SIZE=[299,299]


def check_folder(dir_name):
    if not os.path.exists(dir_name):
        os.makedirs(dir_name)
    return dir_name



class CosMThetaCalculator():
    def __init__(self, margin):
        super(CosMThetaCalculator, self).__init__()
        self.m=margin
        self.C_m_2n = tf.Variable(binom(self.m, range(0, self.m + 1, 2)), dtype='float32')  # C_m{2n}
        self.cos_powers = tf.Variable(range(self.self.m, -1, -2), dtype='float32')  # m - 2n
        self.sin2_powers = tf.Variable(range(self.cos_powers.shape[0]), dtype='float32')  # n
        self.signs = np.ones(self.m // 2 + 1, dtype='float32')
        self.signs[1::2] = -1.0 # 1, -1, 1, -1, ...
        self.signs = tf.Variable(self.signs, dtype='float32')

    def __call__(self,cos_theta):
        sin2_theta = 1.0 - cos_theta**2
        # cos^{m - 2n}
        cos_terms = tf.math.pow(tf.tile(tf.expand_dims(cos_theta, axis=1),
                                        multiples=[1, self.cos_powers.shape[0]]),
                                tf.tile(tf.expand_dims(self.cos_powers, axis=0),
                                        multiples=[cos_theta.shape[0], 1]))

        # sin2^{n}
        sin2_terms = tf.math.pow(tf.tile(tf.expand_dims(sin2_theta, axis=1),
                                         multiples=[1, self.sin2_powers.shape[0]]),
                                 tf.tile(tf.expand_dims(self.sin2_powers, axis=0),
                                         multiples=[sin2_theta.shape[0], 1]))
                                         
        signs = tf.tile(tf.expand_dims(self.signs, axis=0),
                        multiples=[cos_theta.shape[0], 1])
        C_m_2n = tf.tile(tf.expand_dims(self.C_m_2n, axis=0),
                         multiples=[cos_theta.shape[0], 1])
        cos_m_theta = tf.math.reduce_sum(signs * C_m_2n*cos_terms*sin2_terms, axis=1)  # summation of all terms
        return cos_m_theta
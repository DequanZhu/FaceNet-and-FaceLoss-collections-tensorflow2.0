from __future__ import division
import argparse
import sys
# from progressbar import *
import tqdm
import os
import tensorflow as tf
import numpy as np
from tensorflow.python import keras
from facenet import FaceNet
from datasets import create_datasets_from_tfrecord
# from losses import Center_Loss
from utils import check_folder


def parse_arguments(argv):
    description = "facenet train options"
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument('--model_name', type=str, default='InceptionResNetV2')
    parser.add_argument('--restore', action='store_true',
                        help='Whether to restart training from checkpoint ')
    parser.add_argument('--max_nrof_epochs', type=int, default=20,
                        help='The number of epochs to run')
    parser.add_argument('--nrof_classes', type=int, default=9278,
                        help='The number of identities')
    parser.add_argument('--batch_size', type=int,
                        default=32, help='The size of batch')
    parser.add_argument('--image_size', type=int,
                        default=160, help='The size of input image')
    parser.add_argument('--embedding_size', type=int,
                        default=128, help='The size of feature to embedding')
    parser.add_argument('--checkpoint_dir', type=str, default='./checkpoint/',
                        help='Directory name to save the checkpoints')
    parser.add_argument('--train_log_dir', type=str, default='./logs/',
                        help='Directory name to save training logs')
    parser.add_argument('--datasets', type=str, default='../data/train_tfrcd/',
                        help='Directory name to load training data')
    parser.add_argument('--split_ratio', type=float, default=0.9,
                        help='The ratio of training data for split data')
    parser.add_argument('--gpu_ids', type=int, nargs='+',
                        default='0', help='gpu ids to use')
    parser.add_argument('--learning_rate', type=float, default=2e-3,
                        help='Initial learning rate. If set to a negative value a learning rate ')
    parser.add_argument('--center_loss_weight', type=float, default=0.5,
                        help=' The weight of center-loss to total-loss')
    parser.add_argument('--center_loss_alfa', type=float,
                        help='Center update rate for center loss.', default=0.5)
    return parser.parse_args(argv)


class Center_Loss():
    def __init__(self,alpha,nrof_classes,embedding_size):
        super(Center_Loss,self).__init__()
        self.alpha=alpha
        self.centers=tf.zeros(shape=[nrof_classes,embedding_size],dtype=tf.float32)


    def __call__(self,inputs, label):
        embedding_size = tf.shape(inputs)[1]
        label=tf.expand_dims(label,axis=1)
        centers_batch = tf.gather_nd(self.centers, label)
        diff = (1 - self.alpha) * (centers_batch - inputs)
        centers = tf.tensor_scatter_nd_sub(self.centers, label, diff)
        loss = tf.math.reduce_mean(tf.math.square(inputs - centers_batch))
        return loss



def train(opt):
    model = FaceNet(opt, num_classes=opt.nrof_classes).model
    train_datasets, val_datasets = create_datasets_from_tfrecord(
        opt.datasets, opt.batch_size, opt.split_ratio)
    model.compile(optimizer=keras.optimizers.Adam(lr=1e-3, beta_1=0.9, beta_2=0.999, epsilon=0.1),
                  loss=[Center_Loss, keras.losses.SparseCategoricalCrossentropy()]
                  )

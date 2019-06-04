from __future__ import division
import argparse
import sys
import datetime
from progressbar import *
import os
# import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import scipy.misc
from tensorflow.python import keras
from tensorflow.python.keras import layers, optimizers, metrics, Sequential
from tensorflow.python.keras.applications import ResNet50
from facenet import FaceNet
from datasets import create_datasets_from_tfrecord
from losses import Center_Loss





# @tf.function()
def train_one_step(model, train_acc_metric, loss_fun, optimizer,
                   batch_images, batch_labels, center_loss_weight):
    with tf.GradientTape() as tape:
        features = model['embedding'](batch_images)
        embedding = tf.math.l2_normalize(features, axis=1, epsilon=1e-10)
        center_loss = loss_fun(embedding, batch_labels)
        prediction = model['logits'](batch_images)
        train_acc_metric(batch_labels, prediction)
        cross_entropy_loss = keras.losses.SparseCategoricalCrossentropy(prediction, batch_labels)
        total_loss = center_loss_weight * center_loss + cross_entropy_loss

    gradients = tape.gradient(total_loss, model['predictor'].trainable_variables)
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))
    return center_loss, total_loss


def train(opt):
    facenet = FaceNet(opt,num_classes=9279)
    model = facenet.model
    train_datasets = create_datasets_from_tfrecord(opt.train_data,opt.batch_size)



def parse_arguments(argv):
    description = "facenet train options"
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument('--model_name', type=str, default='InceptionResNetV2')
    parser.add_argument('--restore', action='store_true',
                        help='Whether to restart training from checkpoint ')
    parser.add_argument('--epoches', type=int, default=20,
                        help='The number of epochs to run')
    parser.add_argument('--nrof_classes', type=int, default=9279,
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
    parser.add_argument('--train_data', type=str, default='../data/train_tfrcd/',
                        help='Directory name to load training data')
    parser.add_argument('--val_data', type=str, default='../data/val_tfrcd/',
                        help='Directory name to load validate data')
    parser.add_argument('--gpu_ids', type=int, nargs='+',
                        default='0', help='gpu ids to use')       
    parser.add_argument('--learning_rate', type=float,default=2e-4,
                        help='Initial learning rate. If set to a negative value a learning rate ' )                 
    parser.add_argument('--center_loss_weight', type=float,default=0.95,
                         help=' The weight of center-loss to total-loss' )                 
    return parser.parse_args(argv)


if __name__ == '__main__':
    option = parse_arguments(sys.argv[1:])
    train(option)

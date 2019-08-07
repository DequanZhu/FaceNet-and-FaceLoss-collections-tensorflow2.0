from __future__ import division
import os
import sys
import numpy as np
import tensorflow as tf
import tensorflow.python.keras as keras
from tensorflow.python.keras.optimizers import schedules, Adam
from tensorflow.python.keras.losses import SparseCategoricalCrossentropy
from tensorflow.python.keras.metrics import SparseCategoricalAccuracy
from tensorflow.python.keras.callbacks import ModelCheckpoint, TensorBoard
from .facenet import FaceNet
from options.train_options import TrainOptions
from .losses import *
from .datasets import create_datasets_from_tfrecord


class Trainer(object):
    def __init__(self, opt):
        self.args = args
        self.model = self.create_model()
        self.train_datasets, self.nrof_train = create_datasets_from_tfrecord(tfrcd_dir=args.datasets,
                                                                             batch_size=args.batch_size,
                                                                             phase='train')

        self.val_datasets, self.nrof_val = create_datasets_from_tfrecord(tfrcd_dir=args.datasets,
                                                                         batch_size=args.batch_size,
                                                                         phase='val')
        self.train_loader, self.val_loader = self.data_gen(
            self.train_datasets), self.data_gen(self.val_datasets)

        self.lr_schedule = schedules.ExponentialDecay(args.learning_rate,
                                                      decay_steps=10000,
                                                      decay_rate=0.96,
                                                      staircase=True)

        self.optimizer = Adam(learning_rate=self.lr_schedule,
                              beta_1=0.9, beta_2=0.999, epsilon=0.1)

    def create_model(self):
        opt = self.args
        img_size = opt.image_size
        feature_extractor = FaceNet(opt.backbone, img_size, opt.embedding_size, opt.nrof_classes).model
        input = tf.keras.Input(shape=(img_size, img_size, 3), name='face')
        embedding = feature_extractor(input)
        loss_layer = ArcFaceSoftmaxLinear(opt.nrof_classes, opt.embedding_size, opt.margin, opt.feature_scale)
        logits = loss_layer(embedding)
        model = keras.Model(input, logits, name='facenet')
        return model

    def data_gen(train_data):
        for batch_images, batch_annos in train_data:
            yield (batch_images, batch_annos)

    def train(self):
        opt = self.args
        mirrored_strategy = tf.distribute.MirroredStrategy()
        with mirrored_strategy.scope():
            tensorboard_cbk = TensorBoard(log_dir=opt.log_dir)
            checkpoint_cbk = ModelCheckpoint(opt.ckpt_path, monitor='val_acc',)
            self.model.compile(optimizer=self.optimizer,
                               loss=['categorical_crossentropy'],
                               metrics=['categorical_accuracy'])

            history = model.fit(self.train_loader,
                                batch_size=opt.batch_size,
                                epochs=opt.max_epoch,
                                validation_data=self.val_loader,
                                validation_freq=opt.eval_interval,
                                callbacks=[tensorboard_cbk, checkpoint_cbk])

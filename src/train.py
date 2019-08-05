from __future__ import division
import os
import sys
import numpy as np
import tensorflow as tf
from tensorflow.keras.optimizers import schedules, Adam
from tensorflow.python.keras.losses import SparseCategoricalCrossentropy
from tensorflow.python.keras.metrics import SparseCategoricalAccuracy
from facenet import FaceNet
from options.train_options import TrainOptions
from losses import *
from datasets import create_datasets_from_tfrecord
from utils import check_folder
from progressbar import *


class Trainer(object):
    def __init__(self, args):
        self.args=args
        self.model = FaceNet(args.backbone, args.image_size, args.embedding_size, args.nrof_classes).model
        self.train_datasets, self.nrof_train = create_datasets_from_tfrecord(tfrcd_dir=args.datasets,
                                                                             batch_size = args.batch_size,
                                                                             phase='train')

        self.val_datasets, self.nrof_val = create_datasets_from_tfrecord(tfrcd_dir=args.datasets,
                                                                         batch_size =  args.batch_size,
                                                                         phase='val')
        self.lr_schedule = schedules.ExponentialDecay(args.learning_rate,
                                                      decay_steps=10000,
                                                      decay_rate=0.96,
                                                      staircase=True)

        self.optimizer = Adam(learning_rate=self.lr_schedule, beta_1=0.9, beta_2=0.999, epsilon=0.1)
        self.checkpoint = tf.train.Checkpoint(epoch=tf.Variable(0,dtype=tf.int64),
                                              n_iter=tf.Variable(0,dtype=tf.int64), 
                                              best_pred=tf.Variable(0.0,dtype=tf.float32),
                                              optimizer=self.optimizer,
                                              model=self.model)
        self.manager = tf.train.CheckpointManager(self.checkpoint, args.checkpoint_dir, max_to_keep=3)
        check_folder(args.log_dir)
        self.train_summary_writer = tf.summary.create_file_writer(args.log_dir)

    # @tf.function()
    def train_one_step(self, train_acc_metric, loss_layer, batch_examples):
        with tf.GradientTape() as tape:
            batch_images, batch_labels = batch_examples
            features = self.model(batch_images)
            embedding = tf.math.l2_normalize(features, axis=1, epsilon=1e-10)
            logits = loss_layer(embedding, batch_labels)
            loss = SparseCategoricalCrossentropy()(batch_labels, logits)
            train_acc_metric(batch_labels, logits)
        gradients = tape.gradient(loss, self.model.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, self.model.trainable_variables))
        return loss

    def training(self, epoch):
        opt = self.args
        loss_layer = ArcFaceSoftmaxLinear(opt.nrof_classes, opt.embedding_size, opt.margin, opt.feature_scale)
        train_acc_metric = SparseCategoricalAccuracy()
        widgets = ['train :', Percentage(), ' ', Bar('#'), ' ',Timer(), ' ', ETA(), ' ']
        pbar = ProgressBar(widgets=widgets, max_value=int(self.nrof_train//opt.batch_size)+1).start()
        for batch_id, batch_examples in pbar(enumerate(self.train_datasets)):
            loss = self.train_one_step(train_acc_metric, loss_layer, batch_examples)
            with self.train_summary_writer.as_default():
                tf.summary.scalar('total_loss', loss, self.checkpoint.n_iter)
            self.checkpoint.n_iter.assign_add(1)
        pbar.finish() 
        train_acc = train_acc_metric.result()
        print('\nTraining acc over epoch {}: {:.4f}'.format(epoch, train_acc))
        with self.train_summary_writer.as_default():
            tf.summary.scalar('train/acc', train_acc_metric.result(), self.checkpoint.epoch)
        train_acc_metric.reset_states()
        save_path = self.manager.save()
        print('save checkpoint to {}'.format(save_path))


    def validate(self, epoch):
        widgets = ['validate :', Percentage(), ' ', Bar('#'), ' ',Timer(), ' ', ETA(), ' ']
        pbar = ProgressBar(widgets=widgets, max_value=int(self.nrof_val//self.args.batch_size)+1).start()
        val_acc_metric = SparseCategoricalAccuracy()
        for batch_id, (batch_images_validate, batch_labels_validate) in pbar(enumerate(self.val_datasets)):
            prediction = self.model(batch_images_validate)
            val_acc_metric(batch_labels_validate, prediction)
        pbar.finish() 
        val_acc = val_acc_metric.result()
        print('\nvalidate acc over epoch {}: {:.4f}'.format(epoch, val_acc))
        with self.train_summary_writer.as_default():
            tf.summary.scalar('val/acc', val_acc_metric.result(),self.checkpoint.epoch)
        self.checkpoint.epoch.assign_add(1)
        
        val_acc_metric.reset_states()

        if(val_acc > self.checkpoint.best_pred):
            self.checkpoint.best_pred = val_acc
            with open(os.path.join(self.checkpoint_dir, 'best_pred.txt'), 'w') as f:
                f.write(str(self.best_pred))
            self.model.save(os.path.join(self.checkpoint_dir, 'best_model.h5'))


def main(argv):
    opt = TrainOptions(argv).parse()
    trainer = Trainer(opt)
    start_epoch = 0
    if opt.restore:
        start_epoch = trainer.checkpoint.restore(trainer.manager.latest_checkpoint)
    for epoch in range(start_epoch, opt.max_epoch):
        # trainer.training(epoch)
        if not opt.no_val and epoch % opt.eval_interval == (opt.eval_interval - 1):
            trainer.validate(epoch)


if __name__ == '__main__':
    main(sys.argv[1:])

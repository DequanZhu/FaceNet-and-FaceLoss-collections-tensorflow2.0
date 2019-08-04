from __future__ import division
import os
import sys
import numpy as np
import tensorflow as tf
from tensorflow.python.keras import optimizers 
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
        self.args = args
        self.nclass=args.nclass
        self.dataset_dir=args.dataset_dir
        self.model = FaceNet(self.args).model
        self.train_datasets,self.nrof_train= create_datasets_from_tfrecord(args.train_tfrcd, args.batch_size)
        self.val_datasets,self.nrof_val =create_datasets_from_tfrecord(args.val_tfrcd, args.batch_size)
        self.lr_schedule = optimizers.schedules.ExponentialDecay(args.learning_rate,
                                                                 decay_steps=10000,
                                                                 decay_rate=0.96,
                                                                 staircase=True)
 
        self.optimizer = optimizers.Adam(learning_rate=self.lr_schedule, beta_1=0.9, beta_2=0.999, epsilon=0.1)
        self.checkpoint = tf.train.Checkpoint(epoch=tf.Variable(0), optimizer=self.optimizer, model=self.model)
        self.manager = tf.train.CheckpointManager(checkpoint, args.checkpoint_dir, max_to_keep=3)
        check_folder(args.train_log_dir)
        self.train_summary_writer = tf.summary.create_file_writer(args.train_log_dir)



    @tf.function()
    def train_one_step(self, train_acc_metric, loss_layer, batch_examples):
        with tf.GradientTape() as tape:
            batch_images,batch_labels=batch_examples
            outputs = self.model(batch_images)
            features = outputs['embedding']
            embedding = tf.math.l2_normalize(features, axis=1, epsilon=1e-10)
            logits = loss_layer(embedding, batch_labels)
            loss = SparseCategoricalCrossentropy()(batch_labels,logits)
            train_acc_metric(batch_labels, logits)
        gradients = tape.gradient(loss, model.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, model.trainable_variables))
        return loss


    def training(self,epoch):
        opt=self.args
        loss_layer = ArcFaceLoss(opt.margin,opt.feature_scale)
        train_acc_metric = SparseCategoricalAccuracy()
        widgets = ['train :', Percentage(), ' ', Bar('#'), ' ', Timer(), ' ', ETA(), ' ']
        pbar = ProgressBar(widgets=widgets,max_value=int(self.nrof_train//opt.batch_size)+1).start()
        for batch_id, batch_examples in pbar(enumerate(self.train_datasets)):
            center_loss, total_loss = train_one_step(train_acc_metric, loss_layer, batch_examples)
            with train_summary_writer.as_default():
                tf.summary.scalar('center_loss', center_loss)
                tf.summary.scalar('total_loss', total_loss)
        pbar.close()
        train_acc = train_acc_metric.result()
        print('Training acc over epoch {}: %s' % (epoch, float(train_acc)))
        with train_summary_writer.as_default():
            tf.summary.scalar('train_acc', train_acc_metric.result())
        train_acc_metric.reset_states()
        save_path = manager.save()
        print('save checkpoint to {}'.format(save_path))
        checkpoint.epoch.assign_add(1)



    def validate(self,epoch):
        widgets = ['validate :', Percentage(), ' ', Bar('#'), ' ', Timer(), ' ', ETA(), ' ']
        pbar = ProgressBar(widgets=widgets,max_value=int(self.nrof_val//self.args.batch_size)+1).start()
        val_acc_metric = SparseCategoricalAccuracy()
        for batch_id, (batch_images_validate, batch_labels_validate) in pbar(enumerate(self.val_datasets)):
            prediction = model(batch_images_validate)
            val_acc_metric(batch_labels_validate, prediction)
        pbar.close()
        val_acc = val_acc_metric.result()
        print('validate acc over epoch {}: %s' % (epoch, float(val_acc)))
        with train_summary_writer.as_default():
            tf.summary.scalar('val_acc', val_acc_metric.result())
        val_acc_metric.reset_states()
        if(val_acc>self.best_pred):
            self.best_pred=val_acc
            with open(os.path.join(self.checkpoint_dir, 'best_pred.txt'), 'w') as f:
                f.write(str(self.best_pred))
            self.model.save(os.path.join(self.checkpoint_dir,'best_model.h5'))


def main():
    opt=TrainOptions().parse()
    trainer=Trainer(opt)
    start_epoch = 0
    if opt.restore:
        start_epoch = trainer.checkpoint.restore(trainer.manager.latest_checkpoint)
    for epoch in range(start_epoch, opt.max_nrof_epochs):
        trainer.training(epoch)
    if not trainer.opt.no_val and epoch % opt.eval_interval == (opt.eval_interval - 1):
        trainer.validation(epoch)



if __name__ == '__main__':
    main()


from __future__ import division
import argparse
import sys
import os
import tensorflow as tf
import numpy as np
from tensorflow.python import keras
from facenet import FaceNet
from datasets import create_datasets_from_tfrecord
from losses import Center_Loss
from utils import check_folder
from progressbar import *


# tf.debugging.set_log_device_placement(True)

@tf.function()
def train_one_step(model, train_acc_metric, loss_fun, optimizer,
                   batch_examples, center_loss_weight):
    with tf.GradientTape() as tape:
        batch_images,batch_labels=batch_examples
        outputs = model(batch_images)
        features = outputs['embedding']
        embedding = tf.math.l2_normalize(features, axis=1, epsilon=1e-10)
        center_loss = loss_fun(embedding, batch_labels)
        prediction = outputs['logits']
        train_acc_metric(batch_labels, prediction)
        cross_entropy_loss = tf.keras.losses.SparseCategoricalCrossentropy()(batch_labels,prediction)
        total_loss = center_loss_weight * center_loss + cross_entropy_loss
    gradients = tape.gradient(total_loss, model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))
    return center_loss, total_loss


def train(opt):
    facenet = FaceNet(opt, num_classes=opt.nrof_classes)
    model = facenet.model
    lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(opt.learning_rate,
                                                                decay_steps=10000,
                                                                decay_rate=0.96,
                                                                staircase=True)
    optimizer = tf.keras.optimizers.Adam(learning_rate=lr_schedule, beta_1=0.9, beta_2=0.999, epsilon=0.1)

    checkpoint = tf.train.Checkpoint(step=tf.Variable(0), epoch=tf.Variable(0), optimizer=optimizer, model=model)
    manager = tf.train.CheckpointManager(checkpoint, opt.checkpoint_dir, max_to_keep=3)
    check_folder(opt.train_log_dir)
    train_summary_writer = tf.summary.create_file_writer(opt.train_log_dir)

    start_epoch = 0
    if opt.restore:
        start_epoch = checkpoint.restore(manager.latest_checkpoint)
    loss_fun = Center_Loss(alpha=opt.center_loss_alfa, nrof_classes=opt.nrof_classes,embedding_size=opt.embedding_size)
    train_acc_metric = tf.keras.metrics.SparseCategoricalAccuracy()
    val_acc_metric = tf.keras.metrics.SparseCategoricalAccuracy()
    train_datasets,val_datasets,nrof_train,nrof_val = create_datasets_from_tfrecord(opt.datasets, opt.batch_size,opt.split_ratio)

    for epoch in range(start_epoch, opt.max_nrof_epochs):
        widgets = ['train :', Percentage(), ' ', Bar('#'), ' ', Timer(), ' ', ETA(), ' ']
        pbar = ProgressBar(widgets=widgets,max_value=int(nrof_train//opt.batch_size)+1).start()
        for batch_id, batch_examples in pbar(enumerate(train_datasets)):
            center_loss, total_loss = train_one_step(model, train_acc_metric, loss_fun, optimizer,
                                                     batch_examples, opt.center_loss_weight)
            checkpoint.step.assign_add(1)
            step = int(checkpoint.step)
            if step % 400 == 0:
                with train_summary_writer.as_default():
                    tf.summary.scalar('center_loss', center_loss, step=step)
                    tf.summary.scalar('total_loss', total_loss, step=step)
        pbar.close()
        train_acc = train_acc_metric.result()
        print('Training acc over epoch {}: %s' % (epoch, float(train_acc)))

        widgets = ['validate :', Percentage(), ' ', Bar('#'), ' ', Timer(), ' ', ETA(), ' ']
        pbar = ProgressBar(widgets=widgets,max_value=int(nrof_val//opt.batch_size)+1).start()
        for batch_id, (batch_images_validate, batch_labels_validate) in pbar(enumerate(val_datasets)):
            prediction = model(batch_images_validate)
            val_acc_metric(batch_labels_validate, prediction)
        pbar.close()
        val_acc = train_acc_metric.result()
        print('validate acc over epoch {}: %s' % (epoch, float(val_acc)))

        with train_summary_writer.as_default():
            tf.summary.scalar('train_acc', train_acc_metric.result(), step=step)
            tf.summary.scalar('val_acc', train_acc_metric.result(), step=step)
        train_acc_metric.reset_states()
        val_acc_metric.reset_states()
        save_path = manager.save()
        print('save checkpoint to {}'.format(save_path))
        checkpoint.epoch.assign_add(1)


def parse_arguments(argv):
    description = "facenet train options"
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument('--model_name', type=str, default='InceptionResNetV2')
    parser.add_argument('--restore', action='store_true',
                        help='Whether to restart training from checkpoint ')
    parser.add_argument('--max_nrof_epochs', type=int, default=20,
                        help='The number of epochs to run')
    parser.add_argument('--nrof_classes', type=int, default=9277,
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


if __name__ == '__main__':
    option = parse_arguments(sys.argv[1:])
    train(option)

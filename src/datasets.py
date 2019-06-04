# coding=utf-8
import glob
import sys
import tensorflow as tf
import os
import argparse
import numpy as np
import random
from progressbar import *



IMG_SIZE = [160, 160]


def create_image_example(image_path, label):
    image_string = open(image_path, 'rb').read()
    image_shape = tf.image.decode_jpeg(image_string, channels=3).shape
    feature = {
        'image_shape': tf.train.Feature(int64_list=tf.train.Int64List(value=image_shape)),
        'id': tf.train.Feature(int64_list=tf.train.Int64List(value=[label])),
        'image_raw': tf.train.Feature(bytes_list=tf.train.BytesList(value=[image_string]))
    }
    example_proto = tf.train.Example(features=tf.train.Features(feature=feature))
    return example_proto


def write_to_tfrecord(write_path, image_paths, images_labels):
    widgets = ['writing to {} :'.format(write_path), Percentage(), ' ', Bar('#'), ' ', Timer(), ' ', ETA(), ' ']
    pbar = ProgressBar(widgets=widgets, max_value=len(image_paths)).start()
    with tf.io.TFRecordWriter(write_path) as writer:
        for filename, label in pbar(zip(image_paths, images_labels)):
            try:
                tf_example = create_image_example(filename, label)
                writer.write(tf_example.SerializeToString())
            except:
                print('fail to read image {}'.format(filename))
                continue



def parse_image_function(example_proto):
    image_feature_description = {
        'image_shape': tf.io.FixedLenFeature(shape=[3, ], dtype=tf.int64),
        'id': tf.io.FixedLenFeature(shape=[], dtype=tf.int64),
        'image_raw': tf.io.FixedLenFeature(shape=[], dtype=tf.string),
    }
    parsed_example = tf.io.parse_single_example(example_proto, image_feature_description)
    image_string = parsed_example['image_raw']
    image = tf.image.decode_jpeg(image_string, channels=3)
    image = preprocess_image(image)
    label = parsed_example['id']-2
    return image, label


def preprocess_image(image):
    image = tf.image.resize(image, IMG_SIZE)
    image = tf.image.random_flip_left_right(image)
    image /= 255.0
    return image



def create_tfrecord_data(src_dir, dest_dir,nrof_imgs_per_file):
    if not os.path.exists(dest_dir):
        os.makedirs(dest_dir)
    image_paths = glob.glob(src_dir+'/*/*.jpg')
    random.seed(0)
    random.shuffle(image_paths)
    image_ids = [path.split('/')[-2] for path in image_paths]
    image_ids = [int(label[1:]) for label in image_ids]
    nrof_tfrcd_files = len(image_paths)//nrof_imgs_per_file + 1
    for i in range(nrof_tfrcd_files):
        save_path = os.path.join(dest_dir, 'train_{:04d}.tfrecords'.format(int((i+1)*nrof_imgs_per_file)))
        start = max(0, i*nrof_imgs_per_file)
        end = min((i+1)*nrof_imgs_per_file, len(image_paths))
        write_to_tfrecord(save_path, image_paths[start:end], image_ids[start:end])





def create_datasets_from_tfrecord(tfrcd_dir,batch_size=32,split_ratio=0.9):
    file_paths = os.listdir(tfrcd_dir)
    file_paths = [os.path.join(tfrcd_dir, file_path)for file_path in file_paths]
    nrof_train=int(len(file_paths)*split_ratio)+1
    nrof_val=len(file_paths)-nrof_train
    nb_train=nrof_train*50000
    nb_val=nrof_val*50000
    AUTOTUNE = tf.data.experimental.AUTOTUNE
    train_raw_image_dataset = tf.data.TFRecordDataset(file_paths[0:nrof_train])
    val_raw_image_dataset = tf.data.TFRecordDataset(file_paths[nrof_train:])
    parsed_train_image_dataset = train_raw_image_dataset.map(parse_image_function)
    parsed_val_image_dataset = val_raw_image_dataset.map(parse_image_function)
    train_datasets = parsed_train_image_dataset.batch(batch_size).prefetch(AUTOTUNE)
    val_datasets = parsed_val_image_dataset.batch(batch_size).prefetch(AUTOTUNE)
    return train_datasets,val_datasets,nb_train,nb_val



def parse_arguments(argv):
    description = "options for generate tfrecords format data"
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument('--batch_size', type=int,
                        default=128, help='The size of batch')
    parser.add_argument('--train_datasets', type=str,
                        default='../datasets/vggface2/train', help='The direction where to save raw jpeg train images')
    parser.add_argument('--train_tfrcd', type=str,
                        default='../data/train_tfrcd', help='The direction where to save tfrecords format training data')
    parser.add_argument('--nrof_imgs_per_file', type=int,
                        default=50000, help='The number of images to write to single tfrecords file')
    return parser.parse_args(argv)


def main(args):
    create_tfrecord_data(args.train_datasets,args.train_tfrcd, args.nrof_imgs_per_file)


if __name__ == "__main__":
    args = parse_arguments(sys.argv[1:])
    main(args)
    print('hello')

    # create_tfrecord_data(src_dir='../datasets/vggface2/train',dest_dir='../train_tfrcd1')
    # train_ds=create_datasets_from_tfrecord('../train_tfrcd')
    # for images,ids in train_ds:
    #     pprint.pprint(images.shape)
    #     pprint.pprint(ids.shape)

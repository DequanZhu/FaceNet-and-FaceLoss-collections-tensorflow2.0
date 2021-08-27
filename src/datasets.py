# coding=utf-8
import glob
import sys
import tensorflow as tf
import os
import argparse
import numpy as np
import random
import pandas as pd
from progressbar import *
from utils import check_folder


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
    label = parsed_example['id']
    return image, label


def preprocess_image(image):
    image = tf.image.resize(image, IMG_SIZE)
    image = tf.image.random_flip_left_right(image)
    image /= 255.0
    return image



def create_tfrecord_data(src_dir, dest_dir,nrof_imgs_per_file,split_ratio=0.95):
    nrof_class=len(os.listdir(src_dir))
    image_paths = glob.glob(src_dir+'/*/*.jpg')
    nrof_total=len(image_paths)
    random.seed(0)
    random.shuffle(image_paths)
    image_ids = [path.split('/')[-2] for path in image_paths]
    image_ids = [int(label[1:]) for label in image_ids]
    nrof_train=int(nrof_total*split_ratio)
    nrof_val=nrof_total-nrof_train
    nrof_train_tfrcd_files = nrof_train//nrof_imgs_per_file + 1
    nrof_val_tfrcd_files = nrof_val//nrof_imgs_per_file + 1
    df=pd.DataFrame({'train_num':nrof_train,'val_num':nrof_val,'class_num':nrof_class},index=[0])
    check_folder(dest_dir)
    df.to_csv(os.path.join(dest_dir,'info.csv'),index=False)
    for i in range(nrof_train_tfrcd_files):
        save_dir=os.path.join(dest_dir,'train')
        check_folder(save_dir)
        save_path = os.path.join(save_dir, 'train_{:04d}.tfrecords'.format(int((i+1)*nrof_imgs_per_file)))
        start = max(0, i*nrof_imgs_per_file)
        end = min((i+1)*nrof_imgs_per_file, len(image_paths))
        write_to_tfrecord(save_path, image_paths[start:end], image_ids[start:end])

    for i in range(nrof_val_tfrcd_files):
        save_dir=os.path.join(dest_dir,'val')
        check_folder(save_dir)
        save_path = os.path.join(save_dir, 'val_{:04d}.tfrecords'.format(int((i+1)*nrof_imgs_per_file)))
        start = max(0, i*nrof_imgs_per_file)
        end = min((i+1)*nrof_imgs_per_file, len(image_paths))
        write_to_tfrecord(save_path, image_paths[start:end], image_ids[start:end])


def create_datasets_from_tfrecord(tfrcd_dir,batch_size=32,phase='train'):
    if phase=='train':
        df=pd.read_csv(os.path.join(tfrcd_dir,'info.csv'))
        nrof_samples=df['train_num'][0]
    if phase=='val':
        df=pd.read_csv(os.path.join(tfrcd_dir,'info.csv'))
        nrof_samples=df['val_num'][0]
    
    file_paths = os.listdir(os.path.join(tfrcd_dir,phase))
    file_paths = [os.path.join(tfrcd_dir, phase, file_path)for file_path in file_paths]
    AUTOTUNE = tf.data.experimental.AUTOTUNE
    raw_image_dataset = tf.data.TFRecordDataset(file_paths)
    parsed_image_dataset = raw_image_dataset.map(parse_image_function,num_parallel_calls=tf.data.experimental.AUTOTUNE)
    datasets = parsed_image_dataset.batch(batch_size).prefetch(AUTOTUNE)
    return datasets,nrof_samples



def parse_arguments(argv):
    description = "options for generate tfrecords format data"
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument('--batch_size', type=int,
                        default=128, help='The size of batch')
    parser.add_argument('--src_dir', type=str,
                        default='/home/zdq/facenet_zdq/datasets/vggface2/test', help='The direction where to save raw jpeg train images')
    parser.add_argument('--dest_dir', type=str,
                        default='/home/zdq/vgg_tfrcd/', help='The direction where to save tfrecords format training data')
    parser.add_argument('--nrof_imgs_per_file', type=int,
                        default=500, help='The number of images to write to single tfrecords file')
    return parser.parse_args(argv)


def main(args):
    create_tfrecord_data(args.src_dir,args.dest_dir, args.nrof_imgs_per_file)
    # train_data,train_num=create_datasets_from_tfrecord(tfrcd_dir=args.dest_dir,batch_size=512,phase='train')
    # for batch_examples, batch_id in train_data:
    #     print(batch_id)

if __name__ == "__main__":
    args = parse_arguments(sys.argv[1:])
    main(args)

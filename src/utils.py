import pathlib
import glob
import tensorflow as tf
import os
import random
BATCH_SIZE = 128
IMG_SIZE=[299,299]


def check_folder(dir_name):
    if not os.path.exists(dir_name):
        os.makedirs(dir_name)
    return dir_name
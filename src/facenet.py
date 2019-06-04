from __future__ import division
import argparse
import numpy as np
import tensorflow as tf
from tensorflow.python import keras
from tensorflow.python.keras import layers, optimizers, metrics, Sequential

model_sets = {'InceptionResNetV2': keras.applications.InceptionResNetV2,
              'Resnet50': keras.applications.ResNet50,
              'InceptionV3': keras.applications.InceptionV3
              }


class FaceNet():
    def __init__(self, opt, num_classes):
        super(FaceNet, self).__init__()
        self.model_name = opt.model_name
        self.model = self.create_model(self.model_name, opt.image_size, opt.embedding_size, num_classes)

    def create_model(self, model_name, input_size, embedding_size,num_classes, include_top=False):
        base_network = model_sets[model_name](input_shape=(input_size, input_size, 3),
                                              include_top=include_top,
                                              weights='imagenet')

        base_network.trainable = True
        inputs = base_network.input
        x = base_network(inputs)
        x = layers.GlobalAveragePooling2D()(x)
        embedding = layers.Dense(embedding_size,name='embedding')(x)
        logits = layers.Dense(num_classes,name='logits')(embedding)
        model = keras.Model(inputs=inputs, outputs={'embedding':embedding, 'logits':logits})
        model.summary()
        return model

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
    def __init__(self, opt):
        super(FaceNet, self).__init__()
        self.model = self.create_model(opt.backbone, opt.image_size, opt.embedding_size, opt.nrof_classes)

    def create_model(self, backbone, input_size, embedding_size,num_classes, include_top=False):
        base_network = model_sets[backbone](input_shape=(input_size, input_size, 3),
                                              include_top=include_top,
                                              weights='imagenet')

        base_network.trainable = True
        inputs = base_network.input
        x = base_network(inputs)
        x = layers.GlobalAveragePooling2D()(x)
        embedding = layers.Dense(embedding_size,name='embedding')(x)
        # logits = layers.Dense(num_classes,name='logits')(embedding)
        # model = keras.Model(inputs=inputs, outputs={'embedding':embedding, 'logits':logits})
        model = keras.Model(inputs=inputs, outputs=embedding)
        model.summary()
        return model
      
      
if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--model-name',type=str, default='Resnet50')
    parser.add_argument('--image-size',type=int, default=160)
    parser.add_argument('--num_classes',type=int, default=100)
    parser.add_argument('--embedding_size',type=int, default=512)
    opt = parser.parse_args()
    model=FaceNet(opt)
